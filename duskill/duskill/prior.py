"""DuSkill Prior Implementation""" 

from functools import partial
from typing import Dict, List, Optional, Tuple, Type, Union, Callable

import time
import gym
import numpy as np
import jax
import jax.numpy as jnp
from jax import nn
import haiku as hk
import optax
import wandb

from sb3_jax.common.norm_layers import RunningNormLayer
from sb3_jax.common.utils import get_dummy_obs, get_dummy_act
from sb3_jax.common.save_util import load_from_zip_file, save_to_zip_file
from sb3_jax.common.jax_utils import jit_optimize_with_state, stop_grad

from duskill.common.prior import SkillPrior
from duskill.common.mlp_vae import Encoder
from duskill.common.lstm_vae import LSTMEncoder
from duskill.common.du_vae import (
    ddpm_schedules,
    MLPDiffusionModel, 
    HGDAEOutput,
    PrefEncoder,
    PrefPrior,
    GuidedDiffusionDecoder,
    HierarchicalGuidedDiffusionAutoEncoder,
)
from duskill.utils.buffers import SkillBuffer


class DuSkillPrior(SkillPrior):
    """Stage-1 DuSkill model: a guided-diffusion skill/preference prior.

    Trains three components on an offline multi-preference dataset:
      - `sk_dae`: LSTM skill encoder + guided diffusion action decoder,
      - `sk_pri`: skill prior p(sk_z | s_1),
      - `pf_pri`: preference prior p(pf_z | sk_mean).
    The auto-encoder loss is `diffusion loss + beta * sk_kl + pf_beta * pf_kl`;
    each prior is fit with its own KL to the corresponding (detached) posterior.
    """
    # `predict` emits one primitive action per call (no h-step action sequence).
    closed_loop = True

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        epochs: int,
        learning_rate: float = 3e-4,
        batch_size: int = 128,
        h: int = 10,
        beta: float = 1e-3,
        pf_beta: float = 5e-4,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        latent_dim: int = 10,
        squash_z: bool = False,
        replay_buffer: Type[SkillBuffer] = SkillBuffer,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Callable[[jax.Array], jax.Array] = nn.leaky_relu,
        # diffusion parameters
        n_denoise: int = 50,
        betas: Tuple[float,...] = (1e-4, 0.02),
        is_linear: bool = True,
        guide_wei: float = 0.1,
        drop_rate: float = 0.1,
        # # # # # # # # # # # 
        tensorboard_log: Optional[str] = None,
        wandb_log: Optional[str] = None,
        verbose: int = 0,
        _init_setup_model: bool = True,
        seed: int = 1,
    ):
        super(DuSkillPrior, self).__init__(
            epochs,
            observation_space,
            action_space,
            learning_rate,
            batch_size,
            h,
            beta,
            embed_dim,
            hidden_dim,
            latent_dim,
            squash_z,
            replay_buffer,
            norm_obs=True,
            tensorboard_log=tensorboard_log,
            wandb_log=wandb_log,
            verbose=verbose,
            seed=seed,
        )
        
        self.pf_beta = pf_beta
        if net_arch is None:
            net_arch = dict(enc=[128,128], dec=[128,128], pri=[128,128])
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        # diffusion noise is same as action length
        self.noise_dim = self.action_space.shape[-1] 
        

        # diffusion model parameters
        self.n_denoise = n_denoise
        self.betas = betas
        self.is_linear = is_linear
        self.ddpm_dict = ddpm_schedules(
            self.betas[0], self.betas[1], self.n_denoise, self.is_linear)  
        self.guide_wei = guide_wei
        self.drop_rate = drop_rate
        
        # for prediction 
        self.p_h = 0
        self.p_obs = None
        self.p_sk_emb = None
        self.p_pf_emb = None

        # logger object
        if self.wandb_log is not None:
            wandb_config = dict(
                algo='duskill-prior',
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                h=self.h,
                beta=self.beta,
                pf_beta=self.pf_beta,
                net_arch=self.net_arch,
                latent_dim=self.latent_dim,
                noise_dim=self.noise_dim,
                squash_z=self.squash_z,
                n_denoise=self.n_denoise,
                guide_wei=self.guide_wei,
                drop_rate=self.drop_rate,
                betas=self.betas,
                is_linear=self.is_linear,
                seed=self.seed,
            )
            wandb.init(
                config=wandb_config,
                notes="DuSkill Skill Prior",
            )
            wandb.run.name = self.wandb_log
       
        if _init_setup_model:
            self._build(learning_rate)

    def _build_sk_dae(self) -> hk.Module:
        """Build skill encoder & diffusion based decoder."""
        # Build Skill Encoder 
        sk_encoder = LSTMEncoder(
            self.embed_dim,
            self.hidden_dim,
            self.latent_dim,
            self.activation_fn
        )
        # Builde Pref Encoder
        pf_encoder = PrefEncoder(
            self.net_arch['enc'],
            self.embed_dim,
            self.latent_dim,
            self.activation_fn
        )
        # Build Decoder
        sk_du = MLPDiffusionModel(
            self.net_arch['dec'],
            self.embed_dim,
            self.action_space.shape[-1],
            self.activation_fn,
        )
        pf_du = MLPDiffusionModel(
            self.net_arch['dec'],
            self.embed_dim,
            self.action_space.shape[-1],
            self.activation_fn,
        )
        
        decoder = GuidedDiffusionDecoder(
            sk_du, 
            pf_du,
            self.n_denoise,
            self.guide_wei,
            self.drop_rate,
            self.ddpm_dict,
        )

        return HierarchicalGuidedDiffusionAutoEncoder(
            sk_encoder=sk_encoder,
            pf_encoder=pf_encoder,
            decoder=decoder,
            squash_z=self.squash_z,
        )

    def _build_sk_pri(self) -> hk.Module:
        """Build skill prior."""
        return Encoder(
            self.net_arch['pri'],
            self.latent_dim,
            self.activation_fn,
        )

    def _build_pf_pri(self) -> hk.Module:
        """Build pref prior."""
        return PrefPrior(
            self.net_arch['pri'],
            self.embed_dim,
            self.latent_dim,
            self.activation_fn,
        )

    def _build(self, learning_rate: float) -> None: 
        
        def fn_sk_dae(
            obs: jax.Array,
            act: jax.Array,
            obt: jax.Array, 
            x_t: jax.Array, # diffusion input x_t
            t: jax.Array,   # denoise timestep
            p: jax.Array,   
            denoise: bool,  # run denoise
            sk_mean: jax.Array, 
            sk_std: jax.Array,
            pf_mean: jax.Array,
            pf_std: jax.Array,
            deterministic: bool,
        ): 
            sk_dae = self._build_sk_dae()
            return sk_dae(obs, act, obt, x_t, t, p, denoise, sk_mean, sk_std, pf_mean, pf_std, deterministic)
        
        sk_dae_params, self.sk_dae = hk.transform_with_state(fn_sk_dae)
        dummy_obs, dummy_act = get_dummy_obs(self.observation_space), get_dummy_act(self.action_space)
        dummy_obs_seq = jnp.repeat(dummy_obs, self.h, axis=0).reshape(1, self.h, -1)
        dummy_act_seq = jnp.repeat(dummy_act, self.h, axis=0).reshape(1, self.h, -1)
        dummy_x_t = jax.random.normal(next(self.rng), shape=(1, self.noise_dim))
        dummy_t = jnp.array([[1 / self.n_denoise]])
        dummy_p = jnp.array([[1 / 10]]) 
        
        self.sk_dae_params, self.sk_dae_state = sk_dae_params(
            next(self.rng), dummy_obs_seq, dummy_act_seq, dummy_obs, dummy_x_t, dummy_t, dummy_p, 
            False, None, None, None, None, False)
        self.sk_dae_optimizer = optax.adam(learning_rate=self._get_scheduler(self.epochs))
        self.sk_dae_optimizer_state = self.sk_dae_optimizer.init(self.sk_dae_params)
        
        def fn_sk_pri(obs: jax.Array):
            sk_pri = self._build_sk_pri()
            return sk_pri(obs)
        
        sk_pri_params, self.sk_pri = hk.transform_with_state(fn_sk_pri)
        self.sk_pri_params, self.sk_pri_state = sk_pri_params(next(self.rng), dummy_obs)
        self.sk_pri_optimizer = optax.adam(learning_rate=self._get_scheduler(self.epochs))
        self.sk_pri_optimizer_state = self.sk_pri_optimizer.init(self.sk_pri_params)
        
        def fn_pf_pri(p: jax.Array):
            pf_pri = self._build_pf_pri()
            return pf_pri(p)
        
        dummy_pf_pri = jax.random.normal(next(self.rng), shape=(1, self.latent_dim))
        pf_pri_params, self.pf_pri = hk.transform_with_state(fn_pf_pri)
        self.pf_pri_params, self.pf_pri_state = pf_pri_params(next(self.rng), dummy_pf_pri)
        self.pf_pri_optimizer = optax.adam(learning_rate=self._get_scheduler(self.epochs))
        self.pf_pri_optimizer_state = self.pf_pri_optimizer.init(self.pf_pri_params)

        # build normalization layer
        if self.norm_obs:
            self.norm_layer = RunningNormLayer(self.observation_space.shape)

    def learn(self, epochs: int, gradient_steps: int = 1, log_interval: int = None) -> None:
        
        self.start_time = time.time()
        self._n_epochs, self._n_updates = 0, 0

        for epoch in range(epochs):
            dae_losses, du_losses, sk_kl_losses, pf_kl_losses = [], [], [], []
            sk_pri_losses, pf_pri_losses = [], []

            for gradient_step in range(gradient_steps):
                replay_data = self.replay_buffer.sample(self.batch_size) # [batch, h, s]
                observations = replay_data.observations
                actions = replay_data.actions
                prefs = replay_data.prefs

                if self.norm_obs:
                    observations = self.preprocess_obs(observations, training=True)

                # 1. learning skill dae
                self.sk_dae_optimizer_state, self.sk_dae_params, self.sk_dae_state, dae_loss, dae_info = jit_optimize_with_state(
                    self._sk_dae_loss,
                    self.sk_dae_optimizer,
                    self.sk_dae_optimizer_state,
                    self.sk_dae_params,
                    self.sk_dae_state,
                    None,
                    observations=observations,
                    actions=actions,      
                    observations_t=observations,
                    actions_t=actions,
                    prefs=prefs,
                    rng=next(self.rng),
                )
                
                dae_losses.append(np.array(dae_info['dae_loss']))
                du_losses.append(np.array(dae_info['du_loss']))
                sk_kl_losses.append(np.array(dae_info['sk_kl_loss']))
                pf_kl_losses.append(np.array(dae_info['pf_kl_loss']))

                # 2. learning skill prior
                self.sk_pri_optimizer_state, self.sk_pri_params, self.sk_pri_state, sk_pri_loss, sk_pri_info = jit_optimize_with_state(
                    self._sk_pri_loss,
                    self.sk_pri_optimizer,
                    self.sk_pri_optimizer_state,
                    self.sk_pri_params,
                    self.sk_pri_state,
                    None,
                    observations=observations,
                    q_mean=stop_grad(dae_info['q_sk_mean']),
                    q_std=stop_grad(dae_info['q_sk_std']),
                    rng=next(self.rng)
                )
                sk_pri_losses.append(np.array(sk_pri_info['pri_loss']))
                
                # 3. learning pref prior
                self.pf_pri_optimizer_state, self.pf_pri_params, self.pf_pri_state, pf_pri_loss, pf_pri_info = jit_optimize_with_state(
                    self._pf_pri_loss,
                    self.pf_pri_optimizer,
                    self.pf_pri_optimizer_state,
                    self.pf_pri_params,
                    self.pf_pri_state,
                    None,
                    sk_mean=stop_grad(dae_info['q_sk_mean']),
                    q_mean=stop_grad(dae_info['q_pf_mean']),
                    q_std=stop_grad(dae_info['q_pf_std']),
                    rng=next(self.rng)
                )
                pf_pri_losses.append(np.array(pf_pri_info['pri_loss']))

            self._n_epochs += 1
            self._n_updates += gradient_steps
            
            # checking action mse ...
            if self._n_updates % 100 == 0:
                observations = stop_grad(observations)
                observations = observations.reshape(self.batch_size * self.h, -1)
                actions = actions.reshape(self.batch_size * self.h, -1)
                # action reconstruction : encoder -> decoder
                (_, _, _, _, dae_actions, _), _ = self._sk_dec( \
                    observations, dae_info['q_sk_mean'], dae_info['q_sk_std'], 
                    dae_info['q_pf_mean'], dae_info['q_pf_std'], 
                    self.sk_dae_params, self.sk_dae_state, next(self.rng))
                dae_mse = jnp.mean(jnp.square(dae_actions - actions))
                # action reconstruction : prior -> decoder
                (_, _, _, _, pri_actions, _), _ = self._sk_dec( \
                    observations, sk_pri_info['p_sk_mean'], sk_pri_info['p_sk_std'], 
                    pf_pri_info['p_pf_mean'], pf_pri_info['p_pf_std'], 
                    self.sk_dae_params, self.sk_dae_state, next(self.rng))
                pri_mse = jnp.mean(jnp.square(pri_actions - actions))
                dae_mse, pri_mse = float(dae_mse), float(pri_mse)
                self.logger.record("train/vae/mse", dae_mse)
                self.logger.record("train/pri/mse", pri_mse)
                if self.wandb_log is not None:
                    wandb.log({"train/vae/mse": dae_mse})
                    wandb.log({"train/pri/mse": pri_mse})

            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/vae/loss", np.mean(dae_losses))
            self.logger.record("train/vae/du_loss", np.mean(du_losses))
            self.logger.record("train/vae/sk_kl_loss", np.mean(sk_kl_losses))
            self.logger.record("train/vae/pf_kl_loss", np.mean(pf_kl_losses))
            self.logger.record("train/pri/sk_loss", np.mean(sk_pri_losses))
            self.logger.record("train/pri/pf_loss", np.mean(pf_pri_losses))

            if self.wandb_log is not None:
                wandb.log({
                    "train/vae/loss": np.mean(dae_losses),
                    "train/vae/du_loss": np.mean(du_losses),
                    "train/vae/sk_kl_loss": np.mean(sk_kl_losses),
                    "train/vae/pf_kl_loss": np.mean(pf_kl_losses),
                    "train/pri/sk_loss": np.mean(sk_pri_losses),
                    "train/pri/pf_loss": np.mean(pf_pri_losses),
                })

            if log_interval is not None and self._n_epochs % log_interval == 0:
                self._dump_logs()

    @partial(jax.jit, static_argnums=0)
    def _sk_dae(
        self,
        obs: jax.Array, 
        act: jax.Array,
        obt: jax.Array,
        x_t: jax.Array, 
        t: jax.Array,
        p: jax.Array,
        params: hk.Params, 
        state: hk.Params,
        rng=None,
    ) -> HGDAEOutput:
        return self.sk_dae(params, state, rng, 
            obs=obs, act=act, obt=obt, x_t=x_t, t=t, p=p, denoise=False, 
            sk_mean=None, sk_std=None, pf_mean=None, pf_std=None, deterministic=False)
    
    @partial(jax.jit, static_argnums=0)
    def _sk_dae_with_emb(
        self,
        obt: jax.Array,
        x_t: jax.Array,
        t: jax.Array,
        sk_mean: jax.Array,
        sk_std: jax.Array,
        pf_mean: jax.Array,
        pf_std: jax.Array,
        params: hk.Params,
        state: hk.Params,
        rng=None,
    ) -> HGDAEOutput:
        return self.sk_dae(params, state, rng,
            obs=None, act=None, obt=obt, x_t=x_t, t=t, p=None, denoise=False,
            sk_mean=sk_mean, sk_std=sk_std, pf_mean=pf_mean, pf_std=pf_std, deterministic=False)

    @partial(jax.jit, static_argnums=0)
    def _sk_dec(
        self, 
        obt: jax.Array, 
        sk_mean: jax.Array, 
        sk_std: jax.Array,
        pf_mean: jax.Array,
        pf_std: jax.Array,
        params: hk.Params, 
        state: hk.Params, 
        rng=None,
    ) -> HGDAEOutput:
        return self.sk_dae(params, state, rng, 
            obs=None, act=None, obt=obt, x_t=None, t=None, p=None, denoise=True, 
            sk_mean=sk_mean, sk_std=sk_std, pf_mean=pf_mean, pf_std=pf_std, deterministic=True)

    @partial(jax.jit, static_argnums=0)
    def _sk_dec_with_x_t(
        self, 
        obt: jax.Array,
        x_t: jax.Array,
        sk_mean: jax.Array, 
        sk_std: jax.Array,
        pf_mean: jax.Array,
        pf_std: jax.Array,
        params: hk.Params, 
        state: hk.Params, 
        rng=None,
    ) -> HGDAEOutput:
        return self.sk_dae(params, state, rng, 
            obs=None, act=None, obt=obt, x_t=x_t, t=None, p=None, denoise=True, 
            sk_mean=sk_mean, sk_std=sk_std, pf_mean=pf_mean, pf_std=pf_std, deterministic=True)

    @partial(jax.jit, static_argnums=0)
    def _sk_pri(
        self, 
        obs: jax.Array, 
        params: hk.Params, 
        state: hk.Params, 
        rng=None,
    ) -> Tuple[jax.Array, jax.Array]:
        return self.sk_pri(params, state, rng, obs)
    
    @partial(jax.jit, static_argnums=0)
    def _pf_pri(
        self,
        prefs: jax.Array,
        params: hk.Params,
        state: hk.Params,
        rng=None,
    ) -> Tuple[jax.Array, jax.Array]:
        return self.pf_pri(params, state, rng, prefs)

    @partial(jax.jit, static_argnums=0)
    def _sk_dae_loss(
        self,
        params: hk.Params,
        state: hk.Params,
        observations: jax.Array, # h+1 
        actions: jax.Array,
        observations_t: jax.Array, # h
        actions_t: jax.Array,
        prefs: jax.Array, # h
        rng=None,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        
        batch_size = observations.shape[0]
        
        # preference, retrieve first one
        prefs = prefs[:,0,:]    
        
        # The skill encoder consumes the [batch, h, dim] sequences as-is; the
        # latents are repeated h times inside the model instead of here.

        # These are for target action_t & conditional value
        # [batch, h, obs_dim] -> [batch * h , act_dim] 
        actions_t = actions_t.reshape(batch_size * self.h, -1)
        observations_t = observations_t.reshape(batch_size * self.h, -1) 

        # independent keys for the diffusion noise, the timestep draw and the
        # model's own reparameterization sampling
        k_noise, k_ts, k_model = jax.random.split(rng, 3)

        # sample random noise
        noise = jax.random.normal(k_noise, shape=(batch_size * self.h, actions_t.shape[-1]))
        # sample timestep t
        _ts = jax.random.randint(k_ts, (batch_size * self.h, 1), minval=1, maxval=self.n_denoise+1)
        # get x_t
        x_t = self.ddpm_dict['sqrtab'][_ts] * actions_t + self.ddpm_dict['sqrtmab'][_ts] * noise
        
        (sk_mean, sk_std, pf_mean, pf_std, noise_pred, _), new_state = self._sk_dae(
            obs=observations, act=actions, obt=observations_t, x_t=x_t, t=_ts / self.n_denoise, p=prefs,
            params=params, state=state, rng=k_model)
        
        # diffion loss: run diffusion model to predict noise
        du_loss = jnp.mean(jnp.square(noise - noise_pred), axis=-1)
        du_loss = jnp.mean(du_loss)

        # regularization term: skill encoder, pref encoder 
        sk_kl_loss = 0.5 * jnp.mean(-jnp.log(jnp.square(sk_std)) - 1. + jnp.square(sk_std) + jnp.square(sk_mean), axis=-1)
        pf_kl_loss = 0.5 * jnp.mean(-jnp.log(jnp.square(pf_std)) - 1. + jnp.square(pf_std) + jnp.square(pf_mean), axis=-1)
        sk_kl_loss, pf_kl_loss = jnp.mean(sk_kl_loss), jnp.mean(pf_kl_loss)
        kl_loss = self.beta * sk_kl_loss + self.pf_beta * pf_kl_loss

        loss = du_loss + kl_loss

        info = {
            'dae_loss': loss,
            'du_loss': du_loss,
            'sk_kl_loss': sk_kl_loss,
            'pf_kl_loss': pf_kl_loss,
            'q_sk_mean': sk_mean,
            'q_sk_std': sk_std,
            'q_pf_mean': pf_mean,
            'q_pf_std': pf_std,
            'actions': actions,
            'logits': noise_pred,
        }
        return loss, (new_state, info)
    
    @partial(jax.jit, static_argnums=0)
    def _sk_pri_loss(
        self,
        params: hk.Params,
        state: hk.Params,
        observations: jax.Array, 
        q_mean: jax.Array, 
        q_std: jax.Array, 
        rng=None,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """KL loss between skill prior and skill encoder."""
        observations = observations[:,0,:] # retrieve s_1
        
        # skill prior output
        (p_mean, p_std), new_state = self._sk_pri(observations, params, state, rng)
        
        # KL divergence kl(q,p)
        kl_loss = jnp.mean((jnp.log(p_std) -jnp.log(q_std)) + \
                ((jnp.square(q_std) + jnp.square(q_mean - p_mean)) / (2 * jnp.square(p_std))) - 0.5, axis=-1)
        
        loss = jnp.mean(kl_loss)

        info = {
            'pri_loss': loss,
            'kl_loss': jnp.mean(kl_loss),
            'p_sk_mean': p_mean,
            'p_sk_std': p_std,
        }
        return loss, (new_state, info)
    
    @partial(jax.jit, static_argnums=0)
    def _pf_pri_loss(
        self,
        params: hk.Params,
        state: hk.Params,
        sk_mean: jax.Array,
        q_mean: jax.Array,
        q_std: jax.Array,
        rng=None,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """ KL loss between pref prior and pref encoder."""        
        # pref prior output
        (p_mean, p_std), new_state = self._pf_pri(sk_mean, params, state, rng)

        #  KL divergence kl(q,p)
        kl_loss = jnp.mean((jnp.log(p_std) - jnp.log(q_std)) + \
            ((jnp.square(q_std) + jnp.square(q_mean - p_mean)) / (2 * jnp.square(p_std))) - 0.5, axis=-1)
        
        loss = jnp.mean(kl_loss)

        info = {
            'pri_loss': loss,
            'kl_loss': jnp.mean(kl_loss),
            'p_pf_mean': p_mean,
            'p_pf_std': p_std,
        }
        return loss, (new_state, info)

    def predict_prior(self, obs: jax.Array) -> Tuple[jax.Array, jax.Array]:
        if self.norm_obs:
            obs = self.preprocess_obs(obs)
        (sk_mean, sk_std), _ = self._sk_pri(obs, self.sk_pri_params, self.sk_pri_state, next(self.rng))
        (pf_mean, pf_std), _ = self._pf_pri(sk_mean, self.pf_pri_params, self.pf_pri_state, next(self.rng))
        return (sk_mean, sk_std, pf_mean, pf_std)

    def predict_reset(self) -> None:
        self.p_h = 0

    def predict(self, obs: jax.Array, sk_emb: jax.Array = None, pf_emb: jax.Array = None, deterministic: bool = True):
        if obs is not None and len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
        if sk_emb is not None and len(sk_emb.shape) == 1:
            sk_emb = sk_emb.reshape(1, -1)
        if pf_emb is not None and len(pf_emb.shape) == 1:
            pf_emb = pf_emb.reshape(1, -1)

        if self.norm_obs:
            obs = self.preprocess_obs(obs)    

        if self.p_h == 0: # fisrt step
            self.p_obs = obs
            if sk_emb is not None:
                self.p_sk_emb = sk_emb
            if pf_emb is not None:
                self.p_pf_emb = pf_emb

        if sk_emb is None:
            (sk_mean, sk_std), _ = self._sk_pri(self.p_obs, self.sk_pri_params, self.sk_pri_state, next(self.rng)) # obs_1
            (pf_mean, pf_std), _ = self._pf_pri(sk_mean, self.pf_pri_params, self.pf_pri_state, next(self.rng))
            (_, _, _, _, actions, action_infos), _ = self._sk_dec(
                obs, sk_mean, sk_std, pf_mean, pf_std, self.sk_dae_params, self.sk_dae_state, next(self.rng)) # obs_t, z
        else:
            (sk_mean, sk_std, pf_mean, pf_std, actions, action_infos), _ = self._sk_dec(
                obs, self.p_sk_emb, None, self.p_pf_emb, None,
                self.sk_dae_params, self.sk_dae_state, next(self.rng))
        
        self.p_h += 1
        if self.p_h == self.h: # reset step
            self.p_h = 0
        
        actions = np.array(actions)
        # sk_std / pf_std are None when the caller supplies the embeddings directly
        info = {
            'p_sk_mean': np.array(sk_mean),
            'p_sk_std': None if sk_std is None else np.array(sk_std),
            'p_pf_mean': np.array(pf_mean),
            'p_pf_std': None if pf_std is None else np.array(pf_std),
        }
        return actions, None, info

    def _test(self, verbose: bool = False) -> None:
        replay_data = self.replay_buffer.sample(batch_size=10)
        observations = replay_data.observations
        
        if self.norm_obs:
            observations = self.preprocess_obs(observations)

        loss, (_, info) = self._sk_dae_loss(
            self.sk_dae_params, 
            self.sk_dae_state, 
            observations,
            replay_data.actions, 
            observations,
            replay_data.actions,
            replay_data.prefs,
            next(self.rng)
        )
        for i in range(self.h):
            prior_action, _, p_info = self.predict(replay_data.observations[:,i,:])
            mse = jnp.mean(jnp.square(info['actions'][:,i,:] - prior_action))
            print(f"GRT: {info['actions'][:,i,:]}")
            print(f"PRD: {prior_action}")
            print(f"MSE: {mse}")
        self.p_h = 0 

        if verbose:
            np.set_printoptions(precision=3, suppress=True)
            print("=" * 20)
            print(f"ENC: ~ {info['q_sk_mean'][0]} \n ~ {info['q_sk_std'][0]}")
            print(f"PRI: ~ {p_info['p_sk_mean'][0]} \n ~ {p_info['p_sk_std'][0]}")
            print(f"MSE Loss: {info['du_loss']:.5f}")
            print(f"KL  Loss: {info['sk_kl_loss']}")
            print(f"VAE Loss: {info['dae_loss']:.5f}")
            print("=" * 20)

    def save(self, path: str):
        data = self.__dict__.copy()
        exclude = self._excluded_save_data()
        for param_name in exclude:
            data.pop(param_name, None)

        params_dict = {}
        params_dict['sk_dae_params'] = self.sk_dae_params
        params_dict['sk_dae_state'] = self.sk_dae_state
        params_dict['sk_pri_params'] = self.sk_pri_params
        params_dict['sk_pri_state'] = self.sk_pri_state
        params_dict['pf_pri_params'] = self.pf_pri_params
        params_dict['pf_pri_state'] = self.pf_pri_state
        save_to_zip_file(path, data=data, params=params_dict, verbose=1)

        if self.norm_obs:
            self.norm_layer.save(path + '/norm_layer')
    
    @classmethod
    def load(cls, path: str, drop_rate: float = None):
        data, params = load_from_zip_file(
            path,
            custom_objects=None,
            verbose=1,
            print_system_info=True,
        )

        model = cls(
            observation_space=data['observation_space'],
            action_space=data['action_space'],
            epochs=None,
            _init_setup_model=False,
        )

        model.__dict__.update(data)
        if drop_rate is not None:
            model.drop_rate = drop_rate
        model._build(data['learning_rate'])

        model.sk_dae_params = params['sk_dae_params']
        model.sk_dae_state = params['sk_dae_state']
        model.sk_pri_params = params['sk_pri_params']
        model.sk_pri_state = params['sk_pri_state']
        model.pf_pri_params = params['pf_pri_params']
        model.pf_pri_state = params['pf_pri_state']
        if model.norm_obs:
            model.norm_layer = model.norm_layer.load(path + '/norm_layer')
        return model
