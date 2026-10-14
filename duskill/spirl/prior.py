"""SPiRL Implementation""" 

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
from duskill.common.mlp_vae import VAEOutput, Encoder, Decoder, VariationalAutoEncoder
from duskill.common.lstm_vae import LSTMEncoder
from duskill.utils.buffers import SkillBuffer


class SPiRLPrior(SkillPrior):
    """Skill prior of SPiRL (Pertsch et al., 2020).

    Learns an LSTM skill encoder, an *open-loop* skill decoder that emits a whole
    h-step action sequence from a single latent z, and a state-conditioned prior p(z|s).
    """

    closed_loop = False

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        epochs: int,
        learning_rate: float = 3e-4,
        batch_size: int = 128,
        h: int = 10,
        beta: float = 1.0,
        embed_dim: int = 128,   # embed dim
        hidden_dim: int = 128,  # lstm dim
        latent_dim: int = 10,   # skill dim
        squash_z: bool = False, # squash z with tanh
        replay_buffer: Type[SkillBuffer] = None,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Callable[[jax.Array], jax.Array] = nn.leaky_relu,
        tensorboard_log: Optional[str] = None,
        wandb_log: Optional[str] = None,
        verbose: int = 0,
        _init_setup_model: bool = True,
        seed: int = 1,
    ):
        super(SPiRLPrior, self).__init__(
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
        
        if net_arch is None:
            net_arch = [dict(enc=[64, 64], dec=[64, 64], pri=[64, 64])]
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        
        # logger object
        if self.wandb_log is not None:
            wandb_config = dict(
                algo='spirl-prior',
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                h=self.h,
                beta=self.beta,
                net_arch=self.net_arch,
                latent_dim=self.latent_dim,
                seed=self.seed,
            )
            wandb.init(
                config=wandb_config,
                notes="SPiRL Skill Prior",
            )
            wandb.run.name = self.wandb_log

        if _init_setup_model:
            self._build(learning_rate)
    
    def _build_sk_vae(self) -> hk.Module:
        """Build skill encoder & decoder."""
        encoder = LSTMEncoder(
            self.embed_dim,
            self.hidden_dim,
            self.latent_dim, 
            self.activation_fn,
        )
        decoder = Decoder(
            self.net_arch['dec'],
            self.action_space.shape[-1] * self.h,
            self.activation_fn,
        )
        return VariationalAutoEncoder(
            encoder=encoder,
            decoder=decoder,
        )
   
    def _build_sk_pri(self) -> hk.Module:
        """Build skill prior."""
        prior = Encoder(
            self.net_arch['pri'],
            self.latent_dim,
            self.activation_fn
        )
        return prior

    def _build(self, learning_rate: float) -> None:

        def fn_sk_vae(
            obs: jax.Array, 
            act: jax.Array, 
            mean: jax.Array, 
            std: jax.Array, 
            deterministic: bool,
        ):
            sk_vae = self._build_sk_vae()
            return sk_vae(obs, act, None, mean, std, deterministic)
        
        sk_vae_params, self.sk_vae = hk.transform_with_state(fn_sk_vae)
        dummy_obs, dummy_act = get_dummy_obs(self.observation_space), get_dummy_act(self.action_space)
        dummy_obs_seq = jnp.repeat(dummy_obs, self.h, axis=0).reshape(1, self.h, -1)
        dummy_act_seq = jnp.repeat(dummy_act, self.h, axis=0).reshape(1, self.h, -1)
        
        self.sk_vae_params, self.sk_vae_state = sk_vae_params(
            next(self.rng), dummy_obs_seq, dummy_act_seq, None, None, False)
        self.sk_vae_optimizer = optax.adam(learning_rate=self._get_scheduler(self.epochs))
        self.sk_vae_optimizer_state = self.sk_vae_optimizer.init(self.sk_vae_params)
        
        def fn_sk_pri(obs: jax.Array):
            sk_pri = self._build_sk_pri()
            return sk_pri(obs)
        
        sk_pri_params, self.sk_pri = hk.transform_with_state(fn_sk_pri)
        self.sk_pri_params, self.sk_pri_state = sk_pri_params(next(self.rng), dummy_obs)
        self.sk_pri_optimizer = optax.adam(learning_rate=self._get_scheduler(self.epochs))
        self.sk_pri_optimizer_state = self.sk_pri_optimizer.init(self.sk_pri_params)
        
        # build normalization layer
        if self.norm_obs:
            self.norm_layer = RunningNormLayer(self.observation_space.shape)

    def learn(self, epochs: int, gradient_steps: int = 1, log_interval: int = None) -> None:
        self.start_time = time.time()
        
        self._n_epochs, self._n_updates = 0, 0
        for epoch in range(epochs):
            vae_losses, mse_losses, kl_losses = [], [], []
            pri_losses = []

            for gradient_step in range(gradient_steps):
                replay_data = self.replay_buffer.sample(self.batch_size) # [batch, h, s]
                # obtain skills
                observations = replay_data.observations
                actions = replay_data.actions
                
                if self.norm_obs:
                    observations = self.preprocess_obs(observations, training=True)

                # 1. learning skill vae
                self.sk_vae_optimizer_state, self.sk_vae_params, self.sk_vae_state, vae_loss, vae_info = jit_optimize_with_state(
                    self._sk_vae_loss,
                    self.sk_vae_optimizer,
                    self.sk_vae_optimizer_state,
                    self.sk_vae_params,
                    self.sk_vae_state,
                    None,
                    observations=observations,
                    actions=actions,
                    rng=next(self.rng)
                )
                
                vae_losses.append(np.array(vae_info["vae_loss"]))
                mse_losses.append(np.array(vae_info["mse_loss"]))
                kl_losses.append(np.array(vae_info["kl_loss"]))
                
                # 2. learning skill prior
                self.sk_pri_optimizer_state, self.sk_pri_params, self.sk_pri_state, pri_loss, pri_info = jit_optimize_with_state(
                    self._sk_pri_loss,
                    self.sk_pri_optimizer,
                    self.sk_pri_optimizer_state,
                    self.sk_pri_params,
                    self.sk_pri_state,
                    None,
                    observations=observations,
                    q_mean=stop_grad(vae_info['q_mean']),
                    q_std=stop_grad(vae_info['q_std']),
                    rng=next(self.rng)
                )
                pri_losses.append(np.array(pri_info["pri_loss"]))

            self._n_epochs += 1
            self._n_updates += gradient_steps
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/vae/loss", np.mean(vae_losses))
            self.logger.record("train/vae/mse_loss", np.mean(mse_losses))
            self.logger.record("train/vae/sk_kl_loss", np.mean(kl_losses))
            self.logger.record("train/vae/mse", np.mean(mse_losses))
            self.logger.record("train/pri/sk_loss", np.mean(pri_losses))
            
            if self.wandb_log is not None:
                wandb.log({
                    "train/vae/loss": np.mean(vae_losses),
                    "train/vae/mse_loss": np.mean(mse_losses),
                    "train/vae/sk_kl_loss": np.mean(kl_losses),
                    "train/vae/mse": np.mean(mse_losses),
                    "train/pri/sk_loss": np.mean(pri_losses),
                })

            if log_interval is not None and self._n_epochs % log_interval == 0:
                self._dump_logs()

    @partial(jax.jit, static_argnums=0)
    def _sk_vae(
        self, 
        obs: jax.Array, 
        act: jax.Array, 
        params: hk.Params, 
        state: hk.Params, 
        rng=None
    ) -> VAEOutput:
        return self.sk_vae(params, state, rng, 
            obs=obs, act=act, mean=None, std=None, deterministic=False)

    @partial(jax.jit, static_argnums=0)
    def _sk_vae_with_emb(
        self, 
        mean: jax.Array, 
        std: jax.Array, 
        params: hk.Params, 
        state: hk.Params, 
        rng=None
    ) -> VAEOutput:
        return self.sk_vae(params, state, rng, 
            obs=None, act=None, mean=mean, std=std, deterministic=False)
    
    @partial(jax.jit, static_argnums=0)
    def _sk_dec(
        self, 
        mean: jax.Array, 
        std: jax.Array, 
        params: hk.Params,
        state: hk.Params, rng=None
    ) -> VAEOutput:
        return self.sk_vae(params, state, rng, 
            obs=None, act=None, mean=mean, std=std, deterministic=True)

    @partial(jax.jit, static_argnums=0)
    def _sk_pri(
        self, 
        obs: jax.Array, 
        params: hk.Params, 
        state: hk.Params, rng=None
    ) -> Tuple[jax.Array, jax.Array]:
        return self.sk_pri(params, state, rng, obs)

    @partial(jax.jit, static_argnums=0)
    def _sk_vae_loss(
        self,
        params: hk.Params,
        state: hk.Params,
        observations: jax.Array,
        actions: jax.Array,
        rng=None,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """ELBO loss: E_p[log(x)] - KL(d||q), where q ~ N(0,1)."""
        batch_size = observations.shape[0]
        
        (mean, std, logits), new_state = self._sk_vae(observations, actions, params, state, rng)

        # reconstruction term: use mse loss to approximate log-likelihood
        actions = actions.reshape(batch_size, -1)
        mse_loss = jnp.mean(jnp.square(actions - logits), axis=-1) 
        mse_loss = jnp.mean(mse_loss)

        # regularization term: KL divergence between Gaussians N(mean, std) & N(0,1)
        kl_loss = 0.5 * jnp.mean(-jnp.log(jnp.square(std)) - 1. + jnp.square(std) + jnp.square(mean), axis=-1)
        kl_loss = self.beta * jnp.mean(kl_loss)

        loss = mse_loss + kl_loss

        info = {
            'vae_loss': loss,
            'mse_loss': mse_loss,
            'kl_loss': kl_loss,
            'q_mean': mean,
            'q_std': std,
            'actions': actions,
            'logits': logits,
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
        """KL loss between skill prior and skill encoder """
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
            'p_mean': p_mean,
            'p_std': p_std,
        }
        return loss, (new_state, info)
    
    def predict_prior(self, obs: jax.Array) -> Tuple[jax.Array, jax.Array]:
        if self.norm_obs:
            obs = self.preprocess_obs(obs)
        (mean, std), _ = self._sk_pri(obs, self.sk_pri_params, self.sk_pri_state, next(self.rng))
        return mean, std
    
    def predict_reset(self):
        return

    def predict(self, obs: jax.Array, emb: jax.Array = None, deterministic: bool = True):
        if obs is not None and len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
        if emb is not None and len(emb.shape) == 1:
            emb = emb.reshape(1, -1)

        if self.norm_obs:
            obs = self.preprocess_obs(obs)

        # use prior embedding + decoder actions
        if emb is None:
            (mean, std), _ = self._sk_pri(obs, self.sk_pri_params, self.sk_pri_state, next(self.rng))
            (_, _, actions), _ = self._sk_dec(mean, std, self.sk_vae_params, self.sk_vae_state, next(self.rng))
        else:
            (mean, std, actions), _ = self._sk_dec(emb, None, self.sk_vae_params, self.sk_vae_state, next(self.rng))
        
        actions = np.array(actions)
        info = {
            'p_mean': np.array(mean),
            'p_std': np.array(std)
        }
        return actions, None, info  # [action_size * h], do not reshape 

    def _test(self, verbose: bool = False) -> None: 
        replay_data = self.replay_buffer.sample(1)
    
        if self.norm_obs:
            observations = self.preprocess_obs(replay_data.observations, training=False)

        loss, (_, info) = self._sk_vae_loss(
            self.sk_vae_params, 
            self.sk_vae_state, 
            observations, 
            replay_data.actions, 
            next(self.rng)
        )        
        prior_action, _, p_info = self.predict(observations[:,0,:])

        if verbose:
            np.set_printoptions(precision=3, suppress=True)
            print("=" * 20)
            print(f"GRT: {info['actions']}")
            print(f"ENC: {info['logits']} \n~ {info['q_mean']}")
            print(f"PRI: {prior_action} \n~ {p_info['p_mean']}")
            print(f"MSE Loss: {info['mse_loss']:.5f}")
            print(f"KL  Loss: {info['kl_loss']}")
            print(f"VAE Loss: {info['vae_loss']:.5f}")
            print("=" * 20)

    def save(self, path: str):
        data = self.__dict__.copy()
        exclude = self._excluded_save_data()
        for param_name in exclude:
            data.pop(param_name, None)

        params_dict = {}
        params_dict['sk_vae_params'] = self.sk_vae_params
        params_dict['sk_vae_state'] = self.sk_vae_state
        params_dict['sk_pri_params'] = self.sk_pri_params
        params_dict['sk_pri_state'] = self.sk_pri_state
        save_to_zip_file(path, data=data, params=params_dict, verbose=1)
        
        if self.norm_obs:
            self.norm_layer.save(path + '/norm_layer')

    @classmethod
    def load(cls, path: str):
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
        model._build(data['learning_rate'])

        model.sk_vae_params = params['sk_vae_params']
        model.sk_vae_state = params['sk_vae_state']
        model.sk_pri_params = params['sk_pri_params']
        model.sk_pri_state = params['sk_pri_state']
        
        if model.norm_obs:
            model.norm_layer = model.norm_layer.load(path + '/norm_layer')

        return model
