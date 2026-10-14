from functools import partial
from typing import Dict, List, Optional, Tuple, Type, Union, Callable
from copy import deepcopy

import time
import wandb
import gym
import numpy as np
import jax
import jax.numpy as jnp
from jax import nn
import haiku as hk
import optax

from sb3_jax.common.norm_layers import RunningNormLayer
from sb3_jax.common.utils import get_dummy_obs
from sb3_jax.common.save_util import load_from_zip_file, save_to_zip_file
from sb3_jax.common.jax_utils import jit_optimize_with_state

from duskill.duskill.prior import DuSkillPrior
from duskill.common.imitation import SkillIm
from duskill.common.mlp_vae import Encoder
from duskill.common.du_vae import (
    HGDAEOutput,
    PrefPrior, 
    MLPDiffusionModel,
    GuidedDiffusionDecoder,
    HierarchicalGuidedDiffusionAutoEncoder,
)
from duskill.utils.buffers import SkillBuffer
from duskill.utils.evaluations import evaluate_skill_policy


class DuSkillIM(SkillIm):
    """Stage-2 DuSkill model: few-shot imitation on top of a frozen prior.

    A small skill head and pref head (warm-started from the prior's `sk_pri` /
    `pf_pri`) are fit by behaviour cloning through the diffusion decoder, using
    the same noise-prediction loss as stage 1. With `tune_dec` the decoder is
    rebuilt locally so it can be fine-tuned as well (see `tune_dec_type`).
    """
    # `predict` emits one primitive action per call (no h-step action sequence).
    closed_loop = True

    def __init__(
        self,
        env: gym.Env,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        epochs: int = None,
        learning_rate: float = 3e-4,
        batch_size: int = 128,
        replay_buffer: Type[SkillBuffer] = None,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Callable[[jax.Array], jax.Array] = nn.leaky_relu,
        # Pretrained Skill Prior
        prior_path: str = None, # path for skill prior
        drop_rate: float = 0.0,
        tune_dec: bool = False,
        tune_dec_type: int = 0, # 1: tune pref decoder only, 2: tune both decoders
        # # # # # # # # # # # #
        eval_interval: int = 10,
        eval_episodes: int = 10,
        eval_path: str = None,
        tensorboard_log: Optional[str] = None,
        wandb_log: Optional[str] = None,
        verbose: int = 0,
        _init_setup_model: bool = True,
        seed: int = 0,
    ):
        super(DuSkillIM, self).__init__(
            env,
            observation_space,
            action_space,
            epochs,
            learning_rate,
            batch_size,
            replay_buffer,
            norm_obs=True,
            prior_path=prior_path,
            tune_dec=tune_dec,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
            eval_path=eval_path,
            tensorboard_log=tensorboard_log,
            wandb_log=wandb_log,
            verbose=verbose,
            seed=seed,
        )

        if net_arch is None:
            net_arch = dict(sk_pi=[64, 64], pf_pi=[64, 64], dec=[128, 128])
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.drop_rate = drop_rate

        self.tune_dec_type = tune_dec_type

        if self.wandb_log is not None:
            wandb_config = dict(
                algo='duskill-imitation',
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                net_arch=self.net_arch,
                prior_path=self.prior_path,
                drop_rate=self.drop_rate,
                tune_dec=self.tune_dec,
                tune_dec_type=self.tune_dec_type,
                seed=self.seed,
            )
            wandb.init(
                config=wandb_config,
                notes="DuSkill One-shot Imitation",
            )
            wandb.run.name = self.wandb_log

        if _init_setup_model:
            self._build(learning_rate)
    
    def _build_sk_pi(self) -> Encoder:
        """ Build high-level policy for skill embeddings. """
        return Encoder(
            self.net_arch['sk_pi'],
            self.latent_dim,
            self.activation_fn,
        )

    def _build_pf_pi(self) -> PrefPrior:
        """ Build high-level policy for pref embeddings. """
        return PrefPrior(
            self.net_arch['pf_pi'],
            self.embed_dim,
            self.latent_dim,
            self.activation_fn,
        )

    def _build_dec(self) -> hk.Module:
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

        if self.tune_dec_type == 0:
            raise ValueError(f"tune_dec_type should be 1 or 2, but {self.tune_dec_type}")
        decoder = GuidedDiffusionDecoder(
            sk_du, 
            pf_du,
            self.n_denoise,
            self.guide_wei,
            self.drop_rate,
            self.ddpm_dict,
            freeze_sk_dec=True if self.tune_dec_type == 1 else False,
        )

        return HierarchicalGuidedDiffusionAutoEncoder(
            sk_encoder=None,
            pf_encoder=None,
            decoder=decoder,
            squash_z=self.sk_pri.squash_z,
        )

    def _build(self, learning_rate: float) -> None:

        # DuSkill: load the frozen stage-1 prior and inherit its embedding space
        # (the guidance drop rate is overridden by this model's own setting)
        self.sk_pri = DuSkillPrior.load(self.prior_path, drop_rate=self.drop_rate)
        print("Loaded Squash Z:", self.sk_pri.squash_z)
        self.h = self.sk_pri.h
        self.noise_dim = self.sk_pri.noise_dim
        self.embed_dim = self.sk_pri.embed_dim
        self.latent_dim = self.sk_pri.latent_dim
        self.n_denoise = self.sk_pri.n_denoise
        self.ddpm_dict = self.sk_pri.ddpm_dict
        self.guide_wei = self.sk_pri.guide_wei
        
        def fn_pi(
            obs: jax.Array,
            obt: jax.Array,
            x_t: jax.Array,
            t: jax.Array,
            denoise: bool,
            deterministic: bool,
        ):
            sk_pi = self._build_sk_pi()
            pf_pi = self._build_pf_pi()

            sk_mean, sk_std = sk_pi(obs)
            pf_mean, pf_std = pf_pi(sk_mean) # no reparameterization
            if self.tune_dec:
                sk_dae = self._build_dec()
                return sk_dae(obs=None, act=None, o_t=obt, x_t=x_t, t=t, p=None, denoise=denoise, 
                    sk_mean=sk_mean, sk_std=sk_std, pf_mean=pf_mean, pf_std=pf_std, deterministic=deterministic)
            return HGDAEOutput(sk_mean, sk_std, pf_mean, pf_std, None, None)
         
        params, self.pi = hk.transform_with_state(fn_pi)
        dummy_obs = get_dummy_obs(self.observation_space)
        dummy_x_t = jax.random.normal(next(self.rng), shape=(1, self.noise_dim))
        dummy_t = jnp.array([[1 / self.n_denoise]])
        self.params, self.state = params(
            next(self.rng), dummy_obs, dummy_obs, dummy_x_t, dummy_t, False, False)
        self.optimizer = optax.adam(learning_rate=learning_rate)
        self.optimizer_state = self.optimizer.init(self.params)
        
        # DuSkill: Initialize skill policy with skill prior
        for key, val in self.sk_pri.sk_pri_params.items():
            self.params[key] = deepcopy(val)
        for key, val in self.sk_pri.sk_pri_state.items():
            self.state[key] = deepcopy(val)
        # DuSkill: Initialize pref policy with pref prior
        for key, val in self.sk_pri.pf_pri_params.items():
            self.params[key] = deepcopy(val)
        for key, val in self.sk_pri.pf_pri_state.items():
            self.state[key] = deepcopy(val)
        # DuSkill: Initialize decoder
        if self.tune_dec:
            for key, val in self.sk_pri.sk_dae_params.items():
                if 'diffusion' in key:
                    self.params[key] = deepcopy(val)
            for key, val in self.sk_pri.sk_dae_state.items():
                if 'diffusion' in key:
                    self.state[key] = deepcopy(val)

        # DuSkill: Initialize normalization layer
        if self.sk_pri.norm_obs:
            self.norm_pi = RunningNormLayer.load(self.prior_path + '/norm_layer')
        else:
            self.norm_obs = False

    def learn(self, epochs: int, gradient_steps: int = 1, log_interval: int = None) -> None:

        self.start_time = time.time()
        self._n_epochs, self._n_updates = 0, 0

        for epoch in range(epochs):
            losses, du_losses = [], []

            for gradient_step in range(gradient_steps):
                replay_data = self.replay_buffer.sample(self.batch_size) # [batch, h, s]
                observations = replay_data.observations
                actions = replay_data.actions
                
                if self.norm_obs:
                    observations = self.preprocess_pi(observations, training=False)
                
                # 1. learining high-level policy
                self.optimizer_state, self.params, self.state, loss, info = jit_optimize_with_state(
                    self._loss,
                    self.optimizer,
                    self.optimizer_state,
                    self.params,
                    self.state,
                    None,
                    observations=observations,
                    actions=actions,
                    rng=next(self.rng),
                )
                losses.append(np.array(info['loss']))
                du_losses.append(np.array(info['du_loss']))

            self._n_epochs += 1
            self._n_updates += gradient_steps
            
            # evaluate for every intervals
            avg_rew, avg_suc = None, None
            if self._n_updates == gradient_steps or self._n_updates % self.eval_interval == 0:
                print(f"Evaluating at {self._n_updates}")
                avg_rew, std_rew, avg_suc, std_suc = evaluate_skill_policy(
                    self,
                    self.env,
                    eval_episodes=self.eval_episodes,
                    deterministic=True,
                    ret_success=True,
                    ret_std=True,
                    save_path=self.eval_path+'/eval.txt',
                )
                avg_suc, std_suc = np.mean(avg_suc), np.mean(std_suc)
                self.logger.record("eval/avg_success", avg_suc)
                self.logger.record("eval/avg_reward", avg_rew)
                
                if self.eval_results['avg_rew'] <= avg_rew:
                    self.eval_results['avg_rew'] = avg_rew
                    self.eval_results['std_rew'] = std_rew
                    self.eval_results['avg_suc'] = avg_suc
                    self.eval_results['std_suc'] = std_suc
                    self.best_params = deepcopy(self.params)
                    self.best_state = deepcopy(self.state)

            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/loss", np.mean(losses))
            self.logger.record("train/du_loss", np.mean(du_losses))
            self.logger.record("train/learning_rate", self.learning_rate)
            if self.wandb_log is not None:
                wandb.log({
                    "train/loss": np.mean(losses),
                    "train/du_loss": np.mean(du_losses),
                    "eval/avg_success": avg_suc,
                    "eval/avg_reward": avg_rew,
                })

            if log_interval is not None and self._n_epochs % log_interval == 0:
                self._dump_logs()

    @partial(jax.jit, static_argnums=0)
    def _pi(
        self, 
        obs: jax.Array,
        obt: jax.Array,
        x_t: jax.Array,
        t: jax.Array,
        params: hk.Params, 
        state: hk.Params,
        rng=None,
    ) -> HGDAEOutput:
        return self.pi(params, state, rng,
            obs=obs, obt=obt, x_t=x_t, t=t, 
            denoise=False, deterministic=False)

    @partial(jax.jit, static_argnums=0)
    def _pi_dec(
        self,
        obs: jax.Array,
        obt: jax.Array,
        params: hk.Params,
        state: hk.State,
        rng=None,
    ) -> HGDAEOutput:
        return self.pi(params, state, rng,
            obs=obs, obt=obt, x_t=None, t=None, 
            denoise=True, deterministic=True)

    @partial(jax.jit, static_argnums=0)
    def _loss(
        self,
        params: hk.Params,
        state: hk.Params,
        observations: jax.Array, # h
        actions: jax.Array,
        rng=None,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        
        batch_size = observations.shape[0]

        # [batch, h, dim] -> [batch * h, dim]
        observations_t = observations.reshape(batch_size * self.h, -1)
        actions_t = actions.reshape(batch_size * self.h, -1)
        # [batch, h, dim] -> [batch, dim] -> [batch * h, dim]
        observations = observations[:,0,:]
        observations = jnp.repeat(observations, self.h, axis=0)

        # independent keys for the diffusion noise, the timestep draw and the
        # model's own reparameterization sampling
        k_noise, k_ts, k_model = jax.random.split(rng, 3)
        noise = jax.random.normal(k_noise, shape=(batch_size * self.h, actions_t.shape[-1]))
        _ts = jax.random.randint(k_ts, (batch_size * self.h, 1), minval=1, maxval=self.n_denoise+1) # 1 <= ts <= n_denoise
        x_t = self.ddpm_dict['sqrtab'][_ts] * actions_t + self.ddpm_dict['sqrtmab'][_ts] * noise

        # run high-level policy
        (sk_mean, sk_std, pf_mean, pf_std, noise_pred, _), new_state = self._pi(
            observations, observations_t, x_t, _ts / self.n_denoise, params, state, k_model)
     
        if not self.tune_dec:
            # run skill dec with embeddings
            (_, _, _, _, noise_pred, _), _ = self.sk_pri._sk_dae_with_emb(
                obt=observations_t, x_t=x_t, t=_ts/self.n_denoise, 
                sk_mean=sk_mean, sk_std=sk_std, pf_mean=pf_mean, pf_std=pf_std,
                params=self.sk_pri.sk_dae_params, state=self.sk_pri.sk_dae_state, rng=k_model
            )

        # diffion loss: run diffusion model to predict noise
        du_loss = jnp.mean(jnp.square(noise - noise_pred), axis=-1)
        du_loss = jnp.mean(du_loss)
        
        # loss summation
        loss = du_loss
        
        info = {
            'loss': loss,
            'du_loss': du_loss,
        }
        return loss, (new_state, info)
    
    def _test(self, verbose: bool = False) -> None:
        replay_data = self.replay_buffer.sample(10)
        # [batch, h, dim] -> [batch, dim]
        observations = replay_data.observations[:,0,:]
        prefs = replay_data.prefs[:,0,:]

        if self.norm_obs:
            observations = self.preprocess_pi(observations, training=False)
        
        if verbose:
            np.set_printoptions(precision=3, suppress=True)
            print("=" * 20)
            print(f"GT {prefs}")
            print("=" * 20)
    
    def predict_reset(self) -> None:
        self.p_h = 0

    def predict(self, obs: jax.Array, deterministic: bool = True):
        if obs is not None and len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
        
        if self.norm_obs: 
            obs = self.preprocess_pi(obs, training=False)
        
        if self.p_h == 0: # first step
            self.p_obs = obs
        
        (sk_mean, sk_std, pf_mean, pf_std, actions, _), _ = self._pi_dec(
            self.p_obs, obs, self.params, self.state, next(self.rng))

        if not self.tune_dec:
            (_, _, _, _, actions, _), _ = self.sk_pri._sk_dec(
                obs, sk_mean, sk_std, pf_mean, pf_std, 
                self.sk_pri.sk_dae_params, self.sk_pri.sk_dae_state, next(self.rng))
        
        self.p_h += 1
        if self.p_h == self.h:
            self.p_h = 0

        actions = np.array(actions)
        return actions, None, {}

    def save(self, path: str):
        self._save_eval(path)

        data = self.__dict__.copy()
        exclude = self._excluded_save_data()
        for param_name in exclude:
            data.pop(param_name, None)

        params_dict = {}
        params_dict['params'] = self.params
        params_dict['state'] = self.state
        params_dict['best_params'] = self.best_params
        params_dict['best_state'] = self.best_state
        save_to_zip_file(path, data=data, params=params_dict, verbose=1)
        
        if self.norm_obs:
            self.norm_pi.save(path + '/norm_pi')

    @classmethod
    def load(cls, path: str):
        data, params = load_from_zip_file(
            path,
            custom_objects=None,
            verbose=1,
            print_system_info=True,
        )

        model = cls(
            env=None,
            observation_space=data['observation_space'],
            action_space=data['action_space'],
            epochs=None,
            _init_setup_model=False,
        )

        model.__dict__.update(data)
        model._build(data['learning_rate'])

        model.params = params['params']
        model.state = params['state']
        model.best_params = params.get('best_params', None)
        model.best_state = params.get('best_state', None)
        if model.norm_obs:
            model.norm_pi = model.norm_pi.load(path + '/norm_pi')
        return model
