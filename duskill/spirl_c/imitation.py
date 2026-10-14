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

from duskill.spirl_c.prior import SPiRLcPrior
from duskill.common.imitation import SkillIm
from duskill.common.mlp_vae import Encoder, Decoder, VariationalAutoEncoder
from duskill.utils.buffers import SkillBuffer
from duskill.utils.evaluations import evaluate_skill_policy


class SPiRLcIM(SkillIm):
    """Few-shot imitation on top of a pretrained closed-loop SPiRL prior (Pertsch et al., 2020).

    A high-level policy pi(z|s_0) is initialised from the skill prior and fine-tuned by
    behaviour cloning through the closed-loop skill decoder, which is re-queried with the
    current observation at every step of the skill.
    """

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
        tune_dec: bool = False, # whether finetune decoder
        eval_interval: int = 10,
        eval_episodes: int = 10,
        eval_path: str = None,
        tensorboard_log: Optional[str] = None,
        wandb_log: Optional[str] = None,
        verbose: int = 0,
        _init_setup_model: bool = True,
        seed: int = 0,
    ):
        super(SPiRLcIM, self).__init__(
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
            net_arch = dict(sk_pi=[64, 64], sk_dec=[64, 64])
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        
        if self.wandb_log is not None:
            wandb_config = dict(
                algo='spirl_c-imitation',
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                net_arch=self.net_arch,
                prior_path=self.prior_path,
                tune_dec=self.tune_dec,
                seed=self.seed,
            )
            wandb.init(
                config=wandb_config,
                notes="SPiRLc One-shot Imitation",
            )
            wandb.run.name = self.wandb_log

        if _init_setup_model:
            self._build(learning_rate)

    def _build_sk_pi(self) -> Encoder:
        return Encoder(
            self.net_arch['sk_pi'],
            self.latent_dim,
            self.activation_fn,
        )
    
    def _build_sk_dec(self) -> VariationalAutoEncoder:
        decoder = Decoder(
            self.net_arch['sk_dec'],
            self.action_space.shape[-1],
            self.activation_fn,
        )
        return VariationalAutoEncoder(
            encoder=None,
            decoder=decoder,
            closed_loop=True,
        )

    def _build(self, learning_rate: float) -> None:

        # SPiRLc: Initialize embedding space z.
        self.sk_pri = SPiRLcPrior.load(self.prior_path)
        self.h = self.sk_pri.h
        self.latent_dim = self.sk_pri.latent_dim

        def fn_sk_pi(
            obs: jax.Array, 
            obt: jax.Array,
            deterministic: bool,
        ):
            sk_pi = self._build_sk_pi()
            mean, std = sk_pi(obs)
            if self.tune_dec:
                sk_dec = self._build_sk_dec()
                return sk_dec(obs=None, act=None, obt=obt, mean=mean, std=std, deterministic=deterministic)
            return (mean, std, None)
        
        params, self.sk_pi = hk.transform_with_state(fn_sk_pi)
        self.params, self.state = params(
            next(self.rng), get_dummy_obs(self.observation_space), get_dummy_obs(self.observation_space), False)
        self.optimizer = optax.adam(learning_rate=learning_rate)
        self.optimizer_state = self.optimizer.init(self.params)
        
        # SPiRLc: Initialize skill policy with skill prior
        for key, val in self.sk_pri.sk_pri_params.items():
            self.params[key] = deepcopy(val)
        for key, val in self.sk_pri.sk_pri_state.items():
            self.state[key] = deepcopy(val)
        # Initialize decoder
        if self.tune_dec:
            for key, val in self.sk_pri.sk_vae_params.items():
                if 'decoder' in key:
                    self.params[key] = deepcopy(val)
            for key, val in self.sk_pri.sk_vae_state.items():
                if 'decoder' in key:
                    self.state[key] = deepcopy(val)
        
        # SPiRLc: Initialize normalization layer
        if self.sk_pri.norm_obs:
            self.norm_pi = RunningNormLayer.load(self.prior_path + '/norm_layer')
        else:
            self.norm_pi = None
            self.norm_obs = False
    
    def learn(self, epochs: int, gradient_steps: int = 1, log_interval: int = None) -> None:

        self.start_time = time.time()
        self._n_epochs, self._n_updates = 0, 0

        for epoch in range(epochs):
            losses, mse_losses = [], []

            for gradient_step in range(gradient_steps):
                replay_data = self.replay_buffer.sample(self.batch_size) # [batch, h, s]
                observations = replay_data.observations
                actions = replay_data.actions
                prefs = replay_data.prefs

                if self.norm_obs:
                    observations = self.preprocess_pi(observations, training=False)
                
                # 1. learning high-level policy
                self.optimizer_state, self.params, self.state, loss, info = jit_optimize_with_state(
                    self._loss,
                    self.optimizer,
                    self.optimizer_state,
                    self.params,
                    self.state,
                    None,
                    observations=observations,
                    actions=actions,
                    prefs=prefs,
                    rng=next(self.rng),
                )
                losses.append(np.array(info['loss']))
                mse_losses.append(np.array(info['mse_loss']))
        
            self._n_epochs += 1
            self._n_updates += gradient_steps
            
            # evaluate for every intervals
            avg_suc, avg_rew = None, None
            if (self._n_updates == gradient_steps or self._n_updates % self.eval_interval == 0):
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
            self.logger.record("train/mse_loss", np.mean(mse_losses))
            self.logger.record("train/learning_rate", self.learning_rate)
            if self.wandb_log is not None:
                wandb.log({
                    "train/loss": np.mean(losses),
                    "train/mse_loss": np.mean(mse_losses),
                    "eval/avg_success": avg_suc,
                    "eval/avg_reward": avg_rew,
                })

            if log_interval is not None and self._n_epochs % log_interval == 0:
                self._dump_logs()

    @partial(jax.jit, static_argnums=0)
    def _sk_pi(
        self,
        obs: jax.Array, 
        obt: jax.Array, 
        params: hk.Params, 
        state: hk.Params,
        rng=None,
    ):
        return self.sk_pi(params, state, rng, obs, obt, deterministic=False)

    @partial(jax.jit, static_argnums=0)
    def _sk_pi_dec(
        self,
        obs: jax.Array, 
        obt: jax.Array, 
        params: hk.Params, 
        state: hk.Params,
        rng=None,
    ):
        return self.sk_pi(params, state, rng, obs, obt, deterministic=True)
    
    @partial(jax.jit, static_argnums=0)
    def _loss(
        self,
        params: hk.Params,
        state: hk.Params,
        observations: jax.Array, # h
        actions: jax.Array,
        prefs: jax.Array,
        rng=None,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        
        batch_size = observations.shape[0]

        observations_t = observations.reshape(batch_size * self.h, -1)
        observations = observations[:,0,:] # retrieve s_1

        # run high-level policy
        (sk_mean, sk_std, logits), new_state = self._sk_pi(
            observations, observations_t, params, state, rng)
            
        if not self.tune_dec:
            # run skill dec with embeddings
            (_, _, logits), _ = self.sk_pri._sk_vae_with_emb(
                obt=observations_t, mean=sk_mean, std=sk_std, 
                params=self.sk_pri.sk_vae_params, state=self.sk_pri.sk_vae_state, rng=rng
            )
        
        # reconstruction loss: run vae decoder to predict action
        actions = actions.reshape(batch_size * self.h, -1)
        mse_loss = jnp.mean(jnp.square(actions - logits), axis=-1)
        mse_loss = jnp.mean(mse_loss)

        loss = mse_loss
        
        info = {
            'loss': loss,
            'mse_loss': mse_loss,
        }
        return loss, (new_state, info)
    
    def _test(self, verbose: bool = False) -> None:
        if verbose:
            np.set_printoptions(precision=3, suppress=True)
    
    def predict_reset(self) -> None:
        self.p_h = 0

    def predict(self, obs: jax.Array, deterministic: bool = True):
        if obs is not None and len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
        
        if self.norm_obs: 
            obs = self.preprocess_pi(obs, training=False)
        
        if self.p_h == 0:
            self.p_obs = obs

        (sk_mean, sk_std, actions), _ = self._sk_pi_dec(
            self.p_obs, obs, self.params, self.state, next(self.rng))
        if not self.tune_dec:
            (_, _, actions), _ = self.sk_pri._sk_dec(
                obs, sk_mean, sk_std,
                self.sk_pri.sk_vae_params, self.sk_pri.sk_vae_state, next(self.rng))

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
            _init_setup_model=False,
        )

        model.__dict__.update(data)
        model._build(data['learning_rate'])

        model.params = params['params']
        model.state = params['state']
        if model.norm_obs:
            model.norm_pi = model.norm_pi.load(path + '/norm_pi')
        return model
