"""FIST Implementation""" 

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
from duskill.common.mlp_vae import (
    VAEOutput, 
    Encoder, 
    Decoder, 
    VariationalAutoEncoder,
    ContrastiveEncoder,
    ContrastiveFutureState,
)
from duskill.common.lstm_vae import LSTMEncoder
from duskill.utils.buffers import SkillBuffer
from duskill.utils.evaluations import evaluate_skill_policy


class FISTPrior(SkillPrior):
    """Few-shot Imitation with Skill Transition models (FIST, Hakhamaneshi et al., 2022).

    An LSTM skill encoder with a *closed-loop* decoder, plus a future-conditioned prior
    p(z | s_t, s_{t+h}) and a contrastive distance function used at test time to retrieve
    the nearest demo state as the goal s_{t+h}. The same class serves both the pretraining
    stage and the `finetune=True` few-shot imitation stage.
    """

    closed_loop = True

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
        # for imitation
        finetune: bool = False,
        env: gym.Env = None,
        eval_interval: int = 10,
        eval_episodes: int = 10,
        eval_path: str = None,
        tensorboard_log: Optional[str] = None,
        wandb_log: Optional[str] = None,
        verbose: int = 0,
        _init_setup_model: bool = True,
        seed: int = 1,
    ):
        super(FISTPrior, self).__init__(
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
        
        self.finetune = finetune
        self.env = env
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.eval_path = eval_path
        self.eval_results = {
            'avg_rew': 0,
            'std_rew': 0,
            'avg_suc': 0,
            'std_suc': 0,
        }

        # for prediction 
        self.p_h = 0
        self.p_obs = None
        self.p_obh = None
        self.z_demo = None

        # logger object
        if self.wandb_log is not None:
            wandb_config = dict(
                algo='fist-prior' if not self.finetune else 'fist-imitation',
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
                notes="FIST Skill Prior" if not self.finetune else 'FIST One-shot Imitation',
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
            self.action_space.shape[-1],
            self.activation_fn,
        )
        return VariationalAutoEncoder(
            encoder=encoder,
            decoder=decoder,
            closed_loop=True,
            squash_z=self.squash_z,
        )
   
    def _build_sk_pri(self) -> hk.Module:
        """Build skill prior."""
        prior = Encoder(
            self.net_arch['pri'],
            self.latent_dim,
            self.activation_fn
        )
        return prior
    
    def _build_dist(self) -> ContrastiveFutureState:
        encoder = ContrastiveEncoder(
            hidden_dim=128,
            feature_dim=32,
        )
        return ContrastiveFutureState(
            encoder,
            feature_dim=32,
        )

    def _build(self, learning_rate: float) -> None:

        def fn_sk_vae(
            obs: jax.Array, 
            act: jax.Array,
            obt: jax.Array,
            mean: jax.Array, 
            std: jax.Array, 
            deterministic: bool,
        ):
            sk_vae = self._build_sk_vae()
            return sk_vae(obs, act, obt, mean, std, deterministic)
        
        sk_vae_params, self.sk_vae = hk.transform_with_state(fn_sk_vae)
        dummy_obs, dummy_act = get_dummy_obs(self.observation_space), get_dummy_act(self.action_space)
        dummy_obs_seq = jnp.repeat(dummy_obs, self.h, axis=0).reshape(1, self.h, -1)
        dummy_act_seq = jnp.repeat(dummy_act, self.h, axis=0).reshape(1, self.h, -1)
            
        self.sk_vae_params, self.sk_vae_state = sk_vae_params(
            next(self.rng), dummy_obs_seq, dummy_act_seq, dummy_obs, None, None, False)
        self.sk_vae_optimizer = optax.adam(learning_rate=self._get_scheduler(self.epochs))
        self.sk_vae_optimizer_state = self.sk_vae_optimizer.init(self.sk_vae_params)
        
        def fn_sk_pri(obs: jax.Array, obs_h: jax.Array):
            obs = jnp.concatenate((obs, obs_h), axis=-1)
            sk_pri = self._build_sk_pri()
            return sk_pri(obs)
        
        sk_pri_params, self.sk_pri = hk.transform_with_state(fn_sk_pri)
        self.sk_pri_params, self.sk_pri_state = sk_pri_params(next(self.rng), dummy_obs, dummy_obs)
        self.sk_pri_optimizer = optax.adam(learning_rate=self._get_scheduler(self.epochs))
        self.sk_pri_optimizer_state = self.sk_pri_optimizer.init(self.sk_pri_params)
        
        def fn_dist(obs: jax.Array, obs_h: jax.Array, ret: bool):
            dist = self._build_dist()
            return dist(obs, obs_h, ret)
        dist_params, self.dist = hk.transform_with_state(fn_dist)
        self.dist_params, self.dist_state = dist_params(next(self.rng), dummy_obs, dummy_obs, False)
        self.dist_optimizer = optax.adam(learning_rate=self._get_scheduler(self.epochs))
        self.dist_optimizer_state = self.dist_optimizer.init(self.dist_params)

        # build normalization layer
        if self.norm_obs:
            self.norm_layer = RunningNormLayer(self.observation_space.shape)
   
    def learn(self, epochs: int, gradient_steps: int = 1, log_interval: int = None) -> None:
        self.start_time = time.time()
        
        self._n_epochs, self._n_updates = 0, 0
        for epoch in range(epochs):
            vae_losses, mse_losses, kl_losses = [], [], []
            pri_losses = []
            dist_losses = []

            for gradient_step in range(gradient_steps):
                replay_data = self.replay_buffer.sample(self.batch_size) # [batch, h, s]
                # obtain skills
                observations = replay_data.observations
                actions = replay_data.actions
                
                if self.norm_obs:
                    if not self.finetune:
                        observations = self.preprocess_obs(observations, training=True)
                    else:
                        observations = self.preprocess_obs(observations, training=False)

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
                    observations_t=observations,
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

                # 3. learning distance function
                if not self.finetune:
                    self.dist_optimizer_state, self.dist_params, self.dist_state, dist_loss, dist_info = jit_optimize_with_state(
                        self._dist_loss,
                        self.dist_optimizer,
                        self.dist_optimizer_state,
                        self.dist_params,
                        self.dist_state,
                        None,
                        observations=observations,
                    )
                    dist_losses.append(dist_info["dist_loss"])
            
            self._n_epochs += 1
            self._n_updates += gradient_steps
            
            # checking action mse ...
            if self._n_updates % 100 == 0:
                observations = stop_grad(observations)
                observations = observations.reshape(self.batch_size * self.h, -1)
                actions = actions.reshape(self.batch_size * self.h, -1)
                # action reconstruction : encoder -> decoder
                (_, _, vae_actions), _ = self._sk_dec( \
                    observations, vae_info['q_mean'], vae_info['q_std'], 
                    self.sk_vae_params, self.sk_vae_state, next(self.rng))
                vae_mse = jnp.mean(jnp.square(vae_actions - actions))
                # action reconstruction : prior -> decoder
                (_, _, pri_actions), _ = self._sk_dec( \
                    observations, pri_info['p_mean'], pri_info['p_std'], 
                    self.sk_vae_params, self.sk_vae_state, next(self.rng))
                pri_mse = jnp.mean(jnp.square(pri_actions - actions))
                vae_mse, pri_mse = float(vae_mse), float(pri_mse)
                self.logger.record("train/vae/mse", vae_mse)
                self.logger.record("train/pri/mse", pri_mse)
                if self.wandb_log is not None:
                    wandb.log({
                        "train/vae/mse": vae_mse,
                        "train/pri/mse": pri_mse,
                    })

            # evaluate for every intervals 
            avg_suc, avg_rew = None, None
            if self.finetune and (self._n_updates == gradient_steps or self._n_updates % self.eval_interval == 0):
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

            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/vae/loss", np.mean(vae_losses))
            self.logger.record("train/vae/mse_loss", np.mean(mse_losses))
            self.logger.record("train/vae/sk_kl_loss", np.mean(kl_losses))
            self.logger.record("train/vae/mse", np.mean(mse_losses))
            self.logger.record("train/pri/sk_loss", np.mean(pri_losses))
            if not self.finetune:
                self.logger.record("train/dist/dist_loss", np.mean(dist_losses))

            if self.wandb_log is not None:
                if not self.finetune:
                    wandb.log({
                        "train/vae/loss": np.mean(vae_losses),
                        "train/vae/mse_loss": np.mean(mse_losses),
                        "train/vae/sk_kl_loss": np.mean(kl_losses),
                        "train/vae/mse": np.mean(mse_losses),
                        "train/pri/sk_loss": np.mean(pri_losses),
                        "train/dist/dist_loss": np.mean(dist_losses),
                    })
                else:
                    wandb.log({
                        "train/loss": np.mean(vae_losses),
                        "train/mse_loss": np.mean(mse_losses),
                        "train/sk_kl_loss": np.mean(kl_losses),
                        "train/mse": np.mean(mse_losses),
                        "train/sk_loss": np.mean(pri_losses),
                        "eval/avg_success": avg_suc,
                        "eval/avg_reward": avg_rew,
                    })


            if log_interval is not None and self._n_epochs % log_interval == 0:
                self._dump_logs()

    @partial(jax.jit, static_argnums=0)
    def _sk_vae(
        self, 
        obs: jax.Array, 
        act: jax.Array, 
        obt: jax.Array,
        params: hk.Params, 
        state: hk.Params, 
        rng=None
    ) -> VAEOutput:
        return self.sk_vae(params, state, rng, 
            obs=obs, act=act, obt=obt,
            mean=None, std=None, deterministic=False)
    
    @partial(jax.jit, static_argnums=0)
    def _sk_dec(
        self, 
        obt: jax.Array,
        mean: jax.Array, 
        std: jax.Array, 
        params: hk.Params, 
        state: hk.Params, 
        rng=None
    ) -> VAEOutput:
        return self.sk_vae(params, state, rng, 
            obs=None, act=None, obt=obt,
            mean=mean, std=std, deterministic=True)

    @partial(jax.jit, static_argnums=0)
    def _sk_pri(
        self, 
        obs: jax.Array,
        obs_h: jax.Array,
        params: hk.Params, 
        state: hk.Params, 
        rng=None
    ) -> Tuple[jax.Array, jax.Array]:
        return self.sk_pri(params, state, rng, obs, obs_h)
    
    @partial(jax.jit, static_argnums=0)
    def _dist(
        self,
        obs: jax.Array,
        obs_h: jax.Array,
        params: hk.Params,
        state: hk.Params,
        rng=None
    ) -> jax.Array:
        return self.dist(params, state, rng, obs, obs_h, False)

    @partial(jax.jit, static_argnums=0)
    def _dist_enc(
        self,
        obs: jax.Array,
        params: hk.Params,
        state: hk.Params,
        rng=None
    ) -> jax.Array:
        return self.dist(params, state, rng, obs, None, True)

    @partial(jax.jit, static_argnums=0)
    def _sk_vae_loss(
        self,
        params: hk.Params,
        state: hk.Params,
        observations: jax.Array,
        actions: jax.Array,
        observations_t: jax.Array,
        rng=None,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """ELBO loss: E_p[log(x)] - KL(d||q), where q ~ N(0,1)."""
        batch_size = observations.shape[0]
       
        observations_t = observations_t.reshape(batch_size * self.h, -1)

        (mean, std, logits), new_state = self._sk_vae(
            obs=observations, act=actions, obt=observations_t, 
            params=params, state=state, rng=rng)

        # reconstruction term: use mse loss to approximate log-likelihood
        actions = actions.reshape(batch_size * self.h, -1)
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
        observations_0 = observations[:,0,:] # retrieve s_0
        observations_h = observations[:,self.h-1,:] # retrieve s_h
        
        # skill prior output
        (p_mean, p_std), new_state = self._sk_pri(observations_0, observations_h, params, state, rng)
        
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
    
    @partial(jax.jit, static_argnums=0)
    def _dist_loss(
        self,
        params: hk.Params,
        state: hk.Params,
        observations: jax.Array,
        rng=None
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Contrastive (InfoNCE) loss for the future-state distance function."""
        observations_0 = observations[:,0,:] # retrieve s_1
        observations_h = observations[:,self.h-1,:] # retrieve s_h

        # dist output
        logits, new_state = self._dist(observations_0, observations_h, params, state, rng)
        # positive keys are on the diagonal
        positive_logits = logits.diagonal()
        log_sum_exp_logits = jnp.log(jnp.sum(jnp.exp(logits), axis=-1))
        ce_loss = -positive_logits.squeeze() + log_sum_exp_logits

        loss = jnp.mean(ce_loss)

        info = {
            'dist_loss': loss,
            'ce_loss': jnp.mean(ce_loss),
        }
        return loss, (new_state, info)

    def predict_prior(self, obs: jax.Array, obs_h: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Skill prior p(z | s_t, s_{t+h}); `obs_h` is the (retrieved) future state."""
        if self.norm_obs:
            obs = self.preprocess_obs(obs)
            obs_h = self.preprocess_obs(obs_h)
        (mean, std), _ = self._sk_pri(obs, obs_h, self.sk_pri_params, self.sk_pri_state, next(self.rng))
        return mean, std
    
    def predict_reset(self):
        self.p_h = 0
    
    def set_z_demo(self, demo):
        print(f"Setting z demo {demo.shape}")
        self.demo = self.preprocess_obs(demo, training=False)
        self.z_demo, _ = self._dist_enc(self.demo, self.dist_params, self.dist_state, next(self.rng))

    def predict(self, obs: jax.Array, deterministic: bool = True):
        if obs is not None and len(obs.shape) == 1:
            obs = obs.reshape(1, -1)

        if self.norm_obs:
            obs = self.preprocess_obs(obs, training=False)
        
        if self.p_h == 0: # first step
            self.p_obs = obs
            z_obs, _ = self._dist_enc(obs, self.dist_params, self.dist_state, next(self.rng))
            dists = jnp.sum(jnp.square(z_obs - self.z_demo), axis=-1)
            min_idx = jnp.argmin(dists)
            self.p_obh = self.demo[min_idx].reshape(1, -1)

        (mean, std), _ = self._sk_pri(self.p_obs, self.p_obh, self.sk_pri_params, self.sk_pri_state, next(self.rng))
        (_, _, actions), _ = self._sk_dec(
            obs, mean, std, self.sk_vae_params, self.sk_vae_state, next(self.rng))
        
        self.p_h += 1
        if self.p_h == self.h: # reset step
            self.p_h = 0
        
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
            observations,
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
    
    def _save_eval(self, path) -> None:
        with open(path + '/max_rew.txt', 'a') as f:
            f.write(f"Max Reward: {self.eval_results['avg_rew']:.3f} +/- {self.eval_results['std_rew']:.3f}\n")
            f.write(f"Success: {self.eval_results['avg_suc']:.1f} +/- {self.eval_results['std_suc']:.1f}")

    def save(self, path: str):
        if self.finetune:
            self._save_eval(path)

        data = self.__dict__.copy()
        params_dict = {}
        exclude = self._excluded_save_data()
        for param_name in exclude:
            data.pop(param_name, None)

        params_dict['sk_vae_params'] = self.sk_vae_params
        params_dict['sk_vae_state'] = self.sk_vae_state
        params_dict['sk_pri_params'] = self.sk_pri_params
        params_dict['sk_pri_state'] = self.sk_pri_state
        params_dict['dist_params'] = self.dist_params
        params_dict['dist_state'] = self.dist_state
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
        model.dist_params = params['dist_params']
        model.dist_state = params['dist_state']
        
        if model.norm_obs:
            model.norm_layer = model.norm_layer.load(path + '/norm_layer')

        return model

    def _load(self, path: str):
        data, params = load_from_zip_file(
            path,
            custom_objects=None,
            verbose=1,
            print_system_info=True,
        )

        self.sk_vae_params = params['sk_vae_params']
        self.sk_vae_state = params['sk_vae_state']
        self.sk_pri_params = params['sk_pri_params']
        self.sk_pri_state = params['sk_pri_state']
        self.dist_params = params['dist_params']
        self.dist_state = params['dist_state']
        
        if self.norm_obs:
            self.norm_layer = self.norm_layer.load(path + '/norm_layer')
