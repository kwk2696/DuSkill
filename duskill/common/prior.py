from abc import ABC, abstractmethod
from typing import List, Optional, Type

import time
import gym
import jax
import jax.numpy as jnp
import haiku as hk 
import optax

from sb3_jax.common.buffers import BaseBuffer
from sb3_jax.common.utils import configure_logger

from duskill.utils.utils import print_b


class SkillPrior(ABC):
    """ BaseClass for Prior """
    def __init__(
        self,
        epochs: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        learning_rate: float = 3e-4,
        batch_size: int = 128,
        h: int = 10,            # skill length
        beta: float = 1.0,      # kl coeff
        embed_dim: int = 128,   # embedding dim
        hidden_dim: int = 128,  # lstm dim
        latent_dim: int = 10,   # skill dim
        squash_z: bool = False, # suqash z with tanh
        replay_buffer: Type[BaseBuffer] = None,
        norm_obs: bool = True, 
        tensorboard_log: Optional[str] = None,
        wandb_log: Optional[str] = None,
        verbose: int = 0,
        seed: int = 1,
    ):

        # envs
        self.epochs = epochs
        self.observation_space = observation_space
        self.action_space = action_space
        
        # hyperparams
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.h = h
        self.beta = beta
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.squash_z = squash_z

        # buffers
        self.replay_buffer = replay_buffer
        self.norm_obs = norm_obs
        
        # seedings
        self.seed = seed
        self.rng = hk.PRNGSequence(seed)
        
        # loggings
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.logger = configure_logger(self.verbose, self.tensorboard_log, "run", True)
        self.wandb_log = wandb_log
        
        # print required epochs
        try:
            epoch = self._get_epoch()
            print_b(f"<< {epoch} needed for 1 rollout of buffer >>")
            print_b(f"<< For default at least {epoch * 200} needed >>")
        except AttributeError:  # no replay buffer attached yet (e.g. when loading)
            pass

    def _get_epoch(self) -> None:
        return self.replay_buffer.num_timesteps / self.batch_size
    
    def _get_scheduler(self, epochs: int):
        if epochs is None:
            return self.learning_rate
        # 0.001 - 0.0005 - 0.00025 - 0.000125 - 0.0000625
        return optax.piecewise_constant_schedule(
            init_value=self.learning_rate,
            boundaries_and_scales={
                epochs*0.15: 0.5,
                epochs*0.25: 0.5,
                epochs*0.50: 0.5,
                epochs*0.75: 0.5})

    @abstractmethod
    def _build(self, learning_rate: float) -> None:
        """ Building Model """

    @abstractmethod
    def learn(self, epochs: int, gradient_steps: int = 1, log_interval: int = None) -> None:
        """ Learning Algorithm """

    def preprocess_obs(self, obs: jax.Array, training: bool = False) -> jnp.ndarray:
        if self.norm_obs:
            temp = None
            if len(obs.shape) > 2: 
                temp = obs.shape
                obs = obs.reshape(-1, obs.shape[-1])
            obs = self.norm_layer(obs, training=training)
            if temp is not None:
                obs = obs.reshape(*temp)
            return obs
        return obs

    def _dump_logs(self) -> None:
        time_elapsed = time.time() - self.start_time
        fps = int((self._n_epochs - 0) / (time_elapsed + 1e-8))
        self.logger.record("time/epoch", self._n_epochs, exclude="tensorboard")
        self.logger.record("time/fps", fps)
        self.logger.dump(step=self._n_epochs)

    def _excluded_save_data(self) -> List[str]:
        return [
            'replay_buffer',
            'rng',
            'tensorboard_log',
            'logger',
            'wandb_log',
            'ddpm_dict',
            'sk_vae',
            'sk_dae',
            'sk_pri',
            'pf_pri',
            'dist',
            'demo',
            'z_demo',
        ]
