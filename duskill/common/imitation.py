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


class SkillIm(ABC):
    """ BaseClass for Skill Imitation """
    def __init__(
        self,
        env: gym.Env,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        epochs: int = None,
        learning_rate: float = 3e-4,
        batch_size: int = 128,
        replay_buffer: Type[BaseBuffer] = None,
        norm_obs: bool = True, 
        prior_path: str = None, # path for skill prior
        tune_dec: bool = False,
        eval_interval: int = 10,
        eval_episodes: int = 10,
        eval_path: str = None,
        tensorboard_log: Optional[str] = None,
        wandb_log: Optional[str] = None,
        verbose: int = 0,
        seed: int = 0,
    ):

        # envs
        self.env = env
        self.observation_space = observation_space
        self.action_space = action_space

        # hyperparams
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # buffers
        self.replay_buffer = replay_buffer
        self.norm_obs = norm_obs

        # prior
        self.prior_path = prior_path
        self.tune_dec = tune_dec 

        # evaluations
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.eval_path = eval_path
        self.eval_results = {
            'avg_rew': 0,
            'std_rew': 0,
            'avg_suc': 0,
            'std_suc': 0,
        }
        self.best_params = None
        self.best_state = None

        # for prediction
        self.p_h = 0 
        self.p_obs = None
        # # # # # # # # # 

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
            print_b(f"<< For default at least {epoch * 50} needed >>")
        except AttributeError:  # no replay buffer attached yet (e.g. when loading)
            pass

    def _get_epoch(self) -> None:
        return self.replay_buffer.num_timesteps / self.batch_size

    def _get_scheduler(self, epochs: int):
        if epochs is None:
            return self.learning_rate
        return optax.piecewise_constant_schedule(
            init_value=self.learning_rate,
            boundaries_and_scales={
                epochs*0.25:0.5,
                epochs*0.5:0.5,
                epochs*0.75:0.5})

    @abstractmethod
    def _build(self, learning_rate: float) -> None:
        """ Building Model """

    @abstractmethod
    def learn(self, epochs: int, gradient_steps: int = 1, log_interval: int = None) -> None:
        """ Learning Algorithm """
    
    def preprocess_pi(self, obs: jax.Array, training: bool = False) -> jnp.ndarray:
        if self.norm_obs:
            temp = None
            if len(obs.shape) > 2:
                temp = obs.shape
                obs = obs.reshape(obs.shape[0] * obs.shape[1], -1)
            obs = self.norm_pi(obs, training=training)
            if temp is not None:
                obs = obs.reshape(*temp)
            return obs
        return obs

    def _save_eval(self, path) -> None:
        with open(path + '/max_rew.txt', 'a') as f:
            f.write(f"Max Reward: {self.eval_results['avg_rew']:.3f} +/- {self.eval_results['std_rew']:.3f}\n")
            f.write(f"Success: {self.eval_results['avg_suc']:.1f} +/- {self.eval_results['std_suc']:.1f}")

    def _dump_logs(self) -> None:
        time_elapsed = time.time() - self.start_time
        fps = int((self._n_epochs - 0) / (time_elapsed + 1e-8))
        self.logger.record("time/epoch", self._n_epochs, exclude="tensorboard")
        self.logger.record("time/fps", fps)
        self.logger.dump(step=self._n_epochs)
    
    def _excluded_save_data(self) -> List[str]:
        return [
            'env',
            'replay_buffer',
            'rng',
            'tensorboard_log',
            'logger',
            'wandb_log',
            'sk_pri',
            'ddpm_dict',
            'pi',
            'sk_pi',
            'params',
            'state',
            'norm_pi',
        ]
