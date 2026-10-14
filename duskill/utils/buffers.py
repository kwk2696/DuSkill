import pickle
from typing import Optional, Dict, Union, List, NamedTuple

import random
import numpy as np
import jax.numpy as jnp
from gym import spaces

from stable_baselines3.common.vec_env import VecNormalize
from sb3_jax.common.buffers import BaseBuffer

from duskill.utils.utils import print_b

# Preferences are stored as small integer speed indices; divide by this to get a ~[0, 1] value.
PREF_SCALE = 10


class SkillBufferSamples(NamedTuple):
    observations: jnp.ndarray
    observations_with_next: jnp.ndarray
    actions: jnp.ndarray
    prefs: jnp.ndarray


class SkillBuffer(BaseBuffer):
    """Skill buffer."""
    def __init__(
        self,
        h: int, # skill length
        norm_obs: bool = False,
        trajectories: Dict[str, np.ndarray] = None,
        observation_space: spaces.Space = None,
        action_space: spaces.Space = None,
    ):
        super(SkillBuffer, self).__init__(None, observation_space, action_space)
        self.h = h
        self.norm_obs = norm_obs
        self.trajectories = trajectories

        self.setup(verbose=True)

    def setup(self, verbose: bool = False) -> None:
        # pass if traj not initialized
        if self.trajectories is None: return
        
        observations, traj_lengths = [], []
        for path in self.trajectories:
            observations.append(path['observations'])
            traj_lengths.append(len(path['observations']))
        traj_lengths = np.array(traj_lengths)

        observations = np.concatenate(observations, axis=0)
        self.obs_mean, self.obs_std = np.mean(observations, axis=0), np.std(observations, axis=0) + 1e-6
        self.num_trajs = len(traj_lengths)
        self.num_timesteps = sum(traj_lengths)
        
        if verbose:
            print_b('=' * 16 + ' Trajectory Infos ' + '=' * 16)
            print(f'\t{self.num_trajs} trajectories, {self.num_timesteps} timesteps found')
            print_b('=' * 50)
        
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> SkillBufferSamples:
        batch_inds = np.random.choice(
            np.arange(self.num_trajs),
            size=batch_size,
            replace=True
        )
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> SkillBufferSamples:
        observations, observations_with_next, actions, prefs = [], [], [], []

        for i in range(len(batch_inds)):
            traj = self.trajectories[int(batch_inds[i])]
            si = np.random.randint(0, traj['actions'].shape[0] - 1)

            # get sequence from dataset
            observations.append(traj['observations'][si:si + self.h].reshape(1, -1, self.obs_dim))
            observations_with_next.append(traj['observations'][si:si + self.h+1].reshape(1, -1, self.obs_dim))
            actions.append(traj['actions'][si:si + self.h].reshape(1, -1, self.act_dim))
            if 'prefs' in traj.keys():
                p = traj['prefs'][si].repeat(self.h).reshape(1, -1, 1) / PREF_SCALE
                prefs.append(p)
            
            # padding
            tlen = observations[-1].shape[1]
            observations[-1] = np.concatenate([np.zeros((1, self.h - tlen, self.obs_dim)), observations[-1]], axis=1)
            tlen = observations_with_next[-1].shape[1]
            observations_with_next[-1] = np.concatenate([np.zeros((1, self.h+1 - tlen, self.obs_dim)), observations_with_next[-1]], axis=1)
            tlen = actions[-1].shape[1]
            actions[-1] = np.concatenate([np.ones((1, self.h - tlen, self.act_dim)) * 0, actions[-1]], axis=1)

            # norm obs
            if self.norm_obs:
                observations[-1] = (observations[-1] - self.obs_mean) / self.obs_std
                observations_with_next[-1] = (observations_with_next[-1] - self.obs_mean) / self.obs_std
        
        data = (
            np.concatenate(observations, axis=0).astype(np.float32),
            np.concatenate(observations_with_next, axis=0).astype(np.float32),
            np.concatenate(actions, axis=0).astype(np.float32),
            np.concatenate(prefs, axis=0).astype(np.float32) if 'prefs' in traj.keys() else 0,
        )
        return SkillBufferSamples(*tuple(map(self.to_jnp, data)))
    
    def load_traj(self, path: Union[str, List[str]]):
        if not isinstance(path, List):
            path = [path]
        trajs = []
        for p in path:
            with open(p, 'rb') as f:
                traj = pickle.load(f)
            trajs.extend(traj)
        # let's shuffle the list
        random.shuffle(trajs)
        self.trajectories = trajs
        self.setup(verbose=True)
