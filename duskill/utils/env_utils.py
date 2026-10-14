"""Meta-World Complex (MC) environment, expert and dataset helpers.

The benchmark used in the paper chains ``n`` Meta-World skills
(``puck``, ``drawer``, ``button``, ``door``; ``handle`` and ``lever`` are held
out as novel skills) into a single long-horizon episode.  Three *non-functional*
(nfunc) domains are defined on top of that chain:

* ``SP`` -- speed.   The nfunc level sets the commanded end-effector speed and
  the reward checks that each sub-task finishes within a speed-dependent number
  of steps.
* ``EN`` -- energy.  Actions are integrated into a velocity (``self.vel``) and
  the reward checks the accumulated control energy of each sub-task.  The
  observation is extended with the current 3-D velocity.
* ``WD`` -- wind.    A constant 2-D disturbance is added to every action and
  switched every 175 steps.  The observation is extended with the previous
  hand position.

Environment classes are named ``MC<n>-<domain>[-<config idx>]``, e.g.
``MC4-SP`` or ``MC4-SP-3``.  When a numeric config index is present,
:func:`get_task_config` resolves the task/nfunc split from
``configs/metaworld_complex/MC<n>-<domain>/config_<idx>.json`` and the
``--config_cls`` command line option is **ignored**.  Without an index the file
named by ``--config_cls`` (default ``config.json``) is used.
"""

import os
import re
from typing import Optional, Union, List

import copy
import json
import tqdm
import random
import pickle as pkl
from itertools import permutations

import cv2
import gym
import numpy as np
from metaworld.policies.policy import Policy as MetaWorldExp

from duskill.envs.metaworld_complex import SawyerComplexEnvV2
import duskill.envs.metaworld_complex.policies as metaworld_complex_exp
from duskill.utils.utils import print_y, print_b, print_r, make_dir


""" MetaWorld (Complex) Environment """

# Standard deviation of the Gaussian noise added to expert actions.
EXPERT_NOISE_STD = 0.005

# Number of "return" steps the reset controller is run for before the episode
# starts.  Keyed by the full environment class, with a per-domain fallback.
RETURN_LENGTH = {
    'MC4-SP-0': 40,
    'MC4-SP-1': 30,
    'MC4-SP-2': 20,
    'MC4-SP-3': 18,
    'MC4-SP-4': 18,
    'MC4-EN-0': 40,
    'MC4-EN-1': 30,
    'MC4-EN-3': 18,
    'MC4-EN-4': 18,
    'MC4-WD-0': 40,
    'MC4-WD-1': 30,
    'MC4-WD-3': 18,
    'MC4-WD-4': 18,
    # per-domain fallback
    'MC4-SP': 20,
    'MC4-EN': 20,
    'MC4-WD': 20,
}

# Non-functionality levels selectable per skill; the integers stored in the
# config files index into these lists.
SPEED_NFUNCS = [0.325, 0.350, 0.375, 0.400, 0.425, 0.450, 0.475, 0.500, 0.525]
ENERGY_NFUNCS = [0.325, 0.350, 0.375, 0.400, 0.425, 0.450, 0.475, 0.500, 0.525]
WIND_NFUNCS = [-0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20]

# Skills used by the benchmark; `handle` and `lever` are novel (test only).
TASK_SET = ['puck', 'drawer', 'button', 'door']
NOVEL_TASK_SET = ['handle', 'lever']


# Class for MetaWorld
class MetaWorldEnv(gym.Env):
    def __init__(self, env_cls: str, env, task, sparse_reward=True, max_steps=200, deterministic=True, seed=777):
        random.seed(seed)
        np.random.seed(seed)  # the env and the scripted experts use the global NumPy RNG
        self.env_cls = env_cls
        self.env = env      # environment
        self.task = task    # list of tasks
        self.sparse_reward = sparse_reward # goal conditioned reward
        self.max_steps = max_steps
        self.deterministic = deterministic  # deterministic task

        self.observation_space = self.env.observation_space
        self.action_space = gym.spaces.Box(
            self.env.action_space.low,
            self.env.action_space.high,
            dtype=np.float32
        )
    
    def reset(self):
        task = self.task[0] if self.deterministic else random.choice(self.task)
        self.env.set_task(task)
        self.timesteps = 0
        return self.env.reset()

    def step(self, action):
        action = np.clip(action, -1, 1)
        obs, rew, done, info = self.env.step(action)
        self.timesteps += 1
        
        # Done if success / max step
        rewS = 1 if info['success'] else 0
        info['rewS'] = rewS
        info['is_success'] = [True] if info['success'] else [False]
        if info['success'] or self.timesteps == self.max_steps:
            done = True

        return obs, rewS, done, info
    
    def render(self, offscreen=True, camera_name='corner3', resolution=(640, 480)):
        image = self.env.render(offscreen, camera_name, resolution)
        return image


class MetaWorldComplexEnv(MetaWorldEnv):
    def __init__(self, env_cls, env, task, n, exp_nfunc, nfunc_limit, novel=None, sparse_reward=True, max_steps=1200, deterministic=True, seed=777):
        super(MetaWorldComplexEnv, self).__init__(env_cls, env, task, sparse_reward, max_steps, deterministic, seed)
        self.env._partially_observable = False
        self.task = task
        if exp_nfunc is None:
            self.exp_nfuncs = None
        else:  # this can be multiple ...
            self.exp_nfuncs = [exp_nfunc] if len(np.array(exp_nfunc).shape) == 1 else exp_nfunc
        self.nfunc_limit = nfunc_limit
        self.novel = novel
        # `novel` marks an evaluation episode in which only the single novel
        # skill is executed, so the observation covers one sub-task only.
        num_obs_tasks = len(task) if self.novel is None else 1

        # 34 entries per sub-task + 4 extra; EN/WD append a 3-vector
        # (current velocity / previous hand position).
        if 'SP' in env_cls:
            obs_dim = 34 * num_obs_tasks + 4
        elif 'EN' in env_cls or 'WD' in env_cls:
            obs_dim = 34 * num_obs_tasks + 4 + 3
        else:
            raise ValueError(f'Not a valid non-functionality {env_cls}')
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.num_tasks = n
        self.max_steps = 400 * n
    
    def reset(self):
        self.timesteps = 0
        self._step, self._steps = 0, [] # memorize how many steps required
        self.rewA, self.rewAs = 0., [] # memorize rewA
        self.success = 0
        self.mode = 0
        obs = self.env.reset()
        return_policy = metaworld_complex_exp.CustomSawyerReturnV2Policy()

        if 'EN' in self.env_cls:
            self.delta = .1
        if 'WD' in self.env_cls:
            self.delta = .1
            if self.exp_nfuncs is None:
                self.exp_nfunc = None  # no wind sampled -> zero wind
            else:
                idx = np.random.choice([i for i in range(len(self.exp_nfuncs))], size=1)
                self.exp_nfunc = self.exp_nfuncs[idx[0]]

        return_length = None
        if self.env_cls in RETURN_LENGTH:
            return_length = RETURN_LENGTH[self.env_cls]
        else:
            family = '-'.join(self.env_cls.split('-')[:2])
            if family in RETURN_LENGTH:
                return_length = RETURN_LENGTH[family]
        if return_length is None:
            raise ValueError(
                f"No return length configured for {self.env_cls}; "
                f"known classes: {sorted(RETURN_LENGTH)}")

        for i in range(return_length):
            act = return_policy.get_action(self.env._last_stable_obs, prior_skill=self.env.skill_list[0])
            act = np.clip(act, -1, 1)
            next_obs, rew, done, info = self.env.step(act.copy()) 
            obs = next_obs

        if 'EN' in self.env_cls:
            self.vel = [0, 0, 0]
            self.prev_obs = [0, 0, 0]
            obs = np.concatenate((obs, self.vel), axis=-1)
            self.prev_obs = copy.deepcopy(obs[:3])
        if 'WD' in self.env_cls:
            self.prev_obs = [0, 0, 0]
            obs = np.concatenate((obs, self.prev_obs), axis=-1)
            self.prev_obs = copy.deepcopy(obs[:3])
        return obs

    def step(self, action):
        action = np.clip(action, -1, 1)
        
        if 'SP' in self.env_cls:
            self.rewA += np.linalg.norm(action[:3])
        if 'EN' in self.env_cls:
            self.rewA += np.sum(np.square(action[:-1]))*100
            vel = [0, 0, 0]
            vel[0] = self.vel[0] + action[0] * self.delta
            vel[1] = self.vel[1] + action[1] * self.delta
            vel[2] = self.vel[2] + action[2] * self.delta
            action = np.append(np.array(vel), action[-1])
        if 'WD' in self.env_cls:
            if self.timesteps != 0 and self.timesteps % 175 == 0 and self.mode < 3:
                self.mode += 1
            self.rewA += np.linalg.norm(action[:3])
            if self.exp_nfunc is not None:
                wind = self.exp_nfunc[self.mode]
                action = action + np.array([-wind, -wind, 0, 0])

        obs, rew, done, info = self.env.step(action)
        self.timesteps += 1
        self._step += 1
        # Done if success / max step
        rewS = 0

        if info['success']:
            # speed configuration
            if 'SP' in self.env_cls:
                if self.nfunc_limit is not None:
                    limit, thresh  = self.nfunc_limit[self.success], 5
                    diff = np.abs(self._step - limit)
                    if self._step > 10: # aviod sequence violation
                        if diff <= thresh:
                            rewS = 1.0
                        elif diff <= thresh*2:
                            rewS = 1.0 - (diff - thresh) * 0.1
                        else:
                            # if agent does not meet the speed limit -> Then success 0
                            rewS = 0.0
                        self.success += 1
                        self._steps.append(self._step)
                        self.rewAs.append(rewS - 0.5)
                        self._step = 0
                        self.rewA = 0
                    else:
                        rewS = 0.0
                        self.success += 0
                else: # if no step limit
                    rewS = 1
                    self.success += 1
                    self._steps.append(self._step)
                    self.rewAs.append(self.rewA / self._step)
                    self._step = 0
                    self.rewA = 0.
            if 'EN' in self.env_cls:
                if self.nfunc_limit is not None:
                    cur_task = self.env.skill_list[self.success]
                    if cur_task == 'puck':
                        limit, thresh = self.nfunc_limit[self.success], 30
                    elif cur_task == 'drawer':
                        limit, thresh = self.nfunc_limit[self.success], 40
                    elif cur_task == 'button':
                        limit, thresh = self.nfunc_limit[self.success], 50
                    elif cur_task == 'door':
                        limit, thresh = self.nfunc_limit[self.success], 60
                    else:
                        raise ValueError(f"unknown skill {cur_task}")

                    diff = np.abs(self.rewA - limit)
                    if self._step > 10: # aviod sequence violation
                        if diff <= thresh:
                            rewS = 1.0
                        elif diff <= thresh*2:
                            rewS = 1.0 - (diff - thresh) * (1 / (thresh*2))
                        else:
                            rewS = 0.5
                        self.success += 1
                        self._steps.append(self._step)
                        self.rewAs.append(self.rewA) 
                        self._step = 0
                        self.rewA = 0.
                    else:
                        rewS = 0.0
                        self.success += 0
                else: # if no nfunc limit
                    rewS = 1
                    self.success += 1
                    self._steps.append(self._step)
                    self.rewAs.append(self.rewA) 
                    self._step = 0
                    self.rewA = 0.
            if 'WD' in self.env_cls:
                rewS = 1
                self.success += 1
                self._steps.append(self._step)
                self.rewAs.append(self.rewA) 
                self._step = 0
                self.rewA = 0.
         
        info['steps'] = self._steps + [0 for _ in range(self.num_tasks - self.success)]
        info['is_success'] = [True for _ in range(self.success)] + [False for _ in range(self.num_tasks - self.success)]
        info['rew'] = {
            'rewS': rewS, 
            'rewA': self.rewAs + [0 for _ in range(self.num_tasks - self.success)],}
        # if success all the tasks
        if self.success == self.num_tasks or self.timesteps >= self.max_steps:
            done = True

        if 'EN' in self.env_cls:
            self.vel = vel
            obs = np.concatenate((obs, self.vel), axis=-1)
            self.prev_obs = copy.deepcopy(obs[:3])
        if 'WD' in self.env_cls:
            obs = np.concatenate((obs, self.prev_obs), axis=-1)
            self.prev_obs = copy.deepcopy(obs[:3])
        return obs, rewS * 2, done, info


# Make MetaWorld environment
def make_metaworld_env(
    env_cls: str,               # MC<n>-<domain>[-<config idx>], e.g. MC4-SP-3
    task: List[str] = None,     # ordered list of sub-task (skill) names
    nfunc: List[float] = None,  # non-functionality
    env_exp: bool = False,
    novel: bool = None,
    seed: Optional[int] = 777,
) -> gym.Env:

    m = re.match(r'MC(\d+)', env_cls)
    if m is None:
        raise ValueError(
            f"Not a valid Meta-World Complex class {env_cls}; "
            "expected 'MC<n>-<domain>[-<config idx>]'")
    n = int(m.group(1))

    env = SawyerComplexEnvV2(task)
    # get non-functionality
    nfunc_limit, exp_nfunc = None, None
    if nfunc is not None:
        _, task_nfunc, _ = get_task_list(env_cls)
        exp_nfunc = [task_nfunc[t][s] for t, s in zip(task, nfunc)]
        path = f'./datasets/metaworld_complex/{env_cls}/'
        nfunc_str = [str(s) for s in nfunc]
        path += '-'.join(task) + '-' + '-'.join(nfunc_str)
        if os.path.exists(path):
            if 'SP' in env_cls:
                with open(path + '/traj_20_step.json') as f:
                    nfunc_limit = json.load(f)['avg']
            if 'EN' in env_cls:
                with open(path + '/traj_20_rewA.json') as f:
                    nfunc_limit = json.load(f)['avg']
    print_r(f"Loading Speed Limit as {nfunc_limit}")
    env = MetaWorldComplexEnv(env_cls, env, task, n, exp_nfunc, nfunc_limit, novel, seed=seed)

    if env_exp: # return with expert
        return env, make_metaworld_exp(env_cls, task, exp_nfunc)
    return env


# Make MetaWorld expert (shortly, exp)
def make_metaworld_exp(
    env_cls: str,
    task: List[str] = None,
    task_nfunc: List[float] = None,
) -> Union[MetaWorldExp, List[MetaWorldExp]]:

    if 'SP' in env_cls:
        metaworld_complex_exps = {
            'puck': metaworld_complex_exp.CustomSpeedSawyerPlateSlideSideV2Policy,
            'drawer': metaworld_complex_exp.CustomSpeedSawyerDrawerCloseV2Policy,
            'button': metaworld_complex_exp.CustomSpeedSawyerButtonPressV2Policy,
            'door': metaworld_complex_exp.CustomSpeedSawyerDoorOpenV2Policy,
            # novel
            'handle': metaworld_complex_exp.CustomSpeedSawyerHandlePressV2Policy,
            'lever': metaworld_complex_exp.CustomSpeedSawyerLeverPullV2Policy,
        }
    elif 'EN' in env_cls:
        metaworld_complex_exps = {
            'puck': metaworld_complex_exp.CustomEnergySawyerPlateSlideSideV2Policy,
            'drawer': metaworld_complex_exp.CustomEnergySawyerDrawerCloseV2Policy,
            'button': metaworld_complex_exp.CustomEnergySawyerButtonPressV2Policy,
            'door': metaworld_complex_exp.CustomEnergySawyerDoorOpenV2Policy,
        }
    elif 'WD' in env_cls:
        metaworld_complex_exps = {
            'puck': metaworld_complex_exp.CustomWindSawyerPlateSlideSideV2Policy,
            'drawer': metaworld_complex_exp.CustomWindSawyerDrawerCloseV2Policy,
            'button': metaworld_complex_exp.CustomWindSawyerButtonPressV2Policy,
            'door': metaworld_complex_exp.CustomWindSawyerDoorOpenV2Policy,
        }
    else:
        raise ValueError(f'Not a valid non-functionality {env_cls}')

    if task_nfunc is None:
        task_nfunc = [None for _ in task]

    return [metaworld_complex_exps[sub_task](nfunc=nfunc) \
    for sub_task, nfunc in zip(task, task_nfunc)]


# Make MetaWorld Complex Task List
def get_task_list(
    env_cls: str,
    shuffle: bool = False,
) -> List[str]:

    if 'MC' not in env_cls:
        raise ValueError(f'Not a MetaWorld Complex {env_cls}')

    task_set = list(TASK_SET)
    # 1. Speed
    if 'SP' in env_cls:
        task_nfunc = {t: list(SPEED_NFUNCS) for t in task_set + NOVEL_TASK_SET}
    elif 'EN' in env_cls:
        task_nfunc = {t: list(ENERGY_NFUNCS) for t in task_set}
    elif 'WD' in env_cls:
        task_nfunc = {t: list(WIND_NFUNCS) for t in task_set}
    else:
        raise ValueError(f'Not a valid non-functionality {env_cls}')

    tasks_not_shuffled = []
    for task in permutations(task_set, 4):
        tasks_not_shuffled.append(task)
    return task_set, task_nfunc, tasks_not_shuffled


def get_task_config(
    env_cls: str,
    env_config: str = 'config.json',
):
    cls = env_cls.split('-')
    if len(cls) == 2:
        cpath = f'./configs/metaworld_complex/{env_cls}/{env_config}'
    else:
        # A numeric suffix on --env_cls pins the config file; --config_cls is ignored.
        cpath = f"./configs/metaworld_complex/{'-'.join(cls[:-1])}/config_{int(cls[-1])}.json"
        print_y(f"--env_cls {env_cls} pins {cpath}; --config_cls {env_config} is ignored")

    with open(cpath) as f:
        config = json.load(f)
    return config['train_tasks'], config['test_tasks'], \
            config['train_nfuncs'], config['test_nfuncs']


# Collect Expert Trajectories
def collect_metaworld_exp(
    env_cls: str, 
    env: Union[MetaWorldEnv, MetaWorldComplexEnv],
    exp: Union[MetaWorldExp, List[MetaWorldExp]],
    exp_nfunc: Union[float, List[float]] = None,
    exp_noise: bool = False,
    n_episodes: int = 20,
    save_video: str = False,
    save_path: str = None,
    verbose: str = False,
) -> None:
    _, task_nfunc, _ = get_task_list(env_cls)
    
    # trajectories to save
    trajs = []
    noise_std = EXPERT_NOISE_STD

    is_success, epi_length, sub_length, rewA = [], [], [], []
    for epi in tqdm.tqdm(range(n_episodes), total=n_episodes, desc='Episode'):
        # reset experts if EN
        if 'EN' in env_cls:
            for ex in exp:
                ex.reset()

        observations, next_observations, actions, images = [], [], [], []
        rewards, dones, infos, prefs = [], [], [], []
        sub_steps, sub_step, step  = [], 0, 0 
        obs = env.reset()

        while True:
            if save_video and epi == (n_episodes-1):
                images.append(env.render(resolution=(640,480)))
            if not isinstance(exp, List): # SingleTask
                act = exp.get_action(obs)
            else: # ComplexTask (MultiTask)
                act = exp[env.success].get_action(env.env._last_stable_obs, obt=obs, p=exp_nfunc) # get current task obs
                prefs.append([task_nfunc[env.task[env.success]].index(exp[env.success].nfunc) + 1]) # add 1
            if exp_noise:
                noise = np.random.normal(0.0, noise_std, size=act.shape) 
                noise[-1] = 0
                act += noise
            if 'WD' in env_cls and env.exp_nfunc is not None: # scaled action with wind
                wind = env.exp_nfunc[env.mode]
                act = act + np.array([wind, wind, 0, 0])
            act = np.clip(act, -1, 1)
            next_obs, rew, done, info = env.step(act.copy())
            step += 1
            sub_step += 1
            observations.append(obs); next_observations.append(next_obs);
            actions.append(act); rewards.append([rew]); dones.append([done]); infos.append(info)
            obs = next_obs
             
            if info['success']:
                sub_steps.append(sub_step)
                sub_step = 0

            if done:
                is_success.append(info['is_success'])
                break

            # MetaWorld Complex returning ...
            if isinstance(exp, List) and info['success']: 
                # Return policy for MetaWorld Complex
                return_policy = metaworld_complex_exp.CustomSawyerReturnV2Policy(env_cls=env_cls, nfunc=exp[env.success].nfunc)
                pref = task_nfunc[env.task[env.success]].index(exp[env.success].nfunc)
                if 'SP' in env_cls: return_length = 40
                elif 'EN' in env_cls: return_length = 80
                elif 'WD' in env_cls: return_length = 40
                else: raise ValueError(f'Not a valid non-functionality {env_cls}')
                
                for i in range(return_length):
                    if save_video and epi == (n_episodes-1) and i != 0:
                        images.append(env.render(resolution=(640,480)))
                    act = return_policy.get_action(env.env._last_stable_obs, obt=obs, p=None, \
                            prior_skill=env.env.skill_list[env.success-1], prior_act=act)
                    if exp_noise:
                        noise = np.random.normal(0.0, noise_std, size=act.shape) 
                        noise[-1] = 0
                        act += noise
                    act = np.clip(act, -1, 1)
                    next_obs, rew, done, info = env.step(act.copy())
                    step += 1
                    sub_step += 1
                    observations.append(obs); next_observations.append(next_obs);
                    actions.append(act); rewards.append([rew]); dones.append([done]); infos.append(info); 
                    prefs.append([pref + 1]) 
                     
                    obs = next_obs

        epi_length.append(step)
        sub_length.append(sub_steps)
        rewA.append(info['rew']['rewA'])
        
        # Make one trajectory
        traj = {
            'observations': np.array(observations),
            'next_observations': np.array(next_observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'infos': np.array(infos),
            'prefs': np.array(prefs), 
        }
        trajs.append(traj)
    
    if verbose:
        print_b("=" * 17 + ' Collect Dataset ' + '=' * 17)
        print(f"\t{n_episodes} episodes average length: {int(np.mean(epi_length))}")
        is_success = np.mean(is_success, axis=0)
        avg_sub_len, std_sub_len = np.mean(sub_length, axis=0), np.std(sub_length, axis=0)
        avg_rewA, std_rewA = np.mean(rewA, axis=0), np.std(rewA, axis=0)
        for i, (sub_task, success, len_mean, len_std) in enumerate(zip(env.task, is_success, avg_sub_len, std_sub_len)):
            print(f"Task {i} {sub_task}\tsuccess rate: {success*100}%, length: {len_mean:.1f} +/- {len_std:.2f}, rewA: {avg_rewA[i]:.2f} +/- {std_rewA[i]:.2f}")
        print(f"Total success rate : {np.mean(is_success)*100}%")
        print_b("=" * 50)
     
    if save_path is not None:
        make_dir(save_path)
        if exp_noise:
            traj_path = save_path + f'/traj_{n_episodes}_noise.pkl'
            step_path = save_path + f'/traj_{n_episodes}_step_noise.json'
            rewA_path = save_path + f'/traj_{n_episodes}_rewA_noise.json'
        else:
            traj_path = save_path + f'/traj_{n_episodes}.pkl'
            step_path = save_path + f'/traj_{n_episodes}_step.json'
            rewA_path = save_path + f'/traj_{n_episodes}_rewA.json'
        with open(traj_path, 'wb') as f:
            pkl.dump(trajs, f)
        with open(step_path, 'w') as f:
            data = {'avg': list(np.mean(sub_length, axis=0)),
                    'std': list(np.std(sub_length, axis=0))}
            json.dump(data, f)
        with open(rewA_path, 'w') as f:
            data = {'avg': list(np.mean(rewA, axis=0)),
                    'std': list(np.std(rewA, axis=0))}
            json.dump(data, f)

    if save_video:
        h, w, c = images[0].shape
        video = cv2.VideoWriter(save_path + '/expert.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (w, h))
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video.write(image)
        video.release()
