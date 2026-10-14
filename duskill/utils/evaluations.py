from typing import Tuple, Union, List

import tqdm

import cv2
import gym
import numpy as np

from duskill.utils.wrappers import SkillWrapper
from duskill.utils.env_utils import make_metaworld_env
from duskill.utils.utils import print_b, print_y


def evaluate_skill_policy(
    policy,
    env: Union[gym.Env],
    eval_episodes: int = 100,
    deterministic: bool = True,
    ret_success: bool = False,
    ret_std: bool = False,
    video_path: str = None,
    save_path: str = None, 
    verbose: bool = True,
) -> Tuple[np.array, np.array]:
    
    tot_reward, tot_success, epi_length, sub_length, rewA = [], [], [], [], []
    images = []

    for epi in tqdm.tqdm(range(eval_episodes), total=eval_episodes, desc='Episode'):
        obs = env.reset()
        try:
            policy.predict_reset() # reset the skill-step counter, if the model keeps one
        except AttributeError:
            pass
        if video_path is not None and epi == (eval_episodes-1):
            images.append(env.render())
        epi_reward, step = 0., 0
        while True:
            action, _, _ = policy.predict(obs, deterministic=deterministic)
            action = np.array(action.copy())
            if video_path is not None and epi == (eval_episodes-1):
                obs, reward, done, info = env.step(action, render=True)
                images.extend(info['images'])
            else:
                obs, reward, done, info = env.step(action)
            epi_reward += reward
            step += 1

            if done:
                break

        tot_reward.append(epi_reward)
        epi_length.append(step)
        
        rewA.append(info['rew']['rewA'])
        if 'steps' in info.keys():
            sub_length.append(info['steps'])
        else:
            sub_length.append([step])
        if 'is_success' in info.keys():
            tot_success.append(info['is_success'])
        else:
            tot_success.append(False)
    
    avg_reward, std_reward = np.mean(tot_reward), np.std(tot_reward)
    avg_success, std_success = np.mean(tot_success, axis=0)*100, np.std(tot_success, axis=0)*100

    # count only the lengths of the sub-tasks that were actually solved
    avg_sub_len, std_sub_len, avg_rewA, std_rewA = [], [], [], []
    for i in range(env.num_tasks):
        s_len, s_a = [], []
        for success, length, rewa in zip(tot_success, sub_length, rewA):
            if success[i]:
                s_len.append(length[i])
                s_a.append(rewa[i])
        avg_sub_len.append(np.mean(s_len)); std_sub_len.append(np.std(s_len))
        avg_rewA.append(np.mean(s_a)); std_rewA.append(np.std(s_a))

    if verbose:
        print_b("=" * 13 + ' Performance Evaluation ' + '=' * 13)
        print(f"\t{eval_episodes} episodes average length: {int(np.mean(epi_length))}")
        for i, (success, _, len_mean, len_std) in enumerate(zip(avg_success, std_success, avg_sub_len, std_sub_len)):
            print(f"\tTask {i} Success Rate: {success:.1f}%, length: {len_mean:.1f} +/- {len_std:.2f}, rewA: {avg_rewA[i]:.2f} +/- {std_rewA[i]:.2f}")
        print(f"\tTotal Success Rate  : {np.mean(avg_success):.1f}%")
        print(f"\tTotal Average Reward: {avg_reward:.3f} +/- {std_reward:.3f}")
        print_b("=" * 50)
    
    if video_path:
        h, w, c = images[0].shape
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), 30, (w, h))
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video.write(image)
        video.release()
        print_y(f"Saved video at {video_path}")
    
    if save_path:
        with open(save_path, 'a') as f:
            f.write("=" * 13 + ' Performance Evaluation ' + '=' * 13 + '\n')
            f.write(f"\t{eval_episodes} episodes average length: {int(np.mean(epi_length))}\n")
            for i, (success, _, len_mean, len_std) in enumerate(zip(avg_success, std_success, avg_sub_len, std_sub_len)):
                f.write(f"\tTask {i} Success Rate: {success:.1f}%, length: {len_mean:.1f} +/- {len_std:.2f}, rewA: {avg_rewA[i]:.2f} +/- {std_rewA[i]:.2f}\n")
            f.write(f"\tTotal Success Rate  : {np.mean(avg_success):.1f}%\n")
            f.write(f"\tTotal Average Reward: {avg_reward:.3f} +/- {std_reward:.3f}\n")
            f.write("=" * 50 + "\n")

    if ret_success:
        if ret_std:
            return avg_reward, std_reward, avg_success, std_success
        return avg_reward, avg_success
        
    return avg_reward, std_reward


def evaluate_metaworld_complex(
    policy,
    env_cls: str,
    task_list: List[List[str]],
    nfunc_list: List[List[int]] = None,
    h: int = None,
    eval_episodes: int = 100,
    deterministic: bool = True,
    save_path: str = None, 
    verbose: bool = True,
) -> None:
    # closed-loop models emit one primitive action per predict(); open-loop ones emit a
    # whole h-step action sequence, so only the latter need the SkillWrapper to unroll it.
    if h is None:
        h = 1 if getattr(policy, 'closed_loop', False) else policy.h
    
    if nfunc_list is None:
        nfunc_list = [None for _ in task_list]

    tot_success = []
    for task, nfunc in zip(task_list, nfunc_list):
        env = make_metaworld_env(
            env_cls=env_cls, 
            task=task, 
            nfunc=nfunc,
            seed=777,
        )
        env = SkillWrapper(env, h=h)
        
        _, _, avg_success, std_success = evaluate_skill_policy(
            policy,
            env,
            eval_episodes,
            deterministic,
            ret_success=True,
            ret_std=True,
            verbose=verbose,
        )

        if save_path is not None:
            with open(save_path, 'a') as f:
                f.write("="*5 + "Task: " + "-".join(task) + "="*5 + "\n")
                for i, (sub_task, success, _) in enumerate(zip(task, avg_success, std_success)):
                    f.write(f"\tTask {i} ({sub_task})\t: {success:.1f}%\n")
                f.write(f"\tTask Avg: {np.mean(avg_success):.1f}%\n")
        tot_success.append(np.mean(avg_success))

    if save_path is not None:
        with open(save_path, 'a') as f:
            f.write(f"\nTotal Success Rate: {np.mean(tot_success):.1f}%")
