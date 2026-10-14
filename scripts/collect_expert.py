"""Collect scripted-expert demonstrations for Meta-World Complex.

Stages:
  (default)  sweep every non-functionality (speed) index for the full task set and dump
             one dataset per speed, as a sanity check on the scripted experts
  --collect  collect, for every train/test task in the environment config, the datasets
             used by the rest of the pipeline: a noisy pretraining set, a clean evaluation
             set, and 1-/3-/5-/10-shot imitation sets

Example:
    python scripts/collect_expert.py --env_cls MC4-SP --collect --n_collect 20
"""
import os
import json

import numpy as np

from duskill import (
    parse_args,
    make_dir,
    make_metaworld_env,
    collect_metaworld_exp,
    get_task_list,
    get_task_config,
    print_r,
    print_b,
    print_y,
)


def print_json_summary(path, keys=('avg',)):
    """Print selected fields of a previously collected dataset summary, if it exists."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        data = json.load(f)
    for key in keys:
        print(data[key])


def test(args):

    # 1. Get all possible task list
    task_set, _, _ = get_task_list(args.env_cls)
    if args.novel is not None:
        if args.novel == 1: task_set[1] = 'handle'
        if args.novel == 2: task_set[2] = 'lever'

    nfunc_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    for i in nfunc_idx:
        # 2. Get non-functionality
        nfunc = [i for _ in range(len(task_set))]
        print_r(f"<< Expert at Speed {nfunc} >>")

        dpath = f'./datasets/metaworld_complex/{args.env_cls}/'
        make_dir(dpath)
        nfunc_str = [str(s) for s in nfunc]
        dpath += '-'.join(task_set) + '-' + '-'.join(nfunc_str)
        print_y(f"Saving at {dpath}")

        # 3. Make environment (env_exp=True also returns the scripted experts)
        env, exp = make_metaworld_env(
            env_cls=args.env_cls,
            task=task_set,
            nfunc=nfunc,
            env_exp=True,
            seed=777,
        )

        # 4. Collect expert datasets
        collect_metaworld_exp(
            env_cls=args.env_cls,
            env=env,
            exp=exp,
            exp_noise=args.exp_noise,
            n_episodes=args.n_collect,
            save_video=args.video,
            save_path=dpath,
            verbose=True,
        )


def collect(args):

    # 1. Get the train/test tasks of the environment config
    train_tasks, test_tasks, train_nfuncs, test_nfuncs = \
            get_task_config(args.env_cls, args.config_cls)

    for i, (task, nfunc_idx) in enumerate(zip(train_tasks + test_tasks, train_nfuncs + test_nfuncs)):
        if i == len(train_tasks):
            print_b("\n==================================\n")
        nfunc = [s for s in nfunc_idx]

        dpath = f'./datasets/metaworld_complex/{args.env_cls}/'
        make_dir(dpath)
        nfunc_str = [str(s) for s in nfunc_idx]
        dpath += '-'.join(task) + '-' + '-'.join(nfunc_str)
        print_y(f"Saving at {dpath}")

        # Resume: skip tasks that were already collected, printing their summary instead.
        if os.path.exists(dpath):
            print_r(f"Already exists ! {dpath}")
            np.set_printoptions(precision=3, suppress=True)
            print_json_summary(dpath + '/traj_1_step.json')
            print_json_summary(dpath + f'/traj_{args.n_collect}_step.json', ('avg', 'std'))
            print(" x x x x x x x x")
            print_json_summary(dpath + '/traj_1_rewA.json')
            print_json_summary(dpath + f'/traj_{args.n_collect}_rewA.json', ('avg', 'std'))
            continue

        # 2. Make environment (env_exp=True also returns the scripted experts)
        env, exp = make_metaworld_env(
            env_cls=args.env_cls,
            task=task,
            nfunc=nfunc,
            env_exp=True,
            novel=args.novel,
            seed=777,
        )

        # 3. Collect: Pretraining demo (noisy)
        collect_metaworld_exp(
            env_cls=args.env_cls,
            env=env,
            exp=exp,
            exp_noise=True,
            n_episodes=args.n_collect,
            save_video=False,
            save_path=dpath,
            verbose=True,
        )

        # 4. Collect: Evaluation demo (clean)
        collect_metaworld_exp(
            env_cls=args.env_cls,
            env=env,
            exp=exp,
            exp_noise=False,
            n_episodes=args.n_collect,
            save_video=False,
            save_path=dpath,
            verbose=True,
        )

        # 5. Collect: One-shot demo
        collect_metaworld_exp(
            env_cls=args.env_cls,
            env=env,
            exp=exp,
            # no need noise for few-shot demos
            exp_noise=False,
            n_episodes=1,
            save_video=args.video,
            save_path=dpath,
            verbose=True,
        )

        # 6. Collect: 3-shot demo
        collect_metaworld_exp(
            env_cls=args.env_cls,
            env=env,
            exp=exp,
            exp_noise=False,
            n_episodes=3,
            save_video=args.video,
            save_path=dpath,
            verbose=True,
        )

        # 7. Collect: 5-shot demo
        collect_metaworld_exp(
            env_cls=args.env_cls,
            env=env,
            exp=exp,
            exp_noise=False,
            n_episodes=5,
            save_video=False,
            save_path=dpath,
            verbose=True,
        )

        # 8. Collect: 10-shot demo
        collect_metaworld_exp(
            env_cls=args.env_cls,
            env=env,
            exp=exp,
            exp_noise=False,
            n_episodes=10,
            save_video=False,
            save_path=dpath,
            verbose=True,
        )


if __name__ == "__main__":
    args = parse_args()
    if args.collect:
        collect(args)
    else:
        test(args)
