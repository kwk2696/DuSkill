"""Behaviour-cloning baseline: flat (non-hierarchical) policy pretraining and few-shot
fine-tuning on Meta-World Complex.

Stages (select with the flags below; both may be combined in one run):
  --train_prior   behaviour-clone a flat policy on the full multi-task dataset
  --train_imit    fine-tune that policy on a few demonstrations of one test task
Add --evaluate to load the corresponding checkpoint and evaluate it instead of training.

Example:
    python scripts/main_bc.py --env_cls MC4-SP --train_prior --epochs 20000 --wandb
    python scripts/main_bc.py --env_cls MC4-SP --train_imit --task_idx 0 --shot 5
"""
import pickle

from duskill import (
    parse_args,
    make_dir,
    save_params,
    make_metaworld_env,
    get_task_config,
    evaluate_skill_policy,
    evaluate_metaworld_complex,
    print_r,
)
from duskill.utils.wrappers import SkillWrapper
from duskill.utils.callbacks import EvalCallback
from sb3_jax.common.buffers import OfflineBuffer
from sb3_jax.common.norm_layers import RunningNormLayer
from sb3_jax.bc.policies import MlpPolicy
from sb3_jax.bc.bc import BC


def make_offline_buffer(env, paths):
    buffer_size = 0 
    for path in paths:
        with open(path, "rb") as f:
            trajs = pickle.load(f)
        for traj in trajs:
            buffer_size += len(traj['actions'])
    print(f"Buffer size: {buffer_size}")
    
    buff = OfflineBuffer(
        buffer_size=buffer_size,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    for path in paths:
        with open(path, "rb") as f:
            trajs = pickle.load(f)
        for traj in trajs:
            obs = traj['observations'][0]
            for j in range(len(traj['actions'])):
                prev_obs = obs
                obs, act, rew = traj['next_observations'][j], traj['actions'][j], traj['rewards'][j]
                done = True if j == (len(traj['actions'])-1) else False
                buff.add(prev_obs, obs, act, rew, done, None)
    assert buffer_size == buff.buffer_size
    return buff


def train_prior(args):
    
    # 1. Get Task List
    train_tasks, test_tasks, train_nfuncs, test_nfuncs \
        = get_task_config(args.env_cls, args.config_cls) 
    
    ppath = f'./models/bc/{args.env_cls}/{args.tag0}/prior'
    wpath = f'bc/prior/{args.env_cls}/{args.tag0}'
    make_dir(ppath)

    if not args.evaluate:
        save_params(ppath, args)

        env = make_metaworld_env(
            env_cls=args.env_cls,
            task=train_tasks[0],
            nfunc=train_nfuncs[0],
            seed=777
        )
        
        dpath_list = []
        for task, nfunc in zip(train_tasks, train_nfuncs):
            nfunc = [str(s) for s in nfunc]
            dpath = f'./datasets/metaworld_complex/{args.env_cls}/'
            dpath += '-'.join(task) # task_names
            dpath += '-'
            dpath += '-'.join(nfunc)
            dpath += '/traj_20_noise.pkl'
            dpath_list.append(dpath)
            print(f"Loading ... {dpath}")
        buff = make_offline_buffer(env, dpath_list)

        # Hyper-parameters not exposed on the CLI are the paper settings.
        bc = BC(
            policy=MlpPolicy,
            env=env,
            replay_buffer=buff,
            learning_rate=1e-4,
            batch_size=args.batch_size,
            loss_type='neglogp',
            policy_kwargs=dict(
                log_std_init=0.,
                net_arch=[128]*5,
                normalization_class=RunningNormLayer,
                use_dist=True,
            ),
            wandb_log=wpath if args.wandb else None,
            verbose=1,
            seed=args.seed,
        )
        bc.learn(total_timesteps=args.epochs, log_interval=int(args.epochs/50))
        bc.save(ppath)
    else:
        print_r("<< BC Prior Evaluation on Train/Test Tasks >>")
        bc = BC.load(ppath)

        # BC is a flat policy: it emits a single primitive action per predict().
        evaluate_metaworld_complex(
            bc,
            args.env_cls,
            train_tasks,
            h=1,
            eval_episodes=args.n_episodes,
            deterministic=True,
            save_path=ppath+'/eval_train_tasks.txt',
        )
        evaluate_metaworld_complex(
            bc,
            args.env_cls,
            test_tasks,
            h=1,
            eval_episodes=args.n_episodes,
            deterministic=True,
            save_path=ppath+'/eval_test_tasks.txt',
        )


def train_imit(args):
    # 1. Make Environment    
    train_tasks, test_tasks, train_nfuncs, test_nfuncs = \
            get_task_config(args.env_cls, args.config_cls)
    task = test_tasks[args.task_idx]
    nfunc = test_nfuncs[args.task_idx]
    
    algo = 'imitation-tune'
    ppath = f'./models/bc/{args.env_cls}/{args.tag0}/prior'
    wpath = f'bc/{algo}/{args.env_cls}/{args.task_cls}{str(args.task_idx).zfill(2)}/{args.tag0}/{args.tag1}/'
    mpath = f'./models/bc/{args.env_cls}/{args.tag0}/{algo}/' 
    mpath += '-'.join(task) + f'-{args.task_cls}-{str(args.task_idx).zfill(2)}' + f'/{args.tag1}'
    make_dir(mpath)
    if not args.evaluate:
        save_params(mpath, args)

        str_nfunc = [str(s) for s in nfunc]

        # Whether to use nfunc limit for environment
        if args.no_nfunc: nfunc = None
        print_r(f"Speed Configuration ... {nfunc}")

        env = make_metaworld_env(
            env_cls=args.env_cls,
            task=task,
            nfunc=nfunc,
            seed=777,
        )
        env = SkillWrapper(env)

        dpath = f'./datasets/metaworld_complex/{args.env_cls}/'
        dpath += '-'.join(task) # task_names
        dpath += '-'
        dpath += '-'.join(str_nfunc)
        dpath += f'/traj_{args.shot}.pkl' 
        print(f"Loading ... {dpath}")
        buff = make_offline_buffer(env, [dpath])

        # BC.load() rebuilds the model from the checkpoint, so the training settings that are
        # not part of the checkpoint have to be re-applied afterwards.
        bc = BC.load(ppath, env=env)
        bc.replay_buffer = buff
        bc.learning_rate = 1e-4
        bc.wandb_log = wpath if args.wandb else None
        bc.batch_size = args.batch_size

        callback = EvalCallback(env=env, eval_path=mpath, eval_freq=int(args.epochs/20), eval_episodes=args.n_episodes)
        bc.learn(total_timesteps=args.epochs, log_interval=int(args.epochs/50), callback=callback)
        bc.save(mpath)
        evaluate_skill_policy(
            bc, 
            env, 
            eval_episodes=args.n_episodes, 
            deterministic=True, 
            save_path=mpath+'/eval.txt',
        )
    else:
        # Evaluation
        print_r("<< BC One-shot Imitation Evaluation on Test Task >>")
        bc = BC.load(mpath)
        
        # Whether to use nfunc limit for environment
        if args.no_nfunc: nfunc = None
        print_r(f"Speed Configuration ... {nfunc}")

        env = make_metaworld_env(
            env_cls=args.env_cls,
            task=task,
            nfunc=nfunc,
            seed=777
        )
        env = SkillWrapper(env)

        # Save video
        if args.video:
            evaluate_skill_policy(
                bc, 
                env, 
                eval_episodes=1, 
                deterministic=True, 
                video_path=mpath+'/'+'-'.join(task)+'.avi',
            )
            return 

        # Save evaluation
        evaluate_skill_policy(
            bc,
            env,
            eval_episodes=args.n_episodes,
            deterministic=True,
            save_path=mpath+'/reval.txt',
        )


if __name__ == "__main__":
    args = parse_args()
    if args.train_prior:
        print_r("<< BC Prior Learning >>")
        train_prior(args)
    if args.train_imit:
        print_r("<< BC One-shot Imitation >>")
        train_imit(args)
