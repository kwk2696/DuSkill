"""Closed-loop SPiRL (SPiRLc) baseline: skill-prior pretraining and few-shot imitation
on Meta-World Complex.

Stages (select with the flags below; both may be combined in one run):
  --train_prior   pretrain the closed-loop skill VAE + prior on the multi-task dataset
  --train_imit    few-shot behaviour cloning of a high-level policy on one test task
Add --evaluate to load the corresponding checkpoint and evaluate it instead of training.

Example:
    python scripts/main_spirl_c.py --env_cls MC4-SP --train_prior --epochs 20000 --wandb
    python scripts/main_spirl_c.py --env_cls MC4-SP --train_imit --task_idx 0 --shot 5
"""
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
from duskill.utils.buffers import SkillBuffer
from duskill.spirl_c.prior import SPiRLcPrior
from duskill.spirl_c.imitation import SPiRLcIM


def train_prior(args):

    # 1. Get Task List
    train_tasks, test_tasks, train_nfuncs, test_nfuncs \
        = get_task_config(args.env_cls, args.config_cls) 
    
    ppath = f'./models/spirl_c/{args.env_cls}/{args.tag0}/prior'
    wpath = f'spirl_c/prior/{args.env_cls}/{args.tag0}'
    make_dir(ppath)
    if not args.evaluate:
        save_params(ppath, args)

        env = make_metaworld_env(
            env_cls=args.env_cls,
            task=train_tasks[0],
            seed=777
        )

        # 2. Make Buffer
        buff = SkillBuffer(
            h=args.h,
            norm_obs=False,
            observation_space=env.observation_space,
            action_space=env.action_space
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
        buff.load_traj(path=dpath_list)

        # 3. Create Model (hyper-parameters not exposed on the CLI are the paper settings)
        prior = SPiRLcPrior(
            observation_space=env.observation_space,
            action_space=env.action_space,
            epochs=args.epochs,
            learning_rate=1e-3,
            batch_size=args.batch_size,
            h=args.h,
            beta=args.beta,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            squash_z=args.squash_z,
            replay_buffer=buff,
            net_arch=dict(
                enc=[128]*5, 
                dec=[128]*5,
                pri=[128]*5,
            ),
            wandb_log=wpath if args.wandb else None,
            verbose=1,
            seed=args.seed,
        )
        prior.learn(epochs=args.epochs, log_interval=1000)
        prior._test(verbose=True)
        prior.save(ppath)
    else:
        prior = SPiRLcPrior.load(ppath)

        if args.task_idx is not None:
            print_r(f"<< SPiRL-c Prior Evaluation on {args.task_cls} {args.task_idx}>>")
            task = train_tasks[args.task_idx] if args.task_cls == "train" \
                else test_tasks[args.task_idx]

            env = make_metaworld_env(
                env_cls=args.env_cls,
                task=task,
                seed=777
            )
            env = SkillWrapper(env)
            evaluate_skill_policy(
                prior, 
                env, 
                eval_episodes=args.n_episodes, 
                deterministic=True, 
                video_path=ppath+'/'+'-'.join(task)+'.avi',
            )
        else:
            # Evaluation
            print_r("<< SPiRLc Prior Evaluation on Train Tasks>>")
            evaluate_metaworld_complex(
                prior,
                args.env_cls,
                train_tasks,
                eval_episodes=args.n_episodes,
                deterministic=True,
                save_path=ppath+'/eval_train_tasks.txt',
            )
            # Evaluation on test tasks
            print_r("<< SPiRLc Prior Evaluation on Test Tasks >>")
            evaluate_metaworld_complex(
                prior,
                args.env_cls,
                test_tasks,
                eval_episodes=args.n_episodes,
                deterministic=True,
                save_path=ppath+'/eval_test_tasks.txt',
            )


def train_imit(args):
    # 1. Make Environment    
    train_tasks, test_tasks, train_nfuncs, test_nfuncs = \
            get_task_config(args.env_cls, args.config_cls)
    if args.task_cls == 'test':
        task = test_tasks[args.task_idx]
        nfunc = test_nfuncs[args.task_idx]
    else:
        task = train_tasks[args.task_idx]
        nfunc = train_nfuncs[args.task_idx]
    
    algo = 'imitation'
    if args.tune_dec:
        algo = 'imitation-tune'
    ppath = f'./models/spirl_c/{args.env_cls}/{args.tag0}/prior'
    wpath = f'spirl_c/{algo}/{args.env_cls}/{args.task_cls}{str(args.task_idx).zfill(2)}/{args.tag0}/{args.tag1}/'
    mpath = f'./models/spirl_c/{args.env_cls}/{args.tag0}/{algo}/' 
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
        obs_space, act_space = env.observation_space, env.action_space
        env = SkillWrapper(env)
        
        # 2. Make Buffer
        buff = SkillBuffer(
            h=args.h,
            norm_obs=False,
            observation_space=obs_space,
            action_space=act_space,
        )

        dpath = f'./datasets/metaworld_complex/{args.env_cls}/'
        dpath += '-'.join(task) # task_names
        dpath += '-'
        dpath += '-'.join(str_nfunc)
        dpath += f'/traj_{args.shot}.pkl'
        print(f"Loading ... {dpath}")
        buff.load_traj(path=dpath)

        spirl_c_im = SPiRLcIM(
            env=env,
            observation_space=obs_space,
            action_space=act_space,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            replay_buffer=buff,
            net_arch=dict(
                sk_pi=[128]*5,
                sk_dec=[128]*5,
            ),
            prior_path=ppath,
            tune_dec=args.tune_dec,
            eval_interval=int(args.epochs/20),
            eval_episodes=args.n_episodes,
            eval_path=mpath,
            wandb_log=wpath if args.wandb else None,
            verbose=1,
            seed=args.seed,
        )
        
        spirl_c_im.learn(epochs=args.epochs, log_interval=int(args.epochs/50))
        spirl_c_im._test(verbose=True)
        spirl_c_im.save(mpath)
        evaluate_skill_policy(
            spirl_c_im,
            env,
            eval_episodes=args.n_episodes,
            deterministic=True,
            save_path=mpath+'/eval.txt',
        )
        print_r(f"Saving {mpath}")
    else:
        # Evaluation
        print_r("<< SPiRLcIM One-shot Imitation Evaluation on Test Task >>")
        spirl_c_im = SPiRLcIM.load(mpath)

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
                spirl_c_im, 
                env, 
                eval_episodes=1, 
                deterministic=True, 
                video_path=mpath+'/'+'-'.join(task)+'.avi',
            )
            return 

        # Save evaluation
        evaluate_skill_policy(
            spirl_c_im,
            env,
            eval_episodes=args.n_episodes,
            deterministic=True,
            save_path=mpath+'/reval.txt',
        )


if __name__ == "__main__":
    args = parse_args()
    if args.train_prior:
        print_r("<< SPiRLc Prior Learning >> ")
        train_prior(args)
    if args.train_imit:
        print_r("<< SPiRLc One-shot Imitation >>")
        train_imit(args)
