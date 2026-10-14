"""Entry point for the two-stage DuSkill pipeline.

Stage 1 - skill prior: pre-train the LSTM skill encoder, the guided diffusion
action decoder and the skill/preference priors on the offline multi-preference
dataset.

    python scripts/main_duskill.py --train_prior --env_cls MC4-SP --tag0 v0

Stage 2 - few-shot imitation: fit a high-level skill/pref policy on a handful of
demonstrations of a target task, on top of the frozen stage-1 decoder.

    python scripts/main_duskill.py --train_imit --env_cls MC4-SP --tag0 v0 \
        --task_cls test --task_idx 0 --shot 5

Add --evaluate to any stage to load the saved model and evaluate it instead of
training, and --wandb to mirror the logs to Weights & Biases.
"""
from duskill import (
    parse_args,
    make_dir,
    make_metaworld_env,
    evaluate_skill_policy,
    evaluate_metaworld_complex,
    get_task_config,
    print_r,
    save_params,
)
from duskill.utils.wrappers import SkillWrapper
from duskill.utils.buffers import SkillBuffer
from duskill.duskill.prior import DuSkillPrior
from duskill.duskill.imitation import DuSkillIM


def train_prior(args):

    # 1. Get Task List
    train_tasks, test_tasks, train_nfuncs, test_nfuncs \
        = get_task_config(args.env_cls, args.config_cls) 
    
    ppath = f'./models/duskill/{args.env_cls}/{args.tag0}/prior'
    wpath = f'duskill/prior/{args.env_cls}/{args.tag0}'
    make_dir(ppath)
    
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
        action_space=env.action_space,
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

    if not args.evaluate:
        save_params(ppath, args)
        
        # 3. Create Model
        prior = DuSkillPrior(
            observation_space=env.observation_space,
            action_space=env.action_space,
            epochs=args.epochs,
            learning_rate=1e-3,
            batch_size=args.batch_size,
            h=args.h,
            beta=args.beta,
            pf_beta=args.pf_beta,
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
            n_denoise=args.n_denoise,
            guide_wei=args.guide_wei,
            drop_rate=args.drop_rate,
            wandb_log=wpath if args.wandb else None,
            verbose=1,
            seed=args.seed,
        )
        prior.learn(epochs=args.epochs, log_interval=1000)
        prior._test(verbose=True)
        prior.save(ppath)
    else:
        prior = DuSkillPrior.load(ppath)
        prior.replay_buffer = buff
        for i in range(10):
            prior._test(verbose=True)

        if args.task_idx is not None:
            print_r(f"<< DuSkill Prior Evaluation on {args.task_cls} {args.task_idx}>>")
            task = train_tasks[args.task_idx] if args.task_cls == "train" \
                else test_tasks[args.task_idx]
            
            nfunc = train_nfuncs[args.task_idx] if args.task_cls == "train" \
                else test_nfuncs[args.task_idx]

            env = make_metaworld_env(
                env_cls=args.env_cls,
                task=task,
                seed=777
            )
            env = SkillWrapper(env)
            avg_reward, std_reward = evaluate_skill_policy(
                prior, 
                env, 
                eval_episodes=args.n_episodes, 
                deterministic=True, 
                video_path=ppath+'/'+'-'.join(task)+'.avi',
            )
        else:
            # Evlauation on train tasks
            print_r("<< DuSkill Prior Evaluation on Train Tasks>>")
            evaluate_metaworld_complex(
                prior,
                args.env_cls,
                train_tasks,
                h=1,
                eval_episodes=args.n_episodes,
                deterministic=True,
                save_path=ppath+'/eval_train_tasks.txt',
            )
            # Evaluation on test tasks
            print_r("<< DuSkill Prior Evaluation on Test Tasks >>")
            evaluate_metaworld_complex(
                prior,
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
    if args.task_cls == 'test':
        task = test_tasks[args.task_idx]
        nfunc = test_nfuncs[args.task_idx]
    else:
        task = train_tasks[args.task_idx]
        nfunc = train_nfuncs[args.task_idx]
    
    algo = 'imitation'
    if args.tune_dec:
        if args.tune_dec_type == 1:
            algo = 'imitation-tune-pref'
        elif args.tune_dec_type == 2:
            algo = 'imitation-tune-both'
        else:
            algo = 'imitation-tune'
    ppath = f'./models/duskill/{args.env_cls}/{args.tag0}/prior'
    wpath = f'duskill/{algo}/{args.env_cls}/{args.task_cls}{str(args.task_idx).zfill(2)}/{args.tag0}/{args.tag1}/'
    mpath = f'./models/duskill/{args.env_cls}/{args.tag0}/{algo}/' 
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

        # keep the intervals positive even for very small --epochs
        log_interval = max(1, int(args.epochs / 50))
        eval_interval = max(1, int(args.epochs / 20))

        duskill_im = DuSkillIM(
            env=env,
            observation_space=obs_space,
            action_space=act_space,
            epochs=args.epochs,
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size,
            replay_buffer=buff,
            net_arch=dict(
                sk_pi=[128]*5,
                pf_pi=[128]*5,
                dec=[128]*5,
            ),
            prior_path=ppath,
            tune_dec=args.tune_dec,
            tune_dec_type=args.tune_dec_type,
            eval_interval=eval_interval,
            eval_episodes=args.n_episodes,
            eval_path=mpath,
            wandb_log=wpath if args.wandb else None,
            verbose=1,
            seed=args.seed,
        )
         
        duskill_im.learn(epochs=args.epochs, log_interval=log_interval)
        duskill_im._test(verbose=True)
        duskill_im.save(mpath)
        avg_reward, std_reward = evaluate_skill_policy(
            duskill_im,
            env,
            eval_episodes=args.n_episodes,
            deterministic=True,
            save_path=mpath+'/eval.txt',
        )
        print_r(f"Saving {mpath}")
    else:
        # Evlauation on train tasks
        print_r("<< DuSkill One-shot Imitation Evaluation on Test Task >>")
        duskill_im = DuSkillIM.load(mpath)

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
            avg_reward, std_reward = evaluate_skill_policy(
                duskill_im, 
                env, 
                eval_episodes=1, 
                deterministic=True, 
                video_path=mpath+'/'+'-'.join(task)+'.avi',
            )
            return

        # Save evaluation
        avg_reward, std_reward = evaluate_skill_policy(
            duskill_im,
            env,
            eval_episodes=args.n_episodes,
            deterministic=True,
            save_path=mpath+'/reval.txt',
        )


if __name__ == "__main__":
    args = parse_args()
    if args.train_prior:
        print_r("<< DuSkill Prior Learning >>")
        train_prior(args)
    if args.train_imit:
        print_r("<< DuSkill One-shot Imitation >>")
        train_imit(args)
