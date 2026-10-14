import argparse
from typing import Any, Dict


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument("--env_cls", type=str, default="MC4-SP")   # MC4-SP | MC4-EN | MC4-WD, optionally with a config suffix, e.g. MC4-SP-3
    parser.add_argument("--task_cls", type=str, default="test") # train, test
    parser.add_argument("--task_idx", type=int, default=None) # Target task index; required for --train_imit. Omit with --train_prior --evaluate to evaluate every train/test task
    parser.add_argument("--config_cls", type=str, default='config.json') # Configuration path
    parser.add_argument("--h", type=int, default=10)            # Skill length
    parser.add_argument("--novel", type=int, default=None)      # Novel task
    # expert
    parser.add_argument("--exp_noise", action="store_true")     # Add Gaussian noise to expert actions
    parser.add_argument("--shot", type=int, default=5)          # Number of demonstrations for few-shot imitation
    parser.add_argument("--n_collect", type=int, default=20)    # Number of expert trajectories to collect for pretraining
    # prior
    parser.add_argument("--epochs", type=int, default=10_000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--beta", type=float, default=5e-4)         # KL regularization
    parser.add_argument("--pf_beta", type=float, default=1e-4)      # KL regularization
    parser.add_argument("--embed_dim", type=int, default=64)        # Embedding dimension
    parser.add_argument("--hidden_dim", type=int, default=128)      # LSTM Hidden dimension
    parser.add_argument("--latent_dim", type=int, default=32)       # Skill dimension
    parser.add_argument("--n_denoise", type=int, default=50)        # Denoising timesteps
    parser.add_argument("--guide_wei", type=float, default=.1)      # Guidance weight
    parser.add_argument("--drop_rate", type=float, default=.1)      # Guidance drop rate
    parser.add_argument("--squash_z", action="store_true")          # Squash z with tanh
    # imitation
    parser.add_argument("--tune_dec", action="store_true")          # Finetuning decoder
    parser.add_argument("--tune_dec_type", type=int, default=0)     # Decoder finetuning variant
    parser.add_argument("--no_nfunc", action="store_true")          # No speed configuration
    # miscs
    parser.add_argument("--train_prior", action="store_true")       # Pretraining skill with dataset
    parser.add_argument("--train_imit", action="store_true")        # One-shot Imitation Learning
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--wandb", action="store_true")             # Log to Weights & Biases
    parser.add_argument("--n_episodes", type=int, default=20)       # Number of episodes used for evaluation
    parser.add_argument("--tag0", type=str, default='v0')
    parser.add_argument("--tag1", type=str, default='v0')
    args = parser.parse_args()
    if args.train_imit and args.task_idx is None:
        parser.error("--task_idx is required with --train_imit")
    return args
