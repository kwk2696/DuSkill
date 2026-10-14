# DuSkill: Robust Policy Learning via Offline Skill Diffusion

Official implementation of **"Robust Policy Learning via Offline Skill Diffusion"**
(Woo Kyung Kim, Minjong Yoo, Honguk Woo), AAAI 2024.
[[Paper]](https://doi.org/10.1609/aaai.v38i12.29217) [[arXiv]](https://arxiv.org/abs/2403.00225)

DuSkill learns a skill space from an offline dataset that covers several source
domains, and uses that skill space to adapt quickly to a target domain from a
few demonstrations. Two components make the skill space robust to domain shift:

* a **hierarchical domain encoder** that splits each skill into a
  domain-invariant embedding (what the skill does) and a domain-variant
  embedding (how the domain changes the way it is executed), and
* a **guided diffusion skill decoder** that generates actions from the two
  embeddings with classifier-free-style guidance, so the decoder can
  extrapolate beyond the behaviours in the dataset.

This release contains the two stages used for the few-shot imitation results in
the paper: skill-prior pretraining and few-shot imitation. It also contains the
SPiRL, SPiRL-c, FIST and BC baselines and the multi-stage Meta-World benchmark
with its speed, energy and wind domain variations.

## Repository layout

```
duskill/
  duskill/        DuSkill: skill prior (prior.py) and few-shot imitation (imitation.py)
  spirl/          SPiRL baseline (open-loop skill decoder)
  spirl_c/        SPiRL-c baseline (closed-loop skill decoder)
  fist/           FIST baseline
  common/         shared modules: diffusion decoder, VAE/LSTM encoders, base classes
  utils/          environment construction, buffers, evaluation, CLI arguments
  envs/
    metaworld_complex/   multi-stage Meta-World environment and scripted experts
    metaworld/           modified copy of Meta-World (see NOTICE)
scripts/
  collect_expert.py      build the offline datasets with scripted experts
  main_duskill.py        DuSkill: --train_prior / --train_imit
  main_spirl.py, main_spirl_c.py, main_fist.py, main_bc.py   baselines
configs/metaworld_complex/   task sequences and domain levels (see configs/README.md)
```

## Installation

Tested with Python 3.8+ on Linux. Two dependencies are not on PyPI and must be
installed from source first, in this order:

```bash
# 1. JAX/Haiku port of Stable-Baselines3
pip install git+https://github.com/kwk2696/sb3-jax-haiku.git

# 2. The Meta-World fork shipped in this repository (upstream Meta-World will not work:
#    the multi-stage environment relies on hooks and assets that exist only in this copy)
pip install -e duskill/envs/metaworld

# 3. Remaining dependencies and this package
pip install -r requirements.txt
pip install -e .
```

Notes:

* The environments use `mujoco-py`, which needs a MuJoCo installation and the
  usual `LD_LIBRARY_PATH` / `MUJOCO_PY_MUJOCO_PATH` setup. See the
  [mujoco-py README](https://github.com/openai/mujoco-py) for your platform.
* `gym==0.21` is required (the code uses the pre-0.26 `step()` API). Installing it
  needs `setuptools<66` and `wheel<0.40`.
* `sb3-jax-haiku` depends on `stable-baselines3<2.0`, which pulls in PyTorch even
  though all models here are JAX.
* Logging to Weights & Biases is optional. Pass `--wandb` to any script to enable it.

All commands below are run from the repository root. Datasets are read from
`./datasets/` and checkpoints are written to `./models/`.

## Benchmark

The benchmark is a multi-stage version of Meta-World. Every task is a sequence
of four sub-tasks (`puck`, `drawer`, `button`, `door`, in a task-specific order),
and every sub-task is executed under one of several *non-functionality* levels
that define the domain:

| `--env_cls` | Domain variation |
| --- | --- |
| `MC4-SP` | speed of the arm |
| `MC4-EN` | energy (acceleration) of the arm |
| `MC4-WD` | wind applied to the arm |

The train/test split of task sequences and the per-sub-task levels are defined
in `configs/metaworld_complex/<env_cls>/config.json`. Appending a numeric suffix
(`MC4-SP-3`) selects `config_3.json` instead; see
[configs/README.md](configs/README.md) for what each variant contains.

## Quick test

To check the installation end to end on a tiny configuration (two training
tasks, one test task, a few hundred epochs), run:

```bash
python scripts/collect_expert.py --env_cls MC4-SP --collect --config_cls config_smoke.json
python scripts/main_duskill.py --train_prior --env_cls MC4-SP --config_cls config_smoke.json \
    --epochs 300 --tag0 smoke --n_episodes 2
python scripts/main_duskill.py --train_imit  --env_cls MC4-SP --config_cls config_smoke.json \
    --epochs 300 --tag0 smoke --task_idx 0 --shot 5 --n_episodes 2
```

The success rates will be near zero at this budget; the point is that data
collection, training, checkpointing and evaluation all run through. The
baseline scripts accept the same arguments.

## 1. Collect the datasets

Scripted experts generate the offline data. This writes, for each train and
test task sequence, a noisy pretraining set (`traj_20_noise.pkl`), a clean
evaluation set (`traj_20.pkl`) and 1/3/5/10-shot imitation sets
(`traj_<k>.pkl`) under
`datasets/metaworld_complex/<env_cls>/<task>-<levels>/`.

```bash
python scripts/collect_expert.py --env_cls MC4-SP --collect --n_collect 20
```

Repeat for `MC4-EN` and `MC4-WD` as needed. The pretraining and evaluation
scripts expect exactly these file names, so keep `--n_collect 20`.

## 2. Train the skill prior

```bash
python scripts/main_duskill.py --train_prior --env_cls MC4-SP \
    --epochs 100000 --beta 5e-4 --guide_wei 0.5 --tag0 d0
```

This trains the LSTM skill encoder, the preference (domain) encoder, the
guided diffusion decoder and the two priors on the noisy pretraining sets of
all training tasks, and saves them to `models/duskill/MC4-SP/d0/prior/`.

Useful options: `--latent_dim` (skill dimension, 32), `--n_denoise` (diffusion
steps, 50), `--guide_wei` (guidance weight), `--drop_rate` (guidance dropout),
`--pf_beta` (KL weight of the domain prior). Add `--evaluate` to load the
saved prior and roll it out on every train and test task instead of training;
add `--task_idx k` to evaluate a single task and record a video of it.

## 3. Few-shot imitation on a target task

```bash
python scripts/main_duskill.py --train_imit --env_cls MC4-SP --tag0 d0 \
    --task_cls test --task_idx 0 --shot 5 \
    --epochs 100000 --learning_rate 1e-3 --drop_rate 0.0 --tag1 v5
```

`--task_idx` indexes the `test_tasks` list of the config (use `--task_cls train`
for a training task), `--shot` selects the
`traj_<shot>.pkl` demonstration file. The high-level policy is trained on top
of the frozen stage-2 decoder. To also fine-tune the decoder, add
`--tune_dec --tune_dec_type 1` (domain branch only) or `--tune_dec_type 2`
(both branches). Results are written next to the checkpoint under
`models/duskill/MC4-SP/d0/imitation*/`, and `--evaluate` re-runs the
evaluation from a saved checkpoint. `--video` records one episode instead.

Evaluation reports success rate on the full four-stage task; the number of
evaluation episodes is set with `--n_episodes` (20).

## Baselines

Every baseline follows the same two-stage interface:

```bash
python scripts/main_spirl.py   --train_prior --env_cls MC4-SP --epochs 100000 --tag0 d0
python scripts/main_spirl.py   --train_imit  --env_cls MC4-SP --tag0 d0 --task_idx 0 --shot 5

python scripts/main_spirl_c.py ...   # SPiRL with a closed-loop decoder
python scripts/main_fist.py    ...   # FIST (semi-parametric few-shot imitation)
python scripts/main_bc.py      ...   # flat behaviour cloning
```

## Notes on this release

* Only the few-shot imitation setting is included. The online RL adaptation
  experiments of the paper and the ablation variants are not part of this
  release.
* Compared with the code used for the paper, two defects were fixed that can
  change training dynamics: the diffusion loss now uses independent PRNG keys
  for the noise, the timestep and the model's reparameterisation, and
  in-training evaluation (best-checkpoint tracking) is no longer gated behind
  80,000 epochs. Numbers may therefore differ slightly from the published
  tables.
* The environment construction reads `datasets/.../traj_20_step.json`
  (speed) or `traj_20_rewA.json` (energy) to set the per-task limit. Run the
  dataset collection before training, otherwise the limit is silently
  disabled.

## Citation

```bibtex
@inproceedings{kim2024duskill,
  title     = {Robust Policy Learning via Offline Skill Diffusion},
  author    = {Kim, Woo Kyung and Yoo, Minjong and Woo, Honguk},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {38},
  number    = {12},
  pages     = {13177--13184},
  year      = {2024},
  doi       = {10.1609/aaai.v38i12.29217}
}
```

## License

This repository is released under the MIT License (see [LICENSE](LICENSE)).
`duskill/envs/metaworld/` is a modified copy of Meta-World (MIT, Meta-World
Team); its license and a list of modifications are kept in that directory.
Note that the `sb3-jax-haiku` dependency is distributed under GPL-3.0.
