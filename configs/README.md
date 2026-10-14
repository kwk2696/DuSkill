# Benchmark configurations

Each config file lists the ordered skill sequences (`train_tasks` / `test_tasks`)
and, for every sequence, one non-functionality (nfunc) level per skill
(`train_nfuncs` / `test_nfuncs`). An nfunc level is an index into the per-skill
level table in `duskill/utils/env_utils.py` (`SPEED_NFUNCS`, `ENERGY_NFUNCS`,
`WIND_NFUNCS`); the levels actually used by the benchmark are `{0, 2, 4, 6, 8}`.

Files live under `configs/metaworld_complex/<MC4-DOMAIN>/`, where the domain is
`SP` (speed), `EN` (energy) or `WD` (wind).

## How a config is selected

`get_task_config(env_cls, config_cls)` resolves the file as follows:

* `--env_cls MC4-SP` (no numeric suffix) — loads
  `configs/metaworld_complex/MC4-SP/<--config_cls>`, default `config.json`.
* `--env_cls MC4-SP-<k>` — **forces** `configs/metaworld_complex/MC4-SP/config_<k>.json`
  and **ignores `--config_cls` entirely**.

## Available configs

| File | Domains | Train | Test | nfunc levels | Notes |
| --- | --- | --- | --- | --- | --- |
| `config.json` | SP, EN, WD | 12 | 45 | 0, 2, 4, 6, 8 | Main benchmark. |
| `config_0.json` | SP, EN, WD | 6 | 4 (SP/WD), 3 (EN) | 4 only | Single nfunc level; SP/WD test set adds one `handle` sequence. |
| `config_1.json` | SP, EN, WD | 6 | 3 | 2, 6 | Two nfunc levels. |
| `config_2.json` | SP only | 12 | 21 | 0, 2, 4, 6, 8 | Same train split as `config.json`; novel-skill test set including `handle` and `lever`. |
| `config_3.json` | SP, EN, WD | 12 | 18 | 0, 2, 4, 6, 8 | Same train split as `config.json`; reduced test set. |
| `config_4.json` | SP, EN, WD | 12 | 9 | 0, 2, 4, 6, 8 | Same train sequences as `config.json` but an alternative train nfunc assignment; reduced test set. |
| `config_smoke.json` | SP only | 2 | 1 | 0, 2, 6 | Tiny config for checking the installation end to end (see README, "Quick test"). |

Train/test counts are numbers of skill sequences (task tuples). Skills are
`puck`, `drawer`, `button` and `door`; `handle` and `lever` appear only in test
sets as novel skills.
