# rlvr

> Auto-maintained by Agent. Do not edit manually.

## Purpose
Houses Reinforcement Learning with Verifiable Rewards (RLVR) curricula, Kaggle Grandmaster notebooks, reproducible research benchmarks, and verifiable post-training workflows.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `README.md` | File | Overview of RLVR curriculum, installation requirements, and Kaggle benchmarks |
| `__init__.py` | File | Module package initialization |
| `kaggle_showcase/` | Dir | [kaggle_showcase/INDEX.md](kaggle_showcase/INDEX.md) - Production Kaggle notebook package and metadata |
| `notebooks/` | Dir | [notebooks/INDEX.md](notebooks/INDEX.md) - Executable Jupyter notebooks demonstrating GRPO and RLVR workflows |
| `research/` | Dir | [research/INDEX.md](research/INDEX.md) - Post-training research monographs and frontier paradigms |
| `tutorials/` | Dir | [tutorials/INDEX.md](tutorials/INDEX.md) - 18-part post-training MLE and verifiable rewards curriculum |

## Invariants & Rules
- Notebooks should adhere to idempotent execution standards so cells run cleanly top-to-bottom.
- Verifiable reward functions must produce deterministic, reproducible boolean or scalar signals.
