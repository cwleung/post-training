# kaggle_showcase

> Auto-maintained by Agent. Do not edit manually.

## Purpose
Maintains self-contained Kaggle Grandmaster showcase notebooks and competition metadata for RLVR post-training solutions.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `README.md` | File | Showcase overview and deployment instructions |
| `kernel-metadata.json` | File | Kaggle CLI kernel configuration and dataset binding metadata |
| `post_training_mle_showcase.ipynb` | File | Complete end-to-end runnable notebook demonstrating post-training MLE techniques |

## Invariants & Rules
- Notebooks must be runnable within Kaggle GPU/TPU resource constraints without external secret leakage.
