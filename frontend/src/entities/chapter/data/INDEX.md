# data (frontend/src/entities/chapter/data)

> Auto-maintained by Agent. Do not edit manually.

## Purpose
Pre-bundled JS curriculum metadata for the three learning tracks (`deepagents`, `rl`, `rlvr`), containing chapter titles, reading times, visualizer bindings, intuitive summaries, and structured table of contents.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `deepagents/` | Dir | 31 chapters covering Agent Harness Engineering and evaluation loops |
| `rl/` | Dir | 18 chapters covering classical RL, PPO, DPO, GRPO, and alignment |
| `rlvr/` | Dir | 18 chapters covering RLVR, verifiable rewards, and Kaggle grandmaster tracks |

## Invariants & Rules
- Data files are dynamically loaded on-demand by `chapterLoader.ts` to minimize initial bundle size.
- Content must remain free of dead anchor links or references to retired prototype routes.
