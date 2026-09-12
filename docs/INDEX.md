# System Documentation & Directory Directory (`docs/INDEX.md`)

> Auto-maintained by Agent. Machine-readable harness memory. Do not edit manually.

> [!IMPORTANT]
> **Agent-Only Memory**: All documentation files under `docs/` and all distributed `INDEX.md` files are strictly **internal harness contracts and memory designed to be read, traversed, and maintained by AI agents**. Keep descriptions dense, structured, factual, and token-efficient.

Welcome to the central documentation index for **DeepAgents**. This repository is an end-to-end framework uniting Agent Harness Engineering, Reinforcement Learning with Verifiable Rewards (RLVR), in-context policy learning (GRPO / PPO / DPO), and a full-featured interactive learning web platform.

---

## 1. Core System Documentation & Topic Guides

| Document | Type | Purpose & Scope |
|---|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Architecture | High-level system architecture, component boundaries, and cross-directory dependency flows |
| [API_SURFACE.md](API_SURFACE.md) | API Spec | Complete reference for REST API routes, CLI tools, entrypoint scripts, and interactive apps |
| [STORAGE.md](STORAGE.md) | Persistence | Specifications for data persistence: model checkpoints, Kairos append-only logs, and schemas |
| [SYSTEM_STATE.md](SYSTEM_STATE.md) | State Log | Append-only execution and sync history tracking updates to directory indexes |

> **Dynamic Topic Creation**: Agents are explicitly empowered to add new topic-specific documents under `docs/` (e.g. `docs/<TOPIC>.md`) when new domains, training protocols, evaluation suites, or algorithms emerge. Any newly created topic file must be registered in the table above and logged in [SYSTEM_STATE.md](SYSTEM_STATE.md).

---

## 2. Repository Directory Hierarchy

Every functional subdirectory in `deepagents` contains a dedicated, auto-maintained `INDEX.md` describing its purpose, file responsibilities, and directory-specific invariants:

| Directory | Subsystem / Role | Local Index |
|---|---|---|
| `rl/` | Reinforcement Learning, GRPO algorithms, in-context policy demo pipelines | [rl/INDEX.md](../rl/INDEX.md) |
| └── `rl/tutorials/` | 18-part full-stack RL & alignment curriculum in Traditional Chinese | [rl/tutorials/INDEX.md](../rl/tutorials/INDEX.md) |
| `rlvr/` | RL with Verifiable Rewards, Kaggle benchmarks, and evaluation notebooks | [rlvr/INDEX.md](../rlvr/INDEX.md) |
| ├── `rlvr/kaggle_showcase/` | Kaggle grandmaster solutions and showcase pipelines | [rlvr/kaggle_showcase/INDEX.md](../rlvr/kaggle_showcase/INDEX.md) |
| ├── `rlvr/notebooks/` | Interactive Jupyter exploration notebooks | [rlvr/notebooks/INDEX.md](../rlvr/notebooks/INDEX.md) |
| ├── `rlvr/research/` | Research scripts and RLVR experimental benchmarks | [rlvr/research/INDEX.md](../rlvr/research/INDEX.md) |
| └── `rlvr/tutorials/` | Step-by-step verifiable reward engineering guides | [rlvr/tutorials/INDEX.md](../rlvr/tutorials/INDEX.md) |
| `web/` | Unified FastAPI backend serving REST APIs and static SPA assets | [web/INDEX.md](../web/INDEX.md) |
| └── `web/routers/` | Modular FastAPI route handlers (DeepAgents, Eval, RL, RLVR, Wiki) | [web/routers/INDEX.md](../web/routers/INDEX.md) |
| `frontend/` | Vite + React + TypeScript web application frontend | [frontend/INDEX.md](../frontend/INDEX.md) |
| └── `frontend/src/` | Frontend feature modules, pages, widgets, and shared components | [frontend/src/INDEX.md](../frontend/src/INDEX.md) |
| `tutorials/` | 31-part deep curriculum covering Harness Engineering and evaluation loops | [tutorials/INDEX.md](../tutorials/INDEX.md) |
| `.agents/` | Antigravity agent configuration, persistent execution rules, and workspace skills | [.agents/INDEX.md](../.agents/INDEX.md) |
| ├── `.agents/rules/` | Continuous agent execution rules (auto-index protocol) | [.agents/rules/INDEX.md](../.agents/rules/INDEX.md) |
| └── `.agents/skills/` | Workspace workflow skills (auto-index, excalidraw-skill, interactive-learning-platform, rubric-evaluation) | [.agents/skills/INDEX.md](../.agents/skills/INDEX.md) |

---

## 3. Directory Invariants

1. **Self-Describing**: Every newly created functional subdirectory must immediately instantiate an `INDEX.md`.
2. **Zero-Drift**: Whenever a file in any of the above directories is modified, added, or removed, its local `INDEX.md` and this catalog must be reconciled in the same turn.
3. **Excluded Directories**: Transient files (`.git/`, `.venv/`, `node_modules/`, `checkpoints/*`, `logs/*`, `__pycache__/`) are never indexed.
