# AGENTS.md — Global Harness Contract for DeepAgents

> This file is the **root harness contract**: global invariants and protocols that are always enforced across this repository.

---

## 0. Prime Directives

1. **Never Fabricate**: If you cannot verify a source, say so. If you cannot confirm a link or endpoint resolves, do not add it. Rigor overrides "being helpful" every time.
2. **Zero-Drift Documentation**: Any file addition, modification, or deletion requires updating the corresponding local `INDEX.md` and relevant global docs in `docs/` in the exact same turn.
3. **Preserve Functionality**: Do not break the FastAPI backend (`serve.py`, `web/app.py`), the React frontend (`frontend/`), or the core RL training scripts (`rl/`).
4. **Agent-Only Readership**: All files in `docs/` and all distributed `INDEX.md` files are strictly **internal harness contracts and machine memory to be read and written by AI agents**. They are not human-facing prose. Keep them dense, factual, and structured.
5. **UvA-DLC Progressive Notebook Standard**: All educational and technical tutorial code cells across the platform (specifically in the Post-Training Track and related curricula) must strictly conform to the progressive 5-stage UvA Deep Learning Notebook Standard (inspired by UvA DL Tutorial 4: Synthetic Batch $\to$ Causal Log-Prob Module $\to$ Vectorized Loss & Telemetry $\to$ Pathological Stress Tests & Failure Simulations $\to$ Remediation & Comparative Ablation). Single, isolated, disconnected code snippets with no runtime context or execution feedback are strictly forbidden. Every Python cell must be immediately accompanied by a realistic console execution output block (`[Execution Output / Telemetry Log]`).

---

## 1. Documentation & Maintenance Protocols

All agents working in this repository must adhere to:
- **Autonomous Documentation Protocol**: See [.agents/rules/auto-index.md](.agents/rules/auto-index.md) and the `auto-index` skill at [.agents/skills/auto-index/SKILL.md](.agents/skills/auto-index/SKILL.md).
- **Subdirectory `INDEX.md` Standard**: Every functional directory contains an auto-maintained `INDEX.md` summarizing its purpose, table of contents (files/subdirs, types, responsibilities), and invariants.
- **Global Documentation Suite (`docs/`)**: The `docs/` directory maintains global system truth:
  - `docs/INDEX.md`: Directory tree, subsystem map, and registered topic documents.
  - `docs/ARCHITECTURE.md`: Architecture boundaries and dependency flow.
  - `docs/API_SURFACE.md`: REST API endpoints, CLI entrypoints, and demo interfaces.
  - `docs/STORAGE.md`: Persistence schemas, log memory, and checkpoints.
  - `docs/SYSTEM_STATE.md`: Append-only synchronization log.
- **Dynamic Topic Creation**: Agents are empowered to create new topic-specific markdown documents directly under `docs/` (e.g. `docs/<TOPIC>.md`) whenever a new subsystem, research track, or protocol is designed. Newly created topic documents must be immediately registered in `docs/INDEX.md` and recorded in `docs/SYSTEM_STATE.md`.

---

## 2. Repository Layout

```
deepagents/
├── AGENTS.md                     ← Global harness contract
├── serve.py                      ← Backend entrypoint (Uvicorn / FastAPI)
├── docs/                         ← Centralized architecture and API documentation
│   ├── INDEX.md
│   ├── ARCHITECTURE.md
│   ├── API_SURFACE.md
│   ├── STORAGE.md
│   ├── SYSTEM_STATE.md
│   └── <TOPIC>.md                ← Dynamic topic documents created by agents
├── rl/                           ← Reinforcement learning & in-context policy learning
│   ├── AGENTS.md                 ← RL-specific domain contract & invariants
│   └── ...
├── rlvr/                         ← RL verification, Kaggle showcase, and notebooks
├── web/                          ← FastAPI API routers, state, and schemas
├── frontend/                     ← React / Vite SPA frontend
├── tutorials/                    ← 31-chapter curriculum on agent harness & evaluation
└── .agents/                      ← Antigravity agent configuration
    ├── rules/auto-index.md       ← Continuous documentation sync rule
    └── skills/                   ← Workspace-scoped workflow skills
```

> Note: Workspace workflow skills are located in `.agents/skills/` (`auto-index`, `excalidraw-skill`, `interactive-learning-platform`, `rubric-evaluation`). Global general-purpose personal skills reside in `~/.gemini/config/skills/`.


For RL-specific policies, refer directly to [rl/AGENTS.md](rl/AGENTS.md).
