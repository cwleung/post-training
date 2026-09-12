# DeepAgents & RL Interactive Web Platform

The web subsystem (`web/`) serves as the unified FastAPI backend for the **DeepAgents & Reinforcement Learning Interactive Learning Platform**, powering REST API services and serving the production React 19 Single Page Application (SPA).

---

## 1. Quick Start

```bash
# 1. Install backend dependencies
pip install -r web/requirements.txt

# 2. Build or verify the React frontend bundle
cd frontend && npm run build && cd ..

# 3. Launch unified development server
python3 serve.py
# Or directly via uvicorn:
# uvicorn web.app:app --reload --port 8000

# 4. Open in browser
# http://127.0.0.1:8000
```

The server exposes an interactive API catalog at `http://127.0.0.1:8000/api`.

---

## 2. Architecture & Directory Layout

```
web/
├── app.py              # FastAPI application mounting /assets, serving web/dist/index.html, and loading routers
├── requirements.txt    # Backend dependencies (fastapi, uvicorn, pydantic, numpy)
├── schemas.py          # Pydantic request and response schemas
├── state.py            # In-memory thread-safe session stores
├── routers/            # Modular domain route handlers
│   ├── deepagents.py   # 31 DeepAgents tutorial chapters metadata and content
│   ├── eval.py         # EvalFramework benchmarks and prompt optimization
│   ├── rlvr.py         # 18 RLVR chapters, code inspection, and notebook catalog
│   ├── toy.py          # 6-action discrete toy GRPO simulation
│   ├── llm.py          # Character-level LLM GRPO & training pipeline
│   ├── offline.py      # Kairos memory logs & offline resampling
│   ├── policy.py       # In-context policy 7 scenarios & anomaly detection
│   ├── tutorial.py     # Step-by-step GRPO mathematical tracer
│   └── wiki.py         # Knowledge graph query & compound synthesis
└── dist/               # Production assets built by Vite from frontend/
    ├── index.html      # SPA shell
    └── assets/         # Bundled JS, CSS, and KaTeX fonts
```

---

## 3. Frontend Integration

The presentation layer is located in `frontend/` (React 19, TypeScript, Tailwind CSS v4, KaTeX, Mermaid).
The legacy 5-page static demo prototype has been fully integrated into the 3-track curriculum:
- **Agents Track (`deepagents`)**: 31 chapters covering Agent Harness Engineering and Evaluation.
- **RL Track (`rl`)**: 18 chapters with interactive code inspectors, KaTeX derivations, and inline labs.
- **RLVR Track (`rlvr`)**: 18 chapters covering Post-Training MLE, verifiable rewards, and Kaggle showcases.

All static assets are built with `npm run build` in `frontend/` and emitted directly into `web/dist/`.

---

## 4. API Endpoints

### DeepAgents & RLVR
- `GET /api/deepagents/tutorials` — List 31 DeepAgents tutorials
- `GET /api/deepagents/tutorials/{id}` — Full markdown of a tutorial
- `GET /api/rlvr/chapters` — List 18 post-training chapters
- `GET /api/rlvr/chapters/{id}` — Full markdown of an RLVR chapter
- `GET /api/rlvr/code` — List production algorithms
- `GET /api/rlvr/notebooks` — List Kaggle/Colab notebooks

### Reinforcement Learning Endpoints (`/api/rl/*`)
- `POST /api/rl/toy/step` — Discrete GRPO optimization step
- `POST /api/rl/llm/generate` — Text rollout generation with reward calculation
- `POST /api/rl/offline/collect` — Log generation for offline replay buffer
- `POST /api/rl/policy/scenario/{n}` — Scenario simulation (1..7) for in-context policy
- `POST /api/rl/tutorial/step` — Single-step GRPO mathematical trace
- `GET /api/rl/wiki/resources` — Curated literature database & benchmark tables

Backward-compatible un-prefixed aliases (`/api/toy`, `/api/llm`, `/api/offline`, `/api/policy`, `/api/tutorial`, `/api/wiki`) are also maintained for legacy compatibility.
