# DeepAgents API Surface Reference (`docs/API_SURFACE.md`)

> Auto-maintained by Agent. Do not edit manually.

This document details all exposed REST API endpoints, CLI entrypoints, and interactive applications provided across the DeepAgents repository.

---

## 1. REST API Endpoints (FastAPI Backend)

Launched via `python3 serve.py` on default port `8000`.

### 1.1 DeepAgents Harness APIs (`/api/deepagents`)
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/deepagents/tutorials` | Lists all 31 tutorial chapters, titles, filenames, and byte sizes |
| `GET` | `/api/deepagents/tutorials/{tutorial_id}` | Retrieves full markdown content of a specific tutorial |

### 1.2 Evaluation Framework APIs (`/api/eval`)
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/eval/status` | Reports evaluation engine status and available benchmark suites |
| `POST` | `/api/eval/run` | Executes an evaluation task with specified prompt and strategy (`hybrid`, `llm_judge`, `rubric`) |
| `POST` | `/api/eval/custom` | Evaluates arbitrary candidate output against task constraints |
| `POST` | `/api/eval/optimize` | Runs GEval / MIPRO-style iterative prompt refinement loop |
| `GET` | `/api/eval/runs` | Returns historical evaluation runs and score distributions |

### 1.3 RLVR Post-Training APIs (`/api/rlvr`)
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/rlvr/curriculum` | Returns structured RLVR learning milestones and verifiable reward types |
| `GET` | `/api/rlvr/challenges` | Lists Kaggle and competitive programming challenge environments |
| `POST` | `/api/rlvr/verify` | Evaluates code submissions against verifiable unit test suites |

### 1.4 Reinforcement Learning APIs (`/api/rl/*`)
| Namespace | Method | Endpoint | Description |
|---|---|---|---|
| `/chapters` | `GET` | `/api/rl/chapters` | Lists all 18 research RL & alignment chapters with titles and sizes |
| `/chapters` | `GET` | `/api/rl/chapters/{chapter_id}` | Retrieves full markdown content of a specific RL chapter |
| `/toy` | `POST` | `/api/rl/toy/train` | Runs toy GRPO optimization iterations on scalar reward objectives |
| `/llm` | `POST` | `/api/rl/llm/generate` | Generates text rollouts with candidate reward calculations |
| `/offline` | `GET` | `/api/rl/offline/status` | Checks offline replay buffer and batch training progress |
| `/policy` | `POST` | `/api/rl/policy/feedback` | Ingests feedback events and updates in-context system prompt policy |
| `/wiki` | `GET` | `/api/rl/wiki/resources` | Returns curated paper database, research links, and benchmark tables |

---

## 2. CLI Entrypoints & Executable Scripts

All commands should be executed from the repository root:

```bash
# 1. Start the unified FastAPI backend server
python3 serve.py
PORT=9000 HOST=0.0.0.0 python3 serve.py

# 2. Run Toy GRPO Algorithm Demonstration
python3 -m rl.grpo_toy

# 3. Character-level LLM GRPO Training
python3 -m rl.grpo_llm_char

# 4. Interactive In-Context Policy Feedback Simulation Demo
python3 -m rl.policy_feedback_demo

# 5. Core Environment & Policy Self-Tests
python3 -m rl.grpo_play_core
python3 -m rl.training_pipeline
```

---

## 3. Web Frontend Development & Build

Located in `frontend/`:

```bash
cd frontend

# Install dependencies
npm install

# Start Vite live reload development server
npm run dev

# Build production bundle into web/dist
npm run build
```

The production assets in `web/dist` are automatically served as static files by `serve.py` at `/` or `/static`.
