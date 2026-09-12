# routers (web/routers)

> Auto-maintained by Agent. Do not edit manually.

## Purpose
Modular FastAPI route handlers organizing endpoints into domain-specific namespaces across agent harnesses, evaluations, RL algorithms, and curriculum.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `__init__.py` | File | Router package exports |
| `deepagents.py` | File | Endpoints for DeepAgents 31-tutorial curriculum metadata and markdown content |
| `eval.py` | File | Endpoints for EvalFramework benchmark execution, custom grading, and prompt optimization |
| `llm.py` | File | Character-level and autoregressive LLM rollout generation endpoints |
| `offline.py` | File | Offline dataset querying, replay buffer inspect, and batch metrics endpoints |
| `policy.py` | File | In-context feedback ingestion and active policy state inspection |
| `rlvr.py` | File | RLVR curriculum overview, programming challenges, and verification APIs |
| `toy.py` | File | Toy GRPO training step and loss convergence simulation endpoints |
| `tutorial.py` | File | Interactive learning platform chapter content and reading progress APIs |
| `wiki.py` | File | Knowledge base querying, curated paper indices, and benchmark tables |

## Invariants & Rules
- Routers must use standard Pydantic request models from `web/schemas.py`.
- Handle unexpected errors cleanly and raise appropriate `HTTPException` status codes.
