# web

> Auto-maintained by Agent. Do not edit manually.

## Purpose
Unified FastAPI backend application serving REST APIs for DeepAgents tutorials, evaluation framework, RL training endpoints, and static web client assets.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `README_web.md` | File | Documentation on backend architecture, routing, and deployment instructions |
| `__init__.py` | File | Module package initialization |
| `app.py` | File | FastAPI application initialization, CORS middleware, router registration, and static mounts |
| `requirements.txt` | File | Python dependencies for the web server (FastAPI, Uvicorn, Pydantic) |
| `schemas.py` | File | Pydantic data schemas and request/response models |
| `state.py` | File | In-memory shared server state and active connection tracker |
| `routers/` | Dir | [routers/INDEX.md](routers/INDEX.md) - Modular FastAPI APIRouter handlers |

## Invariants & Rules
- All new API routes must be grouped under a router in `routers/` and registered with a clear URL prefix.
- Build output `dist/` is transient and excluded from source indexing.
