#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
web/app.py — Unified DeepAgents & RL Interactive Platform Backend
==================================================================
Serves:
1. DeepAgents Harness & Agent Stack APIs (/api/deepagents/*)
2. Eval Framework Benchmarks & Refinement APIs (/api/eval/*)
3. Reinforcement Learning & GRPO APIs (/api/rl/* and legacy /api/*)
4. RLVR Post-Training MLE Curriculum & Code APIs (/api/rlvr/*)
5. Sphinx RTD-style Interactive Learning Platform Frontend (/static)
"""

import sys
from pathlib import Path

# Insert repo root and sub-packages so imports resolve seamlessly
_REPO_ROOT = Path(__file__).resolve().parent.parent
for p in [str(_REPO_ROOT), str(_REPO_ROOT / "rl"), str(_REPO_ROOT / "rlvr")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .routers import (
    toy,
    llm,
    offline,
    policy,
    tutorial,
    wiki,
    eval as eval_router,
    deepagents as deepagents_router,
    rlvr as rlvr_router,
    rl_chapters as rl_chapters_router,
)

app = FastAPI(
    title="DeepAgents & RL/Post-Training Unified Platform",
    description=(
        "Unified platform combining DeepAgents Harness Engineering, "
        "EvalFramework benchmarking, Reinforcement Learning (GRPO/PPO/DPO), "
        "and RLVR Post-Training MLE curriculum."
    ),
    version="2.1.0",
)

# Enable CORS for local cross-origin prototyping
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- DeepAgents, Eval & RLVR APIs ----
app.include_router(deepagents_router.router, prefix="/api/deepagents", tags=["DeepAgents-Harness"])
app.include_router(eval_router.router, prefix="/api/eval", tags=["Eval-Framework"])
app.include_router(rlvr_router.router, prefix="/api/rlvr", tags=["RLVR-Post-Training"])
app.include_router(rl_chapters_router.router, prefix="/api/rl/chapters", tags=["RL-Curriculum"])

# ---- RL APIs (Namespaced /api/rl/*) ----
app.include_router(toy.router, prefix="/api/rl/toy", tags=["RL-Toy"])
app.include_router(llm.router, prefix="/api/rl/llm", tags=["RL-LLM"])
app.include_router(offline.router, prefix="/api/rl/offline", tags=["RL-Offline"])
app.include_router(policy.router, prefix="/api/rl/policy", tags=["RL-Policy"])
app.include_router(tutorial.router, prefix="/api/rl/tutorial", tags=["RL-Tutorial"])
app.include_router(wiki.router, prefix="/api/rl/wiki", tags=["RL-Wiki"])

# ---- Production React 19 Frontend Mount ----
_WEB_DIR = Path(__file__).resolve().parent
_DIST = _WEB_DIR / "dist"

# Mount modern Vite build assets
if (_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(_DIST / "assets")), name="frontend_assets")


@app.get("/")
def root():
    """Returns the modern React 19 SPA index page."""
    index_file = _DIST / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"status": "error", "detail": "Frontend bundle not found. Run 'npm run build' in frontend/."}


@app.get("/api")
def api_catalog():
    """Self-describing API index."""
    return {
        "name": "DeepAgents & RL/Post-Training Unified Interactive Platform API",
        "sites": {
            "deepagents": "/?site=deepagents#da01",
            "research_rl": "/?site=rl#ch01",
            "rlvr": "/?site=rlvr#rlvr01",
        },
        "endpoints": {
            "/api/deepagents/tutorials": "List all 31 DeepAgents tutorials",
            "/api/eval/status": "Evaluation engine credential & benchmark metrics",
            "/api/eval/runs": "Recent evaluation run records",
            "/api/rlvr/chapters": "List all 18 post-training MLE chapters",
            "/api/rlvr/code": "List production post-training algorithm modules",
            "/api/rlvr/notebooks": "List Kaggle & Colab showcase notebooks",
            "/api/rl/chapters": "List all 18 research RL & alignment chapters",
            "/api/rl/toy": "Part 1 — 6-action toy GRPO",
            "/api/rl/llm": "Part 1+1.5 — Character-level LLM GRPO & training pipeline",
            "/api/rl/offline": "Part 1 — Kairos memory & offline resample",
            "/api/rl/policy": "Part 2 — In-context policy 7 scenarios",
            "/api/rl/tutorial": "Part 1 — GRPO math tracer",
            "/api/rl/wiki": "Part 4 — Wiki query and KG verification",
        },
    }
