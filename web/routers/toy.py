#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
routers/toy.py — Part 1：6-action toy GRPO（你當 reward）
========================================================

包 grpo_play_core.PolicyState。互動 loop：
  create → sample → (前端顯示) → step(accepted_idx) → 顯示 result → 重複

PolicyState 持有 theta + rng（mutable），所以用 TOY_SESSIONS 做 per-session 隔離。
"""

import numpy as np
from fastapi import APIRouter, HTTPException

from grpo_play_core import PolicyState, THEMES, K_ACTIONS, G

from .. import schemas
from ..state import TOY_SESSIONS, new_sid, with_lock

router = APIRouter()


def _queries(theme: str):
    return [{"q": q["q"], "correct": q.get("correct")} for q in THEMES[theme]["queries"]]


def _action_descriptions(theme: str):
    if theme != "dashboard":
        return None
    return {str(k): v for k, v in THEMES["dashboard"]["action_descriptions"].items()}


def _initial_probs(theme: str) -> list:
    # uniform：1/K_ACTIONS
    return [round(float(x), 4) for x in np.full(K_ACTIONS, 1.0 / K_ACTIONS)]


# ------------------------------------------------------------
@router.post("/create", response_model=schemas.ToyCreateOut)
def create(body: schemas.ToyCreateIn):
    if body.theme not in THEMES:
        raise HTTPException(400, f"theme 必須是 {list(THEMES.keys())}，收到 {body.theme!r}")
    sid = new_sid()
    state = PolicyState(body.theme, seed=0)
    with with_lock():
        TOY_SESSIONS[sid] = state
    return schemas.ToyCreateOut(
        session_id=sid,
        theme=body.theme,
        name=THEMES[body.theme]["name"],
        action_label=THEMES[body.theme]["action_label"],
        queries=_queries(body.theme),
        action_descriptions=_action_descriptions(body.theme),
        auto_label=THEMES[body.theme]["auto_label"],
        probs=_initial_probs(body.theme),
    )


def _get(sid: str) -> PolicyState:
    state = TOY_SESSIONS.get(sid)
    if state is None:
        raise HTTPException(404, f"session {sid} 不存在（可能已重啟）。請先 /create。")
    return state


@router.post("/sample", response_model=schemas.ToySampleOut)
def sample(body: schemas.ToySampleIn):
    state = _get(body.session_id)
    if not (0 <= body.query_idx < len(state.queries)):
        raise HTTPException(400, f"query_idx 超出範圍（0..{len(state.queries)-1}）")
    with with_lock():
        group = state.sample_group(body.query_idx).tolist()
    return schemas.ToySampleOut(group=group, query_idx=body.query_idx)


@router.post("/step", response_model=schemas.ToyStepOut)
def step(body: schemas.ToyStepIn):
    state = _get(body.session_id)
    if not (0 <= body.query_idx < len(state.queries)):
        raise HTTPException(400, f"query_idx 超出範圍")
    if body.accepted_idx != -1 and not (0 <= body.accepted_idx < len(body.group)):
        raise HTTPException(400, f"accepted_idx 超出範圍（-1 或 0..{len(body.group)-1}）")
    group = np.array(body.group, dtype=int)
    with with_lock():
        res = state.apply_user_choice(body.query_idx, group, body.accepted_idx)
    return schemas.ToyStepOut(**res)


@router.post("/reset", response_model=schemas.ToyCreateOut)
def reset(body: schemas.ToyResetIn):
    if body.theme not in THEMES:
        raise HTTPException(400, f"theme 必須是 {list(THEMES.keys())}")
    sid = body.session_id
    state = PolicyState(body.theme, seed=0)
    with with_lock():
        TOY_SESSIONS[sid] = state
    return schemas.ToyCreateOut(
        session_id=sid,
        theme=body.theme,
        name=THEMES[body.theme]["name"],
        action_label=THEMES[body.theme]["action_label"],
        queries=_queries(body.theme),
        action_descriptions=_action_descriptions(body.theme),
        auto_label=THEMES[body.theme]["auto_label"],
        probs=_initial_probs(body.theme),
    )


@router.get("/state/{sid}")
def get_state(sid: str):
    """額外：回當前某 query 的機率分佈（給前端 refresh 圖用）。"""
    state = _get(sid)
    out = []
    for qi in range(len(state.queries)):
        out.append({
            "query_idx": qi,
            "query": state.queries[qi]["q"],
            "probs": [round(float(x), 4) for x in state.probabilities(qi)],
        })
    return {"session_id": sid, "theme": state.theme,
            "n_rounds": len(state.history), "per_query": out}
