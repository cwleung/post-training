#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
routers/policy.py — Part 2：in-context policy learning（七場景）
================================================================

包 policy_feedback.py + policy_feedback_demo.py 的七個場景。
不靠 stdout capture — 直接呼叫 demo 裡的 session factory（回 event list），
再呼叫底層分析函式（compact / stratified_stats / detect_anomalies / render_agents_md），
回結構化 JSON。

七場景對應 policy_feedback_demo.py 的 S1–S7：
  S1 正常 traffic   S2 學好→KEEP   S3 學壞→AVOID
  S4 reward hacking S5 off-policy   S6 LLM 接線   S7 寫回 AGENTS.md
"""

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException

from kairos_log import KairosLogMemory
from policy_feedback import (
    FeedbackEvent, record_feedback, read_feedback_events,
    stratified_stats, detect_anomalies, compact, _sessionize,
)
import policy_feedback_demo as pfd

from .. import schemas

router = APIRouter()


# ============================================================
# 七場景 metadata + 對應的 sessions factory 序列
# ============================================================
SCENARIOS = [
    {
        "n": 1, "key": "s1", "title": "[S1] 正常 traffic — 分層 win_rate，無異常",
        "what": "混合 L4/L3/L2 三種 intent 的健康 traffic，看分層統計，確認沒觸發異常。",
        "sessions": [("session_normal", "s1"), ("session_learning_well", "s1b")],
    },
    {
        "n": 2, "key": "s2", "title": "[S2] agent 學得好 — compact() 產生 KEEP rules",
        "what": "L4 plan 一直被接受 → win_rate 達標 → 自動產生 KEEP rule（自帶 evidence）。",
        "sessions": [("session_learning_well", "s2a"), ("session_learning_well", "s2b")],
    },
    {
        "n": 3, "key": "s3", "title": "[S3] agent 學得壞 — compact() 產生 AVOID rules",
        "what": "L3 edit 一直被 undo → undo_rate 達標 → 自動產生 AVOID rule。",
        "sessions": [("session_learning_bad", "s3a"), ("session_learning_bad", "s3b")],
    },
    {
        "n": 4, "key": "s4", "title": "[S4] reward-hacking — win_rate 跳升 + query 多樣性崩塌",
        "what": "前一 session 正常，這一 session 全 confirm 但 query 幾乎一模一樣 → 觸發警示。",
        "sessions": [("session_normal", "s4_prev"), ("session_hacking", "s4_now")],
    },
    {
        "n": 5, "key": "s5", "title": "[S5] off-policy — log 混合新舊 policy_version",
        "what": "v0 舊 events 混 v1 新 events → off_policy_note 偵測到（誠實標示，不裝作會 IS）。",
        "sessions": [("session_old_policy", "s5_old"), ("session_normal", "s5_new")],
    },
    {
        "n": 6, "key": "s6", "title": "[S6] LLM 接線 — compact() 接 callable，輸出自然語句",
        "what": "接一個 fake LLM callable，把 heuristic draft 改寫成自然語句 rules_refined。",
        "sessions": [("session_learning_well", "s6a"),
                     ("session_learning_well", "s6b"),
                     ("session_learning_bad", "s6c")],
    },
    {
        "n": 7, "key": "s7", "title": "[S7] 寫回 AGENTS.md — 三層結構，guardrail 不被動到",
        "what": "把 rules render 進 AGENTS.md；System Constraints / User Preferences 原樣不動。",
        "sessions": [("session_learning_well", "s7a"),
                     ("session_learning_well", "s7b"),
                     ("session_learning_bad", "s7c"),
                     ("session_learning_bad", "s7d")],
    },
]


def _run_scenario(n: int) -> schemas.ScenarioOut:
    if not (1 <= n <= 7):
        raise HTTPException(404, "scenario 必須 1..7")
    meta = SCENARIOS[n - 1]

    tmpdir = Path(tempfile.mkdtemp(prefix=f"policy_s{n}_"))
    kairos = KairosLogMemory(tmpdir / f"{meta['key']}.log.md")

    # 用 getattr 抓 demo 裡的 factory，組 events
    all_events: list = []
    for factory_name, sid in meta["sessions"]:
        factory = getattr(pfd, factory_name)
        sess_events = factory(sid)
        for e in sess_events:
            record_feedback(kairos, e)
        all_events.extend(sess_events)

    events = read_feedback_events(kairos)
    use_llm = (n == 6)
    llm_rewrite = pfd.fake_llm_rewrite if use_llm else None
    result = compact(events, min_n=3, llm_rewrite=llm_rewrite)

    compact_out = schemas.CompactOut(
        rules=[schemas.RuleOut(**r) for r in result["rules"]],
        rules_refined=result["rules_refined"],
        stratified_stats=result["stratified_stats"],
        anomaly=result["anomaly"],
        off_policy_note=result["off_policy_note"],
        n_events=result["n_events"],
    )

    agents_md = None
    new_version = None
    if n == 7:
        md, ver = pfd.render_agents_md(result["rules"], old_version="v0")
        agents_md = md
        new_version = ver

    # events 轉成可序列化的 payload（含 reward / intent_level）
    events_payload = [e.to_payload() for e in events]

    return schemas.ScenarioOut(
        n=n, key=meta["key"], title=meta["title"], what=meta["what"],
        sessions_used=[f[0] for f in meta["sessions"]],
        events=events_payload,
        compact=compact_out,
        agents_md=agents_md,
        new_version=new_version,
    )


# ============================================================
# Endpoints
# ============================================================
@router.get("/scenarios", response_model=schemas.ScenariosIndexOut)
def scenarios_index():
    return schemas.ScenariosIndexOut(scenarios=[
        {"n": s["n"], "key": s["key"], "title": s["title"], "what": s["what"]}
        for s in SCENARIOS
    ])


@router.post("/scenario/{n}", response_model=schemas.ScenarioOut)
def scenario(n: int):
    return _run_scenario(n)


@router.post("/compact", response_model=schemas.CompactOut)
def compact_free(body: schemas.CompactIn):
    """自由輸入 events 後跑 compact。"""
    events = []
    for fe in body.events:
        try:
            events.append(FeedbackEvent(
                signal=fe.signal, query=fe.query,
                session_id=fe.session_id, policy_version=fe.policy_version,
            ))
        except ValueError as e:
            raise HTTPException(400, f"無效的 signal {fe.signal!r}：{e}")
    llm_rewrite = pfd.fake_llm_rewrite if body.use_llm else None
    result = compact(events, min_n=3, llm_rewrite=llm_rewrite)
    return schemas.CompactOut(
        rules=[schemas.RuleOut(**r) for r in result["rules"]],
        rules_refined=result["rules_refined"],
        stratified_stats=result["stratified_stats"],
        anomaly=result["anomaly"],
        off_policy_note=result["off_policy_note"],
        n_events=result["n_events"],
    )


@router.post("/agents-md", response_model=schemas.AgentsMdOut)
def agents_md(body: schemas.AgentsMdIn):
    rules = [r.model_dump() for r in body.rules]
    md, ver = pfd.render_agents_md(rules, old_version=body.old_version)
    return schemas.AgentsMdOut(markdown=md, new_version=ver)


@router.get("/signals")
def signals():
    """回 REWARD_BY_SIGNAL 對照表，方便前端 compact 表單用。"""
    from policy_feedback import REWARD_BY_SIGNAL, INTENT_LEVEL_OF
    return {
        "reward_by_signal": REWARD_BY_SIGNAL,
        "intent_level_of": INTENT_LEVEL_OF,
    }
