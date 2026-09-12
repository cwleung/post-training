#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
routers/llm.py — Part 1 + 1.5：char-level LLM GRPO + production pipeline
========================================================================

每個 session 一份 TrainingPipeline（per-session temp dir），多 tab 不互踩。
透過 FastAPI 提供 JSON API，前端用 Chart.js 畫圖呈現互動與指標。

互動 loop：
  create → generate(prompt) → (前端顯示 4 個候選) → accept(idx)/reject() → result_table → 重複
另支援 offline-train（讀該 session 的 kairos log → resample → step）、reset、checkpoint 下載。
"""

import numpy as np
from fastapi import APIRouter, HTTPException
from pathlib import Path

from grpo_llm_char import (
    CharPolicy, VOCAB, V2I, I2V, V,
    encode, decode, generate, seq_logprob,
)
from training_pipeline import TrainingPipeline, TrainingConfig

from .. import schemas
from ..state import LLM_SESSIONS, new_sid, with_lock

router = APIRouter()

# render helper（把不可見 token 用符號表示）
def _render_gen_readable(full_seq, prompt_len):
    gen_ids = full_seq[prompt_len:]
    out = []
    for i in gen_ids:
        c = I2V[i]
        if c == ">":
            out.append("‹END›")
        elif c == " ":
            out.append("␣")
        elif c == "<":
            out.append("‹BOS›")
        else:
            out.append(c)
    return "".join(out)


def _metrics(p: TrainingPipeline) -> schemas.LLMMetricsOut:
    m = p.metrics()
    return schemas.LLMMetricsOut(
        round=m["round"],
        policy_version=m["policy_version"],
        rolling_mean_reward=m.get("rolling_mean_reward"),
        frozen=m["frozen"],
        hacking_detail=m.get("hacking_detail", ""),
        checkpoints=m["checkpoints"],
        beta_kl=m["beta_kl"],
        kl_implemented=m["kl_implemented"],
    )


def _get(sid: str) -> TrainingPipeline:
    p = LLM_SESSIONS.get(sid)
    if p is None:
        raise HTTPException(404, f"session {sid} 不存在（可能已重啟）。請先 /create。")
    return p


# ------------------------------------------------------------
@router.post("/create", response_model=schemas.LLMCreateOut)
def create():
    sid = new_sid()
    # per-session 目錄，避免多 tab 寫同一份 checkpoint/log
    cfg = TrainingConfig(
        checkpoint_dir=str(Path("checkpoints") / sid),
        log_dir=str(Path("logs") / sid),
        kairos_path=str(Path("logs") / sid / "kairos.md"),
        reward_csv=str(Path("logs") / sid / "reward_log.csv"),
    )
    p = TrainingPipeline(cfg)
    p.resume_from_latest()
    with with_lock():
        LLM_SESSIONS[sid] = p
    return schemas.LLMCreateOut(session_id=sid, state=_metrics(p))


@router.get("/state/{sid}", response_model=schemas.LLMStateOut)
def state(sid: str):
    p = _get(sid)
    return schemas.LLMStateOut(metrics=_metrics(p), checkpoints=p.ckpt.list_checkpoints())


# ------------------------------------------------------------
@router.post("/generate", response_model=schemas.LLMGenerateOut)
def generate_candidates(body: schemas.LLMGenerateIn):
    p = _get(body.session_id)
    prompt = body.prompt
    if not prompt:
        raise HTTPException(400, "prompt 不可為空")
    bad = set(prompt) - set(VOCAB)
    if bad:
        raise HTTPException(400, f"這些字元不在 vocab 裡：{sorted(bad)}；vocab={VOCAB[1:]}")

    prompt_ids = [V2I["<"]] + encode(prompt)
    prompt_len = len(prompt_ids)
    with with_lock():
        group = [generate(p.policy, prompt_ids, 8, p.rng) for _ in range(4)]

    candidates = []
    for i, s in enumerate(group):
        candidates.append(schemas.LLMCandidate(
            label=chr(ord('A') + i),
            text=decode(s),
            gen_readable=_render_gen_readable(s, prompt_len),
            logp=float(seq_logprob(p.policy, s, prompt_len)),
        ))
    # 把當前 group 暫存在 pipeline 上（accept/reject 要用）
    with with_lock():
        setattr(p, "_web_group", group)
        setattr(p, "_web_prompt_ids", prompt_ids)
        setattr(p, "_web_prompt_str", prompt)
    return schemas.LLMGenerateOut(prompt=prompt, candidates=candidates)


def _build_result(p: TrainingPipeline, res: dict) -> schemas.LLMResultOut:
    """把 pipeline.step() 的回傳 dict 轉成前端 result_table。"""
    if res.get("frozen"):
        return schemas.LLMResultOut(
            result_table=[], baseline=0.0, accepted_idx=-1,
            state=_metrics(p),
            note=f"系統已凍結（reward hacking）：{res.get('detail', '')}",
        )
    labels = ["A", "B", "C", "D"]
    rows = []
    for i in range(4):
        lb = res["logp_before"][i]
        la = res["logp_after"][i]
        raw_ratio = float(np.exp(la - lb))
        clipped = min(max(raw_ratio, 1 - p.config.eps_clip), 1 + p.config.eps_clip)
        rows.append(schemas.LLMResultRow(
            candidate=labels[i],
            reward=1.0 if i == res["accepted_idx"] else -1.0,
            advantage=float(res["advantages"][i]),
            logp_before=float(lb),
            logp_after=float(la),
            ratio=raw_ratio,
            clipped=abs(raw_ratio - clipped) > 1e-4,
            accepted=(i == res["accepted_idx"]),
        ))
    return schemas.LLMResultOut(
        result_table=rows,
        baseline=float(res["baseline"]),
        accepted_idx=res["accepted_idx"],
        state=_metrics(p),
    )


@router.post("/accept", response_model=schemas.LLMResultOut)
def accept(body: schemas.LLMChoiceIn):
    p = _get(body.session_id)
    group = getattr(p, "_web_group", None)
    prompt_ids = getattr(p, "_web_prompt_ids", None)
    prompt_str = getattr(p, "_web_prompt_str", "")
    if group is None or prompt_ids is None:
        raise HTTPException(400, "還沒 generate。請先 POST /generate。")
    if p.guard.frozen:
        raise HTTPException(409, "系統已凍結（reward hacking 觸發），無法 accept。")
    with with_lock():
        res = p.step(prompt_str, prompt_ids, group, accepted_idx=0)
    return _build_result(p, res)


@router.post("/accept/{idx}", response_model=schemas.LLMResultOut)
def accept_idx(idx: int, body: schemas.LLMChoiceIn):
    """明確指定接受第幾個（0..3），給前端按鈕 A/B/C/D 用。"""
    if not (0 <= idx < 4):
        raise HTTPException(400, "idx 必須 0..3")
    p = _get(body.session_id)
    group = getattr(p, "_web_group", None)
    prompt_ids = getattr(p, "_web_prompt_ids", None)
    prompt_str = getattr(p, "_web_prompt_str", "")
    if group is None or prompt_ids is None:
        raise HTTPException(400, "還沒 generate。請先 POST /generate。")
    if p.guard.frozen:
        raise HTTPException(409, "系統已凍結（reward hacking 觸發），無法 accept。")
    with with_lock():
        res = p.step(prompt_str, prompt_ids, group, accepted_idx=idx)
    return _build_result(p, res)


@router.post("/reject", response_model=schemas.LLMResultOut)
def reject(body: schemas.LLMChoiceIn):
    p = _get(body.session_id)
    group = getattr(p, "_web_group", None)
    prompt_ids = getattr(p, "_web_prompt_ids", None)
    prompt_str = getattr(p, "_web_prompt_str", "")
    if group is None or prompt_ids is None:
        raise HTTPException(400, "還沒 generate。請先 POST /generate。")
    # frozen 時 reject 仍允許（讓 user 能繼續瀏覽，只是不更新）
    if p.guard.frozen:
        return schemas.LLMResultOut(
            result_table=[], baseline=0.0, accepted_idx=-1, state=_metrics(p),
            note="系統已凍結，reject 不會觸發更新。",
        )
    with with_lock():
        res = p.step(prompt_str, prompt_ids, group, accepted_idx=-1)
    return _build_result(p, res)


# ------------------------------------------------------------
@router.get("/reward-curve/{sid}", response_model=schemas.LLMRewardCurveOut)
def reward_curve(sid: str):
    p = _get(sid)
    rows = p.reward_logger.read_all()
    if not rows:
        return schemas.LLMRewardCurveOut(rounds=[], mean_rewards=[], baselines=[], rolling=[])
    rounds = [int(r["round"]) for r in rows]
    means = [float(r["mean_reward"]) for r in rows]
    bases = [float(r["baseline"]) for r in rows]
    # rolling window=5
    window = 5
    rolling: list = [None] * len(means)
    if len(means) >= window:
        roll = np.convolve(means, np.ones(window) / window, mode='valid')
        for j, v in enumerate(roll):
            rolling[j + window - 1] = round(float(v), 4)
    return schemas.LLMRewardCurveOut(
        rounds=rounds, mean_rewards=means, baselines=bases, rolling=rolling
    )


@router.get("/policy-pref/{sid}", response_model=schemas.LLMPolicyPrefOut)
def policy_pref(sid: str):
    """policy 對 3 個 context（after '=', BOS, space）的 top-5 轉移機率。"""
    p = _get(sid)
    ctxs = [(V2I["="], "after '='"), (V2I["<"], "after BOS"), (V2I[" "], "after space")]
    series = []
    for ctx_id, label in ctxs:
        prob = p.policy.probs_at(ctx_id)
        top5 = np.argsort(prob)[::-1][:5]
        series.append({
            "context": label,
            "labels": [I2V[i] for i in top5],
            "values": [round(float(prob[i]), 4) for i in top5],
        })
    return schemas.LLMPolicyPrefOut(
        contexts=[label for _, label in ctxs],
        series=series,
        uniform=round(1.0 / V, 4),
        round=p.round_num,
    )


# ------------------------------------------------------------
@router.post("/offline-train", response_model=schemas.LLMOfflineTrainOut)
def offline_train(body: schemas.LLMOfflineTrainIn):
    """mirror run_offline_training.run_offline：讀該 session 的 kairos log → resample → step。"""
    p = _get(body.session_id)
    if p.guard.frozen:
        raise HTTPException(409, "系統已凍結，無法離線訓練。")

    # 讀 kairos log 裡的 grpo_step 事件（邏輯同 run_offline_training.load_training_samples_from_kairos）
    import re
    samples = []
    for e in p.kairos_logger.kairos.iter_events():
        if e.get("event_type") != "grpo_step":
            continue
        summary = e.get("summary", "")
        pm = re.search(r"prompt='([^']*)'", summary)
        am = re.search(r"accepted=(\w+)", summary)
        sm = re.search(r"policy_v(\d+)", e.get("session_id", ""))
        if not pm or not am:
            continue
        accepted_str = am.group(1)
        accepted_idx = "ABCD".index(accepted_str) if accepted_str in "ABCD" else -1
        samples.append({
            "prompt": pm.group(1),
            "accepted_idx": accepted_idx,
            "policy_version": int(sm.group(1)) if sm else 0,
        })

    if not samples:
        raise HTTPException(400, "kairos log 裡沒有 grpo_step 事件。請先在 LLM 頁互動幾輪。")

    versions = sorted(set(s["policy_version"] for s in samples))
    initial_version = p.version_tracker.current
    iters = []

    with with_lock():
        for epoch in range(body.epochs):
            for i, s in enumerate(samples):
                prompt = s["prompt"]
                prompt_ids = [V2I["<"]] + encode(prompt)
                # resample group（不回放原序列）— 這是 offline GRPO 的關鍵觀念
                group = [generate(p.policy, prompt_ids, 8, p.rng) for _ in range(4)]
                res = p.step(prompt, prompt_ids, group, s["accepted_idx"])
                if res.get("frozen"):
                    iters.append({"epoch": epoch, "i": i, "frozen": True,
                                  "detail": res.get("detail", "")})
                    break
                iters.append({"epoch": epoch, "i": i, "round": res.get("round"),
                              "mean_reward": round(res.get("mean_reward", 0.0), 4)})
            if p.guard.frozen:
                break

    return schemas.LLMOfflineTrainOut(
        iters=iters,
        initial_version=initial_version,
        new_version=p.version_tracker.current,
        frozen=p.guard.frozen,
        n_samples=len(samples),
        off_policy_versions=versions,
    )


@router.post("/reset", response_model=schemas.LLMCreateOut)
def reset(body: schemas.LLMResetIn):
    p = _get(body.session_id)
    with with_lock():
        p.reset()
    return schemas.LLMCreateOut(session_id=body.session_id, state=_metrics(p))


@router.get("/checkpoint/{sid}/{version}")
def checkpoint_file(sid: str, version: int):
    """下載某 version 的 .npz。"""
    p = _get(sid)
    npz_path = Path(p.config.checkpoint_dir) / f"policy_v{version}.npz"
    if not npz_path.exists():
        raise HTTPException(404, f"checkpoint policy_v{version}.npz 不存在")
    from fastapi.responses import FileResponse
    return FileResponse(str(npz_path),
                        filename=f"policy_v{version}.npz",
                        media_type="application/octet-stream")
