#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
routers/tutorial.py — 「從零實作 GRPO」教學 tracer
==================================================

這個 router 的存在意義：讓使用者**單步看一次 GRPO 更新的內部全貌**。
它把 grpo_toy.grpo_update 逐行重現，但每一步都記錄中間值
（baseline / std / advantages / pi_old / 每個 inner-epoch 的 ratio、clip、gradient、obj），
前端就能把「一次更新」像錄影帶一樣播出來。

★ 正確性保證：這裡的數學與 grpo_toy.grpo_update 完全一致（同常數、同 clip、同梯度式），
  差別只在「把中間值記下來回傳」而不是只回 (baseline,std,adv,pi_old)。
  後面的 verify 會對著 grpo_toy 跑同一組 (actions,rewards) 比對 theta 前後差。

互動：
  POST /setup   {n_actions, group_size, lr, eps_clip, inner_epochs, correct_action, seed}
                → 回初始 uniform policy + 可調參數回顯
  POST /sample  {state, prompt_idx?} → 用當前 policy 抽一個 group（用教學專用小 vocab）
  POST /step    {state, actions, rewards} → 跑一次 grpo_update_trace，回完整中間值 + theta 前後
  POST /converge{state, correct_action, n_iters} → 跑 n_iters 輪收斂，回 reward 曲線
"""

import numpy as np
from fastapi import APIRouter, HTTPException

from grpo_toy import softmax, K_ACTIONS  # 用同一個 softmax，保證一致

from .. import schemas
from ..state import new_sid, with_lock

router = APIRouter()

ADV_EPS = 1e-4   # 與 grpo_toy 完全一致


def grpo_update_trace(
    theta_row: np.ndarray,       # shape (K,)：某 prompt 那列 logits
    actions: np.ndarray,         # shape (G,)
    rewards: np.ndarray,         # shape (G,)
    lr: float,
    eps_clip: float,
    inner_epochs: int,
):
    """
    逐行重現 grpo_toy.grpo_update，但記錄所有中間值。
    回傳 dict：baseline, std, advantages, pi_old, pi_after, theta_before, theta_after, epochs[]。
    """
    theta_before = theta_row.copy()
    G = len(actions)

    # 4a. group baseline（GRPO 的靈魂）
    baseline = float(rewards.mean())
    std = float(rewards.std())
    advantages = (rewards - baseline) / (std + ADV_EPS)

    # 4b. 記住舊 policy
    pi_old = softmax(theta_row).copy()

    epochs = []
    # ★ 內部 theta 維持 float64 全精度（與 grpo_toy 一致）；
    #   只有「回傳給前端顯示」的值才 round 到 4 dp，避免累積取整誤差。
    theta_cur = theta_row.astype(float)
    for ep in range(inner_epochs):
        pi_new = softmax(theta_cur)
        grad = np.zeros_like(theta_cur)
        obj = 0.0
        per_a = []   # 每個 group 樣本的 ratio/clip/分支

        for a, A in zip(actions, advantages):
            ratio = float(pi_new[a] / pi_old[a])
            surr1 = ratio * A
            surr2 = float(np.clip(ratio, 1 - eps_clip, 1 + eps_clip)) * A
            take = min(surr1, surr2)
            obj += take
            used_unclipped = bool(surr1 <= surr2)

            # 梯度（只在 unclipped 分支）
            if used_unclipped:
                for b in range(len(theta_cur)):
                    indicator = 1.0 if a == b else 0.0
                    grad[b] += A * ratio * (indicator - pi_new[b])

            per_a.append({
                "action": int(a), "advantage": round(float(A), 4),
                "ratio": round(ratio, 4),
                "surr1": round(float(surr1), 4),
                "surr2": round(float(surr2), 4),
                "take": round(float(take), 4),
                "branch": "unclipped" if used_unclipped else "clipped",
            })

        grad /= G
        obj /= G
        theta_cur = theta_cur + lr * grad   # 全精度累積（梯度上升）

        epochs.append({
            "epoch": ep,
            "pi_new": [round(float(x), 4) for x in pi_new],
            "grad": [round(float(x), 5) for x in grad],
            "obj": round(float(obj), 4),
            "theta_after": [round(float(x), 4) for x in theta_cur],
            "per_action": per_a,
        })

    # 把全精度結果 in-place 寫回外層 theta（模擬 grpo_toy 的 in-place 更新）
    theta_row[:] = theta_cur
    pi_after = softmax(theta_cur)
    return {
        "baseline": round(baseline, 4),
        "std": round(std, 4),
        "advantages": [round(float(x), 4) for x in advantages],
        "pi_old": [round(float(x), 4) for x in pi_old],
        "pi_after": [round(float(x), 4) for x in pi_after],
        "theta_before": [round(float(x), 4) for x in theta_before],
        "theta_after": [round(float(x), 4) for x in theta_cur],
        "epochs": epochs,
    }


# ============================================================
def _uniform_theta(k: int) -> list:
    return [0.0] * k


# ------------------------------------------------------------
@router.post("/setup", response_model=schemas.TutorialSetupOut)
def setup(body: schemas.TutorialSetupIn):
    """建立一個教學 session：可自訂 action 數、group、超參數、正解。"""
    k = int(body.n_actions)
    if not (2 <= k <= 10):
        raise HTTPException(400, "n_actions 必須 2..10")
    if not (2 <= body.group_size <= 16):
        raise HTTPException(400, "group_size 必須 2..16")
    if not (0 <= body.correct_action < k):
        raise HTTPException(400, f"correct_action 必須 0..{k-1}")
    sid = new_sid()
    # 教學用 state：存 theta、rng、超參數、正解
    state = {
        "theta": np.zeros(k),
        "rng": np.random.default_rng(body.seed),
        "lr": body.lr,
        "eps_clip": body.eps_clip,
        "inner_epochs": body.inner_epochs,
        "group_size": body.group_size,
        "correct_action": body.correct_action,
        "k": k,
    }
    from ..state import OFFLINE_STATE  # 借一個 dict 存（tutorial 不需要持久化）
    OFFLINE_STATE[sid] = state
    return {
        "session_id": sid,
        "k": k,
        "group_size": body.group_size,
        "lr": body.lr,
        "eps_clip": body.eps_clip,
        "inner_epochs": body.inner_epochs,
        "correct_action": body.correct_action,
        "pi": [round(float(x), 4) for x in softmax(state["theta"])],
    }


def _get(sid: str):
    from ..state import OFFLINE_STATE
    s = OFFLINE_STATE.get(sid)
    if s is None:
        raise HTTPException(404, f"tutorial session {sid} 不存在。請先 POST /setup。")
    return s


@router.post("/sample")
def sample(body: dict):
    """用當前 policy 抽一個 group。body: {session_id}"""
    sid = body.get("session_id")
    s = _get(sid)
    prob = softmax(s["theta"])
    group = s["rng"].choice(s["k"], size=s["group_size"], p=prob)
    rewards = np.array([1.0 if int(a) == s["correct_action"] else 0.0 for a in group])
    return {
        "actions": group.tolist(),
        "rewards": rewards.tolist(),
        "pi_before": [round(float(x), 4) for x in prob],
    }


@router.post("/step")
def step(body: dict):
    """
    對給定的 (actions, rewards) 跑一次 grpo_update_trace。
    body: {session_id, actions?, rewards?}
    若沒給 actions/rewards，就用當前 policy 現抽一組。
    """
    sid = body.get("session_id")
    s = _get(sid)
    actions = body.get("actions")
    rewards = body.get("rewards")
    if actions is None or rewards is None:
        prob = softmax(s["theta"])
        actions = s["rng"].choice(s["k"], size=s["group_size"], p=prob)
        rewards = np.array([1.0 if int(a) == s["correct_action"] else 0.0 for a in actions])
    actions = np.array(actions)
    rewards = np.array(rewards, dtype=float)
    with with_lock():
        # tracer 內部維持 float64 全精度，且 in-place 寫回 s["theta"]
        # （不要再拿 rounded theta_after 覆蓋，否則跨 step 會累積取整誤差）
        theta_before_full = s["theta"].copy()
        trace = grpo_update_trace(
            s["theta"], actions, rewards,
            s["lr"], s["eps_clip"], s["inner_epochs"],
        )
        # grpo_update_trace 已用 theta_row[:] = theta_cur in-place 更新 s["theta"]，
        # 這裡 theta_after 是顯示用（4dp）；s["theta"] 維持全精度。
    trace["actions"] = actions.tolist()
    trace["rewards"] = rewards.tolist()
    trace["correct_action"] = s["correct_action"]
    return trace


@router.post("/converge")
def converge(body: dict):
    """
    從當前 theta 跑 n_iters 輪收斂（每輪：sample G → reward → update）。
    body: {session_id, n_iters}
    回每輪的 avg reward 與 正解機率，讓前端畫收斂曲線。
    """
    sid = body.get("session_id")
    n = int(body.get("n_iters", 30))
    if not (1 <= n <= 200):
        raise HTTPException(400, "n_iters 必須 1..200")
    s = _get(sid)
    curve = []
    with with_lock():
        for it in range(n):
            prob = softmax(s["theta"])
            actions = s["rng"].choice(s["k"], size=s["group_size"], p=prob)
            rewards = np.array([1.0 if int(a) == s["correct_action"] else 0.0 for a in actions])
            avg_r = float(rewards.mean())
            trace = grpo_update_trace(
                s["theta"], actions, rewards,
                s["lr"], s["eps_clip"], s["inner_epochs"],
            )
            # grpo_update_trace 已 in-place 更新 s["theta"]（全精度），不覆蓋。
            curve.append({
                "iter": it,
                "avg_reward": round(avg_r, 4),
                "correct_prob": round(float(softmax(s["theta"])[s["correct_action"]]), 4),
            })
    return {"curve": curve, "final_pi": [round(float(x), 4) for x in softmax(s["theta"])],
            "correct_action": s["correct_action"]}


@router.get("/verify/{sid}")
def verify(sid: str):
    """
    正確性自檢：對一組固定 (actions,rewards)，本 tracer 與 grpo_toy.grpo_update
    跑出來的 theta 變化必須一致。回傳兩邊的 theta_after 與 max_abs_diff。
    """
    from grpo_toy import grpo_update as gt_update
    s = _get(sid)
    rng = np.random.default_rng(123)
    theta_toy = np.zeros((1, s["k"]))          # grpo_toy 期望 2D
    actions = np.array([0, s["correct_action"], 1, s["correct_action"]])[:s["group_size"]]
    if len(actions) < s["group_size"]:
        actions = np.array(list(actions) + [s["correct_action"]] * (s["group_size"] - len(actions)))
    rewards = np.array([1.0 if int(a) == s["correct_action"] else 0.0 for a in actions])

    # 兩邊都用全精度跑，比較才公平（顯示值才 round）
    theta_row = np.zeros(s["k"])
    theta_row_copy = theta_row.copy()
    trace = grpo_update_trace(theta_row_copy, actions, rewards,
                              s["lr"], s["eps_clip"], s["inner_epochs"])
    # grpo_update_trace in-place 更新 theta_row_copy（全精度）
    gt_update(theta_toy, 0, actions, rewards,
              s["lr"], s["eps_clip"], s["inner_epochs"], rng)

    ours_full = theta_row_copy            # 全精度
    theirs = theta_toy[0]
    diff = float(np.max(np.abs(ours_full - theirs)))
    return {
        "actions": actions.tolist(), "rewards": rewards.tolist(),
        "ours_theta_after": [round(float(x), 4) for x in ours_full],
        "grpo_toy_theta_after": [round(float(x), 4) for x in theirs],
        "max_abs_diff": round(diff, 12),
        "match": diff < 1e-9,
    }
