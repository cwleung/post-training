#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
routers/offline.py — Part 1：Kairos→dataset + resample-not-replay
=================================================================

重現兩個 offline demo 的核心邏輯，回 JSON（不靠 stdout capture）：

  /collect    → demo_log_to_grpo.collect_interaction_log：模擬 agent 互動寫 log
  /dataset    → to_grpo_dataset_by_query + group_relative_advantage：log → GRPO group
  /resample-train → grpo_offline_resample.main()：group 是訓練時 resample 的（★關鍵觀念）
  /kairos-log → 看剛剛 /collect 產生的 log（人類可讀 + JSON）
  /reward-csv → 該 log 對應的 reward_log.csv（如果有）

用單一 OFFLINE_STATE 存 KairosLogMemory（這不是互動式，每次 /collect 重建）。
"""

import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException

from kairos_log import KairosLogMemory, to_grpo_dataset, group_relative_advantage
from grpo_toy import softmax, grpo_update, K_ACTIONS, EPS_CLIP, INNER_EPOCHS
import demo_log_to_grpo as dlg
import grpo_offline_resample as gor

from .. import schemas
from ..state import OFFLINE_STATE

router = APIRouter()


def _kairos() -> KairosLogMemory:
    k = OFFLINE_STATE.get("kairos")
    if k is None:
        raise HTTPException(400, "還沒 collect。請先 POST /collect。")
    return k


# ------------------------------------------------------------
@router.post("/collect", response_model=dict)
def collect(body: schemas.OfflineCollectIn):
    """
    模擬 agent 互動：對 4 個 prompt 各採樣 group_size 個 candidate，
    accept/reject 寫進一份新的 Kairos log（覆蓋前一次）。
    回傳 query → plan_ids mapping。
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="offline_"))
    log_path = tmpdir / "collect.md"
    kairos = KairosLogMemory(log_path)
    theta = np.zeros((dlg.N_PROMPTS, K_ACTIONS))   # uniform 起點
    rng = np.random.default_rng(0)
    query_to_plans = dlg.collect_interaction_log(kairos, theta, body.group_size, rng=rng)

    OFFLINE_STATE.clear()
    OFFLINE_STATE["kairos"] = kairos
    OFFLINE_STATE["query_to_plans"] = query_to_plans
    OFFLINE_STATE["group_size"] = body.group_size
    return {"query_to_plans": query_to_plans, "group_size": body.group_size,
            "log_path": str(log_path)}


# ------------------------------------------------------------
@router.get("/dataset", response_model=schemas.OfflineDatasetOut)
def dataset():
    """log → GRPO dataset（以 query 為 group key）+ group_relative_advantage。"""
    kairos = _kairos()
    query_to_plans = OFFLINE_STATE.get("query_to_plans")
    if not query_to_plans:
        raise HTTPException(400, "沒有 query_to_plans。請先 POST /collect。")

    raw = dlg.to_grpo_dataset_by_query(kairos, query_to_plans)
    groups = []
    n_events = len(list(kairos.iter_events()))
    for g in raw:
        rewards = g["rewards"]
        adv, mean, std = group_relative_advantage(rewards)
        groups.append(schemas.DatasetGroupOut(
            query_id=g["query_id"],
            plan_ids=g["plan_ids"],
            rewards=rewards,
            advantage=[round(float(x), 3) for x in adv],
            baseline=round(float(mean), 4),
            std=round(float(std), 4),
        ))
    return schemas.OfflineDatasetOut(groups=groups, n_events=n_events)


# ------------------------------------------------------------
@router.post("/resample-train", response_model=schemas.OfflineResampleOut)
def resample_train():
    """
    重跑 grpo_offline_resample.main() 的邏輯，回 JSON。
    ★ 教學重點：group 是訓練時 resample 的，不是 log 裡那幾個 plan。
    """
    rng = np.random.default_rng(42)

    # Phase 0：模擬一個 session log（每 query 1 個 plan）
    log = gor.simulate_one_session_log(rng)
    log_rows = [{
        "query": e["query"], "query_idx": e["query_idx"],
        "plan_action": e["plan_action"], "label": e["label"],
        "correct": e["correct"],
    } for e in log]

    # Phase 1：從 log 推導 reward function
    reward_fn, correct_map = gor.derive_reward_function(log)
    correct_map_str = {str(k): v for k, v in sorted(correct_map.items())}

    # Phase 2：GRPO 訓練 — 每個 query 用當前 policy resample G=8 個新候選
    G = 8
    LR = 1.0
    N_QUERIES = len(gor.ALL_QUERIES)
    theta = np.zeros((N_QUERIES, K_ACTIONS))
    N_ITERS = 12

    iters = []
    for it in range(N_ITERS):
        total_reward = 0.0
        total_n = 0
        for entry in log:
            qi = entry["query_idx"]
            prob = softmax(theta[qi])
            group_actions = rng.choice(K_ACTIONS, size=G, p=prob)
            group_rewards = np.array([reward_fn(qi, a) for a in group_actions])
            total_reward += group_rewards.sum()
            total_n += G
            grpo_update(theta, qi, group_actions, group_rewards,
                        lr=LR, eps_clip=EPS_CLIP,
                        inner_epochs=INNER_EPOCHS, rng=rng)
        iters.append({"iter": it, "avg_reward": round(total_reward / total_n, 4)})

    # 收尾：final policy
    final_policy = []
    n_correct = 0
    for qi, (q, correct) in enumerate(gor.ALL_QUERIES):
        prob = softmax(theta[qi])
        best = int(prob.argmax())
        ok = best == correct
        n_correct += int(ok)
        final_policy.append({
            "query": q, "correct": correct,
            "probs": [round(float(p), 3) for p in prob],
            "best": best, "ok": ok,
        })

    return schemas.OfflineResampleOut(
        log_rows=log_rows,
        n_queries=N_QUERIES,
        correct_map=correct_map_str,
        iters=iters,
        final_policy=final_policy,
        n_correct=n_correct,
    )


# ------------------------------------------------------------
@router.get("/kairos-log", response_model=schemas.OfflineLogOut)
def kairos_log():
    """看 /collect 產生的 log：人類可讀視圖 + JSON payload。"""
    kairos = _kairos()
    return schemas.OfflineLogOut(
        human_view=kairos.read(max_lines=500),
        events=list(kairos.iter_events()),
    )


@router.get("/reward-csv")
def reward_csv():
    """如果有 reward_log.csv（offline demo 通常沒有，LLM 頁才有）就回內容。"""
    path = OFFLINE_STATE.get("reward_csv")
    if not path or not Path(path).exists():
        return {"rows": [], "note": "offline demo 不產 reward_log.csv；那是 LLM 頁的產物。"}
    import csv
    with open(path, encoding="utf-8") as f:
        return {"rows": list(csv.DictReader(f))}
