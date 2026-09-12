#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_log_to_grpo.py — 從 Kairos log 走到 GRPO 更新的完整流程
=============================================================

這個 demo 把兩件事串起來：

   真實 agent 互動                    RL 訓練
   ─────────────                     ─────────
   KairosLogMemory.log_event()  ──>  to_grpo_dataset()  ──>  grpo_update()

也就是：先「收集」多個 candidate 的 reward 訊號到 log，
再「回放」成 GRPO 要的 group，最後真的跑一次 GRPO 更新。

== 為什麼這個 demo 對你的 agentic AI 重要 ==
agent 的 policy 是 LLM，RL 幾乎都是 offline 的：
   1. 先讓 agent 跟 user 互動，把「同一個 query 下的多個 candidate + 它們的 accept/reject」
      寫進 Kairos（這是 reward 訊號的唯一正規來源）。
   2. 離線把 log 還原成 GRPO dataset。
   3. 用 GRPO 更新 policy（在 LLM 場景就是更新模型權重）。

這個 demo 用玩具任務（6 個 action 的問答）示範整條鏈。
真正接 LLM 時，只有 step 3 的「policy」要換成 LLM，前兩步完全一樣。

執行：
    python3 demo_log_to_grpo.py
"""

import os
import tempfile

import numpy as np

from kairos_log import (
    KairosLogMemory,
    to_grpo_dataset,
    group_relative_advantage,
)
from grpo_toy import (
    softmax,
    grpo_update,
    K_ACTIONS,
    EPS_CLIP,
    INNER_EPOCHS,
)


# ============================================================
# 玩具任務設定（跟 grpo_toy.py 一致，方便對照）
# ============================================================
PROMPTS = ["1+1=?", "3+1=?", "0+0=?", "2+3=?"]
CORRECT = [2, 4, 0, 5]
N_PROMPTS = len(PROMPTS)


def reward_of(prompt_idx: int, action: int) -> float:
    return 1.0 if action == CORRECT[prompt_idx] else 0.0


# ============================================================
# Step 1：模擬 agent 互動，把候選答案 + accept/reject 寫進 Kairos
#         這一步對應你 orchestrator 裡的「三個 log_event 呼叫點」
# ============================================================
def collect_interaction_log(kairos: KairosLogMemory, theta: np.ndarray,
                            group_size: int = 6, rng=None):
    """
    模擬：對每個 prompt，用「目前 policy」採樣 group_size 個候選答案，
    然後假裝 user 對「答對的候選」按 accept、對「答錯的」按 reject，
    全部寫進 Kairos log。

    真實場景對應：
      - group_size 個候選 = 對同一個 user query 生成多個回答（n > 1 sampling）
      - accept / reject   = user 的實際行為（接受、修改、按 undo）
    """
    rng = rng or np.random.default_rng()
    for p in range(N_PROMPTS):
        prob = softmax(theta[p])
        actions = rng.choice(K_ACTIONS, size=group_size, p=prob)

        # 每個 prompt 視為一個「user query」，用 query 編號當 plan_id 的前綴
        # 注意：真實 GRPO group 的 key 應該是 query 本身（同一 query 下的多個 candidate）。
        # 這裡每個 prompt 採樣的 group_size 個答案，就是同一個 query 下的多個 candidate，
        # 所以我們讓它們共用同一個 query_id。
        query_id = f"q{p}"

        for cand_idx, a in enumerate(actions):
            r = reward_of(p, a)
            # 每個 candidate 一個專屬 plan_id（candidate 內的唯一 key）
            plan_id = f"{query_id}_c{cand_idx}"

            # 模擬 orchestrator 的三個呼叫點
            kairos.log_event(
                "plan_generated", plan_id=plan_id,
                summary=f"{PROMPTS[p]} candidate #{cand_idx} = {a}",
                user_id="alice", session_id="s1",
            )
            kairos.log_event(
                "dashboard_built", plan_id=plan_id, result="success",
                summary=f"candidate #{cand_idx} built", user_id="alice", session_id="s1",
            )
            # reward 訊號：答對 = user_accept，答錯 = user_reject
            kairos.log_event(
                "user_accept" if r == 1.0 else "user_reject",
                plan_id=plan_id,
                quality_tag="user_accepted" if r == 1.0 else "user_rejected",
                summary=f"candidate #{cand_idx} {'accepted' if r==1.0 else 'rejected'}",
                user_id="alice", session_id="s1",
            )

    # 把 query → 它的 candidates 的 mapping 記下來，供 step 2 重組 group 用
    # （因為真實 GRPO group 是「同一 query 下的多個 candidate」）
    return {f"q{p}": [f"q{p}_c{i}" for i in range(group_size)]
            for p in range(N_PROMPTS)}


# ============================================================
# Step 2：把 log 還原成「以 query 為 group key」的 GRPO dataset
# ============================================================
def to_grpo_dataset_by_query(kairos: KairosLogMemory, query_to_plans: dict):
    """
    GRPO 的 group 定義是「同一個 query 下的多個 candidate」。
    Kairos log 是 per-plan_id 存的，所以這裡把同一 query 下的所有 plan 收成一個 group。

    每個 candidate 的 reward = 它「最終」的 reward 訊號
    （取 user_accept/user_reject/undo 那一條；若都沒有就 fallback）。
    """
    # 先拿到 per-plan 的 reward
    by_plan = {}
    events = list(kairos.iter_events())
    for e in events:
        pid = e.get("plan_id")
        if pid is None:
            continue
        by_plan.setdefault(pid, []).append(e)

    # 簡單的 per-plan reward：優先取 user_accept/reject/undo 那條
    def plan_reward(evts):
        from kairos_log import reward_from_event
        # 倒著看，取最後一條有 reward 訊號的
        for e in reversed(evts):
            r = reward_from_event(e)
            if r is not None:
                return r
        return 0.1  # fallback

    dataset = []
    for query_id, plan_ids in query_to_plans.items():
        rewards = [plan_reward(by_plan.get(pid, [])) for pid in plan_ids]
        dataset.append({
            "query_id": query_id,
            "plan_ids": plan_ids,
            "rewards": rewards,
        })
    return dataset


# ============================================================
# Step 3：用還原出來的 GRPO dataset 真的跑一次更新
#         （這就是 offline GRPO 的核心迴圈）
# ============================================================
def main():
    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp()

    # 初始 policy：uniform
    theta = np.zeros((N_PROMPTS, K_ACTIONS))

    print("=" * 70)
    print(" demo: Kairos log → GRPO dataset → grpo_update()")
    print(" （offline GRPO 的最小完整流程）")
    print("=" * 70)

    LR = 1.0
    for outer in range(3):  # 跑 3 輪 collect → train，看 reward 上升
        log_path = os.path.join(tmp, f"round{outer}.md")
        kairos = KairosLogMemory(log_path)

        # --- collect 階段：模擬 agent 互動，寫 log ---
        query_to_plans = collect_interaction_log(
            kairos, theta, group_size=6, rng=rng
        )

        # --- replay 階段：log → GRPO dataset ---
        ds = to_grpo_dataset_by_query(kairos, query_to_plans)

        total_r = sum(sum(g["rewards"]) for g in ds)
        n = sum(len(g["rewards"]) for g in ds)
        print(f"\n[round {outer}] collect 完，log 有 {len(list(kairos.iter_events()))} 個事件")
        print(f"             還原成 {len(ds)} 個 group，平均 reward = {total_r/n:.3f}")

        # --- train 階段：每個 group 跑一次 grpo_update ---
        # 把「同 query 的多個 candidate」當成 group；
        # 但這裡每個 candidate 是「一個 action 採樣」，
        # 對 toy policy 來說，我們把同 query 的 actions + rewards 直接餵進 grpo_update。
        for g in ds:
            p = int(g["query_id"][1:])  # q0..q3 → prompt index
            # 還原每個 candidate 的 action（從 plan_id 的 c{idx} 對回採樣時的 actions）
            # 為了 demo 簡潔，這裡直接用 log 裡 plan_generated 的 summary 反推 action
            actions = []
            for pid in g["plan_ids"]:
                evts = [e for e in kairos.iter_events()
                        if e.get("plan_id") == pid
                        and e.get("event_type") == "plan_generated"]
                # summary 形如 "1+1=? candidate #0 = 2" → 取最後一個整數
                s = evts[0]["summary"]
                actions.append(int(s.split("=")[-1].strip()))
            actions = np.array(actions)
            rewards = np.array(g["rewards"], dtype=float)

            baseline, std, adv, _ = grpo_update(
                theta, p, actions, rewards,
                lr=LR, eps_clip=EPS_CLIP,
                inner_epochs=INNER_EPOCHS, rng=rng,
            )

        if outer == 2:
            print("\n--- 3 輪後 policy 的機率分佈 ---")
            for p in range(N_PROMPTS):
                prob = softmax(theta[p])
                bar = " ".join(f"{a}:{prob[a]:.2f}" for a in range(K_ACTIONS))
                print(f"  {PROMPTS[p]:6s} 正解={CORRECT[p]} | {bar} | "
                      f"最常選={int(prob.argmax())} {'✓' if int(prob.argmax())==CORRECT[p] else '✗'}")

    print("\n" + "=" * 70)
    print(" ✓ 完整流程跑通：")
    print("   orchestrator.log_event() → Kairos log → to_grpo_dataset() → grpo_update()")
    print("=" * 70)


if __name__ == "__main__":
    main()
