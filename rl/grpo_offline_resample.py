#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grpo_offline_resample.py — 解掉「每個 query 都不同、一個 session 很多 plan」的困惑
====================================================================================

核心一句話：
    GRPO 的 group 不是「user 看過的那幾個 plan」，
    而是「訓練時你自己重新抽樣出來的 G 個回答」。

這個 demo 要讓你親眼看到兩件事：

  1. 你的 Kairos log 裡，每個 query 通常只有 1 個 plan（user 不會一次看 8 個版本）。
     → 直接拿 log 當 group 永遠行不通（單一 candidate 沒有相對差異）。

  2. 但這不是問題。GRPO 在「訓練時」會對「同一個 prompt」重新抽 G 個回答，
     這 G 個才是 group。log 只負責提供兩樣東西：
       (a) prompt 來源（真實多樣的 user query）
       (b) reward 訊號 → 推導出一個 reward function / reward model

== 真實場景對照 ==
  你的 Kairos log（一個 session，很多 plan，每個 query 一個 plan）
       │
       ├─ 抽出 prompts          → GRPO 訓練的 prompt 集
       └─ accept/reject labels  → 推導 reward function（rule-based 或訓練一個 RM）
                                   ↓
  GRPO 訓練迴圈：每個 prompt → 當前 policy 抽 G 個 → reward function 打分 → 更新
                 （★ 這 G 個是訓練時「新生成」的，user 從沒看過）

執行：
    python3 grpo_offline_resample.py
"""

import numpy as np

from grpo_toy import softmax, grpo_update, K_ACTIONS, EPS_CLIP, INNER_EPOCHS


# ============================================================
# Phase 0：模擬一個「真實的」Kairos log
#
# 重點：這是一個 session，裡面有「很多不同的 query」，每個 query 「只有一個 plan」。
# 這正是你說的情境：「每個 query 都不同、一個 session 可以有很多 plan」。
# 這份 log 不是 GRPO 的 group 來源，它是 prompt + reward 訊號的來源。
# ============================================================

# 一個比較大的 query 集（10 題），讓你看到「query 很多樣、都不同」完全不是問題。
# 實務上這就是從你 Kairos log 抽出來的真實 user query。
ALL_QUERIES = [
    # (query 字串, 正解 action)
    ("1+1=?", 2),
    ("3+1=?", 4),
    ("0+0=?", 0),
    ("2+3=?", 5),
    ("1+4=?", 5),
    ("2+2=?", 4),
    ("3+3=?", 6 % K_ACTIONS),   # 6 mod 6 = 0，故意讓你看到 reward 的推導
    ("1+0=?", 1),
    ("4+1=?", 5),
    ("0+3=?", 3),
]


def simulate_one_session_log(rng):
    """
    模擬「一個 session」的真實 log：
    每個 query → user 只看到 1 個 plan → 給 accept 或 reject。

    回傳 list of dict:
      { "query": str, "query_idx": int, "plan_action": int, "label": "accept"/"reject" }

    ★ 注意：每個 query 只有「1 個 plan」。
    ★ 這就是為什麼「直接拿 log 當 group」會失敗 —— 單一 candidate 無相對比較。
    """
    log = []
    # 假裝這個 session 的 agent 還很弱（用接近 uniform 的「舊 policy」採樣）
    weak_policy = np.full(K_ACTIONS, 1.0 / K_ACTIONS)
    for qi, (q, correct) in enumerate(ALL_QUERIES):
        action = int(rng.choice(K_ACTIONS, p=weak_policy))
        label = "accept" if action == correct else "reject"
        log.append({
            "query": q,
            "query_idx": qi,
            "correct": correct,
            "plan_action": action,
            "label": label,
        })
    return log


# ============================================================
# Phase 1：從 log 推導 reward function
#
# 這一步對應「你的 accept/reject label → reward model」。
# 玩具版：log 裡被 accept 的 action 就是該 query 的正解，直接建成查表。
# 真實版：拿 (query, action, label) 訓練一個 neural reward model。
# 兩者「介面」完全一樣：reward(query_idx, action) -> float。
# ============================================================
def derive_reward_function(log):
    """
    從 log 推導 reward function。

    玩具做法：
      - 從 log 的 accept label 反推每個 query 的正解 action
      - reward(query_idx, action) = 1.0 if action == 正解 else 0.0

    真實做法（你之後會做的）：
      - 拿 (query_text, candidate, label) 訓練一個 RM
      - reward(query, candidate) = RM(query, candidate)
    """
    correct_map = {}   # query_idx -> correct action
    for entry in log:
        qi = entry["query_idx"]
        # 用 log 提供的「正解」（真實場景裡你可能要用 RM 學出來）
        correct_map[qi] = entry["correct"]

    def reward(query_idx, action):
        return 1.0 if action == correct_map[query_idx] else 0.0

    return reward, correct_map


# ============================================================
# Phase 2：GRPO 訓練 —— group 在這裡「新生成」，不是從 log 重播
# ============================================================
def main():
    rng = np.random.default_rng(42)

    print("=" * 74)
    print(" grpo_offline_resample.py")
    print(" 示範：group 是訓練時 resample 的，不是 log 裡那幾個 plan")
    print("=" * 74)

    # ---- Phase 0：拿到一份「真實」log（一個 session，多 query，每 query 1 plan）----
    log = simulate_one_session_log(rng)
    print(f"\n[Phase 0] 模擬一個 session 的 log：{len(log)} 個 query，每個 query 只有 1 個 plan")
    print("          （這正是你說的情境：一個 session 可以有很多 plan，但每個 query 一個）")
    print()
    print(f"  {'query':10s} {'log裡唯一的plan':14s} {'label':8s}  <- user 只看到這 1 個")
    for e in log:
        print(f"  {e['query']:10s} action={e['plan_action']}        {e['label']:8s}")

    # ---- Phase 1：從 log 推導 reward function ----
    reward_fn, correct_map = derive_reward_function(log)
    print(f"\n[Phase 1] 從 log 推導 reward function（rule-based 版，真實版換成訓練 RM）")
    print("          reward(query_idx, action) = 1.0 if action == 正解 else 0.0")
    print(f"          推導出的正解表：{ {q: correct_map[q] for q in sorted(correct_map)} }")

    # ========================================================
    # Phase 2：GRPO 訓練
    # ★★★ 這裡是整個 demo 的重點 ★★★
    # 每個 query（來自 log）→ 用「當前 policy」抽 G 個新候選 → reward_fn 打分 → 更新
    # ========================================================
    G = 8                       # group size：訓練時每個 query 抽幾個新候選
    LR = 1.0
    N_QUERIES = len(ALL_QUERIES)

    # policy：每個 query 一列 logits。注意 —— log 從來沒提供「group」，group 是這裡現抽的。
    theta = np.zeros((N_QUERIES, K_ACTIONS))

    print(f"\n[Phase 2] GRPO 訓練：G={G}，每個 query 在訓練時「重新抽樣」{G} 個新候選")
    print("          ★ 這 G 個 candidate 是訓練時新生成的，user 從沒看過")
    print("          ★ log 裡那個 plan 只提供了 prompt + 推導 reward 的線索，不參與 group")

    verbose_iters = {0, 1, 4, 9}
    N_ITERS = 12

    for it in range(N_ITERS):
        verbose = it in verbose_iters
        if verbose:
            print("\n" + "#" * 74)
            print(f"# 第 {it} 輪")
            print("#" * 74)

        total_reward = 0.0
        total_n = 0

        for entry in log:
            qi = entry["query_idx"]
            q_text = entry["query"]

            # ★★★ 核心一步：用「當前 policy」抽 G 個新候選 ★★★
            # 這 G 個就是 GRPO 的 group。它們跟 log 裡那 1 個 plan 完全無關。
            prob = softmax(theta[qi])
            group_actions = rng.choice(K_ACTIONS, size=G, p=prob)

            # 用 reward function（從 log 推導來的）給這 G 個打分
            group_rewards = np.array([reward_fn(qi, a) for a in group_actions])
            total_reward += group_rewards.sum()
            total_n += G

            if verbose:
                print(f"\n  Query[{qi}] {q_text!r}  (log 正解={correct_map[qi]})")
                print(f"    log 裡這個 query 的 plan：action={entry['plan_action']} "
                      f"label={entry['label']}   <- 只有 1 個，當不了 group")
                print(f"    訓練時 resample 的 group (G={G})：{group_actions.tolist()}")
                print(f"    reward_fn 打的分數：            {group_rewards.tolist()}")

            # GRPO 更新（跟 toy demo / log demo 用的是完全同一個 grpo_update）
            grpo_update(
                theta, qi, group_actions, group_rewards,
                lr=LR, eps_clip=EPS_CLIP,
                inner_epochs=INNER_EPOCHS, rng=rng,
            )

        if it % 2 == 0 or it == N_ITERS - 1:
            print(f"[iter {it:2d}] 全 query 平均 reward = {total_reward/total_n:.3f}   "
                  f"(1.0 = 每個候選都答對)")

    # ========================================================
    # 收尾：看 policy 學到什麼 + 重申關鍵觀念
    # ========================================================
    print("\n" + "=" * 74)
    print(" 訓練結束 — policy 學到的結果")
    print("=" * 74)
    n_correct = 0
    for qi, (q, correct) in enumerate(ALL_QUERIES):
        prob = softmax(theta[qi])
        best = int(prob.argmax())
        ok = best == correct
        n_correct += ok
        bar = " ".join(f"{a}:{prob[a]:.2f}" for a in range(K_ACTIONS))
        print(f"  {q:8s} 正解={correct} | {bar} | 最常選={best} {'✓' if ok else '✗'}")
    print(f"\n  {n_correct}/{len(ALL_QUERIES)} 題正解被選成最高機率。")

    print("\n" + "-" * 74)
    print(" 三個關鍵觀念回顧（對應你的問題）：")
    print("-" * 74)
    print("""
  Q: 「每個 query 都不同」會不會讓 GRPO 失效？
  A: 不會。GRPO 永遠是「同一個 query 內部」的 G 個候選互相比，
     query 再怎麼獨特都沒關係。多樣的 query 反而是好事 → 訓練 prompt 集。

  Q: 「一個 session 很多 plan」怎麼用？
  A: 一個 session 的 N 個 plan = N 個獨立訓練 prompt。
     它們不是「一個 group」，而是「N 個各自會展開成 group 的 prompt」。
     本 demo 的 log 有 10 個 query → 訓練時每輪就跑 10 個 group。

  Q: log 裡一個 query 只有 1 個 plan 怎麼辦？
  A: 那個 plan 本來就不該當 group。它的用途是：
       (a) 提供這個 query 當訓練 prompt
       (b) 用 accept/reject label 推導 reward function
     group 是訓練時 policy 自己抽樣出來的 G 個新候選（本 demo G=8）。
""")


if __name__ == "__main__":
    main()
