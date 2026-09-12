#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grpo_play_core.py — 互動式 GRPO demo 的共用核心
=================================================

這個模組提供 PolicyState 與互動式環境邏輯，供 FastAPI 後端（web/routers/toy.py）
與前端 live demo 調用。
把「policy、主題資料、GRPO 更新邏輯」集中在這裡，UI 各自展現。

核心設計：
  - PolicyState: 當前 policy 的狀態（logits + 主題），可序列化給 UI 用
  - THEMES: 兩個主題（math / dashboard），可切換
  - sample_group(): 對一個 query 抽 G 個候選（這就是 GRPO 的 group）
  - apply_user_choice(): 收 user 的 accept/reject → 算 advantage → 更新 policy

== 為什麼要這樣設計 ==
你要親眼看到 group 是「policy 當場抽樣出來的 G 個新候選」，
而不是 log 裡的舊 plan。所以核心動作就兩個：
  1. sample_group(): 抽出新 group（讓 user 從裡面挑）
  2. apply_user_choice(): user 挑完 → reward → advantage → 更新
"""

import numpy as np

# 跟前面的 toy demo 共用同一份 GRPO 更新邏輯（單一 source of truth）
from grpo_toy import (
    softmax, grpo_update,
    K_ACTIONS, EPS_CLIP, INNER_EPOCHS,
)

G = 4   # group size：每次給 user 看 4 個候選（UI 上太擠的話調小）


# ============================================================
# 主題資料
# ============================================================
THEMES = {
    # ----------------------------------------------------------
    # 主題 1：純數學題
    # 每個 query 有一個「正解 action」。reward = 答對就是 +1。
    # 最抽象、最純粹，用來摸懂機制。
    # ----------------------------------------------------------
    "math": {
        "name": "數學題",
        "action_label": "數字答案",
        "queries": [
            {"q": "1 + 1 = ?",          "correct": 2},
            {"q": "3 + 1 = ?",          "correct": 4},
            {"q": "0 + 0 = ?",          "correct": 0},
            {"q": "2 + 3 = ?",          "correct": 5},
            {"q": "1 + 4 = ?",          "correct": 5},
        ],
        # action -> 顯示文字（數學題就顯示數字本身）
        "render_action": lambda a: str(a),
        # user 選了某個 action，回傳它「作為答案」時的 reward
        "reward_of": lambda query, a: 1.0 if a == query["correct"] else 0.0,
        # 這個主題裡「user 接受某候選」= 該候選剛好是正解（自動判定）
        # （UI 還是會讓 user 自己選，但 reward 由正解決定，方便體驗 group-relative）
        "auto_label": True,
    },

    # ----------------------------------------------------------
    # 主題 2：Copilot / dashboard 風格
    # action 對應「某種 dashboard 配方」。reward 由 user 主觀選擇決定。
    # 完全對應你的 agentic AI 場景：user accept/reject 就是 reward。
    # ----------------------------------------------------------
    "dashboard": {
        "name": "Copilot dashboard",
        "action_label": "dashboard 配方",
        "queries": [
            {"q": "做一個 Austin market overview dashboard"},
            {"q": "幫我看 Dallas 新屋銷售"},
            {"q": "比較 Houston 跟 Austin 的定價"},
        ],
        # action -> 描述文字。6 個 action 對應 6 種配方風格。
        # 注意：這裡「沒有絕對正確答案」，reward 完全由 user 的選擇決定。
        # 這才是 GRPO 真正的精神 —— group 內相對比較，不需要 ground truth。
        "action_descriptions": {
            0: "2 個 component，極簡風（只要 overview）",
            1: "3 個 component，含定價圖",
            2: "3 個 component，含庫存 + 銷售",
            3: "4 個 component，含 YoY 比較",
            4: "5 個 component，完整分析（定價+庫存+YoY+趨勢）",
            5: "2 個 component，但加很多 filter",
        },
        "render_action": lambda a: THEMES["dashboard"]["action_descriptions"][a],
        # dashboard 主題沒有 ground truth：reward 由 user 選擇決定
        "reward_of": None,      # 用 apply_user_choice 裡的「被選中=+1, 其他=-1」
        "auto_label": False,
    },
}


# ============================================================
# PolicyState：policy 的完整狀態，可丟給 UI
# ============================================================
class PolicyState:
    """
    當前 policy 的狀態。

    theta: (N_queries, K_ACTIONS) 的 logits 矩陣。
           每一列 = 一個 query 的 policy（6 個 action 的機率）。
    theme: 'math' 或 'dashboard'。
    """
    def __init__(self, theme: str, seed: int = 0):
        assert theme in THEMES
        self.theme = theme
        self.queries = THEMES[theme]["queries"]
        self.theta = np.zeros((len(self.queries), K_ACTIONS))
        self.rng = np.random.default_rng(seed)
        self.history = []   # 記錄每一輪的 (query, group, accepted, advantage)

    # ---- 當前某個 query 的 policy 機率 ----
    def probabilities(self, query_idx: int) -> np.ndarray:
        return softmax(self.theta[query_idx])

    # ---- 抽 G 個候選（這就是 GRPO 的 group）----
    def sample_group(self, query_idx: int, g: int = G) -> np.ndarray:
        """對 query 抽 g 個 action。回傳形狀 (g,) 的 int 陣列。"""
        prob = self.probabilities(query_idx)
        return self.rng.choice(K_ACTIONS, size=g, p=prob)

    # ========================================================
    # 核心：apply_user_choice
    #   收 user 的選擇 → 形成 reward → 算 advantage → 更新 policy
    #   回傳一個 dict，告訴 UI 剛剛發生了什麼（給顯示用）
    # ========================================================
    def apply_user_choice(
        self,
        query_idx: int,
        group: np.ndarray,
        accepted_idx: int,   # user 接受的是 group 裡第幾個（-1 = 全拒絕）
    ) -> dict:
        """
        group:       sample_group 回傳的候選陣列
        accepted_idx: user 接受第幾個候選（0..G-1）；-1 表示全拒絕

        reward 規則：
          - math 主題（auto_label=True）：reward 由正解決定，不是 user。
            但這裡我們仍用「user 接受哪個」當訊號，只是為了體驗 group-relative。
            ※ 為了讓 demo 更有教學意義，math 主題下 reward 直接 = 正解判定。
          - dashboard 主題：被 user 接受 = +1.0，其他 = -1.0。
            （這就是「user 行為 = reward」的真實 GRPO 場景）

        回傳 dict：
          {
            "rewards":     [float]*G,
            "baseline":    float,
            "advantages":  [float]*G,
            "prob_before": [float]*K,
            "prob_after":  [float]*K,
            "accepted_action": int (或 None),
          }
        """
        theme = THEMES[self.theme]
        N = len(group)

        # ---- 1. 形成 reward ----
        rewards = np.zeros(N)
        if theme["auto_label"]:
            # math 主題：用「正解」自動判定。
            # 注意：這是教學簡化，讓使用者不用自己判斷哪個對。
            # 真實場景裡 math reward 是 rule-based（答對就是 1），跟 dashboard 一樣不需要 user 打分。
            q = self.queries[query_idx]
            correctness = np.array([theme["reward_of"](q, a) for a in group], dtype=float)
            # 答對 +1，答錯 -1（轉成 accept/reject 視角，跟 dashboard 一致）
            rewards = np.where(correctness > 0.5, 1.0, -1.0)
        else:
            # dashboard：被 user 接受 = +1.0，其他 = -1.0。
            # 這就是「user 行為 = reward」的真實 GRPO 場景，沒有 ground truth。
            rewards[:] = -1.0
            if accepted_idx >= 0:
                rewards[accepted_idx] = 1.0

        # ---- 2. 記住更新前的 policy 機率 ----
        prob_before = self.probabilities(query_idx).copy()

        # ---- 3. GRPO 更新（用共用核心的 grpo_update）----
        grpo_update(
            self.theta, query_idx, group, rewards,
            lr=1.0, eps_clip=EPS_CLIP,
            inner_epochs=INNER_EPOCHS, rng=self.rng,
        )

        prob_after = self.probabilities(query_idx)

        # ---- 4. 算 group 的 advantage（給 UI 顯示用，跟 grpo_update 內部一致）----
        baseline = rewards.mean()
        std = rewards.std()
        adv = (rewards - baseline) / (std + 1e-4)

        result = {
            "group":          group.tolist(),
            "rewards":        rewards.tolist(),
            "baseline":       float(baseline),
            "advantages":     [round(float(x), 3) for x in adv],
            "prob_before":    [round(float(x), 3) for x in prob_before],
            "prob_after":     [round(float(x), 3) for x in prob_after],
            "accepted_action": int(group[accepted_idx]) if accepted_idx >= 0 else None,
        }
        self.history.append({
            "query_idx": query_idx,
            **result,
        })
        return result


# ============================================================
# 一個極小的 self-test（python3 grpo_play_core.py）
# ============================================================
if __name__ == "__main__":
    print("== 測試 math 主題 ==")
    s = PolicyState("math")
    group = s.sample_group(0)
    print(f"  query[0] = {s.queries[0]['q']}, 正解 = {s.queries[0]['correct']}")
    print(f"  抽出的 group: {group.tolist()}")
    # 假裝 user 接受第 0 個
    res = s.apply_user_choice(0, group, accepted_idx=0)
    print(f"  rewards:   {res['rewards']}")
    print(f"  baseline:  {res['baseline']:.3f}")
    print(f"  advantages:{res['advantages']}")
    print(f"  正解機率: {res['prob_before'][2]} -> {res['prob_after'][2]}")

    print("\n== 測試 dashboard 主題 ==")
    s2 = PolicyState("dashboard")
    group2 = s2.sample_group(0)
    print(f"  query[0] = {s2.queries[0]['q']}")
    print(f"  抽出的 group (action 編號): {group2.tolist()}")
    for i, a in enumerate(group2):
        print(f"    [{i}] action={a}: {THEMES['dashboard']['render_action'](a)}")
    # 假裝 user 接受第 2 個
    res2 = s2.apply_user_choice(0, group2, accepted_idx=2)
    print(f"  rewards:   {res2['rewards']}")
    print(f"  advantages:{res2['advantages']}")
    print(f"  被接受的 action={res2['accepted_action']} 機率: "
          f"{res2['prob_before'][res2['accepted_action']]} -> {res2['prob_after'][res2['accepted_action']]}")
    print("\n✓ core OK")
