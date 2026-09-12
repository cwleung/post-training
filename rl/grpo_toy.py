#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO (Group Relative Policy Optimization) 最小玩具 Demo
======================================================

用「18 歲 + 繁中」的角度，把 GRPO 拆到只剩骨頭。
全程只用 numpy，沒有 torch、沒有 black box，每一步的數字都會印出來給你看。

== 我們在解的玩具任務 ==
想像一個超迷你的「數學問答 agent」：
    - 有 N 個 prompt（問題），每題有一個正確答案。
    - policy（策略）對每個 prompt 會給 6 個候選答案（0~5）一組機率。
    - reward（獎勵）：答對 = 1.0，答錯 = 0.0。完全 binary，跟 DeepSeek R1 的
      rule-based reward 精神一樣。

== GRPO 在做的事（一句話）==
對同一個 prompt，讓 policy「一次吐 G 個候選答案」組成一個 group，
只看「這個 group 內部誰比誰好」（相對），來更新 policy。
不需要訓練額外的 critic / value network。

== 跟你（agentic AI / memory project）的關係 ==
你之後的 agent，policy 就是 LLM 本體。
這個 demo 裡的「logits 向量」== LLM 的權重；
「sample G 個候選」== 對同一個 user query 生成多個回答；
「reward」== user_accept / user_reject / 正確性；
「group baseline」== 你不需要 critic，只要在這組回答裡算平均。
學會這個，再回頭看 Kairos/Conversation log → GRPO dataset 就會很自然。

執行：
    python3 grpo_toy.py
"""

import numpy as np

# ============================================================
# 1. 超參數（每一個都會在訓練中用到，這裡先認識它們）
# ============================================================
SEED = 0                 # 固定亂數，讓結果可重現（demo 才好講）

N_PROMPTS = 4            # 有幾個「問題」。實務上這是 dataset 裡的 prompt 數量。
K_ACTIONS = 6            # 每個問題有 6 個候選答案（0,1,2,3,4,5）。實務上是詞表。
GROUP_SIZE = 8           # G：對同一個 prompt 一次採樣幾個答案組成一個「group」。
                         #   ↑ 這就是 GRPO 名字裡的 "Group"。

LR = 1.0                 # 學習率：每一步 policy 要移動多大步。
EPS_CLIP = 0.2           # PPO/GRPO 的 clip 範圍。限制每次更新不要走太遠（穩定訓練）。
                         #   ↑ 名字裡的 "Optimization" 偷偷借用了 PPO 的 clip。
INNER_EPOCHS = 4         # 同一個 group 的資料要重複用幾次（PPO 風格的 multi-epoch）。
N_ITERS = 60             # 總共跑幾輪 outer loop。

ADV_EPS = 1e-4           # 算 advantage 時避免除以 0 的小常數。

# 玩具任務：每個 prompt 對應的正確答案（reward = 1.0 的那個 action）
PROMPTS = ["1+1=?", "3+1=?", "0+0=?", "2+3=?"]
CORRECT = [2, 4, 0, 5]   # 正確答案分別是 2, 4, 0, 5


# ============================================================
# 2. Reward function（獎勵函數）
# ============================================================
def reward(prompt_idx: int, action: int) -> float:
    """答對 1.0，答錯 0.0。binary、rule-based，跟 DeepSeek 風格一致。"""
    return 1.0 if action == CORRECT[prompt_idx] else 0.0


# ============================================================
# 3. Policy（策略）= 一組 logits，經 softmax 變成機率
# ============================================================
def softmax(logits: np.ndarray) -> np.ndarray:
    """把 logits（任意實數）轉成「加起來 = 1」的機率分佈。"""
    z = logits - logits.max()           # 數值穩定技巧（不影響結果）
    e = np.exp(z)
    return e / e.sum()


def sample_actions(prob: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """從目前 policy 的機率分佈，抽 n 個答案。這 n 個就是一個 group。"""
    return rng.choice(K_ACTIONS, size=n, p=prob)


# ============================================================
# 4. GRPO 核心更新（這是最值得讀的一段）
# ============================================================
def grpo_update(
    theta: np.ndarray,          # 整個 policy 的 logits（shape: N_PROMPTS x K_ACTIONS）
    prompt_idx: int,            # 這一步要更新哪個 prompt 的那一列 logits
    actions: np.ndarray,        # 這個 group 裡採樣到的 G 個答案
    rewards: np.ndarray,        # 這 G 個答案各自的 reward
    lr: float,
    eps_clip: float,
    inner_epochs: int,
    rng: np.random.Generator,
):
    # ---- 4a. group baseline（GRPO 的靈魂）---------------------------------
    # baseline = 這個 group 裡所有 reward 的「平均」。
    # 我們不訓練 critic/value network，直接拿 group 平均當基準線。
    # 這是 GRPO 跟 PPO 最大的結構差異。
    baseline = rewards.mean()
    # group std（標準差）：用來把 advantage「正規化」，讓學習更穩。
    #   GRPO 論文裡用的是：A_i = (r_i - mean) / (std + eps)
    std = rewards.std()
    advantages = (rewards - baseline) / (std + ADV_EPS)
    # 白話：
    #   比 group 平均好的答案  -> advantage > 0 -> policy 要「增加」它出現的機率
    #   比 group 平均差的答案  -> advantage < 0 -> policy 要「降低」它出現的機率
    #   整組都一樣好/一樣爛    -> std≈0 -> advantage≈0 -> 這組學不到東西（正常！）

    # ---- 4b. 記住「舊 policy」= 採樣當下的機率 ----------------------------
    # PPO/GRPO 的 ratio = π_new / π_old，需要先存一份採樣時的 π_old。
    pi_old = softmax(theta[prompt_idx]).copy()

    # ---- 4c. PPO-style 的 clipped 更新（multi-epoch）---------------------
    # 同一批 group 資料重複用好幾次，每次都用 clip 防止 policy 偏離太遠。
    for _ in range(inner_epochs):
        pi_new = softmax(theta[prompt_idx])     # 更新後的當前 policy 機率
        grad = np.zeros(K_ACTIONS)              # 要算的梯度（對 logits）
        obj = 0.0                               # 這次的 surrogate objective（越大越好）

        for a, A in zip(actions, advantages):
            # ratio：新 policy 對「這個答案」的機率 是 舊 policy 的幾倍
            ratio = pi_new[a] / pi_old[a]

            # PPO 的 clipped surrogate（兩條取最小）：
            #   surr1 = ratio * A            （正常的 policy gradient）
            #   surr2 = clip(ratio) * A      （把 ratio 鉗在 [1-eps, 1+eps]）
            # 取 min = 「不讓更新佔太多便宜」，這就是 clip 的作用。
            surr1 = ratio * A
            surr2 = np.clip(ratio, 1 - eps_clip, 1 + eps_clip) * A
            take = min(surr1, surr2)
            obj += take

            # 梯度：只有在「沒被 clip」的那條路上才有梯度（PPO 的 stop-gradient 概念）。
            # 數學：對 logit b 微分，d(ratio*A)/dθ_b = A * ratio * ( 1[a==b] - π_new[b] )
            if surr1 <= surr2:                  # 走 unclipped 這條才有梯度
                for b in range(K_ACTIONS):
                    indicator = 1.0 if a == b else 0.0
                    grad[b] += A * ratio * (indicator - pi_new[b])

        grad /= len(actions)                    # 對 group 內取平均
        obj /= len(actions)

        # 因為 objective 越大越好，所以「梯度上升」。
        theta[prompt_idx] += lr * grad

    return baseline, std, advantages, pi_old


# ============================================================
# 5. 主訓練迴圈
# ============================================================
def main():
    rng = np.random.default_rng(SEED)

    # 初始 policy：每個 action 機率差不多（logits 全 0 → uniform）。
    # 剛開始 agent 還沒學過，所以亂猜。
    theta = np.zeros((N_PROMPTS, K_ACTIONS))

    print("=" * 70)
    print(" GRPO 玩具 Demo — 開始")
    print(" 任務：", PROMPTS)
    print(" 正解：", [f"{p}={c}" for p, c in zip(PROMPTS, CORRECT)])
    print(" 每個 prompt 採樣", GROUP_SIZE, "個答案組成一個 group")
    print("=" * 70)

    verbose_iters = {0, 1, 2, 9, 29, N_ITERS - 1}   # 只在這幾輪詳細列印，避免洗版

    for it in range(N_ITERS):
        verbose = it in verbose_iters
        if verbose:
            print("\n" + "#" * 70)
            print(f"# 第 {it} 輪（outer iteration）")
            print("#" * 70)

        total_reward = 0.0

        # 對「每一個 prompt」都跑一次 GRPO 更新
        for p in range(N_PROMPTS):
            prob = softmax(theta[p])
            actions = sample_actions(prob, GROUP_SIZE, rng)
            rewards = np.array([reward(p, a) for a in actions])
            total_reward += rewards.sum()

            if verbose:
                print(f"\n  Prompt[{p}] {PROMPTS[p]!r}  正解={CORRECT[p]}")
                print(f"    採樣到的 group (G={GROUP_SIZE}): {actions.tolist()}")
                print(f"    每個答案的 reward:             {rewards.tolist()}")

            # 進行 GRPO 更新
            baseline, std, advantages, pi_old = grpo_update(
                theta, p, actions, rewards, LR, EPS_CLIP, INNER_EPOCHS, rng
            )

            if verbose:
                print(f"    group baseline (平均 reward):  {baseline:.3f}")
                print(f"    group std:                     {std:.3f}")
                print(f"    advantage (相對優勢):          {[round(float(x),3) for x in advantages]}")
                print(f"    更新前 π_old:                  {[round(float(x),3) for x in pi_old]}")
                prob_new = softmax(theta[p])
                print(f"    更新後 π_new:                  {[round(float(x),3) for x in prob_new]}")
                # 白話解讀：正解的機率應該慢慢變大
                print(f"    -> 正解(action={CORRECT[p]}) 機率: "
                      f"{pi_old[CORRECT[p]]:.3f} → {prob_new[CORRECT[p]]:.3f}")

        # 每幾輪印一次整體分數，方便看收斂
        if it % 10 == 0 or it == N_ITERS - 1:
            avg = total_reward / (N_PROMPTS * GROUP_SIZE)
            print(f"[iter {it:3d}] 平均 reward = {avg:.3f}   "
                  f"(1.0 = 每次都答對)")

    # ========================================================
    # 6. 收斂後看一下結果
    # ========================================================
    print("\n" + "=" * 70)
    print(" 訓練結束 — 看看 policy 學到什麼")
    print("=" * 70)
    correct_count = 0
    for p in range(N_PROMPTS):
        prob = softmax(theta[p])
        best = int(prob.argmax())
        ok = best == CORRECT[p]
        correct_count += ok
        bar = " ".join(f"{a}:{prob[a]:.2f}" for a in range(K_ACTIONS))
        print(f"  {PROMPTS[p]:6s} 正解={CORRECT[p]} | policy 機率: {bar} | "
              f"最常選={best} {'✓' if ok else '✗'}")
    print(f"\n  {correct_count}/{N_PROMPTS} 題的 policy 把正解選成最高機率。")

    print("\n  重點回顧：")
    print("   - GRPO = 每個 prompt 採樣 G 個答案 -> 用 group 平均當 baseline")
    print("   - advantage = reward - baseline（再除以 group std 正規化）")
    print("   - 更新時套 PPO 的 clip，避免 policy 一步走太遠")
    print("   - 完全不需要 critic / value network（省一大塊）")
    print("\n  下一步想深入？看同資料夾 README.md，有衔接到你 memory/agent 專案的說明。")


if __name__ == "__main__":
    main()
