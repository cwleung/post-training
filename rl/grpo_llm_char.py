#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grpo_llm_char.py — 從 toy GRPO 走到真實 LLM GRPO 的關鍵一步
==============================================================

這份 demo 跟 grpo_toy.py 的差別，就是「DeepSeek R1 的 GRPO 跟 PPO 的差別」縮小版：

  grpo_toy.py：
      policy = 6 個 action 的機率分佈
      candidate = 從 6 個選一個                ← 離散單步
      logprob  = log(π[action])

  grpo_llm_char.py（這份）：
      policy = 字元表上每個 char 的 logits（給定前面已生成的字）
      candidate = 自回歸生成「一整段字串」       ← 序列
      logprob  = Σ_t log π[char_t | char_<t]    ← 整段序列的 log 機率
      ratio    = exp(logprob_new - logprob_old) ← DeepSeek R1 loss 公式長這樣

== 為什麼這一步關鍵 ==
你看 DeepSeek R1 / 任何 LLM GRPO 論文，loss 一定有這項：

      ratio_t(θ) = exp( logπ_θ(y_t|x,y_<t) − logπ_old(y_t|x,y_<t) )

單步 toy demo 看不到「序列機率」「token-level log-prob」這些概念。
這份 demo 把它們具體化，讓你以後看論文的 loss 公式不會跳太快。

== 玩具任務 ==
學一個字元級 LM 來生成「正確的數字答案」。
  - prompt = "1+1="
  - 正確回應 = "2"
  - policy 從頭學（初始 logits=0 → 起手亂碼）

執行：
    python3 grpo_llm_char.py
"""

import numpy as np

# ============================================================
# 1. 超參數
# ============================================================
SEED = 0

# 字元表（超小 vocab，方便看數字）
#   '<' = BOS / prompt 起點
#   '>' = EOS（生成到此停止）
#   '0'-'9' = 數字
#   其他可選空白、'+'、'='，讓 prompt 也編碼得進去
VOCAB = ["<", ">", "+", "=", " ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
V2I = {c: i for i, c in enumerate(VOCAB)}
I2V = {i: c for i, c in enumerate(VOCAB)}
V = len(VOCAB)

GROUP_SIZE = 8          # G：每個 prompt 抽幾段候選字串
MAX_LEN = 4             # 生成最長幾個 char（含 EOS）
LR = 1.0
EPS_CLIP = 0.2
INNER_EPOCHS = 4
ADV_EPS = 1e-4
N_ITERS = 120

# 玩具任務：prompt → 正確回應字串
TASKS = [
    ("1+1=", "2"),
    ("3+1=", "4"),
    ("0+0=", "0"),
    ("2+3=", "5"),
    ("1+4=", "5"),
]


# ============================================================
# 2. Policy：一個 (context_char -> logits over vocab) 的查表
#
# 真實 LLM 是幾十億參數的 transformer；這裡用一個小 table 代替。
# 但「介面」完全一樣：給定上文，輸出 vocab 上的 logits。
# ============================================================
class CharPolicy:
    """
    最簡的自回歸 policy：
        logits(context_char) -> 機率 over vocab

    用「上一個 char」當 context（1-gram LM）。
    真實 LLM 用整個上文（前面所有 token），這裡簡化成 1 步，
    但序列機率、token-level logprob、ratio 的計算方式跟真實 LLM 完全一樣。
    """
    def __init__(self):
        # theta[v_prev, v_curr] = 給定上一個 char 是 v_prev，輸出 v_curr 的 logit
        self.theta = np.zeros((V, V))

    def logits_at(self, prev_id: int) -> np.ndarray:
        """給定上一個 token id，回傳當前位置的 logits。"""
        return self.theta[prev_id]

    def probs_at(self, prev_id: int) -> np.ndarray:
        z = self.theta[prev_id]
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()


# ============================================================
# 3. 序列機率的核心：自回歸採樣 + token-level log-prob
# ============================================================
def encode(s: str) -> list:
    """字串 -> token id 列表。"""
    return [V2I[c] for c in s]


def decode(ids: list) -> str:
    return "".join(I2V[i] for i in ids)


def generate(policy: CharPolicy, prompt_ids: list, max_len: int,
             rng: np.random.Generator) -> list:
    """
    自回歸生成：給定 prompt，往後生 token，直到 EOS 或 max_len。

    回傳「完整序列」（prompt + 生成部分），最後一個通常是 EOS。
    這就是 LLM 的 sample：每一步根據當前 logits 抽下一個 token。
    """
    seq = list(prompt_ids)
    eos = V2I[">"]
    for _ in range(max_len):
        prev = seq[-1]
        prob = policy.probs_at(prev)
        nxt = int(rng.choice(V, p=prob))
        seq.append(nxt)
        if nxt == eos:
            break
    return seq


def seq_logprob(policy: CharPolicy, full_seq: list, prompt_len: int) -> float:
    """
    ★★★ LLM-flavored GRPO 的關鍵函式 ★★★

    一段序列的 log 機率 = Σ_t log π(token_t | token_<t)，只算「生成部分」。

    - full_seq   = prompt + generated
    - prompt_len = prompt 的長度（不納入 logprob，因為 prompt 不是模型生成的）

    對應 DeepSeek R1 loss 公式裡的 log π_θ(y_t | x, y_<t)。
    """
    lp = 0.0
    for t in range(prompt_len, len(full_seq)):
        prev = full_seq[t - 1]
        prob = policy.probs_at(prev)
        lp += np.log(prob[full_seq[t]] + 1e-12)
    return lp


def seq_logprob_grad(policy: CharPolicy, full_seq: list, prompt_len: int,
                     coef: float) -> np.ndarray:
    """
    對 policy.theta 的梯度，帶一個係數 coef（會塞 advantage * ratio）。

    ∂log π(y)/∂θ[prev, curr] = coef * (1[prev,curr 在序列裡出現] - π[curr|prev])

    也就是「序列中每個 (prev→curr) 轉移都加分，但要扣掉機率本身」。
    這跟真實 transformer LM 的 cross-entropy 梯度結構完全一樣。
    """
    grad = np.zeros((V, V))
    for t in range(prompt_len, len(full_seq)):
        prev = full_seq[t - 1]
        curr = full_seq[t]
        prob = policy.probs_at(prev)
        grad[prev] += -prob * coef            # 對所有 curr 的 softmax 分母項
        grad[prev, curr] += coef              # 分子項：實際出現的那個
    return grad


# ============================================================
# 4. Reward：這個任務用「exact match」當 reward（rule-based）
# ============================================================
def reward_of(full_seq: list, prompt_len: int, target: str) -> float:
    """
    把生成部分解碼出來（去掉 EOS），打分。

    ★ 教學重點：reward shaping（這是 RL 的核心技巧之一）★

    純 exact-match（對=1, 錯=0）會踩到「cold-start」：
    起手 policy 亂生成，G 段幾乎全錯 → 全部 reward=0 → std=0 → advantage=0 → 學不到。
    這就是 DeepSeek R1 / 任何 LLM GRPO 在難任務上的真正難點之一。

    解法（這份 demo 用的）：partial reward。
      - 第一個字就答對 → 給 0.5（鼓勵「至少開頭對」）
      - 整段正確 → 給 1.0
      - 完全錯 → 0.0
    這樣 group 內會出現差異，policy 就有訊號可學。

    真實 LLM 場景換成 learned RM 或 LLM-as-judge；介面一樣：reward(seq) -> float。
    """
    gen = full_seq[prompt_len:]
    if gen and gen[-1] == V2I[">"]:
        gen = gen[:-1]
    pred = decode(gen).strip()
    if pred == target:
        return 1.0
    if pred and pred[0] == target[0]:    # 第一個字對，給部分分
        return 0.5
    return 0.0


# ============================================================
# 5. LLM-style GRPO 更新（這是 DeepSeek R1 loss 的最小版）
# ============================================================
def grpo_update_llm(policy: CharPolicy, prompt_ids: list, target: str,
                    group_seqs: list, rewards: np.ndarray,
                    lr: float, eps_clip: float, inner_epochs: int,
                    rng: np.random.Generator):
    """
    ★★★ DeepSeek R1 loss 的最小實作 ★★★

    group_seqs: list of full sequences (prompt + generated)，G 段
    rewards:    每段的 reward，shape (G,)

    流程：
      1. group baseline = mean(rewards)；advantage = (r - mean)/std
      2. 記住每段的 logprob_old（採樣當下的 policy）
      3. multi-epoch PPO-style clip 更新：
           ratio = exp(logprob_new - logprob_old)
           loss  = -mean( min(ratio*A, clip(ratio)*A) )
      4. 對 theta 做 gradient ascent
    """
    prompt_len = len(prompt_ids)
    G = len(group_seqs)

    # group baseline（GRPO 的靈魂：取代 value network）
    baseline = rewards.mean()
    std = rewards.std()
    advantages = (rewards - baseline) / (std + ADV_EPS)

    # 採樣當下的 logprob（舊 policy 對這段序列的機率）
    logp_old = np.array([seq_logprob(policy, s, prompt_len) for s in group_seqs])

    for _ in range(inner_epochs):
        # 當前 policy 對每段序列的 logprob
        logp_new = np.array([seq_logprob(policy, s, prompt_len) for s in group_seqs])
        # ★ ratio = exp(logp_new - logp_old) ← DeepSeek R1 公式長這樣
        ratio = np.exp(logp_new - logp_old)

        # clipped surrogate
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
        take = np.minimum(surr1, surr2)

        # 梯度：只在 unclipped 那條才有梯度（PPO stop-gradient 概念）
        grad = np.zeros((V, V))
        for i, seq in enumerate(group_seqs):
            if surr1[i] <= surr2[i]:
                # grad of (ratio_i * A_i) w.r.t. theta
                # = A_i * ratio_i * ∂logp_new_i / ∂theta
                grad += advantages[i] * ratio[i] * \
                        seq_logprob_grad(policy, seq, prompt_len, coef=1.0)
        grad /= G

        policy.theta += lr * grad

    return baseline, std, advantages, logp_old


# ============================================================
# 6. 主訓練迴圈
# ============================================================
def main():
    rng = np.random.default_rng(SEED)
    policy = CharPolicy()

    print("=" * 74)
    print(" grpo_llm_char.py — 從 toy GRPO 走到真實 LLM GRPO")
    print(" policy 是字元級自回歸 LM（1-gram），candidate 是「整段序列」")
    print(" ratio = exp(logp_new - logp_old)  ← DeepSeek R1 loss 公式長這樣")
    print("=" * 74)
    print(f" vocab: {''.join(VOCAB)}  (V={V})")
    print(f" 任務：{len(TASKS)} 題，policy 要學會生成正確答案")
    print(f" group size G={GROUP_SIZE}，max_len={MAX_LEN}")

    verbose_iters = {0, 1, 5, 20, 60, N_ITERS - 1}

    for it in range(N_ITERS):
        verbose = it in verbose_iters
        if verbose:
            print("\n" + "#" * 74)
            print(f"# 第 {it} 輪")
            print("#" * 74)

        total_reward = 0.0
        total_n = 0

        for prompt, target in TASKS:
            prompt_ids = [V2I["<"]] + encode(prompt)   # 起頭加 BOS

            # 對同一個 prompt 抽 G 段候選字串（這就是 GRPO 的 group）
            group_seqs = [generate(policy, prompt_ids, MAX_LEN, rng)
                          for _ in range(GROUP_SIZE)]
            rewards = np.array([reward_of(s, len(prompt_ids), target)
                                for s in group_seqs])
            total_reward += rewards.sum()
            total_n += GROUP_SIZE

            if verbose:
                print(f"\n  prompt={prompt!r} target={target!r}")
                for i, s in enumerate(group_seqs):
                    gen = decode(s[len(prompt_ids):])
                    r = rewards[i]
                    lp = seq_logprob(policy, s, len(prompt_ids))
                    print(f"    [{i}] {prompt}{gen:6s}  "
                          f"reward={r}  logp_old={lp:+.3f}")

            baseline, std, adv, logp_old = grpo_update_llm(
                policy, prompt_ids, target, group_seqs, rewards,
                lr=LR, eps_clip=EPS_CLIP, inner_epochs=INNER_EPOCHS, rng=rng)

            if verbose:
                print(f"    group baseline={baseline:.3f} std={std:.3f}")
                print(f"    advantages={[round(float(x),3) for x in adv]}")
                # 看更新後 logp 怎麼變
                for i, s in enumerate(group_seqs):
                    lp_new = seq_logprob(policy, s, len(prompt_ids))
                    ratio = float(np.exp(lp_new - logp_old[i]))
                    mark = "  ← 被強化" if adv[i] > 0 and r_i(rewards, i) > 0 else ""
                    print(f"    [{i}] logp {logp_old[i]:+.3f} -> {lp_new:+.3f}  "
                          f"ratio={ratio:.3f}{mark}")

        if it % 10 == 0 or it == N_ITERS - 1:
            print(f"[iter {it:3d}] 平均 reward = {total_reward/total_n:.3f}   "
                  f"(1.0 = 每段都答對)")

    # ========================================================
    # 收尾：看 policy 學到什麼 + 對應 DeepSeek R1 公式
    # ========================================================
    print("\n" + "=" * 74)
    print(" 訓練結束 — policy 學到的轉移機率")
    print("=" * 74)
    eos = V2I[">"]
    correct_count = 0
    for prompt, target in TASKS:
        prompt_ids = [V2I["<"]] + encode(prompt)
        # greedy 生成看結果
        seq = list(prompt_ids)
        for _ in range(MAX_LEN):
            prev = seq[-1]
            prob = policy.probs_at(prev)
            nxt = int(prob.argmax())
            seq.append(nxt)
            if nxt == eos:
                break
        pred = decode(seq[len(prompt_ids):])
        ok = pred.replace(">", "").strip() == target
        correct_count += ok
        print(f"  {prompt:6s} target={target} | greedy 生成: {pred!r:8s} "
              f"{'✓' if ok else '✗'}")

    print(f"\n  {correct_count}/{len(TASKS)} 題 greedy 答對。")
    print(f"\n  ★ 教學觀察：你可能注意到 policy 傾向「全部都輸出同一個答案」")
    print(f"    （例如全部輸出 '5>'）。這是兩個真實因素造成的：")
    print(f"    1. policy 是 1-gram（只看上一個 char），所以 '1+1=' 跟 '3+1='")
    print(f"       在生成起點看到的 context 都是 '='，根本無法區分 → 結果一樣。")
    print(f"       真實 LLM 用整個上文（transformer attention）就沒這問題。")
    print(f"    2. partial reward + group-relative 會把 policy 推向「在這組 prompt")
    print(f"       裡最常答對的那個答案」（這組剛好 5 出現兩次）。")
    print(f"       這就是 LLM GRPO 的 mode-collapse 風險，DeepSeek 用 KL penalty 壓制。")
    print(f"\n  → 這份 demo 的重點不是「解出所有題」，是讓你看到：")
    print(f"     - candidate 是序列（不是離散選項）")
    print(f"     - logprob 是逐 token 加總")
    print(f"     - ratio = exp(logp_new - logp_old)，這就是 DeepSeek R1 loss 的 r_t(θ)")

    print("\n" + "-" * 74)
    print(" 對應 DeepSeek R1 的 GRPO loss 公式：")
    print("-" * 74)
    print("""
  J_GRPO(θ) = E[ min( r_t(θ)·Â_t,  clip(r_t(θ), 1-ε, 1+ε)·Â_t ) ] - β·KL(π_θ‖π_ref)

  其中：
    r_t(θ) = exp( logπ_θ(y_t|x,y_<t) − logπ_θold(y_t|x,y_<t) )
              ↑ 這份 demo 的 ratio = exp(logp_new - logp_old)
              ↑ logp 就是 seq_logprob()，逐 token 加總

    Â_t = group-relative advantage
              ↑ 這份 demo 的 (reward - mean(rewards)) / std
              ↑ 「group」是同一個 prompt 抽出來的 G 段候選序列

    KL(π_θ‖π_ref)  ← 這份 demo 沒做（toy 簡化）
    β              ← 沒做

  你看到的差別：
    - toy demo (grpo_toy.py)：單步 action，ratio = π[a]/π_old[a]
    - 這份 demo：序列 action，ratio = exp(Σ logπ - Σ logπ_old)
    - 真實 LLM：完全同這份 demo，只是 policy 從 1-gram table 換成 transformer
""")


def r_i(rewards, i):
    """helper for verbose print"""
    return rewards[i]


if __name__ == "__main__":
    main()
