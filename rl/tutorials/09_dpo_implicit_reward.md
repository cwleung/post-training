# Chapter 9: 直接偏好優化·DPO 隱式獎勵與對齊 (Direct Preference Optimization)

> *「傳統 RLHF 將對齊割裂為『先訓練獎勵模型、再用 PPO 在線微調』的繁複兩階段，中間伴隨著四個模型的顯存暴擊與在線採樣的極不穩定——DPO 以精妙的數學代換證明：語言模型本身就是自己最好的隱式獎勵模型，無需任何強化學習即可實現端到端偏好對齊。」*

---

## 核心心智模型：從顯式獎勵建模到隱式代數閉式解

在 Stanford 大學 Rafailov 等人於 2023 年發表 DPO 之前，大模型偏好對齊必須經歷兩座大山：
1. **獎勵建模（Reward Modeling）**：先訓練一個評分網絡 $r_\psi(x, y)$。
2. **在線 PPO 訓練**：加載 Actor、Critic、Reference Model 和 Reward Model 四個模型，進行高方差的在線強化學習。

```mermaid
flowchart TD
    subgraph TraditionalRLHF["傳統 3 階段 RLHF 流水線 (高門檻、超顯存)"]
        PAIR1["人類偏好數據集<br/>(x, y_w ≻ y_l)"] --> RM["訓練顯式獎勵模型 r_ψ(x, y)<br/>(二元分類損失)"]
        RM --> PPO["在線 PPO 強化學習<br/>維持 4 個模型實例 (Actor, Critic, Ref, RM)"]
        PPO --> FINAL1["對齊後的策略模型 π_PPO"]
    end

    subgraph DPOMethod["DPO 直接偏好優化 (NeurIPS 2023, 革命性突破)"]
        PAIR2["人類偏好數據集<br/>(x, y_w ≻ y_l)"] --> MATH["數學解析閉式代換<br/>r(x, y) = β ln(π_θ / π_ref) + β ln Z(x)"]
        MATH --> DIRECT["直接在靜態數據集上計算 DPO 損失<br/>(僅需 Actor 與凍結的 Reference Model)"]
        DIRECT --> FINAL2["對齊後的策略模型 π_DPO"]
    end

    classDef rlhf fill:#2d3748,stroke:#4a5568,color:#e2e8f0;
    classDef dpo fill:#1a365d,stroke:#3182ce,stroke-width:1.5px,color:#fff;
    class TraditionalRLHF,PAIR1,RM,PPO,FINAL1 rlhf;
    class DPOMethod,PAIR2,MATH,DIRECT,FINAL2 dpo;
```

---

## 9.1 布拉德利-特里 (Bradley-Terry) 偏好模型與解析代換

給定 Prompt $x$，人類評估者在勝者回答 $y_w$（Winner）與敗者回答 $y_l$（Loser）之間進行二元對比。
**Bradley-Terry 偏好模型**形式化為：
$$P(y_w \succ y_l \mid x) = \sigma\left( r(x, y_w) - r(x, y_l) \right) = \frac{1}{1 + \exp\left( -(r(x, y_w) - r(x, y_l)) \right)}$$

### RLHF 的逆強化學習最優解
在標準 RLHF 中，策略優化的目標為最大化獎勵同時約束 KL 散度：
$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi} \left[ r(x, y) \right] - \beta \mathbb{D}_{\text{KL}}\left( \pi(y \mid x) \parallel \pi_{\text{ref}}(y \mid x) \right)$$

根據凸優化拉格朗日對偶性，此受約束問題的**精確解析最優解（Analytical Closed-Form Solution）**為：
$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$
其中配分函數 $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$ 在高維自回歸空間中是**極其難以計算的（Intractable）**。

### 神來之筆：逆轉代換消去配分函數 $Z(x)$
將上式兩邊取對數，反解出獎勵函數 $r(x, y)$：
$$r(x, y) = \beta \ln \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \ln Z(x)$$

現在，將該隱式獎勵表達式代回 Bradley-Terry 偏好概率差項：
$$r(x, y_w) - r(x, y_l) = \beta \ln \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \beta \ln Z(x) - \left( \beta \ln \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + \beta \ln Z(x) \right)$$
**奇蹟發生了**：與具體回答無關的不可求和項 $\beta \ln Z(x)$ 被**完美精確抵消**！
$$r(x, y_w) - r(x, y_l) = \beta \left( \ln \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \ln \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right)$$

---

## 9.2 DPO 目標函數與梯度推導

將上述隱式表達式直接代入二元交叉熵對數似然損失，得到震撼工業界的 **DPO 損失函數**：
$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \ln \sigma \left( \beta \ln \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \ln \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

### 梯度反向傳播的動態推挽機制
對參數 $\theta$ 求導：
$$\nabla_\theta \mathcal{L}_{\text{DPO}}(\theta) = - \beta \mathbb{E} \left[ \sigma\left( \hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w) \right) \cdot \left( \nabla_\theta \ln \pi_\theta(y_w \mid x) - \nabla_\theta \ln \pi_\theta(y_l \mid x) \right) \right]$$

這個梯度呈現出優雅的雙重力學特質：
1. **推挽方向（Push-Pull）**：$\nabla_\theta \ln \pi_\theta(y_w | x)$ 向上推高勝者回答的生成概率；$\nabla_\theta \ln \pi_\theta(y_l | x)$ 向下壓低敗者回答的生成概率。
2. **自適應難度加權（Dynamic Weighting）**：梯度前置權重為 $\sigma(\hat{r}_l - \hat{r}_w)$。
   - 若模型當前已經精準學會 $y_w \gg y_l$（即隱式獎勵差很大），則 $\sigma(\dots) \to 0$，梯度自動歸零，**避免過度擬合簡單樣本**。
   - 若模型當前依然頑固地認為敗者比勝者更好（$\hat{r}_l > \hat{r}_w$），權重趨近於 1.0，釋放**最大梯度信號進行暴力修正**！

> [!TIP]
> <button class="deep-link-btn" data-target="viz">🔬 啟動 DPO 隱式獎勵實驗室</button>，可以手動拖動溫度 $\beta$ 與偏好概率差值，動態觀察推挽梯度的非線性飽和曲面。

---

## 9.3 可執行的 PyTorch DPO 損失計算模組

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0
) -> tuple[torch.Tensor, dict]:
    """
    向量化計算標準 DPO 損失
    Args:
        policy_chosen_logps:     [B] 當前模型對勝者序列的總 log_prob
        policy_rejected_logps:   [B] 當前模型對敗者序列的總 log_prob
        reference_chosen_logps:  [B] 參考模型對勝者序列的總 log_prob (detach)
        reference_rejected_logps:[B] 參考模型對敗者序列的總 log_prob (detach)
        beta:                    隱式 KL 懲罰係數 (通常 0.05 ~ 0.2)
    """
    # 1. 計算隱式獎勵對數比率
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    # 2. 核心 logits: β * ( (log π_w - log ref_w) - (log π_l - log ref_l) )
    logits = beta * (pi_logratios - ref_logratios)

    # 3. 交叉熵損失: -log(sigmoid(logits))
    # 採用 softplus 避免數值溢出: -log(sigmoid(x)) = softplus(-x)
    if label_smoothing == 0.0:
        losses = F.softplus(-logits)
    else:
        # 支持平滑標籤防範噪聲數據
        losses = (
            -F.logsigmoid(logits) * (1 - label_smoothing)
            - F.logsigmoid(-logits) * label_smoothing
        )

    loss = losses.mean()

    # 4. 監控指標：隱式勝敗者獎勵與 Acc
    with torch.no_grad():
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
        reward_accuracies = (chosen_rewards > rejected_rewards).float().mean()
        reward_margin = (chosen_rewards - rejected_rewards).mean()

    metrics = {
        "dpo_loss": loss.item(),
        "reward_accuracy": reward_accuracies.item(),
        "reward_margin": reward_margin.item(),
        "chosen_reward_mean": chosen_rewards.mean().item(),
        "rejected_reward_mean": rejected_rewards.mean().item(),
    }
    return loss, metrics
```

---

## 9.4 DPO 與現代偏好對齊演算法全景對比

| 演算法 | 依賴顯式獎勵模型 | 依賴在線 Rollout | 依賴 Reference Model | 核心解決的痛點 |
|---|---|---|---|---|
| **經典 PPO (2017)** | 是 (需預訓練 RM) | 是 (在線環境交互) | 是 (計算在線 KL) | 探索能力最強，但超參數敏感、訓練成本極高 |
| **DPO (2023)** | **否 (隱式代換)** | **否 (純離線配對數據)** | **是 (計算 log_ratio)** | 徹底砍掉 RM 與 Critic，訓練穩定如監督微調 |
| **IPO (2023)** | 否 | 否 | 是 | 解決 DPO 在過擬合時導致的隱式獎勵無限發散 |
| **KTO (2024)** | 否 | 否 | 是 | 無需配對數據 $(y_w, y_l)$，支持單條二元反饋 $(\pm 1)$ |
| **SimPO (2024)** | 否 | 否 | **否 (徹底剔除 Ref 模型)** | 消除長度偏差，顯存再省 50% |

---

## 🤔 架構深度思辨與工業界陷阱 (Architectural Insight & Production Pitfalls)

### 問題 1：在 DPO 訓練中，經常觀測到模型的「長度欺騙偏差（Length Bias）」甚至比 PPO 還嚴重——模型傾向於只要輸出足夠長的話術，就能拿到高勝率。其根本數學成因是什麼？
- **解答**：
  1. **序列對數似然的非歸一化累加**：在標準 DPO 實現中，序列對數概率是每個 Token 的簡單求和：$\ln \pi_\theta(y | x) = \sum_{t=1}^{|y|} \ln \pi(y_t | x, y_{<t})$。
  2. **累加偏差擴大效應**：如果勝者回答 $y_w$ 天然比敗者回答 $y_l$ 長得多（例如人類標註時偏好更詳盡的回答），每多一個 Token 就多一次對數累加。即使平均 Token 質量並未提升，長回答的累加數值差值也會顯著放大概率優勢。
  3. **工業界修復方案**：
     - **長度歸一化（Length Normalization）**：在損失中將總 log-prob 除以序列長度 $|y|$（如 SimPO 的核心改進：$\frac{1}{|y|} \log \pi$）。
     - **數據集長度平衡**：在數據管線中嚴格限制 $|y_w|$ 與 $|y_l|$ 的字數長度差不超過 10%。

### 問題 2：DPO 的「似然位移漏洞（Likelihood Displacement Bug）」是指什麼？在什麼情況下 DPO 的效果會顯著劣於在線 PPO？
- **解答**：
  1. **似然位移漏洞**：DPO 的推挽梯度只要求 $\ln \pi(y_w) - \ln \pi(y_l)$ 變大。在極端情況下，模型可能發現「將敗者回答 $y_l$ 的概率壓至極低（$-1000$），同時略微降低勝者回答 $y_w$ 的概率（$-100$）」，此時差值依然擴大（$+900$），DPO 損失完美下降！但這導致**所有高質量回答的生成概率都在同步萎縮，模型最終喪失正常的語言表達能力**。
  2. **劣於在線 PPO 的本質原因**：DPO 是純粹的**離線偏好優化（Off-Policy）**。它無法探索歷史數據集之外的新軌跡。而在線 PPO 隨時用自己最新的策略生成全新的推理思維鏈（Rollout），並在自發探索中修正邏輯。在數學推理、多步代碼生成等探索上限決定能力的硬核領域，**在線 RL（PPO / GRPO）的最終性能上限顯著碾壓純離線 DPO**！

---

## 參考文獻與經典論文

1. **Rafailov, R., et al. (2023).** *Direct preference optimization: Your language model is secretly a reward model.* Advances in Neural Information Processing Systems (NeurIPS 36).
2. **Azar, M. G., et al. (2024).** *A general theoretical paradigm to understand learning from human preferences.* International Conference on Machine Learning (ICML).
3. **Meng, Y., et al. (2024).** *SimPO: Simple preference optimization with a reference-free reward.* arXiv preprint arXiv:2405.14734.
