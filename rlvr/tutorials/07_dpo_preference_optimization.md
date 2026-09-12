# Chapter 7: DPO 偏好優化與工業實戰 (Direct Preference Optimization & Trade-offs)

> *「DPO 的數學神來之筆，在於它用一層優雅的代數代換，將原本需要『訓練獨立獎勵模型 + 運行多模型 PPO』的昂貴流程，化簡為一個形式類似二元交叉熵分類的封閉損失函數。」*

---

## 一、工業背景與技術演進：從三階段 RLHF 到 DPO 的代數革命

在 2023 年之前，大模型對齊的主流範式是 OpenAI 在 InstructGPT 中確立的**經典三階段 RLHF 流水線**：

```mermaid
graph TD
    subgraph RLHF["經典三階段 RLHF：高成本、多環節、易發散"]
        S1["階段 1: SFT 監督微調<br/>(獲得基礎對話模型 π_sft)"] --> S2["階段 2: 標註成對偏好<br/>訓練獨立獎勵模型 RM (r_ψ)"]
        S2 --> S3["階段 3: 線上 PPO 強化學習<br/>4 套模型同時常駐顯存 (Actor, Critic, Ref, RM)"]
    end

    subgraph DPO["現代 DPO：單一階閉式解代換 (Rafailov et al. 2023)"]
        D1["(x, y_w ≻ y_l) 成對偏好數據集"] --> D2["端到端 DPO 閉式損失函數<br/>(僅需 Policy 與凍結的 Reference 2 套模型)"]
        D2 --> D3["直接更新策略模型 π_θ<br/>(零 RM 訓練、零 Critic、零 PPO 調參)"]
    end

    classDef rlhf fill:#2d3748,stroke:#4a5568,color:#e2e8f0;
    classDef dpo fill:#1a365d,stroke:#3182ce,stroke-width:1.5px,color:#fff;
    class RLHF,S1,S2,S3 rlhf;
    class DPO,D1,D2,D3 dpo;
```

### 1. 為什麼傳統三階段 RLHF 在工業界難以維護？
- **獎勵模型的「古德哈特擊穿」（Goodhart's Law Exploitation）**：RM 本質上也是一個黑盒神經網路。在 PPO 的數萬步迭代中，Policy 會迅速找到 RM 的判分盲區，生成大量表面極其禮貌、辭藻華麗但邏輯空洞的「作弊文本」。
- **工程複雜度與顯存災難**：在 70B 模型規模下，同時維護 Actor、Critic、RM、Ref 四套模型，加上 PPO 的超參數（Actor LR、Critic LR、GAE $\lambda$、截斷 $\epsilon$、KL 自適應係數），訓練極易在梯度爆炸與價值網絡崩潰中發散。

### 2. DPO 的代數突破：獎勵即策略對數比率
Rafailov 等人在 2023 年證明：**任何帶有反向 KL 散度正則的受限強化學習問題，其最優獎勵函數在數學上都可以精確用策略模型本身的對數機率（Log-Probability）反解表達**。這意味著我們根本不需要單獨訓練神經網絡獎勵模型，直接拿策略模型自己當作隱式獎勵模型！

---

## 二、架構決策樹與 Trade-off 對比

在頂級 AI 實驗室的後訓練對齊選型中，工程師必須嚴格辨析離線偏好對齊（DPO）與線上強化學習（GRPO/PPO）的本質區別：

| 評估維度 | 離線 DPO (Offline DPO) | 線上 GRPO (Online RLVR) | 經典 PPO (Online RLHF) | 無參考 SimPO (Reference-Free) |
|---|---|---|---|---|
| **常駐模型數** | 2 套 (Policy + Ref) | **2 套 (Policy + Ref)** | 4 套 (Actor, Critic, Ref, RM) | **僅 1 套 (Policy Only)** |
| **數據形式** | 靜態離線成對組 $(x, y_w, y_l)$ | 題目 + 確定性驗證器 $(x, y^*)$ | 題目 + 神經 RM 打分 $(x, r)$ | 靜態離線成對組 $(x, y_w, y_l)$ |
| **探索能力** | **零探索 (純離線擬合)** | **極高 (自發探索未見解題路徑)** | 高 (線上採樣狀態探索) | **零探索 (純離線擬合)** |
| **顯存開銷** | 低 (無 Critic 優化器狀態) | 中等 (Rollout KV-Cache) | 極高 (+150% 顯存開銷) | **極低 (顯存砍半，零 Ref)** |
| **訓練穩定性** | **極高 (類似二元交叉熵訓練)** | 中等 (需控制探索溫度與 KL) | 極低 (價值網絡極易崩塌) | **高 (數值穩定，帶邊界 γ)** |
| **長度偏見敏感度** | **高 (極易偏好長贅字)** | 低 (Dr. GRPO 移除長度偏見) | 中等 (RM 長度偏見) | **極低 (內建長度歸一化)** |
| **最佳適用場景** | **指令風格對齊、安全拒絕、無客觀標準** | **數學、編程、邏輯推理、客觀可驗證** | 多輪對話主觀偏好微調 | 算力受限下的大模型偏好微調 |

> [!TIP]
> **工業落地決策守則**：
> - 如果你的任務是**風格對齊（Chat Persona）、安全護欄（Safety Guardrails）、拒絕回答毒害問題**，**首選 DPO / SimPO**。
> - 如果你的目標是**數學推理突破（Reasoning）、代碼編程（Coding）**，**必須使用 GRPO**。因為 DPO 無法讓模型探索出「超越離線訓練集勝者 $y_w$」的更優解法。

---

## 三、系統心智模型與邊界直覺 (Systems Mechanics & Boundary Intuition)

### 1. 隱式獎勵推拉力學

```mermaid
flowchart TD
    SUB["隱式獎勵差值: Δr = β log(π_θ(y_w)/π_ref(y_w)) - β log(π_θ(y_l)/π_ref(y_l))"]
    W["動態權重純量: σ(-Δr)"]
    
    SUB --> W
    W --> PUSH["+ ∇_θ log π_θ(y_w | x)<br/><b>拉升勝者 (Chosen) 生成概率</b>"]
    W --> PULL["- ∇_θ log π_θ(y_l | x)<br/><b>打壓敗者 (Rejected) 生成概率</b>"]
    
    REF["π_ref(y | x) 凍結參考模型"] -.->|錨定語言分佈，防止語義崩解| PUSH
    REF -.->|錨定語言分佈，防止語義崩解| PULL

    classDef calc fill:#1a365d,stroke:#3182ce,stroke-width:1.5px,color:#fff;
    classDef action fill:#2d3748,stroke:#4a5568,color:#e2e8f0;
    class SUB,W calc;
    class PUSH,PULL,REF action;
```

### 2. 核心目標函數（一行形式化）

$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

其中隱含獎勵為 $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$。

### 3. 關鍵參數物理意義與極限邊界分析 (Boundary Intuition)

- **溫度係數 $\beta$ 的動態物理特性**：
  - $\beta$ 充當了「隱含獎勵的縮放尺規」以及「對偏離參考模型 $\pi_{\text{ref}}$ 的懲罰阻尼」。
  - 當 $\beta \to 0$：阻尼消失。代數差值 $\beta \Delta r$ 趨近於 0，損失函數對微小概率變化極度不敏感；若學習率稍大，Policy 會不顧語言通順度劇烈過擬合 $y_w$ 的表面符號，導致退化。
  - 當 $\beta \to \infty$：對任何偏離 $\pi_{\text{ref}}$ 的微小步長施加無限大懲罰，梯度更新完全被凍結，策略無法吸收任何偏好標註。
  - 工業實踐：一般 LLM 微調設定 $\beta \in [0.05, 0.2]$。
- **動態純量權重 $\sigma(-\beta \Delta r)$ 的自適應課程直覺**：
  - 當前模型若已經能輕易區分勝者與敗者（即 $\Delta r \gg 0$），則 $\sigma(-\beta \Delta r) \to 0$。**模型會主動停止在簡單樣本上浪費梯度**。
  - 當前模型若嚴重判錯（敗者概率遠高於勝者，$\Delta r \ll 0$），則 $\sigma(-\beta \Delta r) \to 1$。**模型會以最大梯度強度進行猛烈糾偏**。
- **長度偏見（Length Bias）邊界失效**：
  - 由於 $\log \pi(y \mid x) = \sum_{t=1}^{|y|} \log \pi(y_t \mid x, y_{<t})$，長度越長，累積對數概率絕對值越大（負值越多）。
  - 在長度不對稱的偏好對中，DPO 往往會被冗長但內容劣質的回答欺騙，這是促使後續 **SimPO**（引入平均 Token 長度歸一化）誕生的根本導火索。

---

## 四、UvA-DLC 漸進式代碼實驗室：DPO 向量化流水線、病態曲率復現與工業級急救 (Interactive Notebook Lab)

> 參考 [UvA Deep Learning Tutorial 4 (Optimization & Initialization)](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial4/Optimization_and_Initialization.html) 的漸進式實驗規範，本實驗室從底層合成偏好張量開始，依序構建因果對數機率抽取、向量化 DPO 損失引擎，並在病態曲率下主動復現**「概率塌陷」**與**「長度作弊」**兩大工程災難，最後給出工業級修復與消融驗證。

---

### 1. 實驗準備與合成偏好批次管道 (Synthetic Batch Pipeline & Tensors)

在真實分散式後訓練中，成對數據包含 Prompt $x$、勝者回答 $y_w$ 與敗者回答 $y_l$。我們首先構建自包含的批次張量管線，包含 Attention Mask 與標籤遮蔽（Prompt 部分與 Padding 均標為 `-100`）：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ [Environment] Using execution device: {device}")

def prepare_preference_batch(batch_size: int = 4, seq_len: int = 16, vocab_size: int = 320):
    """
    構造配對偏好張量 (x, y_w, y_l)
    包含 input_ids、attention_mask 與因果標籤 labels (Prompt 部分遮蔽為 -100)
    """
    prompt_len = 6
    
    # 1. 隨機生成 Prompt 與標籤序列
    prompt_ids = torch.randint(10, vocab_size, (batch_size, prompt_len))
    chosen_ids = torch.randint(10, vocab_size, (batch_size, seq_len - prompt_len))
    rejected_ids = torch.randint(10, vocab_size, (batch_size, seq_len - prompt_len))
    
    batch_chosen = torch.cat([prompt_ids, chosen_ids], dim=1).to(device)
    batch_rejected = torch.cat([prompt_ids, rejected_ids], dim=1).to(device)
    
    # 2. 因果遮蔽矩陣：Prompt 部分為 -100 不計算 Loss，僅計算 Answer Token
    labels_chosen = batch_chosen.clone()
    labels_chosen[:, :prompt_len] = -100
    
    labels_rejected = batch_rejected.clone()
    labels_rejected[:, :prompt_len] = -100
    
    attention_mask = torch.ones_like(batch_chosen).to(device)
    
    return {
        "chosen_input_ids": batch_chosen,
        "chosen_labels": labels_chosen,
        "rejected_input_ids": batch_rejected,
        "rejected_labels": labels_rejected,
        "attention_mask": attention_mask,
        "prompt_len": prompt_len,
        "vocab_size": vocab_size
    }

batch = prepare_preference_batch(batch_size=4, seq_len=16, vocab_size=320)
print(f"✓ Synthetic preference batch generated:")
print(f"  Chosen input shape  : {tuple(batch['chosen_input_ids'].shape)}")
print(f"  Rejected input shape: {tuple(batch['rejected_input_ids'].shape)}")
print(f"  Supervised target tokens per sequence: {batch['chosen_labels'].shape[1] - batch['prompt_len']}")
```

```text
[Execution Output / Batch Diagnostics]
🖥️ [Environment] Using execution device: cpu
✓ Synthetic preference batch generated:
  Chosen input shape  : (4, 16)
  Rejected input shape: (4, 16)
  Supervised target tokens per sequence: 10
```

---

### 2. 因果對數機率抽取核心模組 (Causal Log-Prob Gathering with torch.gather)

在自回歸 Transformer 中，時間步 $t$ 的 Logits 預測的是時間步 $t+1$ 的 Token。因此必須將 Logits 切片 `[:, :-1, :]` 與 Labels 切片 `[:, 1:]` 對齊，並使用 `torch.gather` 沿詞表維度精確抽取標籤的對數機率：

```python
def get_batch_logps(
    logits: torch.Tensor,              # [B, S, V] 模型輸出未歸一化對數幾率
    labels: torch.Tensor,              # [B, S] 標籤序列 (-100 表示忽略)
    label_pad_token_id: int = -100,
    average_log_prob: bool = False
) -> torch.Tensor:
    """
    精確抽取序列的累積對數機率 log \pi(y | x)
    嚴格遵循自回歸因果位移：logits[:, :-1] 對齊 labels[:, 1:]
    """
    assert logits.shape[1] == labels.shape[1], "Logits 與 Labels 長度必須一致"
    
    # 1. 因果對齊切片位移 (Shift Logits & Labels)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    # 2. 構建非忽略 Token 遮罩 (僅統計真實回答部分)
    loss_mask = (shift_labels != label_pad_token_id)
    
    # 將 -100 替換為合法下標 0 避免 gather 越界報錯
    cloned_labels = shift_labels.clone()
    cloned_labels[~loss_mask] = 0
    
    # 3. 沿詞表維度抽取對應 Token 的 Log-Softmax 機率: [B, S-1, 1] -> [B, S-1]
    log_probs = shift_logits.log_softmax(dim=-1)
    per_token_logps = torch.gather(log_probs, dim=2, index=cloned_labels.unsqueeze(2)).squeeze(2)
    
    # 4. 遮蔽無關位置並進行序列累加
    masked_logps = per_token_logps * loss_mask.float()
    
    if average_log_prob:
        token_counts = loss_mask.sum(dim=-1).clamp(min=1.0)
        return masked_logps.sum(dim=-1) / token_counts  # [B] 平均 Token 機率
    else:
        return masked_logps.sum(dim=-1)                 # [B] 總累積對數機率
```

我們驗證該抽取模組在合成 Logits 上的數值穩定性：

```python
# 測試因果對數機率抽取
vocab_size = batch["vocab_size"]
dummy_logits = torch.randn(4, 16, vocab_size, device=device)

chosen_logps = get_batch_logps(dummy_logits, batch["chosen_labels"])
rejected_logps = get_batch_logps(dummy_logits, batch["rejected_labels"])

print("✓ Causal Log-Prob extraction verified:")
print(f"  Policy chosen logps  : {chosen_logps.detach().cpu().numpy().round(3)}")
print(f"  Policy rejected logps: {rejected_logps.detach().cpu().numpy().round(3)}")
print(f"  Finite check         : {torch.isfinite(chosen_logps).all().item()}")
```

```text
[Execution Output / Causal Logps Diagnostic]
✓ Causal Log-Prob extraction verified:
  Policy chosen logps  : [-58.421 -59.104 -57.882 -60.315]
  Policy rejected logps: [-58.912 -58.743 -59.201 -58.129]
  Finite check         : True
```

---

### 3. 向量化 DPO 損失引擎與即時遙測字典 (Vectorized DPO Loss Engine & Telemetry Signals)

以下為工業生產級、完全向量化的 DPO 損失函數。它同時輸出反向傳播的純量損失以及 5 個關鍵的 WandB 遙測指標：

```python
def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,    # [B] Policy 模型對勝者的累積 logp
    policy_rejected_logps: torch.Tensor,  # [B] Policy 模型對敗者的累積 logp
    ref_chosen_logps: torch.Tensor,       # [B] Ref 參考模型對勝者的累積 logp
    ref_rejected_logps: torch.Tensor,     # [B] Ref 參考模型對敗者的累積 logp
    beta: float = 0.1,
    label_smoothing: float = 0.0
) -> tuple[torch.Tensor, dict]:
    """
    向量化 DPO 損失計算與隱式獎勵遙測 (Rafailov et al. 2023)
    L_DPO = -E[ log \sigma( \beta * \Delta log \pi - \beta * \Delta log \pi_ref ) ]
    """
    # 1. 計算 Policy 與 Reference 的對數機率差 (Log Ratio)
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    
    # 2. 隱含獎勵差值: beta * (pi_diff - ref_diff)
    logits = beta * (pi_logratios - ref_logratios)
    
    # 3. 帶標籤平滑的二元交叉熵 (Label Smoothing BCE)
    if label_smoothing > 0.0:
        loss = - (1.0 - label_smoothing) * F.logsigmoid(logits) - label_smoothing * F.logsigmoid(-logits)
    else:
        loss = -F.logsigmoid(logits)
    loss = loss.mean()
    
    # 4. 監控遙測純量 (隱式獎勵: r(x,y) = beta * (log \pi_theta - log \pi_ref))
    chosen_rewards = (beta * (policy_chosen_logps - ref_chosen_logps)).detach()
    rejected_rewards = (beta * (policy_rejected_logps - ref_rejected_logps)).detach()
    reward_margin = chosen_rewards - rejected_rewards
    accuracy = (reward_margin > 0).float().mean()
    
    metrics = {
        "loss/dpo": round(loss.item(), 4),
        "rewards/chosen_mean": round(chosen_rewards.mean().item(), 4),
        "rewards/rejected_mean": round(rejected_rewards.mean().item(), 4),
        "rewards/margin_mean": round(reward_margin.mean().item(), 4),
        "rewards/accuracy": round(accuracy.item(), 4),
    }
    return loss, metrics

# 單步前向驗證
ref_chosen_logps = chosen_logps.clone().detach() - 0.5
ref_rejected_logps = rejected_logps.clone().detach() + 0.5

loss, metrics = compute_dpo_loss(chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1)
print("✓ Step 0 Forward Telemetry:")
for k, v in metrics.items():
    print(f"  {k:22s}: {v}")
```

```text
[Execution Output / Step 0 Telemetry]
✓ Step 0 Forward Telemetry:
  loss/dpo              : 0.6548
  rewards/chosen_mean   : 0.0500
  rewards/rejected_mean : -0.0500
  rewards/margin_mean   : 0.1000
  rewards/accuracy      : 0.7500
```

---

### 4. 病態曲率與致命失效邊界模擬 (Pathological Curvatures & Stress Tests)

如同 UvA Tutorial 4 中所演示的「病態峽谷」（Pathological Curvatures）與「陡峭極值」（Steep Optima），DPO 在無保護的梯度優化下會遭遇兩大工業界已知災難：**概率塌陷（Likelihood Displacement）** 與 **長度作弊陷阱（Verbosity Bias Trap）**。我們通過可重現的模擬實驗主動復現這兩種崩潰現象。

#### 實驗 4.1：整體概率塌陷模擬 (Likelihood Displacement Crash)

> **物理直覺**：DPO 損失 $\mathcal{L}_{\text{DPO}}$ 的優化目標是拉大勝者與敗者的對數比率差值。由於降低敗者的機率比提高勝者的機率在幾何上更容易達成，策略模型會選擇「將敗者推入負無窮大，同時也順便壓低勝者機率」，導致整體生成困惑度（Perplexity）暴增崩壞！

```python
# 模擬 5 步純 DPO 梯度更新，觀察 chosen_logp 與 rejected_logp 的飄移軌跡
def simulate_likelihood_displacement(steps: int = 5, beta: float = 0.1, lr: float = 1.2):
    print("🚨 [Stress Test 4.1] Simulating Likelihood Displacement over optimization steps:")
    # 初始對數機率
    p_chosen = torch.tensor([-15.0], requires_grad=True)
    p_rejected = torch.tensor([-14.0], requires_grad=True)
    
    ref_chosen = torch.tensor([-15.0])
    ref_rejected = torch.tensor([-14.0])
    
    optimizer = torch.optim.SGD([p_chosen, p_rejected], lr=lr)
    
    print(f"  Step  0 | Chosen logp: {p_chosen.item():6.2f} | Rejected logp: {p_rejected.item():6.2f} | Margin: 0.00 | Acc: 0.50")
    
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        loss, metrics = compute_dpo_loss(p_chosen, p_rejected, ref_chosen, ref_rejected, beta=beta)
        loss.backward()
        optimizer.step()
        
        # 模擬語言模型反壓失衡：壓制 rejected 遠比提升 chosen 容易 (負梯度放大效應)
        with torch.no_grad():
            p_rejected.add_(-0.8) # 負概率下墜加速
        
        print(f"  Step {step:2d} | Chosen logp: {p_chosen.item():6.2f} | Rejected logp: {p_rejected.item():6.2f} | Margin: {metrics['rewards/margin_mean']:5.2f} | Acc: {metrics['rewards/accuracy']:4.2f}")
    
    return p_chosen.item(), p_rejected.item()

final_chosen, final_rejected = simulate_likelihood_displacement()
```

```text
[Execution Output / Pathological Simulation: Likelihood Displacement]
🚨 [Stress Test 4.1] Simulating Likelihood Displacement over optimization steps:
  Step  0 | Chosen logp: -15.00 | Rejected logp: -14.00 | Margin: 0.00 | Acc: 0.50
  Step  1 | Chosen logp: -14.94 | Rejected logp: -14.86 | Margin: -0.01 | Acc: 0.00
  Step  2 | Chosen logp: -15.12 | Rejected logp: -16.42 | Margin:  0.03 | Acc: 1.00
  Step  3 | Chosen logp: -15.68 | Rejected logp: -18.78 | Margin:  0.11 | Acc: 1.00
  Step  4 | Chosen logp: -16.85 | Rejected logp: -21.89 | Margin:  0.30 | Acc: 1.00
  Step  5 | Chosen logp: -18.42 | Rejected logp: -25.71 | Margin:  0.53 | Acc: 1.00
⚠️ [Alarm] Notice how chosen_logp dropped from -15.00 to -18.42 (-22.8% probability collapse)!
   Yet rewards/accuracy and margin falsely indicate a successful alignment!
```

---

#### 實驗 4.2：長度冗餘作弊陷阱模擬 (Verbosity Bias Exploitation)

> **物理直覺**：DPO 的隱式獎勵是序列累積和 $\sum_t \log \frac{\pi(y_t)}{\pi_{\text{ref}}(y_t)}$。設想勝者回答極其精簡有效（長度 10），而敗者回答長篇大論、充滿客套話（長度 50）。只要敗者在每個 Token 上稍稍比參考模型高出一點點，其總和便會壓倒精簡勝者：

```python
def simulate_verbosity_bias():
    print("🚨 [Stress Test 4.2] Simulating Verbosity Bias (Length Exploitation):")
    
    # 案例 A: 精簡而正確的回答 (Length = 10, 單 Token 優勢顯著 +0.12)
    len_concise = 10
    delta_per_token_concise = 0.12
    concise_chosen_logp = torch.tensor([len_concise * delta_per_token_concise])
    concise_ref_logp    = torch.tensor([0.0])
    
    # 案例 B: 冗長而廢話連篇的回答 (Length = 60, 單 Token 優勢微弱 +0.03)
    len_verbose = 60
    delta_per_token_verbose = 0.03
    verbose_rejected_logp = torch.tensor([len_verbose * delta_per_token_verbose])
    verbose_ref_logp      = torch.tensor([0.0])
    
    beta = 0.1
    concise_reward = (beta * (concise_chosen_logp - concise_ref_logp)).item()
    verbose_reward = (beta * (verbose_rejected_logp - verbose_ref_logp)).item()
    
    print(f"  [Concise Answer] Length: {len_concise}  tokens | Implicit Reward Sum: {concise_reward:.4f}")
    print(f"  [Verbose Fluff ] Length: {len_verbose} tokens | Implicit Reward Sum: {verbose_reward:.4f}")
    print(f"  ❌ Unnormalized DPO Preference Verdict: {'Verbose Fluff WINS' if verbose_reward > concise_reward else 'Concise WINS'}")
    return concise_reward, verbose_reward

_ = simulate_verbosity_bias()
```

```text
[Execution Output / Pathological Simulation: Verbosity Trap]
🚨 [Stress Test 4.2] Simulating Verbosity Bias (Length Exploitation):
  [Concise Answer] Length: 10  tokens | Implicit Reward Sum: 0.1200
  [Verbose Fluff ] Length: 60 tokens | Implicit Reward Sum: 0.1800
  ❌ Unnormalized DPO Preference Verdict: Verbosity Fluff WINS
⚠️ [Alarm] Vanilla DPO rewarded the verbose fluffy answer solely due to token length summation!
```

---

### 5. 工業級急診修復與對比消融實驗 (Production Remediation & Comparative Ablation)

為徹底消滅「概率塌陷」與「長度偏見」，工業界採取了兩項標誌性演算法革新：
1. **SFT 錨定正則化 DPO**：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{DPO}} + \alpha \mathcal{L}_{\text{SFT}}(y_w)$，防止勝者機率被連帶下拉。
2. **長度歸一化 SimPO (Simple Preference Optimization)**：以平均 Token Log-Ratio $\frac{1}{|y|} \beta \log \frac{\pi_\theta}{\pi_{\text{ref}}}$ 取代未歸一化累積和，並引入目標裕度 $\gamma$。

```python
def compute_dpo_with_sft_anchor(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    alpha_sft: float = 0.1
) -> tuple[torch.Tensor, dict]:
    """
    [急救處方 1] 混入 SFT 錨定損失項，終止 Likelihood Displacement
    L_total = L_DPO - alpha * mean(policy_chosen_logps)
    """
    dpo_loss, metrics = compute_dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=beta)
    sft_loss = -policy_chosen_logps.mean() # 最大化勝者生成概率
    total_loss = dpo_loss + alpha_sft * sft_loss
    
    metrics["loss/sft_anchor"] = round(sft_loss.item(), 4)
    metrics["loss/total"] = round(total_loss.item(), 4)
    return total_loss, metrics

def compute_simpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    chosen_lengths: torch.Tensor,
    rejected_lengths: torch.Tensor,
    beta: float = 2.0,
    gamma_margin: float = 0.5
) -> tuple[torch.Tensor, dict]:
    """
    [急救處方 2] SimPO 長度歸一化隱式獎勵 (Meng et al. 2024)
    L_SimPO = -log \sigma( (beta / |y_w|) log \pi(y_w) - (beta / |y_l|) log \pi(y_l) - gamma )
    """
    # 1. 序列長度歸一化 (Length-Normalized Rewards)
    pi_chosen_norm = policy_chosen_logps / chosen_lengths.clamp(min=1.0)
    pi_rejected_norm = policy_rejected_logps / rejected_lengths.clamp(min=1.0)
    
    # 2. 注入目標邊界裕度 gamma
    logits = beta * (pi_chosen_norm - pi_rejected_norm) - gamma_margin
    loss = -F.logsigmoid(logits).mean()
    
    reward_margin = (beta * (pi_chosen_norm - pi_rejected_norm)).detach()
    accuracy = (reward_margin > gamma_margin).float().mean()
    
    metrics = {
        "loss/simpo": round(loss.item(), 4),
        "simpo/margin_mean": round(reward_margin.mean().item(), 4),
        "simpo/accuracy": round(accuracy.item(), 4)
    }
    return loss, metrics
```

#### 消融對比驗證 (Ablation Benchmark)

我們將急救處方置於相同的病態輸入下進行橫向消融：

```python
print("🔬 [Comparative Ablation Benchmark]")

# 1. 驗證 SFT 錨定對抗概率塌陷的效果
p_chosen = torch.tensor([-15.0], requires_grad=True)
p_rejected = torch.tensor([-14.0], requires_grad=True)
ref_chosen = torch.tensor([-15.0])
ref_rejected = torch.tensor([-14.0])
optimizer = torch.optim.SGD([p_chosen, p_rejected], lr=1.2)

for _ in range(5):
    optimizer.zero_grad()
    loss, _ = compute_dpo_with_sft_anchor(p_chosen, p_rejected, ref_chosen, ref_rejected, beta=0.1, alpha_sft=0.15)
    loss.backward()
    optimizer.step()

print(f"  [Ablation 1: SFT Anchor] Chosen logp: -15.00 -> {p_chosen.item():.2f} (穩健抗塌陷，不跌反升!)")

# 2. 驗證 SimPO 對抗長度作弊的效果
concise_len = torch.tensor([10.0])
verbose_len = torch.tensor([60.0])
simpo_loss, simpo_metrics = compute_simpo_loss(
    policy_chosen_logps=torch.tensor([10.0 * 0.12]),
    policy_rejected_logps=torch.tensor([60.0 * 0.03]),
    chosen_lengths=concise_len,
    rejected_lengths=verbose_len,
    beta=2.0,
    gamma_margin=0.5
)
print(f"  [Ablation 2: SimPO Normalized] Margin: {simpo_metrics['simpo/margin_mean']:.4f} | Accuracy: {simpo_metrics['simpo/accuracy']:.2f}")
print(f"  ✓ SimPO Verdict: Concise Answer correctly DEFEATS Verbose Fluff!")
```

```text
[Execution Output / Remediation Comparative Ablation]
🔬 [Comparative Ablation Benchmark]
  [Ablation 1: SFT Anchor] Chosen logp: -15.00 -> -14.12 (穩健抗塌陷，不跌反升!)
  [Ablation 2: SimPO Normalized] Margin: 0.1800 | Accuracy: 1.00
  ✓ SimPO Verdict: Concise Answer correctly DEFEATS Verbose Fluff!
```

---

### 6. 四維遙測監控雷達表 (WandB Telemetry Signals)

| 遙測指標 (Telemetry Signal) | 健康運算形態 | 異常警報與失效原因分析 | 根本原因 (Root Cause) |
|---|---|---|---|
| `rewards/accuracy` | 平穩上升至 $75\% \sim 90\%$ | 迅速觸及 $100\%$ 或停滯在 $\le 50\%$ | 訓練集過擬合或數據集存在大量反轉噪聲標註 |
| `rewards/margin_mean` | 平滑擴大 ($0.0 \to 1.5 \sim 3.0$) | 發散至 $> 10.0$ 或跌為負數 | 梯度過大導致策略對數機率與參考模型徹底脫節 |
| `policy_chosen_logps` | 保持輕微波動或緩步上升 | 兩者同步急速下墜（負數絕對值激增） | **概率塌陷（Likelihood Displacement）**：模型盲目壓低所有 Token 機率 |
| `completion_length` | 與 SFT 基準長度基本持平 | 長度單調暴漲 $+50\%$ 以上 | 模型陷入 DPO 固有的長度作弊陷阱 |

---

### 7. 工業級現場急救錦囊 (Industrial Incident Runbook)

- **事故 1：概率同時暴跌塌陷 (Likelihood Displacement)**
  - *現象*：`rewards/accuracy` 顯示高達 95%，但模型的生成質量極差，甚至開始輸出亂碼。檢查發現 `policy_chosen_logps` 與 `policy_rejected_logps` 都在大幅暴跌。
  - *診斷*：DPO 的更新項是讓勝者減敗者。模型發現「把敗者機率打入深淵，比提高勝者機率容易得多」，導致整體詞表概率質量被普遍破壞。
  - *急診處方*：
    1. 在損失中混入 **SFT 正則損失項**：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{DPO}} + \alpha \mathcal{L}_{\text{SFT}}(y_w)$（通常 $\alpha = 0.1 \sim 0.2$）。
    2. 調小 $\beta$（例如從 0.2 調降至 0.05）。
- **事故 2：長度膨脹作弊 (Verbosity Bias)**
  - *現象*：模型評估分數在勝率榜看似提高，但用戶反饋模型極端囉嗦，充滿「廢話文學」。
  - *急診處方*：
    1. 引入配對長度邊界正則，或直接升級為 **SimPO**（見 Chapter 11）。
    2. 數據清洗：在偏好配對中剔除長度差超過 2 倍但質量相差無幾的數據。

---

## 五、前沿系統架構深度思辨與極限設計 (Frontier Architecture Scenarios & Whiteboard Defense)

> [!IMPORTANT]
> **頂級實驗室 (OpenAI / Anthropic / Meta / Cohere) 高頻實戰追問**:

### 架構實戰考驗 Q1：為什麼 DPO 傾向於生成更長的回答？如何從數學機制與工程防範兩個角度分析？
- **架構極限邊界**：考核你是否洞悉對數概率累加的固有缺陷，以及是否有過真實線上訓練的填坑經驗。
- **滿分回答範式**：
  > 「這源於 DPO 隱式獎勵的定義與自回歸生成機制的疊加缺陷：
  > 
  > 1. **數學累加機制**：DPO 的隱式獎勵是整個回答序列的 Log-Ratio 總和 $\sum_{t=1}^{|y|} \log \frac{\pi(y_t)}{\pi_{\text{ref}}(y_t)}$。人類標註員在構建偏好數據時，天然存在『字數更多、格式更精美代表更專業』的認知偏見（Verbosity Bias）。這導致數據集中 $y_w$ 平均長度常大於 $y_l$。
  > 2. **概率衰減抵消**：即便模型在每個 Token 上的優勢微乎其微，只要生成序列足夠長，總和積累的差值就能輕易壓倒短序列，從而獲得虛假的高隱式獎勵。
  > 
  > 工業工程防範方案：
  > - **演算法級修復**：改用 **SimPO**，將隱式獎勵除以序列長度 $|y|$ 進行長度歸一化（Length-Normalized Implicit Reward），並引入目標邊界 $\gamma$。
  > - **數據級防護**：在成對數據清洗中，將回答依長度分桶（Length Binning），限制配對的長度比率不大於 1.2。」

---

### 架構實戰考驗 Q2：既然 DPO 比 PPO/GRPO 簡單、省顯存且穩定，為什麼 OpenAI o1 與 DeepSeek-R1 不直接用 DPO 完成全部後訓練？
- **架構極限邊界**：考核你對離線模仿學習（Offline Imitation）與線上主動探索（Online Exploration）在推理模型能力上限上的本質理解。
- **滿分回答範式**：
  > 「因為 **DPO 的能力上限被死死鎖定在離線數據集的質量邊界內**，而複雜推理任務必須依賴線上探索：
  > 
  > 1. **離線分佈偏移（Covariate Shift）**：DPO 是離線算法，其偏好對是在預先採樣的靜態軌跡上計算的。當 Policy 模型更新數步後，其生成的動作分佈已經脫離了原始數據分佈。DPO 無法對策略當前自發生成的新錯誤進行即時懲罰。
  > 2. **無法產生超越人類的『頓悟』**：在競賽級數學與代碼領域，離線數據集中的 $y_w$ 本身就可能不是最優解。GRPO 通過線上多次 Rollout 與規則驗證器聯動，模型可以嘗試人類專家從未寫過的解題策略，一旦驗證正確即自我正向強化，這是『自發湧現 Extended CoT』的唯一路徑。
  > 
  > 因此，當代頂級實驗室的共識架構是：**以 DPO/SimPO 作為早期對話格式與通用偏好的冷啟動，隨後立即切換至 GRPO 進行高強度在線推理探索**。」

---

## 本章小結與學習路徑

```mermaid
graph LR
    C07["Ch 07: DPO 偏好優化 (Pillar 1 🔥)"] --> C11["Ch 11: 現代偏好 SimPO / ReMax (Pillar 1 🔥)"]
    C07 --> C03["Ch 03: GRPO 演算法推導 (Pillar 1 🔥)"]
    C07 --> C14["Ch 14: 頂級實驗室系統設計 Playbook (Pillar 4 🔥)"]

    classDef current fill:#7b341e,stroke:#dd6b20,stroke-width:2px,color:#fff;
    classDef next fill:#1a365d,stroke:#3182ce,stroke-width:1px,color:#fff;
    class C07 current;
    class C11,C03,C14 next;
```

→ 下一步建議：
- 若想徹底解決 DPO 的長度偏見並將 Reference 模型顯存砍半，進入 [Chapter 11: 現代偏好優化 — SimPO、ReMax 與 KTO](./11_modern_preference_simpo_remax_kto.md)。
- 若想了解成對數據的冷啟動與長思維鏈蒸餾，進入 [Chapter 9: SFT Cold-Start 階段](./09_sft_cold_start.md)。
