# GRPO 玩具 Demo（給學 agentic AI 的你）

一個只用 numpy、每一步數字都印出來的 GRPO（Group Relative Policy Optimization）
最小實作。目的是讓你**親眼看到** group / baseline / advantage / clip 在數字層面
長什麼樣子，而不是看一堆公式。

> GRPO 就是 DeepSeek R1 在 RL 階段用的演算法，本質是 PPO 的一個變體：
> **不訓練 critic / value network，改用「同一個 prompt 的一組候選答案的平均分」當 baseline。**

---

## 跑起來

```bash
cd /Users/derekleung/Documents/research-RL
python3 grpo_toy.py
```

只依賴 numpy（通常已經裝好）。不需要 torch、不需要 GPU。

執行後你會看到：
- 平均 reward 從 **0.188 → 1.0**（4 題全對）
- 在第 0、1、2、9、29、59 輪，每一個 prompt 都會印出：
  採樣到的 group、reward、baseline、advantage、π_old → π_new

---

## 怎麼讀這份 demo（配合輸出看最快）

### 第 0 輪發生了什麼（最重要的觀察）

看 `Prompt[1] '3+1=?' 正解=4` 那一段：

```
採樣到的 group: [3, 5, 4, 0, 5, 0, 4, 1]      ← policy 隨機吐出 8 個答案
reward:         [0, 0, 1, 0, 0, 0, 1, 0]      ← 答對=1，答錯=0
baseline:       0.250                           ← 這 8 個的平均 = (1+1)/8
advantage:      [-.577, -.577, +1.732, ...]    ← reward - baseline，再除 std
```

GRPO 的精髓就在這幾行：

| 誰 | advantage | policy 怎麼動 |
|---|---|---|
| 答對的（reward=1） | **+1.732（正）** | 機率**被拉高** |
| 答錯的（reward=0） | **-0.577（負）** | 機率**被壓低** |

所以更新完，正解 `action=4` 的機率從 `0.167 → 0.289`。**它只看了這 8 個答案之間的相對好壞，完全沒碰任何 critic。**

### 為什麼第 9 輪開始「學不動」了？

你會看到第 9 輪之後很多組的 advantage 全是 `0.0`，policy 不再動。這是**對的**：

```
採樣到的 group: [2, 2, 2, 2, 2, 2, 2, 2]   ← 全答對
baseline: 1.000
advantage: [0, 0, 0, ...]                    ← std=0，沒有相對差異可學
```

白話：**當整組答案一樣好，GRPO 學不到東西。** 這是 group-relative 的天性，
不是 bug。實務上 DeepSeek 用 KL penalty、多樣化採樣、難度更高的題目來避免這種「學滿了」的停滯。

---

## 術語對照表（demo 裡的變數 ↔ 論文/你會看到的詞）

| demo 裡 | 論文/通用術語 | 白話 |
|---|---|---|
| `theta` | policy parameters θ / logits | 模型權重。在這個玩具裡就是一個 logits 矩陣 |
| `softmax(theta)` | current policy π_θ | 給定 prompt，每個答案的機率 |
| `GROUP_SIZE` (=G) | group size | 對同一個 prompt 採樣幾個候選答案 |
| `actions` | group / candidate set | 那 G 個採樣出來的答案 |
| `rewards` | reward r(y) | 每個答案的品質分數 |
| `rewards.mean()` | **baseline** | group 平均分，取代 PPO 的 value network |
| `rewards.std()` | group std | 用來把 advantage 正規化 |
| `(r - mean)/(std+eps)` | **advantage / relative advantage** | 相對於組內平均，好多少/差多少 |
| `ratio = π_new/π_old` | importance ratio | 新 policy 對舊 policy 的機率倍數 |
| `EPS_CLIP` (=0.2) | clip range ε | 把 ratio 鉗在 [1-ε, 1+ε]，防 policy 一步走太遠 |
| `min(surr1, surr2)` | clipped surrogate objective | PPO/GRPO 的核心 loss |
| `INNER_EPOCHS` | mini-batch / multi-epoch | 同一組 group 資料重複用幾次 |
| （沒有） | critic / value network V(s) | **GRPO 刻意省掉的東西**，這就是它跟 PPO 的差別 |

---

## 跟你的 agentic AI / memory 專案怎麼接？

這是整份 demo 對你**最實用的部分**。你的目標是「有記憶的 RL agent」，
這個玩具 demo 的每個部件，都能映射到你現有的 memory 系統：

| demo（toy） | 你的 agent（真實） | 對應 |
|---|---|---|
| `theta` logits | LLM 權重 | policy 就是模型本身 |
| `PROMPTS` / `CORRECT` | user query / user_accept-reject signal | 你的 reward 來源是 user 行為 |
| `softmax + sample(G)` | 對同一個 query 生成 G 個回答 | group sampling |
| `reward()` function | KairosLog 裡的 `quality_tag` | `user_accepted=+1`, `rejected=-1` |
| `rewards.mean()` baseline | — | 不用 critic，直接 group 平均 |
| `advantage` | 餵進 GRPO loss 的 label | 決定哪個回答該被強化 |

### 你現在的 log schema 已經很接近 GRPO dataset 了

回想你之前設計的 Kairos/Conversation log：

```
event_type: "user_accept" | "user_reject" | "undo"
plan_id: "plan-123"
quality_tag: "user_accepted"
```

把它轉成 GRPO 要的東西，幾乎是一對一：

```
對同一個 user query：
  group = [所有用相同 plan_id 關聯的 candidate plans]
  reward[i] = +1 if quality_tag=="user_accepted"
             -1 if quality_tag in ["user_rejected","undo"]
              0 if 只是 build 成功但沒 feedback
  baseline = mean(rewards)
  advantage[i] = (reward[i] - baseline) / std
  → 套進 grpo_update() 的 loss
```

換句話說：**這個 toy demo 裡的 `grpo_update()` 函式，就是你未來 LLM GRPO 訓練迴圈的核心**；
toy 只是把它從「LLM + autograd」降維成「logits + 手算梯度」，讓你看得懂。

---

## 推薦的學習順序

1. **跑 demo，盯著第 0 輪的數字看**（baseline、advantage 怎麼來的）
2. 打開 `grpo_toy.py`，讀 `grpo_update()` 這個函式 — 它是整個演算法
3. 嘗試調超參數看會壞掉什麼：
   - `GROUP_SIZE = 1` → 會發現 GRPO 完全失效（沒有 group 就沒有 baseline）
   - `EPS_CLIP = 10.0` → clip 形同虛設，policy 會劇烈震盪
   - `LR = 50.0` → 步長太大，機率會爆掉
4. 想想你的 Kairos log：如果今天要餵給這個 demo，你要把哪些欄位變成 `rewards` 陣列？

---

## 這個 demo 刻意「沒做」的事（真實 GRPO 會有）

為了清楚，省掉了這些。等你懂核心後，這些是下一步：

- **KL penalty**：真實 GRPO 會加一項 `β·KL(π_new || π_ref)`，防止模型偏離原始模型太遠
  （避免 reward hacking）。這個玩具沒加，因為 binary reward 很難 hack。
- **token-level log-prob**：真實 LLM 的 ratio 是「整個回答序列」的機率比，
  不是單一 action。這個 toy 把回答壓成一個 action，所以簡化了。
- **GAE / multiple rewards**：真實場景一個回答可能有多個 reward 信號（正確性、格式、長度）。
- **offline / off-policy correction**：你的 agent 是 offline 收 log 再訓練，
  需要處理「採樣 policy ≠ 訓練 policy」的分布偏差。toy 是 on-policy，所以不用。

---

## 檔案

- `grpo_toy.py` — 主 demo，繁中註解，每步列印
- `kairos_log.py` — `KairosLogMemory` + `log_event()` helper + log→GRPO dataset 函式
- `demo_log_to_grpo.py` — 完整 offline GRPO 流程：log → dataset → update
- `grpo_offline_resample.py` — **解「每個 query 都不同、一個 session 很多 plan」的困惑**（見下方）

---

## 進階：從 Kairos log 走到 GRPO（給 agentic AI 用）

上面那份 toy demo 是 **on-policy**：當場採樣、當場更新。
但真實的 agent 是 **offline** 的 —— 先讓 agent 跟 user 互動、把 reward 訊號存進 log，
之後再回頭訓練。這一節示範這條線。

### 核心觀念：reward 訊號住在哪？

GRPO 的 reward 訊號 **唯一正規來源是 Kairos log 的 `log_event()`**，不是 Conversation。
原因：user 按 undo / 說「這不對」的當下如果沒記下來，事後再也補不回來。
Conversation 的文字本來就在，meta 之後 re-parse 就好；但「會消失的訊號」必須即時記。

### 跑

```bash
python3 kairos_log.py        # 1. log_event helper 的 self-test
python3 demo_log_to_grpo.py  # 2. 完整 offline GRPO 流程
```

### `log_event()` 的三個呼叫點（orchestrator 要插的地方）

```python
kairos.log_event("plan_generated", plan_id=pid, ...)       # 1. 產生候選方案
kairos.log_event("dashboard_built", plan_id=pid,           # 2. 實際建出來
                 result="success", ...)
kairos.log_event("user_accept", plan_id=pid,               # 3. user 的評價 ← reward 來源
                 quality_tag="user_accepted", ...)
```

第 3 點最關鍵 —— `quality_tag` 直接決定這個 candidate 的 GRPO reward：

| quality_tag | reward | 解讀 |
|---|---|---|
| `user_accepted` | +1.0 | 被接受 → 強化 |
| `user_revised` | +0.3 | 有用但需改 → 弱強化 |
| `user_rejected` | -1.0 | 被拒絕/undo → 抑制 |
| （只有 build 成功，無 user feedback） | +0.1 | 弱 fallback |

### 一個關鍵語意陷阱（我實作中踩到並修掉的）

**一個 plan = 一個 candidate**，不是「一條 event = 一個 candidate」。

錯誤版會把同一個 plan 的 `plan_generated` + `dashboard_built` + `user_accept`
三條 event 當成三個 candidate，於是 p1 產生 `[0.1, 1.0]` 兩個 reward —— 這是錯的。

正確版：一個 plan 的 reward = 它收到的**最終 user 評價**（取最後一條
`user_accept` / `user_reject` / `undo`；若都沒有，才用 build 成功的 fallback）。
見 `kairos_log.py` 的 `_final_reward_for_plan()`。

### GRPO 的 group key 是 query，不是 plan

這是另一個容易搞錯的地方。GRPO 的 group 定義是：
**同一個 user query 下的多個 candidate 方案**。

```
query = "幫我做 Austin dashboard"
  ├─ plan A (candidate 1)  → user_accepted  → reward +1.0
  ├─ plan B (candidate 2)  → user_rejected  → reward -1.0
  └─ plan C (candidate 3)  → user_revised   → reward +0.3
                                            ↑ 這三個才是一個 GRPO group
```

如果你的 log 裡 query 跟 plan 是 1:1（每個 query 只有一個方案），
那每個 group 只有 1 個 candidate，advantage 永遠是 0，**學不到東西**。
要有 GRPO 效果，必須對同一個 query 生成多個 candidate（n>1 sampling、A/B、重做）。

`demo_log_to_grpo.py` 的 `to_grpo_dataset_by_query()` 示範了「以 query 為 group key」
的正確重組方式。

### 完整流程圖

```
真實 agent 互動                        RL 訓練（offline）
─────────────────                      ──────────────────
orchestrator                           to_grpo_dataset_by_query()
  ├─ log_event("plan_generated")              ├─ group by query_id
  ├─ log_event("dashboard_built")   ──>       ├─ per-plan: 取最終 reward
  └─ log_event("user_accept/reject")          └─ 組成 (group, rewards)
        │                                            │
        ▼                                            ▼
   KairosLogMemory (檔案)                    group_relative_advantage()
                                                    │
                                                    ▼
                                            grpo_update()  ← 跟 toy demo 同一個函式
```

**最後一格 `grpo_update()` 跟 toy demo 是完全同一個函式。**
差別只在：toy demo 的 group 來自「現場採樣」，這裡的 group 來自「從 log 還原」。
這就是為什麼先學 toy demo 很重要 —— 它是這整條鏈的最後一格。

### demo 的小提醒

`demo_log_to_grpo.py` 跑出來的「平均 reward = -0.75」是**正常的**，不是 bug：
6 個 candidate 裡通常只有 1 個答對（+1）、5 個答錯（-1），平均自然偏負。
GRPO 要的就是這種「組內有差異」，正確答案才會被凸顯出來。
3 輪內 policy 已經把 4 題的正解都推到最高機率了。

---

## 最關鍵的觀念修正：group 是「訓練時 resample」的，不是「log 重播」的

如果你跑完上面那份 `demo_log_to_grpo.py`，可能會冒出一個困惑：

> **「我的真實 Kairos log 裡，每個 query 通常只有 1 個 plan（user 不會一次看 8 個版本），
> 一個 session 倒是有很多不同的 plan。這樣怎麼組 group？」**

這是 GRPO 從玩具搬到真實 agent 時最容易踩錯的觀念。一句話戳破：

> **GRPO 的 group 不是「user 看過的那幾個 plan」，而是「訓練時你自己重新抽樣出來的 G 個回答」。**

跑這個 demo 親眼看：

```bash
python3 grpo_offline_resample.py
```

### 三個關鍵觀念（demo 結尾也會印出來）

| 你的問題 | 答案 |
|---|---|
| 「每個 query 都不同」會不會讓 GRPO 失效？ | **不會**。GRPO 永遠是「同一個 query 內部」的 G 個候選互相比，query 再怎麼獨特都沒關係。多樣的 query 反而是好事 → 訓練 prompt 集。 |
| 「一個 session 很多 plan」怎麼用？ | 一個 session 的 N 個 plan = **N 個獨立訓練 prompt**，不是「一個 group」。每個 prompt 在訓練時各自展開成自己的 group。 |
| log 裡一個 query 只有 1 個 plan 怎麼辦？ | 那個 plan 本來就不該當 group。它的用途是 (a) 提供 prompt、(b) 用 accept/reject 推導 reward function。group 是訓練時 policy 自己抽樣出來的 G 個新候選。 |

### 你的 Kairos log 真正的用途

```
你的 log（一個 session，很多 plan，每 query 1 個 plan）
    │
    ├─ 抽出 prompts          → GRPO 訓練的 prompt 集
    └─ accept/reject labels  → 推導 reward function（rule-based 或訓練一個 RM）
                                ↓
GRPO 訓練迴圈：每個 prompt → 當前 policy 抽 G 個「新」候選 → reward function 打分 → 更新
               （★ 這 G 個是訓練時新生成的，user 從沒看過）
```

`grpo_offline_resample.py` 的輸出會刻意對照「log 裡這個 query 只有 1 個 plan」
vs.「訓練時這個 query 被 resample 成 G=8 個新候選」，兩者完全無關。
這就是為什麼「每個 query 都不同」「一個 session 很多 plan」完全不影響 GRPO ——
**group 從來就不是從 log 裡來的**。

### 跑出來的結果

```
[Phase 0] 10 個 query，每個只有 1 個 plan，全部被 reject（弱 agent）
[Phase 2] 訓練時每輪對 10 個 query 各 resample G=8 個新候選
[iter  0] 平均 reward = 0.200
[iter  6] 平均 reward = 0.838
[iter 11] 平均 reward = 0.988   → 10/10 題正解被選成最高機率
```

### 對你的系統，三條路（推薦順序）

| 路 | 做什麼 | 適合你的時機 |
|---|---|---|
| **A. DPO** | 把 Kairos 的 accept/reject/revision 直接轉成 (preferred, rejected) pair，先訓起來 | **現在**。你的資料形狀直接吃得下，不需要 group |
| **B. GRPO + rule reward** | 從 log 抽 prompts → 訓練時抽 G 個 → 用規則打分（schema 過不過、有沒有指定 component） | 有明確可評分規則時 |
| **C. GRPO + RM** | 先訓 reward model，再 GRPO | 工程量大，但最強。標準 DeepSeek/InstructGPT 做法 |

對 production copilot（不能 live n-sample、但 accept/reject 訊號很豐富），
建議 **A → C**：先用 DPO 把資料用起來、順便訓出 RM，再升級到完整 GRPO。

---

## 🎮 互動式 demo（你來當 user）

前面都是「程式跑給你看」。這份互動 demo 反過來：**你來當那個會給 accept/reject 的 user**，
每輪 policy 抽 4 個候選組成 group，你挑一個接受（或全拒絕），看你的選擇如何把 policy 塑形。

**為什麼要玩這個**：你會親身體驗到 GRPO 的核心觀念 ——
1. group 是 policy 當場抽的新候選，不是 log 裡的舊 plan
2. 你的 accept/reject 就是 reward 訊號
3. 「組內相對比較」是什麼感覺

### 互動體驗方式（統一 Web 平台）

互動式 GRPO demo 核心由 `grpo_play_core.py` 驅動，並透過統一平台提供互動體驗：
- **Web 平台**：執行 `python3 serve.py` 並開啟 `http://127.0.0.1:8000`（或 live demo 頁面），支援即時點選候選方案、Advantage 與機率分佈即時視覺化。
- **後端 API**：`/api/rl/toy/` 提供 session 狀態、候選生成與 step 呼叫。

每輪你會看到 4 個候選（`A`/`B`/`C`/`D`），輸入或點選一個接受：
- 被接受 → reward `+1.0`，advantage 為正 → policy 強化它
- 其他 → reward `-1.0`，advantage 為負 → policy 壓低它
- 全拒絕 → 整組負向訊號

### 兩個主題

- **dashboard**（預設）：6 種 dashboard 配方（極簡、含定價圖、含 YoY…），沒有 ground truth，reward 完全由你的選擇決定。**這才是 GRPO 真實場景**。
- **math**：數學題，reward 由正解自動判定。最純粹，用來摸懂機制。

### 檔案架構（互動版）

```
grpo_play_core.py       ← 共用核心：PolicyState + 主題 + GRPO 更新（單一 source of truth）
web/routers/toy.py      ← FastAPI REST API
```

後端呼叫的是 `PolicyState.apply_user_choice()` → `grpo_update()`，
跟 `grpo_toy.py`、`demo_log_to_grpo.py` 是同一個 `grpo_update()`。
整個 repo 的 GRPO 更新邏輯只有一份。

### ⚠️ 安裝/環境注意事項

如果你跑 Gradio 版遇到這個錯誤：

```
TypeError: argument of type 'bool' is not iterable
```

那是 Gradio 4.44 + 新版 pydantic 的已知衝突（pydantic 產生的 JSON schema
用了 `additionalProperties: bool`，Gradio 4.44 的 `gradio_client` 沒處理）。
**修法：把 pydantic 降到 2.9.x**：

```bash
pip install "pydantic<2.10"
```

另外，這個 repo 在 Python 3.9 上跑，所以 Gradio 最高只能到 4.44.1
（Gradio 5.x 需要 Python 3.10+）。4.44.1 + pydantic 2.9.2 這個組合驗證可跑。

---

## 🧠 LLM-flavored 版：從 toy GRPO 走到 DeepSeek R1 的 loss 公式

前面所有 demo 的 policy 都是「6 個 action 的機率分佈」，每次 sample 就是挑一個數字。
但**真實 LLM 的 GRPO 不是這樣**：candidate 是一整段 token 序列，policy 是逐 token 自回歸生成。

這份 demo 把那個落差補上：

```bash
python3 grpo_llm_char.py
```

### 它跟 toy demo 的差別（這就是「LLM GRPO vs PPO」縮小版）

| | `grpo_toy.py`（單步） | `grpo_llm_char.py`（這份，序列） |
|---|---|---|
| policy | 6 個 action 的 logits | 字元表上每個 char 的 logits（給定上文） |
| candidate | 從 6 個選一個 | 自回歸生成一整段字串 |
| logprob | `log(π[action])` | `Σ_t log π[char_t | char_<t]` ← **逐 token 加總** |
| ratio | `π[a]/π_old[a]` | `exp(logp_new - logp_old)` ← **DeepSeek R1 公式長這樣** |
| reward | exact match (binary) | partial：第一個字對 0.5，全對 1.0 |

### 為什麼這一步關鍵

你看 DeepSeek R1 / 任何 LLM GRPO 論文，loss 一定有這項：

```
r_t(θ) = exp( logπ_θ(y_t | x, y_<t) − logπ_old(y_t | x, y_<t) )

J_GRPO(θ) = E[ min( r_t(θ)·Â_t,  clip(r_t(θ), 1-ε, 1+ε)·Â_t ) ] − β·KL(π_θ‖π_ref)
```

`grpo_toy.py` 看不到「序列機率」「token-level log-prob」這些概念。
`grpo_llm_char.py` 把它們具體化：
- `seq_logprob()` = 逐 token log 機率加總 = 論文裡的 `logπ_θ(y_t|x,y_<t)`
- `ratio = exp(logp_new - logp_old)` = 論文裡的 `r_t(θ)`
- group 還是「同一個 prompt 抽 G 段候選序列」，advantage 還是 group-relative

學完這份，你再看 DeepSeek R1 的 loss 公式就不會覺得跳太快了。

### Demo 刻意展示的兩個 LLM GRPO 真實陷阱

跑完你會看到 policy 傾向「全部輸出同一個答案」（mode collapse）。這是兩個真實因素：

1. **1-gram policy 無法區分 prompt**：`1+1=` 和 `3+1=` 在生成起點看到的 context 都是 `=`，根本無法區分 → 學到的答案一樣。**真實 LLM 用整個上文（transformer attention）就沒這問題**。這就是為什麼真實 LLM policy 是 transformer 而不是 1-gram table。

2. **partial reward + group-relative 的 mode-collapse 風險**：policy 會被推向「在這組 prompt 裡最常答對的那個答案」。**DeepSeek 用 KL penalty `β·KL(π_θ‖π_ref)` 壓制這個**（這份 demo 沒做，是 toy 簡化）。

這兩個「沒解決」的點，都是真實 LLM GRPO 訓練時的核心工程問題。這份 demo 的價值不是「解出所有題」，是讓你**看到序列機率、token-level logprob、ratio 怎麼算** —— 這些是 toy demo 看不到、但 DeepSeek R1 論文裡到處都是的概念。

### 學習路徑總覽（整個 Part 1）

```
grpo_toy.py              ← 演算法本身（單步 action，每步印數字）
    ↓
kairos_log.py            ← reward 訊號的存放處（log_event helper）
    ↓
demo_log_to_grpo.py      ← 把兩者串起來（offline GRPO）
    ↓
grpo_offline_resample.py ← group 是訓練時 resample，不是 log 重播
    ↓
grpo_play*.py            ← 互動式體驗（你當 user）
    ↓
grpo_llm_char.py  ★這份  ← 序列 candidate + token logprob + DeepSeek R1 公式
```

走完這條線，你從「GRPO 是什麼」一路到「DeepSeek R1 的 loss 公式每項在算什麼」，
每一步都有可跑的 demo 印證。

---

## 🎮⚡ 互動式 LLM 版：你當 reward function（最接近真實 copilot）

`grpo_llm_play.py` 把 LLM-flavored 版變成**互動式**：你輸入開放式 prompt，policy 生成 4 段候選回答，你挑一個最喜歡的，你的選擇就是 reward。

```bash
python3 grpo_llm_play.py
```

### 為什麼這版直接回答你「開放式問題怎麼打分」的困惑

你之前問的核心問題：

> 「user 問的開放式問題沒有 ground truth，怎麼給 GRPO 的 reward？」

這份 demo 給你答案：**不需要 ground truth。user 就是 reward。**

```
你輸入 prompt（任何開放式問題，例如 "1+1=" 或 "2+3="）
    ↓
policy 用目前的「語感」生成 4 段候選回答（字元級自回歸）
    [A] 1+1=4 <<7845     (logp=-19.06)
    [B] 1+1=3▢4=90‹END›  (logp=-18.17)
    [C] 1+1=48192149     (logp=-21.46)
    [D] 1+1=91622616     (logp=-21.19)
    ↓
你挑一個（例如 A）→ A 的 reward = +1，其他 = -1
    ↓
GRPO 更新：A 的 logp 從 -19.06 → -16.57（↑ 被強化）
          其他候選的 logp 被壓低（group-relative）
```

**這跟你的 copilot 收 `user_accept` / `user_reject` 完全一樣** —— 沒有 ground truth，user 的選擇就是 reward signal。

### 這版跟其他 demo 的定位差別

| Demo | policy | candidate | reward 來源 | 場景類比 |
|---|---|---|---|---|
| `grpo_toy.py` | 6 個 action | 離散選項 | exact match | 教科書 GRPO |
| `grpo_llm_char.py` | 字元序列 | 序列 | exact match（partial） | DeepSeek R1 自動訓練 |
| `grpo_play.py` | 6 個 action | 離散選項 | **你當 reward** | 體驗 group-relative |
| `grpo_llm_play.py` ★ | 字元序列 | 序列 | **你當 reward** | **最接近真實 copilot** |

`grpo_llm_play.py` 是唯一同時具備「序列生成（像真 LLM）」+「user 當 reward（像你的 copilot）」的版本。

### 玩法提示

- **一開始 policy 是 uniform**（生成的全亂碼），所以前幾輪你是在「亂碼裡挑比較不討厭的」
- **多玩幾輪**，policy 會開始模仿你接受過的字元 pattern
- 離開時會 show policy 學到的轉移機率 top-3，看得出你塑形了什麼偏好
- **vocab 只有 15 個字元**（`<>+= 0123456789`），這是 toy 簡化；真實 LLM 詞表幾萬

### 預期看到的學習效果

如果你連續接受開頭是 `5` 的回答，幾輪後你會看到 policy 對 `=` 之後生成 `5` 的機率明顯高於 uniform（1/15≈0.067）。這就是「你的偏好被 group-relative 學進 policy」的可見證據。

### 跟你 copilot 系統的直接對應

| 這份 demo | 你的 copilot |
|---|---|
| 你輸入 prompt | user 的 query |
| policy 生成 4 段候選 | （概念上）n>1 sampling 多個回答 |
| 你挑 A 接受 | user 按 accept / 說「這個好」 |
| A 的 reward=+1 | `quality_tag = user_accepted` |
| GRPO 更新 policy | offline GRPO 更新 LLM 權重 |

唯一差別：你的 production copilot 不會 live n-sample（成本太高），
所以這一步在實務上是 **offline** 的 —— 從 Kairos log 還原 (prompt, accepted, rejected)，
再批次跑 GRPO。但邏輯完全一樣。

### Web 視覺化與互動

互動體驗已整合進統一 Web 平台：

```bash
python3 serve.py
# 打開 http://127.0.0.1:8000
```

圖表顯示 policy 對幾個關鍵 context（`after '='`、`after BOS`、`after space`）的 top-5 轉移機率，紅線標 uniform baseline。
藍色 bar = 高於 uniform（你塑形出來的偏好），灰色 = 低於 uniform。
這讓你**視覺化看到自己的選擇如何改變 policy**。

---

# Part 1.5 — Production 訓練 Pipeline 的工程化 Scaffolding (Production 工程化)

在將學術或玩具級的 RL 演算法（如 Part 1 中的玩具 GRPO）搬上實際的生產環境時，單單有優化演算法本身（`grpo_update_llm`）是不夠的。真實的 production 系統需要高度健壯的「工程外殼（Scaffolding）」來包裝與防護。

這部分由 `training_pipeline.py` 實作，並透過 Web 後端（`web/routers/llm.py`）提供互動 session 與 offline training API。

### 0. 怎麼跑（完整 offline → resume 循環）

```bash
# 1. 啟動 Web 服務平台
python3 serve.py

# 2. 線上互動：前端或 API 互動，產生 kairos log + checkpoint
#   輸入 prompt → 點 accept/reject → 產生 logs/kairos.md, checkpoints/

# 3. 離線訓練：調用 /api/rl/llm/sessions/{sid}/offline-train
#   讀 log → resample → 跑 GRPO → 存新 checkpoint + bump version
```

這三步就是 production RL 的真實循環：**線上收集 → 離線訓練 → 部署新版本**。

### 1. Production 必備元件

- **`TrainingConfig`**：
  集中管理所有超參數（如 `group_size`, `lr`, `eps_clip`, `inner_epochs`, `beta_kl` 等）與路徑設定。真實生產環境下通常會採用 YAML / Hydra 來進行配置管理，在此我們以 dataclass 表達以維持輕量化與清晰度。

- **`CheckpointManager`**：
  在 RL 訓練過程中，隨時可能因為獎勵函數（Reward Function）設計不當、超參數偏移或噪聲資料導致 Policy 崩塌（Model Collapse）。Checkpoint 系統保證了我們隨時能從上一個版本復原。
  - 儲存路徑結構：`checkpoints/policy_v{N}.npz` 儲存 numpy theta 權重，配有 sidecar JSON (`.json`) 檔案儲存歷史 metrics、metadata 和 timestamp。

- **`RewardLogger`**：
  在無 TensorBoard 依賴的環境下，將訓練輪數、prompt、接受的候選索引、平均 reward、baseline、std 等關鍵指標逐輪寫入 `logs/reward_log.csv`。這為系統在 UI 上繪製實時 Learning Curve 提供了穩定的資料來源。

- **`KairosRunLogger`**：
  包裝 `KairosLogMemory` 將訓練事件持久化成 append-only 的 `grpo_step` 日誌。每個事件記錄 `(prompt, accepted_idx, policy_version)`。**這是離線訓練（Offline Training）的資料來源**。注意：log 刻意只存 prompt + user preference，**不存原本的 candidate 序列** —— 因為 offline 重跑時 group 要用「當前 policy」重新採樣，不是回放 log 裡那組（這是 `grpo_offline_resample.py` 的核心觀念）。

- **`PolicyVersionTracker`**：
  每次 Policy 參數成功寫入 checkpoint 後版本號自動加一（`policy_version += 1`）。這能精準識別 logs 中的數據是否由舊版 policy 生成（Off-policy 偵測，這是 Part 2 的關鍵功能之一）。

- **`RewardHackingGuard`**（整合 `policy_feedback.detect_anomalies`）：
  RL 演算法（特別是 GRPO）極易踩中 Reward Hacking（例如：模型學會生成重覆、投機但能拿高分的字元 pattern）。
  - 將每 N 輪劃分為一個「Session」。
  - 檢測 consecutive sessions 間 win-rate 的突然暴升（win-rate spike）以及生成多樣性的急遽崩塌（diversity collapse）。
  - 一旦觸發，會將 Policy 鎖定為 `frozen=True`（模擬 production 的 `/dream freeze` 行為），暫停參數更新，並在 UI 發出紅色 Banner 警告。

### 2. 離線訓練（Offline Training）與 Gradio 連接

- **`run_offline_training.py`**：
  展示了 RL 訓練在真實世界中絕大多數是離線運作的。它讀取 `logs/kairos.md` 的 `grpo_step` 事件，還原成 `(prompt, accepted_idx, policy_version)` 樣本，**對每個樣本用當前 policy 重新採樣 group**（不回放原序列），跑 `TrainingPipeline.step()` 批次更新，最後存新 checkpoint + bump version + 輸出 reward-hacking 報告。它還會偵測 off-policy 資料（log 來自多個 policy version）並誠實標示（不做重要性採樣校正，跟 `policy_feedback._off_policy_note` 一致）。
- **Gradio 網頁版 UI 升級**：
  將舊的 Module-level globals 統一封裝進單一 `PIPELINE` 實例，在啟動時自動讀取最新 checkpoint。UI 新增了實時 Metrics 面板、Matplotlib 繪製的動態 Learning Curve、Reward Hacking Banner 警告（觸發時凍結 accept 按鈕）以及 Checkpoint 歷史下載功能。

### 3. 刻意「不做」的事（與 Part 2 / 真實 Production 的分工）

- **KL 散度懲罰（KL Penalty / Reference Policy）**：
  本版 config 留有 `beta_kl` 佔位符。在真實 LLM 中，會額外維持一個凍結的 reference policy $\pi_{ref}$，並在優化目標中加入 $\beta \cdot \text{KL}(\pi_\theta \parallel \pi_{ref})$。我們在此刻意不實作它以保持 toy 1-gram 梯度的可讀性，此防護任務目前由外部的 `RewardHackingGuard` 替代。
- **真實 Transformer Policy**：
  Policy 仍然維持為極簡的 1-gram table（僅查上一個字元），而非真實的 Transformer。這屬於方向 C 的主題，在 1-gram 下依然能完整展示所有序列機率計算與 ratio 機制。
- **Session 隔離 / 多 User 支持**：
  Gradio 應用仍是單一 process-global policy，供 demo 本地單人測試。真實 production 中需設計 Per-session / Per-user state 隔離。
- **分散式 / 多 GPU 訓練**：
  完全排除在 Demo 範圍外。

---

# Part 2 — 路線 A：In-Context Policy Learning（production pipeline，不是 GRPO）

上面 Part 1 的四個 demo 是**教科書的 GRPO**：on-policy、玩具任務、每步印數字。
但 production copilot 不能那樣跑——它每個 query 只服務一個 response、
沒有 G 個候選可以互相比。硬把 GRPO 套上去會踩進一個統計陷阱（見下方）。

這個 repo 的第二部分（`policy_feedback.py` + `policy_feedback_demo.py`）
是針對 production 的**誠實路線**：不用 weight update，改用 in-context rules。
這條路線有自己的名字、自己的 tradeoff，不要跟 GRPO 混為一談。

## 跑起來

```bash
python3 policy_feedback_demo.py
```

七個場景一次跑完，每個都印出實際的輸入 log、分層統計、rules、落地後的 AGENTS.md。
不依賴 numpy、不依賴任何 LLM API key（compactor 的 LLM pass 用 fake callable 示範接線）。

## 它「不是 GRPO」的核心原因（這段面試會被問，要先想清楚）

GRPO 的 group baseline 是**同一個 prompt 的 G 個採樣輸出的平均**。
那 G 個輸出可交換、來自同一分布，所以「reward − group mean」是一個有定義的相對優勢。

一個 user session 裡的 events **不可交換**：L4 plan 被 confirm、L3 edit 被 undo、
L2 answer 被接受——是不同的 query、不同的時間、不同的 intent level。
把它們的 reward 平均起來得到的數字沒有任何統計意義。
N=10 解決不了這個問題，N=100 也一樣，因為你在平均蘋果和橘子。

這正是 `grpo_offline_resample.py` 在 README 那段警告的：
「GRPO 的 group 不是『user 看過的那幾個 plan』，
而是『訓練時你自己重新抽樣出來的 G 個回答』」。
Production copilot 不抽樣 → 沒有合法的 group → 不能做 advantage estimation。

## 路線 A 改做什麼（三件誠實的事）

| 做什麼 | 為什麼成立 |
|---|---|
| **分層統計**：按 intent_level（L2/L3/L4）分組後才算 win_rate | 同一層內的 events 大致可比（同一種 user signal、同一種 agent action）。垮層平均是錯的，分層是唯一可辯護的平均方式。 |
| **Reward-hacking 偵測**：跨 session win_rate 跳升 + query 多樣性崩塌 → 觸發警示 | win_rate 突然飆高幾乎都是 degenerate shortcut，不是真的變聰明。觸發時 `/dream` 應該 freeze，不落地新 rules。 |
| **Compactor 提取 KEEP/AVOID rules**：寫進 AGENTS.md，下個 session 注入 system prompt | Policy update 不需要 gradient，只需要文字 rule。LLM 行為因此改變，但權重完全不動。 |

## 完整閉環（demo 的 [S7] 會把這條鏈演給你看）

```
user 按 confirm / undo / abandon
    │  record_feedback() → FeedbackEvent
    ▼
KairosLogMemory (append-only daily .md)
    │  read_feedback_events()
    ▼
stratified_stats()   ← 分層 win_rate
detect_anomalies()   ← reward-hacking 偵測
compact()            ← KEEP/AVOID rules（帶 evidence dict）
    │
    ▼
AGENTS.md ## Learned Behavior Rules  +  policy_version bump
    │  下個 session 的 system prompt 自動注入
    ▼
LLM 行為改變 —— 但權重完全不動，沒有 weight update
```

## 三層分離：為什麼 guardrail 永遠不會被學掉

AGENTS.md 硬分成三個 section，compactor **只能動第三個**：

```markdown
## System Constraints      ← guardrail，永遠不被 RL 改
## User Preferences        ← 使用者設定，永遠不被 RL 改
## Learned Behavior Rules  ← 唯一可以被 /dream 改寫的區塊
```

demo 的 [S7] 有斷言：跑完 compaction 後，前兩個 section 的內容必須字字不動。
這是 in-context policy learning 的硬邊界——一旦放寬，LLM 遲早會「順手」改 guardrail。

## 七個場景對照

| 場景 | 演什麼 | 對應的設計決策 |
|---|---|---|
| S1 | 正常 traffic 的分層 win_rate | 不垮層平均是唯一的可辯護方式 |
| S2 | L4_confirm 持續 → KEEP rule | heuristic compactor 自帶 evidence，不靠 LLM |
| S3 | L3_undo 持續 → AVOID rule | 同上，負向訊號同樣可提取 |
| S4 | win_rate 高但 query 全同 → hacking 警示 | diversity_collapse 是 freeze 的觸發條件 |
| S5 | log 裡混合新舊 policy_version | off-policy 用誠實的版本追蹤，不裝作會 importance sampling |
| S6 | compact() 接受 LLM callable | LLM 是可插換的，但 evidence dict 永遠保留 → 可稽核 |
| S7 | rules 寫回 AGENTS.md 三層結構 | guardrail 不會被學掉（有斷言保護） |

## 面試時怎麼講（誠實版本，不會被追問到答不出來）

> 「我設計了一個 in-context policy learning pipeline。User feedback 被記成
> FeedbackEvent，按 intent_level 分層算 win_rate——不垮層平均，這是我從
> 早期 within-session advantage 的坑學到的。Compactor 從分層統計提取
> KEEP/AVOID rules，帶 evidence dict，寫進 AGENTS.md 的 Learned 區塊，
> 下個 session 自動注入 system prompt。
>
> 為了防 reward hacking，我偵測 win_rate 跳升 + query 多樣性崩塌，
> 觸發時 freeze /dream，policy_version 不動。
>
> 整套是 group-relative formulation inspired by GRPO，但我刻意不做
> weight update——production copilot 每 query 只服務一個 response，
> 沒有 G 個候選可比，做 advantage estimation 統計上不成立。
> 升級到真 RL 的觸發條件我都寫進 README 了。」

**這個講法比宣稱「我實作了 GRPO」強得多**，因為它展現你知道邊界在哪。
xAI 要的是懂 RL 的人，不是會套詞的人。

## 升級到真 RL 的觸發條件（寫在這裡，面試時主動講）

從 in-context rules 升級到 weight-level fine-tuning（DPO 或 GRPO）的條件：

1. 累積 **500+ 標注 events**（目前單 session 只有 3-10）
2. `policy_version` 穩定超過 **2 週沒有 schema 變動**
3. reward-hacking detector **沒有持續觸發**

目前還在 in-context 階段。升級時，同一批 reward logs 可以直接轉成 DPO
preference pairs（adv>0 的 action 作為 chosen，adv<0 的作為 rejected）——
這是路線 A 的資產，不是沈沒成本。

## Part 1 vs Part 2 對照

| | Part 1（GRPO 玩具） | Part 2（路線 A production） |
|---|---|---|
| 目的 | 教學：看 group/baseline/advantage 在數字層面怎麼運作 | 上線：production copilot 的閉環 |
| policy | logits 矩陣（假裝是 LLM 權重） | LLM + 注入的明文 rules |
| group | 訓練時現場抽樣 G 個 | **沒有**（production 不抽樣） |
| weight update | 有（手算梯度） | **沒有**（in-context rules） |
| advantage estimation | 有（group mean baseline） | **不做**（垮層平均無意義） |
| 適合的資料量 | 任意（玩具） | 小資料 regime（單 session 3-10 events） |
| 檔案 | grpo_toy.py, demo_log_to_grpo.py, grpo_offline_resample.py | policy_feedback.py, policy_feedback_demo.py |

兩條路線是**互補**的：Part 1 讓你懂 GRPO 的數學，Part 2 讓你知道在
production 約束下什麼能做、什麼不能做。完整的 RL 工程師兩邊都要會。

## 檔案（完整清單）

- `grpo_toy.py` — Part 1：on-policy GRPO 玩具，每步印數字
- `kairos_log.py` — KairosLogMemory + log→GRPO dataset 函式
- `demo_log_to_grpo.py` — Part 1：log → dataset → update 完整 offline 流程
- `grpo_offline_resample.py` — Part 1：解「group 不是 log 重播」的觀念
- `grpo_play_core.py` — Part 1：互動式探索與環境狀態共用核心
- `grpo_llm_char.py` — Part 1：**LLM-flavored 版**（character-level 自回歸，DeepSeek R1 loss 的最小實作）
- `training_pipeline.py` — Part 1.5：**production 工程外殼**（config / checkpoint / reward log / version / hacking guard）
- `memory_selector.py` — 上下文預算感知記憶讀取層（與 paper/code 鏡像保持一致）
- `policy_feedback.py` — Part 2：路線 A 的核心模組（FeedbackEvent / 分層統計 / 偵測 / compactor）
- `policy_feedback_demo.py` — Part 2：七場景閉環 demo
