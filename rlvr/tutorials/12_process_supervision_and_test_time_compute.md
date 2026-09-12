# Chapter 12: 過程監督與推理算力擴展 (Process Supervision & Test-Time Compute)

> *「如果一個學生寫了 28 步天才般的嚴密數學推導，僅在最後一步發生了筆誤，純結果監督會把這 28 步全部打成零分並施加懲罰——這就是信用分配的悲劇；而過程獎勵模型（PRM）與推理時算力擴展，則是打破這一悲劇的關鍵鑰匙。」*

---

## 核心心智模型：結果監督 (ORM) 的信用分配坍塌

在標準 RLVR 中，模型依賴**結果獎勵模型（ORM, Outcome Reward Model）**：最終答案正確給 1，錯誤給 0。但在長推理鏈中，ORM 存在兩大不可調和的致命盲區：

```mermaid
graph TD
    subgraph ORM["結果監督 (ORM) — 粗粒度稀疏反饋"]
        O1["步驟 1: 邏輯嚴密"] --> O2["步驟 2: 深刻推導"]
        O2 --> O3["步驟 3: 計算粗心筆誤 💥"]
        O3 --> O4["步驟 4: 答案錯誤"]
        O4 --> OR["終端得分: 0 分 ❌<br/>(冤枉誤殺：步驟 1 與 2 的精彩推理被一同抹殺懲罰！)"]
    end

    subgraph PRM["過程監督 (PRM) — 步驟級別細粒度信用分配"]
        P1["步驟 1: 置信度 0.99 ✅"] --> P2["步驟 2: 置信度 0.98 ✅"]
        P2 --> P3["步驟 3: 置信度 0.08 🚨 (及早剪枝！回溯重算)"]
        P3 -.->|剪枝回溯重試| P2
    end

    classDef ok fill:#22543d,stroke:#48bb78,color:#fff;
    classDef bad fill:#742a2a,stroke:#f56565,color:#fff;
    class O1,O2,P1,P2 ok;
    class O3,O4,OR,P3 bad;
```

1. **精彩推導被「連坐冤殺」**：前 28 步思路完全正確，最後一步算錯，整個解答被給予負優勢，懲罰了高階抽象思維。
2. **胡說八道被「僥倖獎勵」**：中間邏輯全錯，最後歪打正著猜對答案，結果給予正優勢，強化了幻覺路徑。

---

## 12.1 Math-Shepherd：免人工標註的蒙特卡羅自動過程打分

在早期，訓練 PRM 需要消耗數百萬美元雇佣數學博士逐步打標（如 OpenAI PRM800K）。**Math-Shepherd**（ACL 2024）提出了一種完全自動化的**蒙特卡羅 Rollout 標註法**：

```text
給定一道題 x 與當前推導的前 t 個步驟前綴 (s_1, s_2, ..., s_t)：
1. 從當前步驟結尾出發，讓模型獨立進行 K=16 次隨機接續採樣 (Rollouts)，直至產生最終答案。
2. 呼叫自動驗證器檢查這 16 次接續採樣中，有多少次成功抵達了正確答案？（記為 M）
3. 該步驟 s_t 的軟性過程得分定義為後續成功率：
                      p(s_t) = M / K
```

```python
def math_shepherd_step_labeler(mock_rollout_results: list[bool]) -> float:
    """計算後續蒙特卡羅 Rollout 的成功率作為過程標籤 p(s_t) = M / K"""
    k = len(mock_rollout_results)
    m = sum(mock_rollout_results)
    return m / k

# 步驟 1: 正確的代數因式分解 (後續 16 次接續中有 15 次答對)
p_step1 = math_shepherd_step_labeler([True] * 15 + [False])

# 步驟 2: 荒謬的邏輯跳躍 (後續 16 次接續中僅 1 次靠蒙答對)
p_step2 = math_shepherd_step_labeler([True] * 1 + [False] * 15)

print(f"步驟 1 過程置信度: {p_step1:.3f} (極高可靠度，思路健全)")
print(f"步驟 2 過程置信度: {p_step2:.3f} (極低置信度，偏離正確解空間)")
# 步驟 1: 0.938 | 步驟 2: 0.062 ➔ 清楚區分邏輯優劣！
```

---

## 12.2 PRM 訓練與整條軌跡評分聚合策略

訓練好的 PRM 會在每個思維步驟的結束標記（如 `\n\n` 或 `</step>`）輸出一個介於 0 與 1 之間的標量信心度。

### 三大軌跡聚合策略與「均值致命陷阱」

```python
import math

# 示範軌跡：前 2 步與第 4 步極其精彩，但第 3 步犯下了毀滅性邏輯硬傷 (0.08 分)
step_scores = [0.98, 0.96, 0.08, 0.99]

prod_score = math.prod(step_scores)             # 連乘 (聯合獨立機率)
min_score  = min(step_scores)                   # 木桶最短板
mean_score = sum(step_scores) / len(step_scores)# 算術平均

print(f"連乘聚合 (Product) : {prod_score:.4f} ➔ 判定 REJECT (拒絕瑕疵證明)")
print(f"短板聚合 (Minimum) : {min_score:.4f} ➔ 判定 REJECT (拒絕瑕疵證明)")
print(f"均值聚合 (Mean)    : {mean_score:.4f} ➔ ⚠️ 致命偽陽性：竟然給出 0.75 的高分！")
```

> [!CAUTION]
> **工業界鐵律：嚴禁在數學與代碼驗證中使用算術平均（Mean）聚合 PRM 分數！**
> 在 10 步的推導中，就算中間出現荒謬的邏輯錯誤，算術平均依然會給出 $0.90$ 的高分，徹底被有毒的錯誤解答欺騙。在嚴密邏輯場景下，**必須嚴格採用 Product（連乘）或 Minimum（木桶最短板）**。

---

## 12.3 測試時算力擴展規律 (Test-Time Compute Scaling)

OpenAI o1、o3 與 DeepSeek-R1 帶給 AI 界最深遠的變革，是確立了 **Inference-Time Search（測試時搜尋算力）的擴展定律**（Snell et al., ICLR 2025）：

> **定理**：在推理階段為 PRM 引導的搜尋投入更多算力，其帶來的準確率增益，可等效於將預訓練模型參數量直接放大 **14 倍**！

```text
算力槓桿比喻：
7B 模型 + PRM 引導的 Best-of-N 搜尋 ──► 數學競賽解題率超越 70B 模型的單次貪婪回答！
```

```python
# 測試時搜尋算力擴展實測模擬 (Best-of-N with PRM Reranking)
candidates_n = [1, 4, 16, 64]
accuracy     = [34.0, 48.5, 62.0, 71.5]
tokens_cost  = [280, 1120, 4480, 17920]

for n, acc, cost in zip(candidates_n, accuracy, tokens_cost):
    print(f"N = {n:<2} 候選採樣 | 準確率: {acc:>5.1f}% | 消耗 Token: {cost:>5} tok | 算力倍率: {cost/tokens_cost[0]:>4.1f}x")
```

### 三大測試時搜尋範式
1. **Best-of-N (BoN)**：平行生成 $N$ 條完整軌跡，利用 PRM 聯合分數 $\prod R(s_t)$ 進行高精度重排序。
2. **MCTS 蒙特卡羅樹搜尋 / Beam Search**：在每個推理步驟展開 $B$ 個分支，由 PRM 提前剪枝淘汰荒謬前綴，大幅節約後續生成長度。
3. **自適應動態算力預算（Adaptive Compute）**：遇到簡單問題單次回答（$N=1$）；遇到奧數難題自動調度 $N=64$ 與樹搜尋，達成成本與精度的動態最佳化。

---

## 🤔 架構深度思辨與工業界陷阱 (Architectural Insight & Production Pitfalls)

> [!IMPORTANT]
> **Frontier Lab 核心考點 (DeepMind / OpenAI / Anthropic MLE)**:
> - **陷阱與對策 (Failure Mode & Fix)**:
>   1. **PRM 聚合中的均值陷阱 (The PRM Mean Fallacy)**: 
>      在多步推理中，若採用算術均值（Mean）聚合各步驟得分，一條包含 10 個步驟的推理解答哪怕在第 4 步犯下致命邏輯錯誤（得分 0.0），其平均分依然可以高達 0.90，導致驗證器被重大漏洞欺騙。
>      *工業界對策*: **在數學與邏輯證明中，必須嚴格採用 Product（連乘）或 Minimum（木桶最短板）聚合**。
>   2. **PRM 搜索過程中的好分數捷徑 (PRM Hacking)**: 
>      在 MCTS 或 Beam Search 中，策略模型會生成看起來「極其自信、引用大量專業定理」的中間偽步驟來騙取 PRM 的高分。
>      *工業界對策*: 採用 **雙驗證器混合架構**：中間步驟由 PRM 剪枝引導，終端狀態必須由 Outcome Verifier（單元測試或 SymPy）進行二值強校驗。
> - **核心技術思辨 (Technical Deep Dive)**:
>   *Q: 什麼是 Test-Time Compute Scaling（測試時算力縮放）？與常規預訓練 Scaling Law 相比，它的邊際收益曲線有何不同？*  
>   *A: 預訓練 Scaling Law 依賴於堆疊參數量和數據量，訓練成本高昂且受限於數據天花板；Test-Time Compute Scaling（如 OpenAI o1 / DeepSeek-R1）將算力傾斜至推理階段，通過延長思維鏈（Long-CoT）、MCTS 樹搜索和 PRM Best-of-N 重排序，在模型參數不變的情況下釋放模型深層的推理潛力。其邊際收益曲線在困難問題上表現出顯著的**乘數效應**（小型 7B 模型經 Test-Time Search 可超越千億參數大模型的單次貪婪輸出），但在簡單知識型問答上會迅速面臨收益遞減。*

---

## 下一步

→ 進入 [Chapter 13: 數據飛輪與基準去污染 (Data Flywheel & Decontamination)](./13_data_flywheel_and_decontamination.md)，實作自動化題目合成、拒絕採樣與 MinHash LSH 防洩漏審計。
