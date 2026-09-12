# tutorials (rlvr/tutorials)

> Auto-maintained by Agent. Do not edit manually.

## Purpose
An 18-part comprehensive technical curriculum organized into **4 Thematic Technical Pillars** focusing on Reinforcement Learning with Verifiable Rewards (RLVR) and LLM Post-Training engineering in Traditional Chinese (繁體中文). Features the 7-Pillar Learning Standard: Intuitive Mental Models, Architectural Decision Framework, Systems Mechanics & Boundary Intuition, 5-Stage Progressive Executable Notebook Laboratories (with realistic execution outputs), 4D Telemetry Radar Tables, Production Incident Runbooks, and Frontier AI Labs Whiteboard Defense Scenarios.

## 4 Thematic Pillars & Contents

### Pillar I · 基礎強化學習與偏好優化 (Foundational RLVR & Preference)
| File | Hotspot | Responsibility / Competency Focus |
|---|:---:|---|
| `03_grpo_algorithm.md` | 🔥 | 同儕評審互評會 vs 隨行考官心智模型、零 Critic 斷頭台、Z-Score 鐘形曲線放大、漸進式 5 階實驗室（全零梯度復現、Dr. GRPO 消除長度偏見消融）、4D 遙測雷達與面試手撕 |
| `07_dpo_preference_optimization.md` | 🔥 | 廚房試吃員 vs 主廚自省心智模型、Elo 偏好博弈、配分函數幽靈消去術、三力拔河受力場、漸進式 5 階實驗室（概率塌陷與長度作弊復現、SimPO 消融）、DPO vs PPO 決策樹與失效排查 |
| `11_modern_preference_simpo_remax_kto.md` | 🔥 | 卸下沉重行囊徒步者心智模型、彈簧門檻與勝出邊界 γ、字數灌水對抗、漸進式 5 階實驗室（長度作弊對抗、邊界飽和、KTO 展望理論消融）、4D 遙測雷達與面試手撕 |
| `02_rewards.md` | | 自動閱卷機 vs 印象分考官心智模型、剝洋蔥符號標準化、天平砝碼退火、漸進式 5 階實驗室（標籤轟炸與空想欺詐防禦、動態退火消融）、4D 遙測雷達與面試手撕 |

### Pillar II · 工業級分佈式系統、顯存與推論極限 (Systems & Scale)
| File | Hotspot | Responsibility / Competency Focus |
|---|:---:|---|
| `10_distributed_systems_verl_vllm.md` | 🔥 | 雙模態負載矛盾、雙離合變速箱心智模型、3D-HybridEngine 零拷貝重分片 (<800ms)、漸進式 5 階實驗室（木桶短板與死鎖復現、FP8 KV 消融）、4D 遙測雷達與面試手撕 |
| `15_lora_qlora_peft.md` | 🔥 | 全量微調顯存牆、透明描圖紙心智模型、NF4 資訊理論等分位數證明、雙重量化 DQ、漸進式 5 階實驗室（4-bit 合併精度漂移復現、70B 單卡心算）、4D 遙測雷達與面試手撕 |
| `04_training.md` | | 拼圖顯存精算心智模型、透明描圖紙 LoRA、水庫調洪梯度累積、漸進式 5 階實驗室（OOM 崩潰復現、4-bit NF4 量化消融）、4D 遙測雷達與面試手撕 |
| `16_inference_optimization_quantization_compilation.md` | | Roofline 雙階段物理瓶頸、AWQ 激活感知 4-bit 量化、投機解碼無損拒絕採樣數學證明、漸進式 5 階實驗室（自適應 K 步長與熔斷調度）、MLA / PagedAttention 內存革新 |

### Pillar III · 數據飛輪、冷啟動與過程監督 (Data Flywheel & Reasoning)
| File | Hotspot | Responsibility / Competency Focus |
|---|:---:|---|
| `09_sft_cold_start.md` | 🔥 | 冷啟動死局物理成因、Prompt -100 標籤因果遮蔽、Goldilocks 甜蜜區 Pass@8 躍遷、漸進式 5 階實驗室（未遮蔽洩漏與長度截斷災難復現、嚴格 XML 淨化器）、Token 熵遙測預警 |
| `01_data.md` | | SFT 模仿 vs RLVR 探索心智模型、火車掛鈎 Left-Padding、Goldilocks 甜蜜區、漸進式 5 階實驗室（Prompt 遮蔽與格式殘損修復）、4D 遙測雷達與面試手撕 |
| `12_process_supervision_and_test_time_compute.md` | | ORM 信用分配坍塌、Math-Shepherd 蒙特卡羅自動打標、PRM 均值聚合偽陽性陷阱、漸進式 5 階實驗室（Best-of-N 重排、雙驗證器混合架構）、Test-Time 算力擴展定律 |
| `13_data_flywheel_and_decontamination.md` | | 五階段數據飛輪閉環、Magpie 自發提問機制、AST 安全沙箱執行、13-Gram 測試集去污染金標、漸進式 5 階實驗室（同義改寫逃逸防禦、MinHash LSH 去重）、Llama 3 六輪混合排程 |
| `06_agentic_rlvr.md` | | 多輪 ReAct 交互閉環、Docker/Firecracker 沙箱隔離、Observation 軌跡掩碼生死線、防刷分步數懲罰、漸進式 5 階實驗室（空轉刷分攻擊防禦、解耦 Actor-Learner 架構） |

### Pillar IV · 線上急救、評估指標與前沿系統設計 (Triage & System Design)
| File | Hotspot | Responsibility / Competency Focus |
|---|:---:|---|
| `14_post_training_systems_and_triage_playbook.md` | 🔥 | 64x H100 集群 70B 模型系統設計、3D-HybridEngine 拓撲精算、SimPO 損失手撕、超幾何無偏 Pass@k 估計、漸進式 5 階實驗室（策略熵崩潰與梯度爆炸秒級急診）、Apple MLE 協同架構 |
| `05_evaluation.md` | | Pass@1 貪婪 vs Pass@k 探索天花板、超幾何無偏估計量組合數學推導、Majority@k 自洽表決、漸進式 5 階實驗室（Reward Hacking 泛化斷層復現、Bootstrap 95% 置信區間）、TPCS 推理經濟學 |
| `08_ablations_and_scaling.md` | | KL 散度 β 掃描、組大小 G 零梯度浪費率定理、Goldilocks 課程學習信息熵最大化、LR 尺度法則、漸進式 5 階實驗室（3-Seed 多種子防偽協議、自適應 KL 門禁） |
| `17_llm_evaluation_benchmarking_prompt_optimization.md` | | 四維評估矩陣、LLM-as-a-Judge 三大偏差消除、G-Eval 機率加權連續評分、漸進式 5 階實驗室（雙向 Swap 評估、動態擾動檢測）、DSPy MIPROv2 提示編譯與 CI/CD 發布門禁 |
| `18_alignment_safety_red_teaming.md` | | PPO vs DPO vs GRPO 對齊穩定性全景、Anthropic 憲政 AI 自我批判與修復、白盒 GCG 與黑盒 PAIR/TAP 自動化紅隊、漸進式 5 階實驗室（困惑度過濾、SmoothLLM 隨機平滑）、三層縱深防禦 |

## Invariants & Rules
- Maintain sequential chapter numbering (01 through 18).
- Strictly adhere to Prime Directive 5: 5-stage progressive executable notebook laboratories with alternating runnable code and console execution outputs (`[Execution Output / ...]`). Single, isolated snippets are strictly forbidden.
- Zero mentions of "UvA" or "UvA-DLC" across the repository.
- Ensure all equations use standard LaTeX notation compatible with web rendering, emphasizing boundary intuition over heavy algebraic derivations.
