# tutorials (rlvr/tutorials)

> Auto-maintained by Agent. Do not edit manually.

## Purpose
An 18-part comprehensive technical curriculum organized into **4 Thematic Technical Pillars** focusing on Reinforcement Learning with Verifiable Rewards (RLVR) and LLM Post-Training engineering in Traditional Chinese (繁體中文). Features the 5-Part Learning Context Anatomy: Industry Evolution, Architectural Decision Matrices, Systems Mechanics & Boundary Intuition, Production Telemetry Runbooks, and Frontier Lab Systems Defense.

## 4 Thematic Pillars & Contents

### Pillar I · 基礎強化學習與偏好優化 (Foundational RLVR & Preference)
| File | Hotspot | Responsibility / Competency Focus |
|---|:---:|---|
| `03_grpo_algorithm.md` | 🔥 | PPO 到 GRPO 工業演進、零 Critic 顯存架構、同儕相對優勢與邊界行為直覺 |
| `07_dpo_preference_optimization.md` | 🔥 | Bradley-Terry 偏好演進、隱式獎勵閉式解、UvA-DLC 漸進式 5 階實驗室（概率塌陷與長度作弊復現、SimPO 消融）、DPO vs PPO 決策樹與失效排查 |
| `11_modern_preference_simpo_remax_kto.md` | 🔥 | 無參考模型 SimPO (節省 50% 顯存)、目標邊界 γ 直覺、ReMax 與 KTO 遙測對齊 |
| `02_rewards.md` | | 確定性規則驗證器、格式懲罰、多目標獎勵組合與 Reward Hacking 防護 |

### Pillar II · 工業級分佈式系統、顯存與推論極限 (Systems & Scale)
| File | Hotspot | Responsibility / Competency Focus |
|---|:---:|---|
| `10_distributed_systems_verl_vllm.md` | 🔥 | 解耦 Rollout/Learner 集群、3D-HybridEngine 動態重分片 (<800ms) 與 NCCL 掛起排查 |
| `15_lora_qlora_peft.md` | 🔥 | 全量微調顯存牆、NF4 資訊理論、雙重量化 DQ、分頁優化器與 70B 顯存精確公式 |
| `04_training.md` | | Unsloth + LoRA + GRPOTrainer 整合，GPU 顯存極限優化實戰 |
| `16_inference_optimization_quantization_compilation.md` | | Roofline 模型、GPTQ/AWQ 4-bit 量化、投機解碼無損驗證定理與 MLX 編譯 |

### Pillar III · 數據飛輪、冷啟動與過程監督 (Data Flywheel & Reasoning)
| File | Hotspot | Responsibility / Competency Focus |
|---|:---:|---|
| `09_sft_cold_start.md` | 🔥 | 推理 Cold-Start 困境、長思維鏈合成蒸餾、格式標籤約束與防範模式坍塌 |
| `01_data.md` | | GSM8K 載入清洗、XML 標籤驗證、Left-Padding 自回歸因果遮蔽與 Prompt Schema |
| `12_process_supervision_and_test_time_compute.md` | | 步驟級 PRM 過程監督、Math-Shepherd 自動標籤、Beam Search 與 MCTS 樹剪枝 |
| `13_data_flywheel_and_decontamination.md` | | Magpie 自生成無 Prompt 數據、MinHash 去重、13-gram 基準測試嚴格去污染 |
| `06_agentic_rlvr.md` | | 多輪工具呼叫、沙箱程式碼執行環境、多輪軌跡中間信用分配 |

### Pillar IV · 線上急救、評估指標與前沿系統設計 (Triage & System Design)
| File | Hotspot | Responsibility / Competency Focus |
|---|:---:|---|
| `14_post_training_systems_and_triage_playbook.md` | 🔥 | Frontier AI Labs 64x H100 系統設計精算、白板手撕損失、線上故障秒級急救 |
| `05_evaluation.md` | | 無偏 Pass@k 組合數學、Majority@k、Bootstrap 置信區間與資料集污染檢測 |
| `08_ablations_and_scaling.md` | | Beta 係數、Group 大小 G、學習率敏感度分析與性能擴展曲線 |
| `17_llm_evaluation_benchmarking_prompt_optimization.md` | | 四維度平衡矩陣、G-Eval 機率加權、LLM-as-a-Judge 偏差消除、DSPy 提示編譯 |
| `18_alignment_safety_red_teaming.md` | | PPO vs DPO vs GRPO 對比、Anthropic 憲政 AI、自動化紅隊 GCG/TAP 與三層安全護欄 |

## Invariants & Rules
- Maintain sequential chapter numbering (01 through 18).
- Ensure all chapters strictly conform to the 5-Part Learning Context Anatomy.
- Ensure all equations use standard LaTeX notation compatible with web rendering, emphasizing boundary intuition over heavy algebraic derivations.
