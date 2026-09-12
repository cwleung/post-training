# tutorials (rlvr/tutorials)

> Auto-maintained by Agent. Do not edit manually.

## Purpose
An 18-part comprehensive technical curriculum organized into **4 Thematic Technical Pillars** focusing on Reinforcement Learning with Verifiable Rewards (RLVR) and LLM Post-Training engineering in Traditional Chinese (繁體中文). Features the 7-Pillar Learning Standard: Intuitive Mental Models, Architectural Decision Framework, Systems Mechanics & Boundary Intuition, 5-Stage Progressive Executable Notebook Laboratories (with realistic execution outputs), 4D Telemetry Radar Tables, Production Incident Runbooks, and Frontier AI Labs Whiteboard Defense Scenarios.

---

## Kaggle Hands-On Execution Standard (100% Verified)

Every single chapter in this curriculum includes a **self-contained, 5-stage progressive executable notebook laboratory** designed specifically to run seamlessly on **Kaggle Notebooks (GPU T4 x2, P100, or standard CPU)**:

1. **Self-Contained Tensor Computing**: All laboratories rely strictly on standard PyTorch (`torch`, `torch.nn.functional`), Python standard library (`ast`, `re`, `math`, `typing`), and lightweight public tooling. Zero proprietary cluster dependencies or cloud API keys required.
2. **Idempotent Sequential Execution**: Cells are designed for cumulative step-by-step execution in a Jupyter notebook:
   - **Stage 1**: Synthetic Batch / Data Stream Generation (`Environment & Corpus Setup`)
   - **Stage 2**: Core Algorithm / Vectorized Module Implementation (`Whiteboard First Principles`)
   - **Stage 3**: Vectorized Loss & Real-time Telemetry Metrics (`Loss Engine & Monitoring Radar`)
   - **Stage 4**: Pathological Stress Tests & Failure Simulations (`Entropy Collapse, Leakage, Drift, OOM`)
   - **Stage 5**: Industrial Remediation & Comparative Ablation (`Production Guardrails & Verifications`)
3. **Verified Zero-Deadlock / Zero-Crash**: Audited via automated sequential sandbox testing (`full_suite_audit.py`). 100% of blocks execute without runtime exceptions, infinite loops, or memory leaks.

---

## 4 Thematic Pillars & Comprehensive Curriculum Index

### Pillar I · 基礎強化學習、數據工程與驗證器 (Data, Rewards & RLVR Foundations)
| File | Hotspot | Title & English Concept | Hands-On Kaggle Lab Scope | Kaggle Env / VRAM |
|---|:---:|---|---|---|
| `01_data.md` | | **資料準備與格式化**<br>*(GSM8K Data & Formatting)* | SFT 模仿 vs RLVR 探索心智模型、火車掛鈎 Left-Padding 向量對齊、Prompt 遮蔽與格式殘損修復 | GPU T4 / CPU (~0.8GB) |
| `02_rewards.md` | | **獎勵工程與驗證器**<br>*(Reward Engineering & Verifiers)* | 自動閱卷機心智模型、符號標準化清洗、空想欺詐防禦、標籤轟炸攔截、動態權重退火調度 | GPU T4 / CPU (~0.8GB) |
| `03_grpo_algorithm.md` | 🔥 | **GRPO 演算法推導**<br>*(The GRPO Algorithm & Dr. GRPO)* | 零 Critic 顯存架構、同儕標準化 Z-Score 優勢、全零梯度復現、Dr. GRPO 消除長度偏見消融 | GPU T4 / CPU (~1.5GB) |
| `04_training.md` | | **訓練流水線與 LoRA 優化**<br>*(Training Pipeline & LoRA)* | 拼圖顯存精算、水庫調洪梯度累積、Unsloth 內存加速、OOM 崩潰復現、4-bit NF4 量化急救消融 | GPU T4 / CPU (~2.2GB) |
| `05_evaluation.md` | | **評估與統計指標**<br>*(Unbiased Pass@k & Evaluation)* | Pass@1 貪婪 vs Pass@k 探索天花板、超幾何無偏估計量組合數學推導、Reward Hacking 泛化斷層復現、Bootstrap 95% CI | GPU T4 / CPU (~0.9GB) |

### Pillar II · 偏好優化、多輪代理與冷啟動蒸餾 (Preference Optimization & Reasoning Bootstrap)
| File | Hotspot | Title & English Concept | Hands-On Kaggle Lab Scope | Kaggle Env / VRAM |
|---|:---:|---|---|---|
| `06_agentic_rlvr.md` | | **Agentic RLVR 與多輪 Rollouts**<br>*(Tool Sandboxes & Credit)* | 多輪 ReAct 交互閉環、Docker/Firecracker 沙箱隔離、Observation 軌跡掩碼生死線、空轉刷分攻擊防禦與懲罰 | GPU T4 / CPU (~1.4GB) |
| `07_dpo_preference_optimization.md` | 🔥 | **DPO 與偏好優化**<br>*(DPO Preference Optimization)* | 廚房試吃員心智模型、Bradley-Terry 閉式解、三力拔河受力場、概率塌陷與長度作弊復現、SimPO 消融 | GPU T4 / CPU (~1.2GB) |
| `08_ablations_and_scaling.md` | | **消融實驗與擴展分析**<br>*(Scaling Laws & Ablations)* | KL 散度 $\beta$ 掃描、組大小 G 零梯度浪費率定理、Goldilocks 課程學習信息熵最大化、3-Seed 多種子防偽協議 | GPU T4 / CPU (~1.1GB) |
| `09_sft_cold_start.md` | 🔥 | **SFT Cold-Start 階段**<br>*(Reasoning Cold-Start Bootstrap)* | 冷啟動死局成因、Prompt -100 因果遮蔽、長度截斷災難復現、嚴格 XML 格式淨化器、Token 熵預警 | GPU T4 / CPU (~1.2GB) |

### Pillar III · 分佈式系統、現代對齊與過程監督 (Distributed Scale, Modern Preference & PRMs)
| File | Hotspot | Title & English Concept | Hands-On Kaggle Lab Scope | Kaggle Env / VRAM |
|---|:---:|---|---|---|
| `10_distributed_systems_verl_vllm.md` | 🔥 | **分佈式系統與 3D-HybridEngine**<br>*(veRL & vLLM Infrastructure)* | 雙離合變速箱心智模型、Rollout/Learner 權重動態重分片 (<800ms)、木桶短板延遲爆炸復現、FP8 KV 緩存消融 | GPU T4 / CPU (~2.0GB) |
| `11_modern_preference_simpo_remax_kto.md` | 🔥 | **現代偏好優化**<br>*(SimPO, ReMax & KTO)* | 無參考模型 SimPO (節省 50% 顯存)、目標邊界 $\gamma$、長度作弊對抗、邊界飽和、KTO 展望理論消融 | GPU T4 / CPU (~1.0GB) |
| `12_process_supervision_and_test_time_compute.md` | | **過程監督與測試期計算**<br>*(PRMs & Search Compute)* | ORM 信用分配坍塌、Math-Shepherd 蒙特卡羅自動打標、PRM 均值聚合偽陽性陷阱、Best-of-N 重排與雙驗證器混合架構 | GPU T4 / CPU (~1.5GB) |
| `13_data_flywheel_and_decontamination.md` | | **數據飛輪與基準去污染**<br>*(Flywheel & Decontamination)* | 五階段閉環、13-Gram 精確匹配測試集去污染、AST 安全沙箱執行、同義改寫逃逸防禦、MinHash LSH 去重 | GPU T4 / CPU (~1.0GB) |

### Pillar IV · 線上事故排查、推論極限與安全對齊 (Triage Playbook, Systems Limits & Safety)
| File | Hotspot | Title & English Concept | Hands-On Kaggle Lab Scope | Kaggle Env / VRAM |
|---|:---:|---|---|---|
| `14_post_training_systems_and_triage_playbook.md` | 🔥 | **分散式架構與事故急診 Playbook**<br>*(64x H100 Triage & SimPO)* | 64x H100 70B 集群系統設計、3D-HybridEngine 拓撲精算、超幾何無偏 Pass@k、策略熵崩潰與梯度爆炸秒級急診 | GPU T4 / CPU (~1.6GB) |
| `15_lora_qlora_peft.md` | 🔥 | **LoRA, QLoRA 與參數高效後訓練**<br>*(PEFT & Memory Arithmetic)* | 全量微調顯存牆、透明描圖紙心智模型、NF4 資訊理論等分位數、4-bit 權重合併精度漂移復現、70B 單卡心算 | GPU T4 / CPU (~1.8GB) |
| `16_inference_optimization_quantization_compilation.md` | | **推論極限優化**<br>*(AWQ, Speculative Decoding & MLX)* | Roofline 雙階段物理瓶頸、AWQ 激活感知 4-bit 量化、投機解碼無損拒絕採樣數學證明、自適應 K 步長與熔斷調度 | GPU T4 / CPU (~1.5GB) |
| `17_llm_evaluation_benchmarking_prompt_optimization.md` | | **評估基準與提示優化**<br>*(4D Evaluation & DSPy)* | 四維評估矩陣、LLM-as-a-Judge 三大偏差消除、G-Eval 機率加權評分、雙向 Swap 評估、DSPy MIPROv2 提示編譯 | GPU T4 / CPU (~1.3GB) |
| `18_alignment_safety_red_teaming.md` | | **RL 對齊、憲政 AI 與紅隊防禦**<br>*(Constitutional AI & Safety)* | PPO vs DPO vs GRPO 對齊穩定性全景、Anthropic 憲政 AI 自我批判、白盒 GCG 攻擊、困惑度過濾與 SmoothLLM 隨機平滑 | GPU T4 / CPU (~1.5GB) |

---

## Invariants & Rules
- Maintain sequential chapter numbering (`01` through `18`).
- Strictly adhere to Prime Directive 5: 5-stage progressive executable notebook laboratories with alternating runnable code and realistic console execution outputs (`[Execution Output / ...]`). Single, isolated snippets are strictly forbidden.
- Zero mentions of "UvA" or "UvA-DLC" across the repository.
- Ensure all equations use standard LaTeX notation compatible with web rendering, emphasizing boundary intuition over heavy algebraic derivations.
- All code blocks must pass automated cell-by-cell execution in Python 3.9+ without external API dependencies.
