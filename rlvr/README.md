# 🧠 Production Post-Training ML Engineer Handbook & Codebase

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Architecture: veRL / vLLM / FSDP2](https://img.shields.io/badge/Architecture-veRL%20%7C%20vLLM%20%7C%20FSDP2-purple.svg)]()
[![Algorithms: SimPO | ReMax | GRPO | PRM](https://img.shields.io/badge/Algorithms-SimPO%20%7C%20ReMax%20%7C%20GRPO%20%7C%20PRM-orange.svg)]()

A comprehensive, production-grade learning repository and reference implementation designed for mastering the **Post-Training Machine Learning Engineer** role at frontier AI labs and enterprise teams (Meta, OpenAI, Anthropic, xAI, ByteDance Seed, Scale AI, Databricks, Mistral).

Moving beyond introductory textbook reproductions (like toy GRPO loops on GSM8K), this repository explores the **actual distributed systems, modern reference-free alignment algorithms, process supervision (PRMs), automated data flywheels, and interview system design frameworks** deployed in production today.

---

## 🏛️ The 5 Pillars of the Post-Training Stack

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                       THE 5 PILLARS OF POST-TRAINING ML ENGINEERING                         │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                             │
│  [PILLAR 1: DISTRIBUTED INFRASTRUCTURE]         [PILLAR 2: MODERN ALIGNMENT ALGORITHMS]     │
│  • Decoupled Rollout & Learner Clusters         • Reference-Free Tuning (SimPO, ORPO)       │
│  • 3D-HybridEngine Dynamic Resharding (veRL)    • Non-Critic Policy Gradients (ReMax, GRPO) │
│  • PagedAttention & Radix Caching (vLLM/SGLang) • Unpaired Telemetry Alignment (KTO)        │
│  • Sequence Packing & FlashAttention-2          • Length-Hacking & Goodhart's Law Defence   │
│                                                                                             │
│  [PILLAR 3: VERIFICATION & PRMs]                [PILLAR 4: INDUSTRIAL DATA FLYWHEELS]       │
│  • Step-Level Process Reward Models (PRMs)      • Prompt-Free Generation (Magpie)           │
│  • Automated Step Labeling (Math-Shepherd)      • Ephemeral Execution Sandboxes (Docker)    │
│  • Generative Critics & Self-Rewarding LLMs     • 13-Gram Decontamination & MinHash Dedupe  │
│  • Multi-Objective Reward Ensembling            • Curriculum & Dynamic Mixture Scheduling   │
│                                                                                             │
│  [PILLAR 5: AGENTS & TEST-TIME SYSTEMS]                                                     │
│  • Multi-Turn Tool-Use & Function Calling Alignment (SWE-bench)                             │
│  • Test-Time Compute Scaling & PRM-Guided Tree Search (MCTS, Beam Search)                   │
│  • Production Incident Triage & System Design Interview Playbook                            │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📚 Complete Curriculum: 18 In-Depth Chapters (5 Progressive Stages)

### Stage I: 數據管線與規則驗證器 (Data & Rule-Based Verifiers)
- 🎯 **里程碑**: 搭建零污染數學校驗與無 Critic 組內優勢引擎
- 💼 **對齊崗位**: Post-Training MLE (Verifiers & RL Core)
- **[01. Data Preparation & Formatting](tutorial/01_data.md)**: GSM8K loading, formatting, token validation, prompt schemas.
- **[02. Reward Engineering & Verifiers](tutorial/02_rewards.md)**: Deterministic verifiers, format penalties, reward composition, hacking prevention.
- **[03. The GRPO Algorithm](tutorial/03_grpo_algorithm.md)**: Full mathematical derivation, REINFORCE connection, token-level KL, DAPO, Dr. GRPO.

### Stage II: 輕量微調與無偏統計評估 (LoRA Fine-Tuning & Unbiased Metrics)
- 🎯 **里程碑**: 跑通 LoRA 訓練迴圈並產出無偏 pass@k 統計置信區間
- 💼 **對齊崗位**: Evaluation & Benchmarking Engineer (Anthropic / OpenAI)
- **[04. Training Pipeline & LoRA Optimization](tutorial/04_training.md)**: Unsloth + LoRA + GRPOTrainer, GPU memory optimization.
- **[05. Evaluation & Statistical Metrics](tutorial/05_evaluation.md)**: Unbiased pass@k, majority@k, bootstrap confidence intervals, contamination analysis.
- **[06. Agentic RLVR & Multi-Turn Rollouts](tutorial/06_agentic_rlvr.md)**: Tool calling, sandboxed code execution, intermediate credit assignment.

### Stage III: 偏好對齊、消融實驗與測試期計算 (DPO, SimPO & Test-Time Compute)
- 🎯 **里程碑**: 掌握無參照對齊與測試期計算擴展定律
- 💼 **對齊崗位**: Alignment & Reasoning Researcher (xAI / DeepMind)
- **[07. DPO & Preference Optimization](tutorial/07_dpo_preference_optimization.md)**: Bradley-Terry derivation, DPO vs PPO vs GRPO, failure modes.
- **[08. Ablations & Scaling Analysis](tutorial/08_ablations_and_scaling.md)**: $\beta$ coefficient, group size $G$, learning rate sensitivity, scaling curves.
- **[09. SFT Cold-Start Stage](tutorial/09_sft_cold_start.md)**: Cold-start problem, reasoning demonstrations, distillation, multi-stage pipelines.

### Stage IV: 分佈式 veRL 架構與過程監督 PRM (Distributed veRL & Process Reward Models)
- 🎯 **里程碑**: 架構解耦 Rollout/Learner 集群並實現 Math-Shepherd 步驟級標註
- 💼 **對齊崗位**: Distributed Post-Training Infrastructure Engineer (Meta / ByteDance / Scale AI)
- **[10. Distributed Systems — veRL, vLLM & 3D-HybridEngine](tutorial/10_distributed_systems_verl_vllm.md)**: Decoupled rollout vs learner clusters, dynamic weight resharding (<800ms), sequence packing, KV cache sizing math.
- **[11. Modern Preference Optimization — SimPO, ReMax & KTO](tutorial/11_modern_preference_simpo_remax_kto.md)**: Reference-free SimPO (50% VRAM cut), critic-free ReMax, unpaired KTO telemetry loss, verbosity hacking defense.
- **[12. Process Supervision & Test-Time Compute](tutorial/12_process_supervision_and_test_time_compute.md)**: Process Reward Models (PRMs), Math-Shepherd automated Monte Carlo step labeling, Best-of-N reranking, test-time scaling laws.
- **[13. Industrial Data Flywheels & Decontamination](tutorial/13_data_flywheel_and_decontamination.md)**: Magpie prompt-free synthesis, execution sandboxing, 13-gram benchmark decontamination, dynamic mixture scheduling.

### Stage V: 企業級推論優化、安全紅隊與 Apple MLE 系統設計 (Enterprise Inference, Red-Teaming & System Design)
- 🎯 **里程碑**: 構建量化投機推理、4D 評估矩陣、自動化紅隊防線並完成前沿大廠系統架構設計
- 💼 **架構目標**: Post-Training MLE & Evaluation Architect / Distributed Systems Scientist
- **[15. LoRA & QLoRA PEFT Engineering](tutorials/15_lora_qlora_peft.md)**: NF4 資訊理論推導、雙重量化 (DQ)、分頁優化器與精確顯存公式。
- **[16. Production Inference Optimisation](tutorials/16_inference_optimization_quantization_compilation.md)**: Roofline 算力邊界、AWQ/GPTQ、投機解碼嚴格無損證明、KV-Cache 與 Apple MLX 統一架構。
- **[17. LLM Evaluation & Benchmarking Strategies](tutorials/17_llm_evaluation_benchmarking_prompt_optimization.md)**: 4D 評估矩陣 (Accuracy/Latency/Safety/Cost)、G-Eval 機率加權評分、MIPROv2 超參數提示詞優化。
- **[18. Safety Alignment & Red-Teaming Methodologies](tutorials/18_alignment_safety_red_teaming.md)**: PPO vs DPO vs GRPO 對齊評比、憲政 AI 自批判修訂、GCG 梯度後綴與 TAP 自動化紅隊防禦邊界。
- **[14. Post-Training Systems & Triage Playbook](tutorials/14_post_training_systems_and_triage_playbook.md)**: 跨大廠分散式系統架構設計、白板推導、生產事故排查與現場實戰。

---

## 🏆 Featured Kaggle Showcase Project

Looking for a publication-ready portfolio project for Kaggle? Check out **[`kaggle_showcase/`](kaggle_showcase/)**:

- **Notebook**: [`kaggle_showcase/post_training_mle_showcase.ipynb`](kaggle_showcase/post_training_mle_showcase.ipynb)
- **Publication Guide**: [`kaggle_showcase/README.md`](kaggle_showcase/README.md)
- **1-Command Deploy**: `kaggle kernels push -p kaggle_showcase/` (via [`kaggle_showcase/kernel-metadata.json`](kaggle_showcase/kernel-metadata.json))
- **Key Technical Demonstrations**: Decoupled veRL systems, Sequence Packing (`cu_seqlens`), ReMax critic-free policy gradients, SimPO (no DPO!), Math-Shepherd PRMs, 4-panel visual analytics dashboard, and interactive reasoning playground!

---

## 📓 Interactive Kaggle Notebooks (Free T4 GPU, 16GB VRAM)

| Notebook | Focus & Modern Stack | Target Accelerator |
|---|---|---|
| **[`kaggle_showcase/post_training_mle_showcase.ipynb`](kaggle_showcase/post_training_mle_showcase.ipynb)** | **🏆 Grandmaster Showcase**: Complete production suite with 4-panel visual analytics dashboard, ReMax, SimPO, PRM & interactive playground. | Kaggle T4 (16GB) |
| **[`notebooks/post_training_mle_kaggle.ipynb`](notebooks/post_training_mle_kaggle.ipynb)** | **The Modern Post-Training MLE Stack**: veRL decoupled rollout architecture, sequence packing (`cu_seqlens`), ReMax critic-free policy gradients, SimPO reference-free preference loss, Math-Shepherd PRMs, and Magpie data flywheel. | Kaggle T4 (16GB) |
| **[`notebooks/rlvr_grpo_kaggle.ipynb`](notebooks/rlvr_grpo_kaggle.ipynb)** | **GRPO Reasoning Foundations**: End-to-end training of Qwen2.5 on GSM8K with verifiable format and correctness rewards. | Kaggle T4 (16GB) |
| **[`notebooks/agentic_rlvr_grpo_kaggle.ipynb`](notebooks/agentic_rlvr_grpo_kaggle.ipynb)** | **Agentic Multi-Turn RLVR**: Tool-use, sub-agent delegation, and environment sandboxing with GRPO. | Kaggle T4 (16GB) |

---

## 💻 Interactive Notebook Implementations

All algorithms and systems are self-contained, heavily documented, and runnable directly on Kaggle T4 GPUs (16GB VRAM) or local Jupyter:

| Notebook | Focus & Modern Stack | Location |
|---|---|---|
| **Grandmaster Showcase** | veRL Decoupled Architecture, ReMax, SimPO, PRM & Interactive Playground | [`kaggle_showcase/post_training_mle_showcase.ipynb`](kaggle_showcase/post_training_mle_showcase.ipynb) |
| **Modern Post-Training MLE Stack** | Sequence Packing (`cu_seqlens`), SimPO, Math-Shepherd PRMs & Magpie Flywheel | [`notebooks/post_training_mle_kaggle.ipynb`](notebooks/post_training_mle_kaggle.ipynb) |
| **GRPO Reasoning Foundations** | End-to-end Qwen2.5 on GSM8K with verifiable format and correctness verifiers | [`notebooks/rlvr_grpo_kaggle.ipynb`](notebooks/rlvr_grpo_kaggle.ipynb) |
| **Agentic Multi-Turn RLVR** | Tool calling, subagent delegation, sandboxed Python runner & multi-turn credit | [`notebooks/agentic_rlvr_grpo_kaggle.ipynb`](notebooks/agentic_rlvr_grpo_kaggle.ipynb) |

---

## ⚡ Quickstart & Interactive Verification

Launch the web platform and explore the interactive 18 chapters and visual simulations:

```bash
# Start the unified FastAPI backend & interactive RTD reader
python3 serve.py
# Open in browser: http://127.0.0.1:8000/#guide?site=rlvr
---

## 🎯 Systems Architecture & Production Playbook

For designing and operating frontier post-training systems, prioritize:

1. **System Design**: Architect a distributed online RL pipeline across 64+ GPUs. Study [Chapter 10](tutorials/10_distributed_systems_verl_vllm.md) and [Chapter 14](tutorials/14_post_training_systems_and_triage_playbook.md).
2. **Algorithm Implementation**: Code SimPO loss ([Chapter 11](tutorials/11_modern_preference_simpo_remax_kto.md)), unbiased pass@k ([Chapter 05](tutorials/05_evaluation.md)), and ReMax advantages.
3. **Production Triage**: Diagnose and resolve CUDA OOMs during rollout, verbosity hacking in DPO/GRPO, and policy entropy collapse. Study Section 14.4 in [Chapter 14](tutorials/14_post_training_systems_and_triage_playbook.md).

---

## 📄 License & Citations

Licensed under the MIT License. If using this codebase for your post-training studies or production workflows, cite the underlying foundational papers compiled in [`phase6_report/references.bib`](deep-research-output/post-training-ml-engineer/phase6_report/references.bib).
