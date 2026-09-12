# Phase 3: Paper Selection & Deep Dive Rationale

This document establishes the rationale for the 10 landmark papers selected for full-text deep reading. Every paper directly addresses a critical operational, algorithmic, or architectural challenge encountered by a **Post-Training Machine Learning Engineer**.

---

## Selected Papers & Rationale Matrix

| # | Paper Title | arXiv ID | Topic Category | Engineering Rationale for Post-Training MLE |
|---|---|---|---|---|
| 1 | **HybridFlow (veRL)** | `2409.19256` | RL Infrastructure & Distributed Systems | Solves the primary post-training throughput bottleneck: decoupling memory-bound rollout generation (vLLM) from compute-bound backward passes (FSDP/Megatron) with 3D-HybridEngine dynamic weight resharding. |
| 2 | **SimPO** | `2405.14734` | Preference Alignment | Eliminates the frozen reference model required by DPO, cutting training VRAM by 50% while incorporating target reward margins and explicit length normalization to prevent verbosity hacking. |
| 3 | **ReMax** | `2310.10505` | Reinforcement Learning Algorithms | Replaces unstable, memory-heavy PPO learned Critic networks with a greedy baseline rollout, slashing GPU memory consumption and training variance on reasoning tasks. |
| 4 | **Math-Shepherd** | `2312.08935` | Process Supervision & Verifiers | Replaces sparse outcome-only rewards with automated step-level credit assignment via Monte Carlo rollouts, eliminating the need for expensive human step annotations. |
| 5 | **KTO** | `2402.01306` | Preference Alignment Formulations | Adapts prospect theory to align LLMs directly on unpaired binary production logs (thumbs up/down) rather than artificial pairwise comparisons. |
| 6 | **ORPO** | `2403.07691` | Monolithic Post-Training | Merges the Supervised Fine-Tuning (SFT) and preference alignment stages into a single objective via odds-ratio penalties, reducing training pipeline complexity and compute overhead. |
| 7 | **The Llama 3 Herd (Sec. 3)** | `2407.21783` | Industrial Alignment Blueprint | The definitive industry case study on multi-round alignment: iterative rejection sampling SFT, DPO, online PPO, data mixture balancing, and safety/helpfulness trade-offs at 405B scale. |
| 8 | **OpenRLHF** | `2405.11143` | Scaled Ray Orchestration | Production engineering framework utilizing Ray to schedule 4 heterogeneous model components (Actor, Critic, Ref, RM) across multi-node GPU clusters with ZeRO-3 and vLLM. |
| 9 | **Magpie** | `2406.08464` | Synthetic Data Synthesis | Exploits autoregressive decoder priors to synthesize millions of high-quality instruction-tuning prompts with zero prompt-engineering cost. |
| 10 | **Scaling Test-Time Compute** | `2408.03314` | Inference-Time Scaling Systems | Quantifies trade-offs between test-time search against verifiers vs. pre-training parameter scaling, defining how post-trained models are deployed and scaled in production. |

---

## Selection Coverage Verification

- **Distributed Systems & Infrastructure**: 2 papers (`veRL`, `OpenRLHF`)
- **Modern Policy & Preference Algorithms**: 4 papers (`SimPO`, `ReMax`, `KTO`, `ORPO`)
- **Process Supervision & Verifiers**: 1 paper (`Math-Shepherd`)
- **Synthetic Data & Industrial Recipes**: 2 papers (`Llama 3`, `Magpie`)
- **Inference Systems & Test-Time Search**: 1 paper (`Scaling Test-Time Compute`)

Total Selected: **10 papers** (meets and exceeds the requirement of ≥8 papers).
Proceeding to full-text deep reading and structured note generation.
