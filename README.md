# DeepAgents & Post-Training MLE / RL Unified Platform

[![Live Platform](https://img.shields.io/badge/Live%20Platform-GitHub%20Pages-2ea44f?style=for-the-badge&logo=github)](https://cwleung.github.io/post-training/)
[![Frontend](https://img.shields.io/badge/Frontend-React%2019%20%7C%20Vite%20%7C%20Tailwind%20v4-blue?style=for-the-badge&logo=react)](frontend/)
[![Backend](https://img.shields.io/badge/Backend-FastAPI%20%7C%20Python%203.11+-009688?style=for-the-badge&logo=fastapi)](web/)
[![Curriculum](https://img.shields.io/badge/Curriculum-67%20Chapters%20%7C%2018%20Simulation%20Labs-purple?style=for-the-badge)]()

A comprehensive, production-grade research and interactive learning platform combining **Agent Harness Engineering**, **EvalFramework Benchmarking & Self-Correction Loops**, and **Frontier Reinforcement Learning & Post-Training Alignment (GRPO / PPO / DPO / SimPO / PRMs)**.

---

## 🌐 Live Interactive Platform

Experience the full interactive web application directly in your browser with zero installation:

👉 **[https://cwleung.github.io/post-training/](https://cwleung.github.io/post-training/)**

* **3 Comprehensive Tracks (67 Chapters)**: DeepAgents, Research Reinforcement Learning, and Production Post-Training MLE.
* **18 Real-Time Parameter Simulation Labs**: Interactive visualizations for CartPole dynamics, PPO clipping, GRPO group advantage, DPO preference dynamics, LoRA VRAM memory estimation, and Token Logits inspection running client-side.
* **Line-by-Line Code Inspector**: Inspect production PyTorch and DeepAgents source code with contextual annotations, symbol lookups, and one-click copy.
* **ReadTheDocs / Jupyter UI/UX**: Dual dark/light theme, fuzzy search (`⌘K` / `Ctrl+K`), and responsive collapsible tree navigation.

---

## 🚀 Quick Start (Local Development)

### Option 1: Full-Stack FastAPI Server
Launches the FastAPI backend serving both REST APIs and the compiled React 19 single-page application:

```bash
# 1. Install backend dependencies
pip install -r web/requirements.txt

# 2. Launch server (defaults to port 8000)
python3 serve.py
# Or using uvicorn directly:
# uvicorn web.app:app --reload --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser. The interactive REST API catalog is available at `/api`.

### Option 2: Frontend Vite Development Server
For rapid UI iteration with Hot Module Replacement (HMR):

```bash
cd frontend
npm install
npm run dev
```
Open **`http://localhost:3000`** in your browser.

### Option 3: Production Build
```bash
cd frontend
npm run build
```
Compiles assets into `web/dist/` with automated chunk splitting and raw markdown bundling.

---

## 🏛️ Repository Architecture

```
deepagents/
├── tutorials/                 # Track 1: 31 DeepAgents Harness Engineering & Evaluation Chapters
├── rl/                        # Track 2: Research RL & In-Context Policy Learning (GRPO, PPO, CQL, SAC)
│   ├── tutorials/             # 18 Rigorous RL Tutorials with PyTorch implementations
│   ├── demo_log_to_grpo.py    # Log-to-GRPO pipeline scaffolding
│   ├── policy_feedback.py     # In-context policy feedback & anomaly detection
│   └── training_pipeline.py   # Distributed RL training loop architecture
├── rlvr/                      # Track 3: Production Post-Training MLE & Verifiable Rewards (RLVR)
│   ├── tutorials/             # 18 Advanced Post-Training & Distributed Systems Chapters
│   ├── notebooks/             # Kaggle / Colab exploration notebooks
│   └── kaggle_showcase/       # Grandmaster-grade showcase pipelines
├── frontend/                  # React 19 + Vite 6 + Tailwind CSS v4 Single Page Application
│   └── src/
│       ├── entities/chapter/  # Dynamic chapter data models & raw markdown bundling
│       ├── entities/simulation/# 18 Interactive simulation lab visualizers
│       ├── entities/manifest/ # Track curriculum registries (deepagents, rl, rlvr)
│       └── widgets/           # Code inspector, reader canvas, and sidebar navigation
├── web/                       # Unified FastAPI backend service (app.py, routers/)
├── docs/                      # Centralized system architecture, API specifications & contracts
└── .agents/                   # Antigravity agent configuration, rules, and workspace skills
```

---

## 📚 Curriculum Tracks (67 Chapters)

### Track 1: DeepAgents — Harness Engineering & Evaluation Framework (31 Chapters)

Moving beyond naive API wrapper wrappers, this track teaches engineers how to construct the operating system beneath autonomous agents: the **execution harness**.

* **Module I: Agent Architecture & Reactive Loops (Ch 01–05)**
  * **[01. Harness Engineering Basics](tutorials/01-harness-engineering-basics.md)**: Harness vs. framework vs. runtime; the four capability pillars; `create_deep_agent` surface.
  * **[02. The DeepAgents Stack](tutorials/02-the-deep-agents-stack.md)**: Middleware ordering, minimal vs. full stacks, replacement vs. insertion merge semantics.
  * **[03. Custom Middleware](tutorials/03-custom-middleware.md)**: `wrap_tool_call`, `AgentMiddleware` hooks, stack execution rules.
  * **[04. Backends & Permissions](tutorials/04-backends-and-permissions.md)**: Virtual filesystems, State/Store/Composite backends, `FilesystemPermission` rules.
  * **[05. Harness Profiles](tutorials/05-harness-profiles.md)**: Provider/model profiles, `excluded_tools`, prompt overrides, subagent constraints.

* **Module II: Multi-Agent Systems & Human-in-the-Loop (Ch 06–10)**
  * **[06. Subagents & Delegation](tutorials/06-subagents-and-delegation.md)**: Task tool delegation, declarative `SubAgent`, structured output schemas.
  * **[07. Context Management](tutorials/07-context-management.md)**: Runtime context, memory compaction, prompt caching, context offloading.
  * **[08. Human-in-the-Loop (HITL)](tutorials/08-human-in-the-loop.md)**: State machine interrupts, decision types, approval flows, replay tokens.
  * **[09. Putting It Together](tutorials/09-putting-it-together.md)**: Building a production-grade research agent with sandboxing and permissions.
  * **[10. Middleware Deep Dive](tutorials/10-middleware-deep-dive.md)**: Onion-model ordering, `ExtendedModelResponse` state writes, `jump_to` guardrails.

* **Module III: Trajectory Evaluation & Safety Guardrails (Ch 11–16)**
  * **[11. Trajectory Evaluation](tutorials/11-trajectory-evaluation.md)**: Deterministic trajectory matching, strict/unordered assertions, error recovery.
  * **[12. Runtime Rubric Grading](tutorials/12-rubric-grading.md)**: `RubricMiddleware` self-critique, referee subagents, evidence gathering.
  * **[13. End-to-End Benchmarking](tutorials/13-end-to-end-benchmarking.md)**: Virtual filesystem side-effect assertions, batch test suites, test pyramids.
  * **[14. Event Streaming & Telemetry](tutorials/14-event-streaming.md)**: Typed Projections real-time streaming, subagent lifecycle cards, SSE adapters.
  * **[15. Multi-Agent Failure Taxonomy](tutorials/15-multi-agent-failure-taxonomy.md)**: MAST 14-mode failure taxonomy, delegation loops, `pass@k` vs. `pass^k`.
  * **[16. Security Red-Teaming](tutorials/16-security-red-teaming.md)**: Indirect prompt injection, tool poisoning, path traversal defenses, ASR metrics.

* **Module IV: Verifiers, Reward Modeling & Judge Alignment (Ch 17–21)**
  * **[17. Eval as an RL Environment](tutorials/17-eval-as-rl-environment.md)**: Verifier design, Goldilocks difficulty calibration, environment decomposition.
  * **[18. LLM-as-Judge & Alignment](tutorials/18-llm-as-judge-alignment.md)**: Cohen's Kappa calibration, position bias mitigation, policy-aware weights.
  * **[19. Eval Infrastructure & CI](tutorials/19-eval-infrastructure.md)**: Living dataset flywheels, Eval-Driven Development (EDD), CI quality gates.
  * **[20. Harness as Context Engineering](tutorials/20-harness-as-context.md)**: Model/Harness/Skill tri-part decomposition, context budget management.
  * **[21. Benchmark Integrity](tutorials/21-benchmark-integrity.md)**: Contamination detection, saturation limits, Goodhart's law defenses.

* **Module V: Production Observability, Debugging & Token Economics (Ch 22–27)**
  * **[22. Observability: Five Evaluation Surfaces](tutorials/22-observability-five-surfaces.md)**: Output, trace, memory, environment, and mechanistic telemetry with OpenTelemetry.
  * **[23. Error Analysis](tutorials/23-error-analysis.md)**: Open coding, axial coding, Pareto frontier analysis, root-cause triage.
  * **[24. Multi-Turn & Long-Horizon Evals](tutorials/24-multi-turn-evaluation.md)**: METR time horizons, session isolation, $\tau^2$-Bench dual control.
  * **[25. RAG Evaluation](tutorials/25-rag-evaluation.md)**: RAG Triad, claim-level evaluation, separating retriever vs. generator failures.
  * **[26. Production Monitoring & Distribution Drift](tutorials/26-production-monitoring.md)**: KL divergence drift detection, eval flywheels, trace diffing.
  * **[27. Cost & Latency Metrics](tutorials/27-cost-latency-metrics.md)**: Pareto frontier optimization, cost decomposition, latency budget slicing.

* **Module VI: Frontier Closed-Loops & Checkpoint Replay (Ch 28–31)**
  * **[29. Tool-Call Evaluation Stack](tutorials/29-tool-call-evaluation-stack.md)**: L1 Selection, L2 Extraction, L3 Assimilation, L4 Recovery layered assessment.
  * **[30. Checkpoint Replay & Deterministic Debugging](tutorials/30-checkpoint-replay.md)**: Snapshot serialization, step interception, deterministic mock injection.
  * **[31. Self-Improving Evaluation Loops](tutorials/31-self-improving-eval-loop.md)**: Generate $\rightarrow$ Critique $\rightarrow$ Revise architecture, cross-task SkillLibrary.
  * **[28. Runnable Code Index](tutorials/28-runnable-examples.md)**: Comprehensive quick-reference guide for all 22 execution modules.

---

### Track 2: Research Reinforcement Learning (18 Chapters)

Deep mathematical derivations paired with production PyTorch implementations and interactive parameter visualizers:

1. **[01. CartPole Physics & Control](rl/tutorials/01_cartpole_physics_and_control.md)**: 2nd-order Lagrangian mechanics, Euler-Cromer integration, phase-space stability.
2. **[02. Multi-Armed Bandits & Exploration](rl/tutorials/02_bandit_exploration_ucb.md)**: Regret bounds, Hoeffding's inequality, Upper Confidence Bounds (UCB1).
3. **[03. Deep Q-Networks (DQN)](rl/tutorials/03_dqn_replay_target.md)**: Bellman optimality operator, prioritized experience replay, Polyak target updates.
4. **[04. Policy Gradients & REINFORCE](rl/tutorials/04_policy_gradient_reinforce.md)**: Likelihood-ratio trick, score function estimator, baseline variance reduction.
5. **[05. Actor-Critic & GAE](rl/tutorials/05_actor_critic_gae.md)**: Advantage estimation, Generalized Advantage Estimation (GAE-$\lambda$), bias-variance trade-off.
6. **[06. Soft Actor-Critic (SAC)](rl/tutorials/06_sac_continuous_control.md)**: Maximum entropy RL, squashed Gaussian reparameterization, dual Q-networks.
7. **[07. Conservative Q-Learning (CQL)](rl/tutorials/07_cql_offline_rl.md)**: Offline RL distribution shift, LogSumExp conservative lower bound, OOD penalties.
8. **[08. Proximal Policy Optimization (PPO)](rl/tutorials/08_ppo_clipped_objective.md)**: Clipped surrogate objective, pessimistic lower bound, KL penalty dynamics.
9. **[09. Direct Preference Optimization (DPO)](rl/tutorials/09_dpo_implicit_reward.md)**: Bradley-Terry preference modeling, analytical closed-form reward substitution.
10. **[10. Group Relative Policy Optimization (GRPO)](rl/tutorials/10_grpo_group_relative.md)**: Critic-free group normalization, Z-score advantage, DeepSeek-Math dynamics.
11. **[11. 3-Stage RLHF Pipeline](rl/tutorials/11_rlhf_3stage_pipeline.md)**: SFT cold-start, Reward Model training, PPO/GRPO online policy alignment.
12. **[12. Reward Hacking & Pitfalls](rl/tutorials/12_reward_hacking_pitfalls.md)**: Goodhart's law in LLMs, length exploitation, dynamic KL freeze gatekeepers.
13. **[13. Test-Time Compute & Pass@k](rl/tutorials/13_test_time_compute_passk.md)**: Inference-time scaling laws, Chen et al. unbiased combinatorial pass@k estimators.
14. **[14. Process Reward Models (PRMs)](rl/tutorials/14_prm_step_supervision_search.md)**: Step-level credit assignment, Math-Shepherd Monte Carlo rollouts, Beam Search pruning.
15. **[15. In-Context Policy Learning](rl/tutorials/15_in_context_policy_harness.md)**: Fast-loop prompt adaptation, Kairos dual-view event memory, KEEP/AVOID rule synthesis.
16. **[16. SWE-RL & Sandbox Verification](rl/tutorials/16_swe_rl_sandbox_patch.md)**: Ephemeral Docker sandboxes, AST syntax verification, patch penalty defenses.
17. **[17. Multimodal RL & KV Cache Optimization](rl/tutorials/17_vlm_multimodal_rl_kvcache.md)**: Dynamic grid token slicing, long-context KV cache analytical memory footprint.
18. **[18. Frontier Systems Design & Roadmap](rl/tutorials/18_frontier_systems_architecture_alignment.md)**: System design trade-offs, scalability boundaries, and production architectural defense.

---

### Track 3: Production Post-Training MLE & RLVR (18 Chapters)

Industrial distributed training architectures and reference-free preference optimization:

* **Pillar I · Foundational RLVR & Preference Optimization**:
  * The GRPO Algorithm & Dr. GRPO derivation (`rlvr03`)
  * Data curation, synthetic generation & decontamination (`rlvr01`)
  * Deterministic rule-based verifiers & reward engineering (`rlvr02`)
  * DPO implicit reward push-pull dynamics (`rlvr07`)
  * Modern reference-free alignment: SimPO, ReMax, KTO (`rlvr11`)
* **Pillar II · Distributed Infrastructure & Scaling**:
  * Cold-start SFT & curated data flywheels (`rlvr09`)
  * Distributed systems: veRL, vLLM PagedAttention & 3D-HybridEngine resharding (`rlvr10`)
  * LoRA, QLoRA & PEFT memory budgeting with exact VRAM models (`rlvr15`)
  * Inference optimization: AWQ/GPTQ quantization, speculative decoding & FlashAttention (`rlvr16`)
* **Pillar III · Verification, Search & Safety**:
  * Agentic RLVR: multi-turn rollouts, sandboxing & credit assignment (`rlvr06`)
  * Process supervision, PRMs & test-time compute search (`rlvr12`)
  * Automated data flywheels & Magpie promptless generation (`rlvr13`)
  * 4D balanced evaluation matrices (Accuracy, Latency, Safety, Cost) (`rlvr17`)
  * Alignment safety, Constitutional AI (RLAIF) & automated red-teaming (GCG/TAP) (`rlvr18`)
* **Pillar IV · Triage, Evaluation & System Design**:
  * Unbiased pass@k, majority@k & bootstrap confidence intervals (`rlvr05`)
  * Hyperparameter sensitivity, group size ablations & compute curves (`rlvr08`)
  * 64x H100 cluster sizing, fault troubleshooting & system design playbook (`rlvr14`)

---

## 🔬 Interactive Simulation Labs Catalog

All 18 simulation labs run **100% client-side** inside the browser using interactive canvas graphics, real-time numerical step integrators, and reactive parameter sliders:

| Simulation Lab | Key Parameters & Controls | Engineering Focus |
|---|---|---|
| **CartPole Physics** | Mass, length, force, integration timestep | 2nd-order Lagrangian mechanics & phase-space stability |
| **Bandit Exploration** | True bandit means, epsilon, UCB $c$ factor | Exploration-exploitation trade-off & cumulative regret |
| **DQN Replay** | Replay buffer size, batch size, target sync frequency $\tau$ | Overestimation bias & temporal difference targets |
| **Policy Gradient** | Learning rate, reward baseline subtractor | Likelihood ratio gradient estimator variance reduction |
| **Actor-Critic GAE** | Discount factor $\gamma$, GAE parameter $\lambda$ | Advantage estimation bias-variance continuum |
| **SAC Continuous** | Target entropy $\mathcal{H}$, temperature $\alpha$ | Maximum entropy exploration & squashed Gaussian bounds |
| **CQL Offline RL** | Out-of-distribution penalty weight $\alpha$, dataset coverage | Distributional shift & pessimistic value regularization |
| **PPO Clipping** | Epsilon clipping threshold $\epsilon$, advantage $A(s,a)$ | Pessimistic surrogate objective & policy drift control |
| **DPO Dynamics** | Reference penalty $\beta$, implicit reward margin | Push-pull probability dynamics without a reward model |
| **GRPO Inspector** | Group size $G$, Z-score advantage normalization | Critic-free group-relative advantage computation |
| **RLHF 3-Stage** | SFT $\rightarrow$ Reward Model $\rightarrow$ PPO alignment | Pipeline coordination & reward over-optimization |
| **Reward Pitfalls** | Goodhart's law severity, dynamic KL threshold | Length bias exploitation & KL divergence freeze gates |
| **Pass@k Estimator** | Sample count $n$, passing samples $c$, target $k$ | Chen et al. unbiased combinatorial pass@k calculation |
| **PRM Beam Search** | Branching factor $b$, step confidence threshold | Step-level credit assignment & dead-branch pruning |
| **Agentic Loops** | Fast in-context loop vs. slow distillation loop | Kairos dual-view memory & KEEP/AVOID rule synthesis |
| **SWE-RL Sandbox** | Patch line count, test pass rate, patch penalty $\lambda$ | AST validation & benchmark suite deletion prevention |
| **LoRA VRAM Calc** | Model parameters, precision (BF16/NF4), rank $r$, batch size | Exact VRAM memory footprint formula & GPU cluster sizing |
| **Token Logits** | Temperature $T$, Top-P nucleus threshold, Top-K filter | Softmax probability calibration & sampling entropy |

---

## 🛠️ Technology Stack

* **Frontend**: React 19, TypeScript 5.7, Vite 6, Tailwind CSS v4, Zustand 5, KaTeX 0.16, Mermaid 10, Lucide Icons.
* **Backend**: FastAPI, Uvicorn, Pydantic v2, PyTorch, NumPy.
* **Deployment**: GitHub Pages (Static SPA with raw markdown code-splitting via GitHub Actions) + optional containerized FastAPI backend.

---

## 📄 License & Attribution

Distributed under the MIT License. Theoretical derivations and instructional architectures adapt foundational works from OpenAI, DeepSeek, Google DeepMind, Anthropic, Meta, and the LangChain ecosystem.