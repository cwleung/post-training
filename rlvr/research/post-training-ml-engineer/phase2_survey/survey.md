# Phase 2: Comprehensive Literature Survey — The Post-Training ML Engineer Landscape

## 1. Survey Overview & Taxonomy

This survey synthesizes 57 curated papers spanning the foundational architectures, modern systems, and production methodologies that define the modern **Post-Training Machine Learning Engineer** role. 

While research scientists focus on novel theoretical objectives, the **Post-Training ML Engineer** is responsible for:
1. Building high-throughput, low-latency distributed training and rollout pipelines.
2. Managing GPU memory bottlenecks across heterogeneous generation and backward-pass clusters.
3. Designing automated data filtering, synthetic generation, and decontamination systems.
4. Implementing stable policy optimization algorithms that resist length hacking and reward collapse.
5. Deploying and evaluating models against rigorous multi-turn and agentic benchmarks.

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                    THE POST-TRAINING ML ENGINEER TAXONOMY                            │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  [THEME 1: SYSTEMS & INFRASTRUCTURE]         [THEME 2: ALIGNMENT ALGORITHMS]         │
│  • Decoupled Actor-Learner Architecture      • Reference-Free Tuning (SimPO, ORPO)   │
│  • 3D-HybridEngine (veRL / HybridFlow)       • Unpaired Loss (KTO)                   │
│  • KV-Cache & PagedAttention (vLLM)          • Non-Critic Policy Gradients (ReMax)   │
│  • ZeRO-3 / FSDP2 / Megatron-LM Scaling      • Online & Iterative DPO                │
│                                                                                      │
│  [THEME 3: VERIFICATION & REWARDS]           [THEME 4: DATA FLYWHEELS & SYNTHESIS]   │
│  • Process Reward Models (PRMs)              • Prompt-Free Generation (Magpie)       │
│  • Automated Step Supervision (Math-Shepherd)• Multi-Turn Agent Data (UltraInteract) │
│  • Mitigation of Reward Hacking & Drift      • Self-Rewarding LLM Iterations         │
│  • Generative Critics & LLM-as-a-Judge       • Decontamination & Filtering           │
│                                                                                      │
│  [THEME 5: AGENTIC & MULTI-TURN RL]          [THEME 6: TEST-TIME INFERENCE COMPUTE]  │
│  • Software Engineering Agents (SWE-bench)   • Best-of-N Verifier Sampling           │
│  • Multi-Turn Sandbox Environments (Docker)  • PRM-Guided Tree Search (MCTS, Beam)   │
│  • Tool-Call Trajectory Credit Assignment    • Test-Time Scaling vs Parameter Count │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Theme 1: Distributed Post-Training Systems & Infrastructure

Autoregressive token generation during RL rollout is **memory-bandwidth bound**, characterized by dynamic sequence lengths, key-value cache bloat, and small batch sizes. In contrast, policy gradient updates (backward pass) are **compute-bound**, requiring large tensor parallel (TP) and pipeline parallel (PP) partitions.

| System | Key Paper / Source | Architecture Innovation | Scaling Limit | Primary Advantage |
|---|---|---|---|---|
| **veRL (HybridFlow)** | Sheng et al. (2024) `[2409.19256]` | 3D-HybridEngine; decoupled vLLM rollout workers & FSDP2/Megatron learners with dynamic resharding | 671B params | Up to 2.8x throughput over legacy colocation; zero idle memory |
| **OpenRLHF** | Hu et al. (2024) `[2405.11143]` | Ray-based distributed orchestration of Actor, Critic, Ref, and Reward models | 70B+ params | Flexible actor placement; native ZeRO-3 and vLLM integration |
| **vLLM (PagedAttention)**| Kwon et al. (2023) `[2309.06180]` | Virtual memory paging for KV cache | Universal serving | Eliminates 60–80% memory fragmentation during rollouts |

### Engineering Implications for Post-Training MLEs:
- **Resharding Overhead**: When alternating between rollout (TP=4) and training (FSDP=8, TP=2), weights must be transferred via NCCL All-to-All. MLEs must optimize communication bandwidth to keep resharding under 5% of step time.
- **Prefix Caching**: Multi-turn agent rollouts share large system prompts and history prefixes. Enabling chunked prefill and automatic prefix caching in vLLM saves 40%+ compute during rollout generation.

---

## 3. Theme 2: Preference Alignment Beyond Vanilla DPO

Direct Preference Optimization (DPO) revolutionized post-training by bypassing reward model training. However, vanilla DPO has three major engineering flaws:
1. **Memory Inefficiency**: Requires holding a frozen reference model in VRAM alongside the active policy.
2. **Length Hacking**: The implicit reward scales linearly with output token length, causing responses to inflate.
3. **Distribution Shift**: Policy generates text that drifts away from the static offline preference dataset.

### Algorithmic Comparison:

| Algorithm | Formulation / Loss Mechanism | Reference Model Needed? | Length Normalized? | Target Problem Solved |
|---|---|---|---|---|
| **DPO** `[2305.18290]` | $\mathcal{L}_{\text{DPO}} = -\log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)$ | **Yes** (Doubles VRAM) | No | Eliminates PPO training instability |
| **SimPO** `[2405.14734]` | $\mathcal{L}_{\text{SimPO}} = -\log \sigma \left( \frac{\beta}{|y_w|}\log \pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log \pi_\theta(y_l|x) - \gamma \right)$ | **No** (50% VRAM saved) | **Yes** (Length normalized) | Eliminates reference model and stops verbosity hacking |
| **ORPO** `[2403.07691]` | $\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}} + \lambda \mathcal{L}_{\text{odds}}$ | **No** | Yes | Merges SFT and preference alignment into a single training run |
| **KTO** `[2402.01306]` | Maximizes utility directly on unpaired binary labels ($y$ is desirable/undesirable) | Yes | Implicit | Works on real-world logs without requiring paired completions |
| **ReMax** `[2310.10505]` | Policy gradient using greedy rollout $\pi_\theta(x)$ as the baseline $b(x)$ | No critic needed | N/A | Cuts PPO GPU memory by 50% without critic training divergence |

---

## 4. Theme 3: Process Reward Models & Step-Level Supervision

In complex reasoning tasks (math, code, multi-hop retrieval), outcome rewards ($r \in \{0, 1\}$) provide extremely sparse gradient feedback. If a 40-step proof has an arithmetic error on step 38, penalizing all 40 steps degrades the model's early logical reasoning.

### Key Innovations:
1. **Let's Verify Step by Step (PRM800K)** `[2305.20050]`: Demonstrated that step-level supervision ("process supervision") drastically outperforms outcome supervision on reasoning benchmarks.
2. **Math-Shepherd** `[2312.08935]`: Automated the creation of PRM training data without human labeling. For each step $s_t$, the system samples $K$ Monte Carlo rollouts to completion. The empirical pass rate estimates $P(\text{correct} | s_t)$, which forms the soft label for step $t$.
3. **Step-DPO** `[2406.18629]`: Applied direct preference optimization at the step boundary, identifying the exact step where a reasoning trajectory diverged from correct to erroneous.

---

## 5. Theme 4: Industrial Data Flywheels & Synthetic Data Engineering

In modern post-training teams, model architecture is largely standardized (decoder-only Transformers with RoPE and SwiGLU). **Competitive differentiation lies almost entirely in data quality, mixture proportions, and automated synthesis pipelines.**

```
[Raw Pre-trained LLM] 
       │
       ▼
 [Magpie Synthesis] ──► Extracts instructions directly from LLM activations (zero prompt cost)
       │
       ▼
 [Task Diversification & Evol-Instruct] ──► Mutates complexity, depth, and constraints
       │
       ▼
 [Automated Verification / Execution] ──► Sandboxed execution (Python, Bash, Unit Tests)
       │
       ▼
 [Decontamination & Benchmark Filtering] ──► 13-gram overlap + MinHash + Embedding similarity
       │
       ▼
 [Balanced Mixture Scheduling] ──► Dynamic weighting: Reasoning (35%), Code (25%), Chat (25%), Safety (15%)
```

### Core Papers:
- **Magpie** `[2406.08464]`: Shows that aligned models will generate user queries autonomously when fed the beginning-of-turn delimiter with no prompt tokens. Yields high-quality instructions without prompt engineering.
- **UltraFeedback** `[2310.01377]`: Scaled multi-dimensional AI feedback (instruction-following, truthfulness, honesty, style) to train robust reward models.
- **Llama 3 Technical Report (Section 3)** `[2407.21783]`: Documents Meta's multi-round flywheel: Cold-start SFT $\to$ Rejection Sampling $\to$ DPO $\to$ Online PPO $\to$ Repeat.

---

## 6. Theme 5: Agentic & Multi-Turn Trajectory RL

Modern post-training ML engineers must align models that interact with software tools, write files, call APIs, and execute in sandboxes.

- **SWE-bench** `[2310.06770]`: Evaluates LLMs on real-world GitHub issues (repo cloning, environment setup, reproduce bug, write patch, pass unit tests).
- **Credit Assignment in Trajectories**: In agentic workflows, an agent takes 15 turns before running the final test. If the test fails, which action was wrong? Post-training engineers use intermediate execution outputs (stdout/stderr, exit codes, linter errors) as auxiliary rewards.

---

## 7. Theme 6: Inference-Time Scaling & Test-Time Search Systems

Post-training does not end when weights are saved. **Test-time compute scaling** bridges training and production inference:

- **Scaling Test-Time Compute Optimally** `[2408.03314]`: Demonstrates that allocating compute at test time via search against verifiers can produce accuracy gains equivalent to scaling pre-training compute by orders of magnitude.
- **Inference Systems Engineering**: Implementing high-throughput Best-of-N sampling, PRM-guided beam search, and speculative verification at scale.

---

## 8. Phase 2 Verification Checklist

- [x] Curated master database `paper_db.jsonl` containing 57 high-impact papers.
- [x] Structured survey across 6 core technical themes covering systems, algorithms, rewards, data, agents, and test-time compute.
- [x] Identified exact mathematical formulations, memory trade-offs, and systems bottlenecks.
