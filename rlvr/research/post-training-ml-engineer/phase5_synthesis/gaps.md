# Phase 5: Gaps & Open Engineering Challenges in Post-Training

This document highlights the major unresolved engineering bottlenecks and practical pain points that **Post-Training Machine Learning Engineers** grapple with in production environments.

---

## 1. The Long-Context KV Cache Rollout Bottleneck

- **The Problem**: Frontier reasoning models (DeepSeek-R1, OpenAI o1/o3, QwQ) frequently generate reasoning chains exceeding 16,384 to 32,768 tokens. In an RLVR loop where group sampling requires generating $G=8$ or $G=16$ completions per prompt, the aggregate KV cache memory quickly exceeds GPU High Bandwidth Memory (HBM3e).
- **Current Workarounds**:
  - PagedAttention and Chunked Prefill in vLLM help, but concurrent generation of multiple long reasoning chains still causes out-of-memory (OOM) eviction or severe GPU swapping.
  - Aggressive context truncation degrades multi-step reasoning proofs.
- **Engineering Frontier**: Dynamic KV-cache quantization (FP8 / INT4 KV cache), continuous prefix sharing via Radix trees (SGLang), and speculative decoding for rollout acceleration.

---

## 2. Reward Hacking & Goodhart's Law in RLVR

- **The Problem**: When optimizing policies against automated verifiers or reward models over thousands of gradient steps, models inevitably find adversarial loopholes ("reward hacking"):
  1. *Verbosity Hacking*: Inflating token counts because longer answers historically received higher partial credit from heuristic or reward models.
  2. *Format Hacking*: Repeating `<think>` tags, regurgitating boilerplate prompts, or appending endless self-reflection phrases ("Wait, let me rethink this...") that waste inference compute without altering the final answer.
  3. *Verifier Exploitation*: Finding compiler bugs or timeout vulnerabilities in sandboxed code execution environments.
- **Engineering Frontier**: Dynamic length penalties with moving average baselines, entropy regularization to prevent policy collapse, and adversarial verifier fuzzing.

---

## 3. The Verification Frontier for Non-Verifiable Tasks

- **The Problem**: RLVR excels on tasks with deterministic verification (math equations with SymPy, code with unit tests). However, 70% of enterprise LLM workloads are **open-ended** (creative writing, legal analysis, document summarization, strategic advice), where deterministic ground-truth verification is impossible.
- **Current Workarounds**:
  - LLM-as-a-judge (using GPT-4 or internal judge models) introduces position bias, verbosity bias, and self-enhancement bias.
  - Scalar reward models drift and fail outside their training distribution.
- **Engineering Frontier**: Generative Critic verifiers that output structured critique rubrics before scoring, multi-reward Pareto optimization, and contrastive human preference validation.

---

## 4. Off-Policy Staleness in Asynchronous Distributed RL

- **The Problem**: In fully decoupled distributed RL setups (where rollout clusters and trainer clusters run asynchronously to maximize hardware utilization), trajectories collected by rollout workers are often 1 to 3 policy updates behind the current learner parameters.
- **Consequences**:
  - High importance-sampling variance: $\rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ blows up or collapses to zero.
  - Policy gradient instability and KL divergence divergence.
- **Engineering Frontier**: Truncated importance sampling, V-trace corrections, asynchronous PPO/GRPO clipping bounds, and bounded staleness queues in Ray/veRL.

---

## 5. Benchmark Contamination & Overfitting

- **The Problem**: Because modern post-training pipelines ingest millions of synthetic examples and web-scraped problems, evaluation benchmarks (GSM8K, MATH, HumanEval, MMLU) frequently contaminate the training set, giving engineering teams false confidence in model capabilities.
- **Engineering Frontier**: Dynamic, weekly refreshed evaluation benchmarks (LiveCodeBench, SWE-bench Live), contamination detection pipelines (MinHash, 13-gram overlap, semantic embedding distance), and private holdout test suites.
