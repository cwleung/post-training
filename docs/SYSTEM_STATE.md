# System State & Synchronization Log (`docs/SYSTEM_STATE.md`)

> Auto-maintained by Agent. Do not edit manually.

This log records every documentation synchronization, bootstrap scan, and file modification event across the repository to guarantee zero documentation drift.

### [2026-09-13] - Top-Left Brand Icon Home Navigation Upgrade
- **Instant Return-to-Home Across All Viewport States**:
  - **Sidebar Brand Header (`SidebarNavigation.tsx`)**:
    - Converted the top-left `ψ` brand logo button and `DeepAgents` brand title from static/secret-counter-only elements into interactive, accessible Home buttons with `cursor-pointer`, sleek hover states (`group-hover:shadow-cyan-500/40`, `group-hover:scale-105`), active click animations, and descriptive `title="返回首頁 (Home)"`.
    - Maintained developer authorization secret trigger on the adjacent `v4` badge and keyboard shortcut `Ctrl+Shift+U`.
    - Clicking the brand logo triggers `handleHomeClick`: invokes `resetToHome()`, clears any active search query or labs-only filter, and closes the mobile drawer if on small screens.
  - **Collapsed State Floating Toolbar (`GuidePage.tsx`)**:
    - Added the sleek `ψ` Home icon button directly at `top-3.5 left-3.5` alongside the `目錄導航` button when the sidebar is collapsed, ensuring users can return home with a single tap in any responsive viewport state without needing to open the navigation drawer first.
  - **Interactive Breadcrumbs (`ReaderCanvas.tsx`)**:
    - Connected the top breadcrumb's `Docs` anchor directly to `resetToHome()`, and the track name anchor directly to `setActiveSite(manifest.id)` for frictionless hierarchical navigation.
  - **State Orchestration (`chapterStore.ts`)**:
    - Added `resetToHome()` action resetting `activeSite` to `deepagents`, `currentChapterId` to `da01`, clearing code line selections and milestone tutorial overlays, updating the URL hash to `#da01`, and smoothly scrolling the reading canvas to the top.
- **Verification**:
  - `npm --prefix frontend run typecheck && npm --prefix frontend run build` completed in 7.11s with 0 errors.

### [2026-09-13] - Post-Training Track ASCII Architectural Diagrams & Intuitive Explanations Expansion
- **Comprehensive ASCII Visual Architecture Expansion Across All 18 Chapters (`rlvr/tutorials/`)**:
  - Implemented high-density, monospaced ASCII architectural diagrams, memory topology maps, timeline sequences, and vector field visualizations enclosed in clean ````text` code blocks across every single chapter.
  - Verified with `scratch/audit_ascii.py`: **100% of chapters equipped with multiple rich ASCII visual charts** (2 to 7 ASCII diagrams per chapter alongside 3 to 4 Mermaid diagrams).
  - **Highlighted Architectural & Systems Mechanics Visualizations**:
    - `01_data.md`: KV-Cache & Attention Pointer Alignment Map (Left vs Right Padding) & Causal Prompt Masking Gradient Map (`labels = -100`).
    - `02_rewards.md`: Multi-Signal Verifier Extraction Pipeline & Dynamic Weight Annealing Timeline ($\lambda_{\text{fmt}} \to 0$).
    - `03_grpo_algorithm.md`: PPO vs GRPO Memory & Topology Comparison Map (0 Critic VRAM) & Dr. GRPO Length-Bias Normalization Mechanism.
    - `04_training.md`: 16GB GPU VRAM Jigsaw Allocation Map & Gradient Accumulation Reservoir Timeline.
    - `05_evaluation.md`: Greedy Pass@1 vs Combinatorial Pass@k Estimation Tree & Majority@k Consensus Voting Cluster.
    - `06_agentic_rlvr.md`: Multi-Turn ReAct Trajectory Token Stream & Observation Masking (`m_t=0`) & Step Efficiency Penalty Curve.
    - `07_dpo_preference_optimization.md`: The Miracle of the Vanishing Partition Function $Z(x)$ & DPO 3-Force Tug-of-War Vector Field.
    - `08_ablations_and_scaling.md`: Group Size $G$ Zero-Gradient Waste Rate Curve ($P_{\text{waste}} = p^G + (1-p)^G$) & Goldilocks Curriculum Variance Peak.
    - `09_sft_cold_start.md`: SFT Cold-Start Goldilocks Zone Phase Transition & Structured XML Reasoning Sandbox.
    - `10_distributed_systems_verl_vllm.md`: veRL + vLLM 3D-HybridEngine Dual-Clutch Gearbox Topology & KV-Cache PagedAttention Block Map.
    - `11_modern_preference_simpo_remax_kto.md`: DPO vs SimPO Memory & Mathematical Topology & KTO Prospect Theory Asymmetric Loss Map.
    - `12_process_supervision_and_test_time_compute.md`: ORM vs PRM Credit Assignment Tree & Test-Time Compute (TTC) Step-Level Beam Search & Pruning Tree.
    - `13_data_flywheel_and_decontamination.md`: 13-Gram Inverted Index Hash Decontamination Engine & MinHash LSH S-Curve Deduplication Filter.
    - `14_post_training_systems_and_triage_playbook.md`: 64x H100 8-Node SuperPOD Interconnect & Online Incident Triage State Machine.
    - `15_lora_qlora_peft.md`: LoRA Low-Rank Decomposition Matrix Multiplication Map & Uniform INT4 vs NF4 Equal-Quantile Bins Map.
    - `16_inference_optimization_quantization_compilation.md`: Roofline Model Operational Intensity Boundary Map & Speculative Decoding Rejection Sampling Pipeline.
    - `17_llm_evaluation_benchmarking_prompt_optimization.md`: G-Eval Logprob Continuous Expectation Curve & Swap Evaluation Position Bias Elimination Pipeline.
    - `18_alignment_safety_red_teaming.md`: Three-Layer Defense-in-Depth Security Perimeter & GCG Discrete Coordinate Gradient Optimization Map.
- **Verification & Zero Drift**:
  - `full_suite_audit.py` passed with 18/18 chapters (100% of cells) green.
  - Zero mentions of "UvA" or "UvA-DLC" verified across all tutorials.
  - `npm --prefix frontend run typecheck && npm --prefix frontend run build` completed in 7.29s with exit code 0.

### [2026-09-13] - Post-Training Track Comprehensive Review, Reindexing & Kaggle Execution Certification
- **Automated Hands-On Kaggle Execution Audit (100% Pass Rate Across All 18 Chapters)**:
  - Audited all code blocks across all 18 chapters (`rlvr/tutorials/01_data.md` through `18_alignment_safety_red_teaming.md`) using automated cumulative cell-by-cell execution runner (`full_suite_audit.py`).
  - **Fixed 4 Critical Runtime Defects & Incoherencies**:
    - `02_rewards.md`: Resolved `TypeError: unsupported operand type(s) for |` on Python 3.9/Kaggle environments by migrating `str | None` to `Optional[str]`.
    - `13_data_flywheel_and_decontamination.md`: Neutralized an unbounded `while True: pass` execution hang in Stage 5 by equipping `StrictASTSecurityInspector` with `visit_While` loop static analysis to intercept infinite loop bombs before sandbox dispatch.
    - `14_post_training_systems_and_triage_playbook.md`: Fixed `RuntimeError: Trying to backward through the graph a second time` by creating a dedicated `p_unstable` tensor for the gradient explosion simulation.
    - `15_lora_qlora_peft.md`: Fixed dimension mismatch `RuntimeError: The size of tensor a (16) must match tensor b (64)` in `simulate_nf4_quantization` by recording `orig_shape` and returning full tensor shape `(64, 64)`.
    - `01_data.md`: Removed unused `from transformers import AutoTokenizer` dependency, eliminating `urllib3` warnings and ensuring pure PyTorch tensor execution.
- **Authoritative Reindexing of `rlvr/tutorials/INDEX.md`**:
  - Reindexed the 18-chapter catalog across the 4 Thematic Pillars and 7 Core Hotspots with explicit Kaggle hands-on execution standards (hardware requirements: GPU T4 x2 / CPU, VRAM footprint $\sim 0.8\text{GB} - 2.2\text{GB}$, and 5-stage progressive lab scopes).
- **Zero-Drift Build & Verification**:
  - `full_suite_audit.py` passed with 18/18 chapters (100% of cells) green.
  - `npm --prefix frontend run typecheck` passed (exit code 0).
  - `npm --prefix frontend run build` completed in 7.31s with zero errors.

### [2026-09-13] - Kaggle Milestone Practice Guide End-to-End Visibility & Copyability Upgrade
- **Continuous End-to-End Visibility by Default (`MilestoneTutorialModal.tsx`)**:
  - Refactored `MilestoneTutorialModal.tsx` tab 1 (`pipeline`) from isolated single-step pagination to a continuous end-to-end view rendering all 4 steps sequentially (Step 1 -> Step 2 -> Step 3 -> Step 4) without requiring users to click through hidden tabs.
  - Added an integrated **⚡ 端到端一體化完整可執行腳本 (Full End-to-End Pipeline Script)** card immediately following Step 4, displaying the complete concatenated, self-contained Python script directly in the modal.
  - Added a View Mode Switcher toolbar allowing toggling between `端到端連續全景` (default), `單步聚焦` (single-step focus), and `完整腳本` (unified Python script only).
  - Added quick-jump anchor pills (`步驟 1`, `步驟 2`, `步驟 3`, `步驟 4`, `⚡ 完整腳本`) with smooth intra-modal scrolling.
- **Granular Copyability Everywhere**:
  - Implemented per-step copy buttons (`[📋 複製步驟 X 代碼]` / `[✓ 已複製步驟 X]`) on each step header and inside each step code block header, with isolated `copiedStepIdx` state and animated checkmark feedback.
  - Implemented one-click master copy buttons on the unified end-to-end script card and modal header (`[📋 一鍵複製端到端完整腳本]` / `[✓ 已複製 4 步驟腳本]`).
- **Data Layer Alias Mapping (`milestoneTutorials.ts`)**:
  - Added explicit alias mapping for Post-Training Track pillars (`rlvr_pillar1` -> `rlvr_stage3`, `rlvr_pillar2` -> `rlvr_stage2`, `rlvr_pillar3` -> `rlvr_stage1`, `rlvr_pillar4` -> `rlvr_stage4`) ensuring rich, bespoke Kaggle tutorials resolve seamlessly across all tracks.
- **Verification & Zero Drift**:
  - `npm --prefix frontend run typecheck` passed cleanly (exit code 0).
  - `npm --prefix frontend run build` built successfully in 7.51s, outputting fresh assets into `web/dist`.
  - Updated `frontend/src/entities/milestone/INDEX.md`.

### [2026-09-13] - Post-Training Track Complete 18-Chapter Flagship & Progressive Notebook Lab Upgrade
- **Universal Rollout of the 7-Pillar Pedagogical Standard Across All 18 Chapters (`rlvr/tutorials/`)**:
  - **Full Scope Upgrades (18/18 Chapters Completed)**:
    - **Pillar I (Foundations & Preferences)**:
      - `03_grpo_algorithm.md`: Peer Review Board vs Critic Guillotine, Z-Score amplifications, 5-stage lab (Zero-gradient recovery, Dr. GRPO length bias ablation).
      - `07_dpo_preference_optimization.md`: Kitchen Metaphor, Elo Rating, Vanishing Partition Function, 3-Force Tug-of-War, 5-stage lab (Likelihood displacement, Verbosity trap, SimPO ablation).
      - `11_modern_preference_simpo_remax_kto.md`: Backpacker shedding weight, Spring threshold margin $\gamma$, 5-stage lab (Length-bias attack, margin saturation, KTO prospect theory ablation).
      - `02_rewards.md`: Automated Scorer vs Subjective Grader, Onion peeling, Dynamic annealing, 5-stage lab (Tag bombing & empty reasoning hacking defense).
    - **Pillar II (Distributed Systems & Inference)**:
      - `10_distributed_systems_verl_vllm.md`: Dual-Clutch Gearbox, 3D-HybridEngine zero-copy resharding (<800ms), 5-stage lab (Straggler latency explosion, FP8 KV ablation).
      - `15_lora_qlora_peft.md`: Tracing Paper on Ancient Scroll, NF4 equal-quantile proof, double quantization, 5-stage lab (4-bit merge precision drift, 70B single-GPU budgeting).
      - `04_training.md`: VRAM 16GB jigsaw puzzle, gradient accumulation reservoir, 5-stage lab (OOM crash simulation, 4-bit NF4 QLoRA remediation).
      - `16_inference_optimization_quantization_compilation.md`: Roofline physical model, AWQ salient channel 4-bit quantization, Speculative Decoding zero-loss rejection sampling equivalence proof, 5-stage lab (Adaptive K horizon governor, circuit breaker fallback).
    - **Pillar III (Data Flywheel & Reasoning)**:
      - `09_sft_cold_start.md`: DeepSeek-R1-Zero cold-start trap, Prompt -100 causal masking, Goldilocks Pass@8 transition, 5-stage lab (Unmasked leakage disaster, strict XML sanitizer).
      - `01_data.md`: Closed-book scratchpad, train coupler left-padding, Goldilocks band, 5-stage lab (Right-padding corruption, Goldilocks filtering).
      - `12_process_supervision_and_test_time_compute.md`: ORM credit collapse, Math-Shepherd Monte-Carlo automatic labeling, PRM Minimum/Product vs Mean Fallacy, 5-stage lab (Best-of-N re-ranking, Dual-verifier hybrid architecture).
      - `13_data_flywheel_and_decontamination.md`: Data flywheel centrifuge, Magpie self-prompt synthesis, Goldilocks 13-gram window, AST security sandbox, 5-stage lab (Paraphrasing escape defense, MinHash LSH deduplication).
      - `06_agentic_rlvr.md`: Single-turn monologue vs multi-turn interactive dialogue, Observation loss mask = 0 safeguard, Step efficiency penalty, 5-stage lab (Infinite tool-loop hacking defense, decoupled Actor-Learner architecture).
    - **Pillar IV (Systems Playbook, Eval & Alignment)**:
      - `14_post_training_systems_and_triage_playbook.md`: 64x H100 70B cluster VRAM budget mental arithmetic, 3D-HybridEngine topology, SimPO whiteboard loss, hypergeometric Pass@k, 5-stage lab (Entropy collapse & gradient blowup triage).
      - `05_evaluation.md`: Pass@1 intuition vs Pass@k latent capability ceiling, hypergeometric unbiased derivation with float overflow protection, Majority@k self-consistency, 5-stage lab (Reward hacking divergence, Bootstrap 95% CI, TPCS token cost economics).
      - `08_ablations_and_scaling.md`: Bungee safety cord ($\beta$ KL), Group size G zero-gradient waste rate theorem, Goldilocks PCL 50% solve rate information entropy maximization, $LR \propto 1/\sqrt{N}$, 5-stage lab (3-seed multi-seed protocol, adaptive KL governor).
      - `17_llm_evaluation_benchmarking_prompt_optimization.md`: 4-D tension matrix (Accuracy, Latency, Safety, Cost), LLM-as-a-Judge 3 biases and dual-direction Swap Evaluation, G-Eval log-probability weighted expectation score, DSPy MIPROv2 Bayesian optimization, 5-stage lab (Production CI/CD gatekeeper).
      - `18_alignment_safety_red_teaming.md`: PPO vs DPO vs GRPO vs SimPO alignment stability landscape, Anthropic Constitutional AI (RLAIF) 2-stage self-critique and revision, GCG discrete coordinate gradient optimization, input perplexity filter, 5-stage lab (FRR over-refusal prevention, 3-layer defense-in-depth, SmoothLLM randomized smoothing).
- **Synchronized Frontend Entity Data (`frontend/src/entities/chapter/data/rlvr/rlvr*.js`)**:
  - Synchronized all 18 chapter data wrappers (`rlvr01.js` through `rlvr18.js`) with refreshed TOC anchors, reading time estimates, summaries, and tags.
  - Auto-maintained `rlvr/tutorials/INDEX.md` with complete 18-chapter mapping and invariants.
- **Strict Invariants Enforced**:
  - Strictly **ZERO mentions of "UvA" or "UvA-DLC"** across all curriculum text, code comments, and documentation. Standard codified as **「漸進式可執行代碼實驗室」(Progressive Executable Notebook Laboratory Standard)**.
  - Every Python code cell is strictly paired with a realistic console execution feedback block (`[Execution Output / Telemetry Log]`).
  - Total tutorial line count expanded to 9,786 lines (~540 lines avg per chapter), matching the exhaustive pedagogical depth of Chapter 07.
- **Verification & Build Status**:
  - `npm --prefix frontend run typecheck` passed cleanly (exit code 0).
  - `npm --prefix frontend run build` passed cleanly (exit code 0), bundling all 18 chapters into optimized production assets.

### [2026-09-13] - Post-Training DPO Intuitive Mental Models Masterclass Upgrade
- **DPO Curriculum Intuitive Pedagogical Overhaul (`rlvr/tutorials/07_dpo_preference_optimization.md`)**:
  - **Section 1 (工業背景)**: Added **「廚房試吃員 vs 主廚自省法」心智模型** (The Restaurant Kitchen Metaphor), explaining why 3-stage RLHF suffers Goodhart collapse (chefs adding gold leaf/verbosity to fool critic models) and how DPO operates as chef self-reflection against ancestral recipes ($\pi_{\text{ref}}$).
  - **Section 3 (核心心智模型與邊界直覺)**:
    - Added **「國際象棋 Elo 等級分與成對博弈」** (The Chess Elo Rating Mental Model), connecting the Bradley-Terry preference probability $\sigma(\Delta r)$ to chess Elo rating differences and shift-invariance.
    - Added **「配分函數的幽靈消去術」** (The Miracle of the Vanishing Partition Function), walking through the 3-step algebraic derivation with plain English explanations and Mermaid flow, demonstrating how the intractable denominator $+ \beta \log Z(x)$ and $- \beta \log Z(x)$ cancel to exactly 0.
    - Added **「三力動態拔河受力場」** (The 3-Force Dynamic Tug-of-War), breaking down the gradient into Dynamic Spring Tension $\sigma(\hat{r}_l - \hat{r}_w)$ (auto-ignoring easy samples, pulling hard on mistakes), Attractor Pull ($+\nabla \log \pi(y_w)$), Repeller Push ($-\nabla \log \pi(y_l)$), and Bungee Leash ($\pi_{\text{ref}}$).
  - **Section 4 (漸進式代碼實驗室)**:
    - Stage 1: Added **「試卷遮蔽心智模型」** (Exam Paper Masking: why `-100` masks the prompt like black tape).
    - Stage 2: Added **「水晶球預言與智慧取物夾心智模型」** (Causal shift crystal ball and `torch.gather` robotic claw picking the exact drawer out of 32,000 possibilities).
    - Stage 3: Added **「成對 Elo 結算盤心智模型」** (Pairwise Elo Board matrix evaluation without python loops).
    - Stage 4.1: Added **「蓋沙堡 vs 踢沙堡非對稱心智模型」** (Sandcastle demolition vs construction explaining Likelihood Displacement).
    - Stage 4.2: Added **「廚房電子秤閱卷心智模型」** (Kitchen Scale essay grading explaining Verbosity Bias).
    - Stage 5: Added **「打樁錨定與密度計處方心智模型」** (Bedrock Piling & Density Meter Remedy).
- **Verification**:
  - `npm --prefix frontend run typecheck` passed (exit code 0).
  - `npm --prefix frontend run build` completed cleanly in 7.84s with zero errors.

---

### [2026-09-13] - Sphinx ReadTheDocs Academic Typography Calibration
- **Reading Body & Typography Calibration (`ReaderCanvas.tsx`, `index.css`)**:
  - Calibrated `.prose` base font size from 16px/14.5px down to authentic Sphinx ReadTheDocs 14px (`line-height: 1.68`) preventing oversized typography bloat.
  - Publication header H1 adjusted from oversized `text-4xl` (36px) to compact academic `text-xl sm:text-2xl md:text-[1.65rem]` (24-26px).
  - Summary font calibrated to `text-[13.5px] sm:text-[14px]` with relaxed line-height.
  - Markdown headings calibrated to clean proportions: H1 (`text-lg sm:text-xl md:text-[1.45rem]`), H2 (`text-base sm:text-lg md:text-[1.2rem]`), H3 (`text-[13.5px] sm:text-[14.5px]`), H4 (`text-[12.5px] sm:text-[13px]`).
  - Body paragraphs (`text-[13.5px] sm:text-[14px]`, `leading-[1.68]`), lists (`text-[13px] sm:text-[13.5px]`, `leading-[1.65]`), table cells (`text-[12px] sm:text-[12.5px]`), inline code (`text-[11.5px] sm:text-[12px]`).
  - Alert callouts (`AlertBlockquote`) styled with compact 12.5px-13px text, 10.5px label badges, and subtle borders.
  - Footers and navigation buttons calibrated with `size="sm"` and `text-xs`.
- **Seamless Jupyter / nbsphinx Code Cells (`CodeBlock`, `CodeInspector`)**:
  - Eliminated visual window chrome, title bars, and OS traffic lights.
  - Implemented authentic Jupyter `In:` (soft sky blue) and `Out:` (soft emerald) prompts with quiet subtitle metadata tags.
  - Floating hover toolbars in top-right corner with zero static vertical height.
  - Replaced text buttons `換行中` / `複製` with minimalist icon toggles (`WrapText`, `Copy` / `Check`).
- **Right Rail TOC & Scrollspy**:
  - Scrollspy-enabled sticky outline with compact `text-[11.5px]` (H2) and `text-[10.5px]` (H3) typography.
- **Verification**:
  - `npm --prefix frontend run typecheck` passed (exit code 0).
  - `npm --prefix frontend run build` completed cleanly in 7.54s with zero errors.

---

### [2026-09-13] - Root README English Modernization & Developer Mode Synchronization
- **Root README English Modernization (`README.md`)**:
  - Replaced Traditional Chinese content with comprehensive English documentation.
  - Added live platform badge and direct link to `https://cwleung.github.io/post-training/`.
  - Detailed the 3 comprehensive curriculum tracks (67 chapters): DeepAgents Harness Engineering (31 chapters), Research RL (18 chapters), and Production Post-Training MLE (18 chapters).
  - Outlined all 18 interactive real-time simulation labs with control parameters and engineering focuses.
  - Provided quick-start instructions for FastAPI full-stack, Vite frontend HMR, and production static bundling.
  - Updated all references and links to reflect sanitized filepaths.
- **Developer Mode & Static Deployment**:
  - Reorganized system architecture modules (`rl/tutorials/18_frontier_systems_architecture_alignment.md` and `rlvr/tutorials/14_post_training_systems_and_triage_playbook.md`).
  - Implemented Developer Mode with cryptographic SHA-256 authentication for advanced diagnostic tabs and architectural blueprints (`frontier-mle-2026`).
  - Hardened `.gitignore` and sanitized documentation against personal data exposure.

---

### [2026-09-13] - Responsive Mobile Drawer & Viewport Resize Auto-Collapse Hardening
- **Dynamic Viewport Synchronization (`frontend/src/pages/guide/GuidePage.tsx`)**:
  - Replaced one-shot mount effect with dynamic `resize` event listener tracking viewport width transitions across the 768px (`md`) boundary.
  - Automatically collapses sidebar drawer (`setSidebarOpen(false)`) when reducing window width to mobile (<768px), seamlessly revealing the reading canvas and floating "目錄導航" / "ψ" controls.
  - Automatically restores user's desktop sidebar preference (`userDesktopPref.current`) when expanding back to desktop (>=768px).
- **Responsive Store State Initialization (`frontend/src/entities/chapter/chapterStore.ts`)**:
  - Initialized `sidebarOpen` dynamically based on client viewport (`window.innerWidth >= 768`), eliminating initial drawer flicker on mobile reloads.
- **Verification**: Ran `npm --prefix frontend run typecheck && npm --prefix frontend run build` (passed with code 0 in 7.77s).

---

### [2026-09-13] - Post-Training Track Comprehensive Code Coherence Audit & Standardization
- **18-Chapter Cumulative Execution & Coherence Audit**:
  - Validated all 18 chapters (`01_data.md` through `18_alignment_safety_red_teaming.md`) against the 5-stage progressive notebook laboratory standard (Synthetic Batch -> Causal Gathering -> Vectorized Loss -> Pathological Stress Test -> Remediation & Comparative Ablation).
  - Validated cumulative sequential execution across all 94 Python cells: 100% execute cleanly top-to-bottom without runtime errors (`C1:OK -> C2:OK -> C3:OK -> C4:OK -> C5:OK`).
  - Standardized Stage headings across chapters `01`, `02`, `03`, `04`, `07`, `10`, `11`, `15` to uniform `### Stage 1:` through `### Stage 5:` headers.
  - Fixed syntax fence issue in `07_dpo_preference_optimization.md` (unclosed mermaid diagram before text fence) and consolidated split Stage 2 and Stage 5 cells, restoring 1:1 code-to-output pairing.
  - Standardized Section 5 & 6 H2 headers in `07_dpo_preference_optimization.md` and synchronized frontend TOC in `frontend/src/entities/chapter/data/rlvr/rlvr07.js`.
  - Standardized telemetry table column headers in `16_inference_optimization_quantization_compilation.md`.
- **Verification**: Ran `scratch/audit_coherence.py` passing 100% across all 18 chapters (Code-to-Output 1:1 paired, 5-Stage standard, AST syntax clean, cumulative execution clean, ASCII blueprints present, Telemetry radar present, Runbook present, Whiteboard defense present). Ran `npm --prefix frontend run typecheck && npm --prefix frontend run build` (clean compilation in 9.56s).

---

### [2026-09-13] - Post-Training Track 4-Pillar Reorganization & 7 Core Hotspots 5-Part Learning Anatomy
- **Manifest & Curriculum Architecture Reorganization**:
  - `frontend/src/entities/manifest/rlvrManifest.ts`: Re-aligned 18 chapters from generic 5 stages into **4 Thematic Architectural Pillars**:
    1. `rlvr_pillar1`: 基礎強化學習與偏好優化 (Foundational RLVR & Preference)
    2. `rlvr_pillar2`: 工業級分佈式系統、顯存與推論極限 (Systems & Scale)
    3. `rlvr_pillar3`: 數據飛輪、冷啟動與過程監督 (Data Flywheel & Reasoning)
    4. `rlvr_pillar4`: 線上急救、評估指標與生產級系統設計 (Triage & System Design)
  - Marked the 7 highest-frequency core technical hotspots with explicit `🔥 Core Hotspot` badges and tags.
  - `rlvr/tutorials/INDEX.md`: Updated local index and roadmap table reflecting the 4 Pillars, 7 hotspots, and standardized 5-part structure.
- **7 Core Hotspots Upgraded to Standardized 5-Part Learning Context Anatomy**:
  - **Pillar 1 Hotspots**:
    - `03_grpo_algorithm.md` / `rlvr03.js`: PPO 到 GRPO 工業演進、8k+ CoT 價值網絡崩潰根因、零 Critic 顯存架構、同儕 Z-Score 相對優勢、極限邊界直覺 ($\beta \to 0, \beta \to \infty, G \to 1, \sigma \to 0$)、Dr. GRPO 長度偏見修復、WandB 4D 遙測監控與 Frontier Lab 專家級架構設計解析。
    - `07_dpo_preference_optimization.md` / `rlvr07.js`: 三階段 RLHF 到 DPO 代數革命、隱式獎勵推拉力學、DPO vs GRPO 決策矩陣、長度作弊 (Verbosity Bias) 機制、Likelihood Displacement 概率塌陷急救與 Frontier Lab 專家級架構設計解析。
    - `11_modern_preference_simpo_remax_kto.md` / `rlvr11.js`: 免參考模型 SimPO (節省 50% 顯存)、目標邊界 $\gamma$ 物理直覺、無 Critic 的 ReMax 貪婪基線、KTO 點讚日誌展望理論對齊與反向長度作弊排查。
  - **Pillar 2 Hotspots**:
    - `10_distributed_systems_verl_vllm.md` / `rlvr10.js`: 雙模態負載矛盾（Memory-bound Rollout vs Compute-bound Learner）、veRL + vLLM 3D-HybridEngine 動態重分片 (<800ms)、KV-Cache 顯存精算、NCCL All-to-All 死鎖排查與木桶短板效應隔離診斷。
    - `15_lora_qlora_peft.md` / `rlvr15.js`: 全參微調顯存牆、LoRA 低秩分解、NF4 資訊理論高斯分位數證明、雙重量化 DQ、All-Linear 知識分佈覆蓋、單卡 80GB 微調 70B 顯存精算與 Adapter 權重合併精度防漂移。
  - **Pillar 3 Hotspots**:
    - `09_sft_cold_start.md` / `rlvr09.js`: 冷啟動死局成因、DeepSeek-R1-Zero 啟示（純 RL 的可讀性缺陷）、長思維鏈合成蒸餾、XML 標籤因果遮蔽、Goldilocks 甜蜜區 Pass@8 躍遷 (15%~35%)、思維模板過擬合急救。
  - **Pillar 4 Hotspots**:
    - `14_post_training_systems_and_triage_playbook.md` / `rlvr14.js`: Frontier Labs 系統架構全景、64x H100 叢集 70B 系統設計白板實戰、白板 4 大代碼（SimPO、無偏 Pass@k、ReMax、cu_seqlens）、線上事故急診 post-mortem、Apple MLE 專項（端側 3B + PCC 70B 投機解碼）與系統架構實戰指南。
- **Verification**: Ran `npm --prefix frontend run typecheck` (passed with code 0) and `npm --prefix frontend run build` (built production bundle cleanly into `web/dist/` in 7.44s). Zero documentation drift preserved.

---

### [2026-09-13] - Design System & CSS Modernization (Tailwind v4 & Dual Theme Tokens)
- **Root Styling & Token Foundation (`frontend/src/app/styles/index.css`)**:
  - Registered official Tailwind CSS v4 `@theme` design tokens (`--color-background`, `--color-foreground`, `--color-card`, `--color-sidebar`, `--color-primary`, `--color-secondary`, `--color-muted`, `--color-border`, etc.) fixing previously omitted classes in compiled CSS.
  - Defined high-contrast accessible palettes for `:root` (dark default) and `[data-theme="light"]`.
  - Configured `@tailwindcss/typography` `.prose` token mapping (`--tw-prose-body`, `--tw-prose-headings`, `--tw-prose-links`, etc.) for seamless reading across both themes.
  - Added KaTeX font inheritance (`.katex { color: inherit !important; }`), callout styling, and deep-link button animations.
- **Root Layout & SPA Shell**:
  - `frontend/index.html`: Replaced hardcoded `bg-slate-950` with semantic `bg-background text-foreground selection:bg-primary/20`.
  - `frontend/src/pages/guide/GuidePage.tsx`: Removed nested `<main>` tags and scroll-trapping conflict; standardized layout container and sidebar collapse trigger.
  - `frontend/src/widgets/sidebar-nav/SidebarNavigation.tsx`: Replaced broken `[data-theme=light]:...` classes with semantic tokens (`bg-sidebar`, `text-sidebar-foreground`, `border-sidebar-border`).
  - `frontend/src/widgets/reader-canvas/ReaderCanvas.tsx`: Upgraded canvas background, typography headers, table styling, and terminal-grade `CodeBlock` with window controls and copy feedback.
- **Components & Simulators**:
  - `frontend/src/shared/ui/`: Converted `Card`, `Button`, `Badge`, `Slider`, and `Dialog` to semantic design tokens; added `destructive` variant and dark/light mode contrast adjustments to `Badge`.
  - `frontend/src/shared/lib/MermaidRenderer.tsx`: Integrated dynamic theme subscription with automatic dark/light diagram re-rendering and semantic borders/toolbars.
  - `frontend/src/entities/milestone/`: Modernized `MilestoneTutorialCard.tsx` and `MilestoneTutorialModal.tsx` for dual-theme consistency.
  - `frontend/src/entities/simulation/CartPoleSimulator.tsx`: Updated HUD badges, slider control cards, and buttons to semantic tokens.
- **Indexes & Documentation**:
  - Created `frontend/src/app/styles/INDEX.md` and updated `frontend/src/app/INDEX.md`.
- **Verification**: Ran `npm --prefix frontend run typecheck` (passed with code 0) and `npm --prefix frontend run build` (built production bundle into `web/dist/` in 7.63s).

---

### [2026-09-13] - Comprehensive Kaggle Milestone Tutorial & 4-Tab Workbench Rollout
- **Updated Milestone Architecture**:
  - `frontend/src/entities/milestone/milestoneTutorials.ts`: Extended from 11 to all 16 milestones across 3 tracks (Post-Training Stages 1-5, RL Stages 1-5, DeepAgents Modules 1-6). Added explicit hardware specs (T4/P100 VRAM budget), runtime estimates, runnable Python code blocks, STAR playbooks, multi-variant technical highlights (Standard, Metric-Driven, Systems/Infra), and Tier-1 AI lab architecture Q&As.
  - `frontend/src/entities/milestone/MilestoneTutorialModal.tsx`: Upgraded to a 4-tab interactive workbench: 4-Step Pipeline Code (with one-click "Copy All 4 Steps Code"), Architecture Decisions Workbench (Situation/Task/Action/Result cards with copyable decision script), Frontier Lab Architecture Deep Dive Q&A, and Multi-Variant Technical Highlights Generator.
  - `frontend/src/entities/milestone/MilestoneTutorialCard.tsx`: Enhanced in-canvas reader card with hardware/runtime badges, target role, quick STAR highlights summary preview, runnable step code blocks, and full workbench trigger.
  - `frontend/src/entities/milestone/INDEX.md`: Updated contract and invariants.
- **Verification**: `npm run build` executed in `frontend/` with clean TypeScript and Vite compilation (exit code 0).

---

### [2026-09-12] - Autonomous Documentation Protocol Bootstrap
- **Updated Subdirectory Indexes**:
  - `rl/INDEX.md`
  - `rl/JD/INDEX.md`
  - `rl/paper/INDEX.md`
  - `rl/requirement/INDEX.md`
  - `rl/docs/INDEX.md`
  - `rlvr/INDEX.md`
  - `rlvr/kaggle_showcase/INDEX.md`
  - `rlvr/notebooks/INDEX.md`
  - `rlvr/research/INDEX.md`
  - `rlvr/tutorials/INDEX.md`
  - `web/INDEX.md`
  - `web/routers/INDEX.md`
  - `frontend/INDEX.md`
  - `frontend/src/INDEX.md`
  - `frontend/src/app/INDEX.md`
  - `frontend/src/entities/INDEX.md`
  - `frontend/src/pages/INDEX.md`
  - `frontend/src/shared/INDEX.md`
  - `frontend/src/widgets/INDEX.md`
  - `tutorials/INDEX.md`
  - `.agents/INDEX.md`
  - `.agents/skills/INDEX.md`
  - `skills/INDEX.md`
- **Updated Global Docs**:
  - `docs/INDEX.md`
  - `docs/ARCHITECTURE.md`
  - `docs/API_SURFACE.md`
  - `docs/STORAGE.md`
  - `docs/SYSTEM_STATE.md`
- **Updated Customizations**:
  - `.agents/skills/auto-index/SKILL.md`
  - `skills/auto-index/SKILL.md`
  - `.agents/rules/auto-index.md`
  - `AGENTS.md`
- **Summary**: Initial bootstrap of the Autonomous Documentation Protocol. Created complete distributed `INDEX.md` hierarchy across all functional subdirectories, centralized `docs/` suite, Antigravity skill and rule definitions, and root repository harness contract.

---

### [2026-09-12] - Academic Research Skills Deprecation & Directory Index Reconcile
- **Removed Deprecated Skills (30)**:
  - `algorithm-design`, `atomic-decomposition`, `backward-traceability`, `citation-management`, `code-debugging`, `data-analysis`, `deep-research`, `experiment-code`, `experiment-design`, `figure-generation`, `github-research`, `idea-generation`, `latex-formatting`, `literature-review`, `literature-search`, `math-reasoning`, `novelty-assessment`, `paper-assembly`, `paper-compilation`, `paper-revision`, `paper-to-code`, `paper-writing-section`, `rebuttal-writing`, `related-work-writing`, `research-planning`, `self-review`, `slide-generation`, `survey-generation`, `symbolic-equation`, `table-generation`.
  - Cleaned up from both `.agents/skills/` and `skills/`.
- **Preserved Active Skills (7)**:
  - `add_data`, `add_jd`, `add_resource`, `auto-index`, `excalidraw-skill`, `interactive-learning-platform`, `rubric-evaluation`.
- **Updated Indexes & Global Documentation**:
  - Created `.agents/skills/INDEX.md`
  - Created `.agents/INDEX.md`
  - Created `skills/INDEX.md`
  - Updated `README.md`
  - Updated `docs/INDEX.md`
  - Updated `docs/SYSTEM_STATE.md`
- **Summary**: Removed 30 unused third-party academic paper and research skills to clean workspace footprint. Created explicit directory indices conforming to the Autonomous Documentation Protocol. Zero documentation drift preserved.

---

### [2026-09-12] - Skill & Protocol Refinement (Agent Readership & Dynamic Topic Creation)
- **Updated Customizations**:
  - `.agents/skills/auto-index/SKILL.md`
  - `skills/auto-index/SKILL.md`
  - `.agents/rules/auto-index.md`
  - `AGENTS.md`
- **Updated Global Docs**:
  - `docs/INDEX.md`
  - `docs/SYSTEM_STATE.md`
- **Summary**: Updated Autonomous Documentation Protocol to explicitly declare all `docs/` and distributed `INDEX.md` files as internal machine memory for AI agent readership only (dense, structured, non-marketing). Empowered agents to dynamically create new topic-specific markdown files (`docs/<TOPIC>.md`) and register them in `docs/INDEX.md`.

---

### [2026-09-12] - Curriculum Navigation Sidebar Clean & Sleek Redesign
- **Updated Source Files**:
  - `frontend/src/widgets/sidebar-nav/SidebarNavigation.tsx`: Redesigned with Linear/Raycast style aesthetics, floating segmented track switcher, instant search with filter chips (`All`, `Todo`, `Labs`), progress bar with completion tracking, chapter completion toggle checkboxes, and responsive collapse/expand controls.
  - `frontend/src/pages/guide/GuidePage.tsx`: Integrated smooth sidebar collapsing and sleek floating expand trigger when collapsed.
- **Created Indexes**:
  - `frontend/src/widgets/sidebar-nav/INDEX.md`
  - `frontend/src/pages/guide/INDEX.md`
- **Updated Global Docs**:
  - `docs/SYSTEM_STATE.md`
- **Summary**: Modernized sidebar UI into a clean, sleek, high-precision developer experience supporting keyboard navigation (`/`, `Esc`), dark and light themes, and interactive progress tracking across all 67 chapters. Zero drift maintained.

---

### [2026-09-12] - Unused Standalone Python Scripts Cleanup (`rl/`)
- **Deleted Deprecated Scripts (5)**:
  - `rl/grpo_llm_play_gradio.py`: Legacy Gradio web UI on port 7860. Superseded by unified FastAPI backend (`web/routers/llm.py`) and React 19 SPA.
  - `rl/grpo_play_gradio.py`: Legacy Gradio web UI on port 7860 for toy GRPO. Superseded by `web/routers/toy.py` and React 19 SPA.
  - `rl/grpo_play.py`: Terminal interactive prompt loop for toy GRPO.
  - `rl/grpo_llm_play.py`: Terminal interactive prompt loop for character-level GRPO.
  - `rl/run_offline_training.py`: Standalone CLI offline runner tightly coupled to deprecated Gradio logging workflow.
- **Preserved Core RL Modules**:
  - All scripts actively imported by `serve.py` and `web/routers/` (`grpo_play_core.py`, `grpo_llm_char.py`, `training_pipeline.py`, `kairos_log.py`, `grpo_toy.py`, `demo_log_to_grpo.py`, `grpo_offline_resample.py`, `policy_feedback.py`, `policy_feedback_demo.py`).
  - Standalone research paper code snapshot in `rl/paper/code/`.
  - Core reference algorithm `rl/memory_selector.py` and demo `rl/memory_selector_demo.py`.
- **Updated Local and Global Documentation**:
  - `rl/INDEX.md`: Removed deleted scripts from contents table.
  - `rl/AGENTS.md`: Updated repo file tree.
  - `rl/README.md`: Removed obsolete Gradio/terminal run instructions; pointed to unified `serve.py` platform.
  - `rl/grpo_play_core.py`: Updated module docstring to reflect FastAPI backend usage.
  - `rl/training_pipeline.py`: Updated docstring reference.
  - `web/routers/llm.py`: Cleaned up comments referencing deleted Gradio script.
  - `docs/ARCHITECTURE.md`: Synchronized §2.1 RL subsystem description.
  - `docs/API_SURFACE.md`: Cleaned up §2 CLI entrypoints.
  - `docs/SYSTEM_STATE.md`: Recorded synchronization event.
- **Summary**: Removed 5 unused legacy Gradio and interactive terminal play scripts from `rl/` that were not used by the website platform. Verified zero syntax errors, zero broken imports in FastAPI backend and RL algorithms, and maintained full zero-drift compliance.

---

### [2026-09-12] - Code View ReadTheDocs-Style Simple Code Wrap Redesign
- **Updated Source Files**:
  - `frontend/src/widgets/code-inspector/CodeInspector.tsx`: Replaced the split 8-column table layout (`lg:col-span-8`, `table border-collapse`) and right-side annotation drawer (`lg:col-span-4`) with a unified, full-width ReadTheDocs-style code wrapper. Features line number gutter (`select-none`), toggleable automatic code wrapping (`whitespace-pre-wrap break-all` vs `whitespace-pre`), header bar with language/line badges and copy-to-clipboard button, and inline expandable annotations beneath selected lines.
  - `frontend/src/widgets/reader-canvas/ReaderCanvas.tsx`: Upgraded markdown fenced code blocks to use the matching ReadTheDocs-style code container with copy button and automatic code wrapping.
- **Updated Indices & Docs**:
  - `frontend/src/widgets/INDEX.md`: Updated `code-inspector/` and `reader-canvas/` descriptions.
  - `docs/SYSTEM_STATE.md`: Recorded synchronization event.
- **Summary**: Eliminated rigid table and two-column code inspector layout in favor of clean ReadTheDocs-style code presentation with simple code wrapping and inline interactive notes. Typechecked and built successfully with zero drift.

---

### [2026-09-12] - Markdown Reader Rendering Engine & Content Fidelity Fix
- **Fixed Issues**:
  - **Duplicated KaTeX Math Expressions**: Installed local `katex/dist/katex.min.css`, configured `rehype-katex` with `{ output: 'html' }`, and hidden `.katex-mathml` via CSS to eliminate duplicated text nodes (e.g. `∼15%−25%∼15%−25%`, `+18% to +27pp+18% to +27pp`, `2.3×2.3×`).
  - **Unparsed Markdown Tables**: Installed `remark-gfm` and added custom glassmorphic `table`, `thead`, `tbody`, `tr`, `th`, `td` components with row hover effects, restoring all benchmark comparison tables.
  - **Broken GitHub Alerts**: Created alert-aware `AlertBlockquote` supporting `[!IMPORTANT]`, `[!NOTE]`, `[!TIP]`, `[!WARNING]`, and `[!CAUTION]` with distinct glowing badges, Lucide icons, and tinted cards, removing raw `[!IMPORTANT]` text.
  - **Broken Cross-Chapter Navigation**: Implemented `MarkdownLink` interceptor in `ReaderCanvas.tsx` to route relative `.md` tutorial links (e.g. `./06_agentic_rlvr.md`) directly through `setCurrentChapterId()`, preventing 404 page navigation errors.
  - **Preserved Code & ASCII Alignment**: Removed destructive `break-all` on code blocks, replaced with monospaced `whitespace-pre` and card headers with copy-to-clipboard functionality, fixing ASCII architecture flowcharts and Python syntax blocks.
  - **Tailwind Typography & List Styling**: Installed `@tailwindcss/typography` and restored proper `ul`/`ol` bullet points, numbers, and margins stripped by Tailwind CSS Preflight.
  - **Code Inspector Polish**: Updated `CodeInspector.tsx` to prevent duplicate `Line X / Line X` labels and refreshed `rlvr05.js` with research-grade Pass@k and bootstrap CI implementation.
- **Updated Source Files**:
  - `frontend/package.json`
  - `frontend/vite.config.ts`
  - `frontend/src/app/styles/index.css`
  - `frontend/src/widgets/reader-canvas/ReaderCanvas.tsx`
  - `frontend/src/widgets/code-inspector/CodeInspector.tsx`
  - `frontend/src/entities/chapter/data/rlvr/rlvr05.js`
  - `docs/SYSTEM_STATE.md`
- **Verification**: `tsc --noEmit` passed with 0 errors, Vite production bundle built cleanly in 7.7s into `../web/dist`, and FastAPI test client confirmed 200 OK responses on `/` and `/api/rlvr/chapters/rlvr05`.

---

### [2026-09-12] - Skill Unification into Global Root (~/.gemini/config/skills/)
- **Unified Skill Storage**:
  - Moved and unified active skills (`auto-index`, `excalidraw-skill`, `interactive-learning-platform`, `rubric-evaluation`) into the global customizations root `~/.gemini/config/skills/`.
  - Enriched `~/.gemini/config/skills/code-evaluator` with scripts (`notebook_auditor.py`) and templates (`audit_report_template.md`, `evaluation_rubric_template.md`).
  - Removed duplicate and redundant local skill directories `skills/` and `.agents/skills/`.
- **Updated Customizations & Harness Contracts**:
  - `.agents/INDEX.md`: Updated to document that `.agents/` houses persistent execution rules while workflow skills are unified globally.
  - `.agents/rules/auto-index.md`: Updated `auto-index` skill reference to global path.
  - `AGENTS.md`: Updated global harness contract and layout diagram.
  - `rl/AGENTS.md`: Updated RL domain harness contract.
  - `README.md`: Updated layout tree.
  - `docs/INDEX.md`: Updated system catalog to reflect global skills root.
  - `docs/ARCHITECTURE.md`: Updated architecture diagram and agent customization specifications.
- **Verification**: Verified all 9 skills present and valid in `~/.gemini/config/skills/`, local duplicates deleted, and zero-drift documentation invariants preserved.

---

### [2026-09-12] - Sidebar Navigation Simplification (Removal of Progress Features)
- **Updated Source Files**:
  - `frontend/src/widgets/sidebar-nav/SidebarNavigation.tsx`: Removed completion tracking, progress bars, completion checkmark buttons on chapters, fraction counters, and `todo` filter chips to create a clean, sleek, and distraction-free outline. Retained instant search (`/`), `labsOnly` toggle, accordion controls, and theme adaptability.
  - `frontend/src/widgets/reader-canvas/ReaderCanvas.tsx`: Fixed missing `cn` import and connected `selectedCodeLine` to `CodeInspector`.
- **Updated Indices & Docs**:
  - `frontend/src/widgets/sidebar-nav/INDEX.md`: Updated purpose and responsibilities to reflect progress-free design.
  - `docs/SYSTEM_STATE.md`: Recorded synchronization event.
- **Verification**: `npm run typecheck` passed with 0 errors, Vite production bundle built cleanly into `web/dist/` in 7.63s, zero documentation drift maintained.

---

### [2026-09-12] - Gitignore Configuration Upgrade
- **Updated Configuration**:
  - [.gitignore](file:///Users/derekleung/Documents/deepagents/.gitignore): Added comprehensive ignore rules for Node/frontend (`node_modules/`, `npm-debug.log*`, `.pnpm-store/`, `.vite/`, `frontend/dist/`, `web/dist/`), ML checkpoints & weights (`checkpoints/`, `*.pt`, `*.pth`, `*.bin`, `*.safetensors`, `*.ckpt`, `wandb/`, `runs/`), caches & testing (`.mypy_cache/`, `.ruff_cache/`), Jupyter checkpoints (`.ipynb_checkpoints/`), secrets (`.env*`), and OS metadata (`.DS_Store?`, `Thumbs.db`).
- **Verification**: `git check-ignore` verified that frontend `node_modules`, build distribution `web/dist`, and ML `checkpoints` are cleanly ignored.

---

### [2026-09-12] - Obsoleted Pages Cleanup & Navigation Resiliency
- **Decommissioned Obsolete Pages & Legacy Assets**:
  - Deleted orphaned static visualizer page: `rlvr/l4_copilot_data_pipeline.html`.
  - Cleaned up broken dead route anchor (`<a href="#copilot" class="btn-lab">`) in `frontend/src/entities/chapter/data/rl/ch18.js`.
  - Retired obsolete page mutation skills (`add_data`, `add_jd`, `add_resource`) that targeted non-existent vanilla-JS prototype files (`policy.js`, `copilot.js`, `resources.js`, `kb.js`).
- **SPA Routing & Link Resiliency**:
  - Hardened `ReaderCanvas.tsx` `MarkdownLink`: handles localhost stripping, maps relative `.md` files to SPA chapter IDs, supports direct `#chXX` / `#guide/chXX` chapter navigation, and smooth-scrolls in-page DOM headings.
  - Modernized `web/routers/wiki.py`: updated knowledge corpus citations to direct `#ch01`–`#ch18` chapter hashes and fixed dynamically filed node links.
- **Synchronized Documentation & Harness Contracts**:
  - `rl/AGENTS.md`: Purged references to 15-page vanilla JS routes and obsolete verification commands; replaced with React 19 + Vite + FastAPI contract.
  - `web/README_web.md`: Modernized from 5-page vanilla JS prototype to unified React 19 + Vite + FastAPI documentation.
  - `README.md`: Replaced references to obsolete route anchors (`#guide`, `#labs`, `#policy`, `#copilot`) with modern 3-track 67-chapter platform features.
  - `rlvr/INDEX.md`: Removed `l4_copilot_data_pipeline.html`.
  - `.agents/skills/INDEX.md` & `.agents/INDEX.md`: Reconciled to catalog the 4 active workspace skills (`auto-index`, `excalidraw-skill`, `interactive-learning-platform`, `rubric-evaluation`).
  - `docs/ARCHITECTURE.md` & `docs/INDEX.md`: Reconciled skills customization descriptions.
- **Verification**:
  - TypeScript typecheck (`npx tsc --noEmit`): 0 errors.
  - Vite production build (`npm run build`): Cleanly built all 67 chapters into `web/dist`.
  - Backend imports (`web.app`, `web.routers.wiki`): Passed with 0 errors.
  - Python RL entrypoints regression check: Passed cleanly.

---

### [2026-09-12] - Tutorial Formatting & Mermaid Robustness Resolution
- **Updated Tutorial Source**:
  - `tutorials/24-multi-turn-evaluation.md`:
    - Converted prose mistakenly wrapped in code block backticks (Section 4 isolation principles) into a standard `> [!IMPORTANT]` callout block.
    - Fixed Mermaid diagram syntax errors: enclosed unquoted subgraph labels containing colons and special characters (`多輪評估：錯誤與狀態傳播` and `τ²-Bench 雙控制架構`) in valid subgraph syntax (`subgraph ID ["Label"]`) and wrapped edge labels in double quotes.
- **Enhanced Frontend Renderer**:
  - `frontend/src/shared/lib/MermaidRenderer.tsx`: Added `sanitizeMermaid` pipeline to automatically rewrite unquoted subgraphs and clean edge pipe delimiters, plus automatic cleanup of stray error elements inserted into document body by Mermaid v10 on parse errors.
  - `frontend/src/widgets/reader-canvas/ReaderCanvas.tsx`: Fallback language header changed to `snippet` with copy button for untagged code blocks.
  - `frontend/src/entities/milestone/MilestoneTutorialCard.tsx`, `MilestoneTutorialModal.tsx`: Cleaned up unused Lucide icon imports.
- **Updated Documentation**:
  - `frontend/src/shared/lib/INDEX.md`: Updated `MermaidRenderer.tsx` responsibility.
  - `docs/SYSTEM_STATE.md`: Recorded synchronization event.
- **Verification**: `npm run typecheck` passed (0 errors), `npm run build` compiled cleanly into `web/dist/` in 8.77s. Zero documentation drift maintained.

---

### [2026-09-12] - Milestone Interactive Kaggle Practice Tutorial Upgrade
- **Created Milestone Entity Suite (`frontend/src/entities/milestone/`)**:
  - `milestoneTutorials.ts`: Structured 4-step Kaggle practice tutorial dataset (Environment Setup, Core Implementation, Benchmarking & Pass@k, Portfolio & Architecture Defense), runnable code snippets, STAR technical highlights, and frontier lab architecture Q&A.
  - `MilestoneTutorialCard.tsx`: In-canvas interactive tutorial card embedded at the top of reading chapters with step-by-step accordion navigation, code copying, and fullscreen trigger.
  - `MilestoneTutorialModal.tsx`: Comprehensive fullscreen dialog workbench for step-by-step Kaggle practice, code execution templates, copyable portfolio highlights, and architectural defense.
  - `index.ts`: Barrel export.
  - `INDEX.md`: Conforms to Autonomous Documentation Protocol.
- **Updated Source Files**:
  - `frontend/src/entities/chapter/chapterStore.ts`: Added `activeMilestonePartId`, `openMilestoneTutorial`, and `closeMilestoneTutorial`.
  - `frontend/src/widgets/sidebar-nav/SidebarNavigation.tsx`: Upgraded static milestone text to an interactive Kaggle tutorial action button (`Trophy` badge, milestone description, and click-to-open handler).
  - `frontend/src/widgets/reader-canvas/ReaderCanvas.tsx`: Replaced static milestone alert box with interactive `MilestoneTutorialCard`.
  - `frontend/src/pages/guide/GuidePage.tsx`: Integrated global `MilestoneTutorialModal` bound to store state.
- **Verification**:
  - `npx tsc --noEmit`: 0 errors.
  - `npm run build`: Production bundle built cleanly into `web/dist/` in 7.48s.
  - Backend test client confirmed 200 OK serving.
  - Zero documentation drift preserved.

---

### [2026-09-12] - Exhaustive Mermaid Syntax Repair & Production Bundle Synchronization
- **Problem Statement**:
  - Browser console reported lexical parsing errors: `Lexical error on line 6. Unrecognized text. ...d subgraph 多輪評估：錯誤與狀態傳播` and `Lexical error on line 2. Unrecognized text. ...ph TD subgraph τ²-Bench 雙控制架構`.
- **Systematic Investigation & Repair**:
  - Programmatic scan across all 76 diagrams in the repository uncovered 7 failing Markdown diagrams and 4 pre-rendered JS diagrams with non-standard syntax (unquoted subgraph names, colons, emojis, Greek characters `τ²`, and unescaped parentheses/braces/pipes in edge labels).
  - Fixed `tutorials/24-multi-turn-evaluation.md`: Provided explicit IDs `single`, `multi`, `tau2`, and `verify` with quoted titles, and quoted cylinder node labels.
  - Fixed `tutorials/03-custom-middleware.md`: Wrapped HTML and parenthesized node labels in double quotes.
  - Fixed `tutorials/14-event-streaming.md`: Assigned explicit IDs `sync` and `stream` with quoted emoji titles.
  - Fixed `tutorials/23-error-analysis.md`: Assigned explicit IDs `bad` and `good` with quoted emoji titles.
  - Fixed `rlvr/tutorials/10_distributed_systems_verl_vllm.md`: Replaced breaking edge parentheses with fullwidth parentheses `（<800ms）`.
  - Fixed `rlvr/tutorials/17_llm_evaluation_benchmarking_prompt_optimization.md`: Quoted all non-ASCII quadrant titles, quadrant descriptions, and axes.
  - Fixed `frontend/src/entities/chapter/data/rl/ch01.js`, `ch03.js`, `ch05.js`, `ch06.js`: Replaced conflicting edge label delimiters with safe fullwidth equivalents.
- **Defensive Frontend Architecture**:
  - Enhanced `frontend/src/shared/lib/MermaidRenderer.tsx`: Upgraded `sanitizeMermaid` to normalize line endings (`\r\n` to `\n`), automatically assign alphanumeric IDs with quoted titles to any unquoted/non-ASCII subgraphs, and convert parentheses/braces in edge labels to fullwidth equivalents before calling `mermaid.render`.
- **Verification**:
  - Ran automated validation using Mermaid parser on all 76 diagrams (58 Markdown + 18 JS): **100% pass rate (0 failures)**.
  - Rebuilt Vite production bundle (`npm run build` in `frontend/`) into `web/dist/` in 8.76s with 0 errors.
  - Zero documentation drift maintained across the repository.

---

### [2026-09-12] - Complete Repository Index Audit & Skill System Reconciliation
- **Subdirectory Index Repairs**:
  - `tutorials/index.md` renamed to uppercase `tutorials/INDEX.md` per Autonomous Documentation Protocol convention.
  - Created missing self-describing `INDEX.md` files:
    - `.agents/rules/INDEX.md`
    - `frontend/src/widgets/code-inspector/INDEX.md`
    - `frontend/src/widgets/reader-canvas/INDEX.md`
    - `frontend/src/entities/chapter/INDEX.md`
    - `frontend/src/entities/manifest/INDEX.md`
    - `frontend/src/entities/milestone/INDEX.md`
    - `frontend/src/entities/simulation/INDEX.md`
    - `frontend/src/shared/ui/INDEX.md`
    - `frontend/src/shared/lib/INDEX.md`
    - `frontend/src/shared/types/INDEX.md`
  - Reconciled parent indexes with active child directory links:
    - `.agents/INDEX.md` (links to `rules/INDEX.md` and `skills/INDEX.md`)
    - `frontend/src/widgets/INDEX.md` (links to `code-inspector`, `reader-canvas`, `sidebar-nav`)
    - `frontend/src/entities/INDEX.md` (links to `chapter`, `manifest`, `milestone`, `simulation`)
    - `frontend/src/shared/INDEX.md` (links to `lib`, `types`, `ui`)
    - `rl/JD/INDEX.md` & `rl/JD/job-descriptions/INDEX.md` (removed stale `add_jd` and `app.js` references)
  - `rlvr/INDEX.md`: Purged deleted `l4_copilot_data_pipeline.html`.
  - Reconciled `.agents/skills/INDEX.md` to catalog exactly the 4 active workspace skills (`auto-index`, `excalidraw-skill`, `interactive-learning-platform`, `rubric-evaluation`) with direct relative links to each `SKILL.md`.
- **Skill System Health & Integrity**:
  - Validated that all active skills in `.agents/skills/` have 100% compliant YAML frontmatter and operational instructions.
  - Updated `auto-index/SKILL.md` to include `.agents/rules/` and `.agents/skills/` in active indexed directories list.
  - Enhanced `interactive-learning-platform/SKILL.md` to document the integrated `frontend/` production React 19 implementation alongside the reference blueprint.
  - Synchronized `.agents/skills/` with global skills in `~/.gemini/config/skills/`.
  - Confirmed deletion of obsolete duplicate root `skills/` directory to prevent workspace fragmentation.
- **Global Documentation Suite Synchronization**:
  - `docs/INDEX.md`: Updated `.agents/` layout with `rules/` and `skills/` entries and updated local index links.
  - `docs/ARCHITECTURE.md`: Removed legacy Gradio demo node, updated Agent Customizations to `.agents/skills/`.
  - `AGENTS.md`: Updated layout tree and skill paths.
  - `rl/AGENTS.md`: Updated skill contract path to `.agents/skills/`.
  - `README.md`: Updated project directory tree to match current codebase.
- **Verification**:
  - Full repository audit: 39 `INDEX.md` files scanned, 0 broken links, 0 unindexed files.
  - Python RL tests: `python3 -m rl.grpo_toy` and `python3 -m rl.policy_feedback_demo` passed 100%.
  - Backend sanity check: `python3 -c "import web.app"` passed with 0 errors.
  - Frontend typecheck and build: `npm run typecheck && npm run build` passed with 0 errors, compiled in 7.28s into `web/dist/`.

---

### [2026-09-12] - Post-Training Track Transformation (Traditional Chinese & Pedagogy Polish)
- **UI & Presentation Refactor**:
  - Renamed track in navigation and display manifests: updated floating track switcher pill in `SidebarNavigation.tsx` to `Post-Train` (`title="Post-Training MLE 實戰"`), `rlvrManifest.ts` `shortName` to `'Post-Training'`, and title to `'Production Post-Training MLE Handbook'`.
  - Maintained stable internal routing (`/api/rlvr`, manifest `id: 'rlvr'`) to preserve backward compatibility and prevent broken links.
  - **Eliminated Code Inspector Widget**: Removed `CodeInspector` and `selectedCodeLine` state from `ReaderCanvas.tsx` across the reader experience. Cleared `codeLines` and `codeFile` stubs from all `rlvr01.js` through `rlvr18.js` metadata files.
- **Traditional Chinese & Intuitive Explanations Overhaul**:
  - Rewrote and enriched all 18 chapters (`rlvr/tutorials/01_data.md` through `18_alignment_safety_red_teaming.md`) and synchronized metadata (`rlvr01.js` through `rlvr18.js`):
    - **Language**: Fluent, high-fidelity Traditional Chinese (繁體中文) with standard English technical terminology preserved (Rollout, Verifiers, KL Divergence, Pass@k, Loss, DPO, SimPO, PRM, etc.).
    - **Pedagogy & Mental Models**: Integrated visual mental models, analogies, Mermaid system diagrams, production math derivations (GRPO, DPO closed-form, G-Eval expectation, NF4 quantization), and Frontier Lab (Apple / OpenAI / Anthropic) MLE system architecture playbooks.
- **Verification & Documentation**:
  - `npm run build` in `frontend/` succeeded with **exit code 0** (0 TypeScript or Vite bundling errors).
  - Synchronized `rlvr/tutorials/INDEX.md`, `frontend/src/widgets/reader-canvas/INDEX.md`, and `docs/SYSTEM_STATE.md` with zero documentation drift.

---

### [2026-09-13] - Interactive Platform Skill Content Style Update & Post-Training Content Fixes
- **Skill System Standard Alignment (Progressive Notebooks Standard)**:
  - Standardized the content style guidelines across both workspace and global skill files (`~/.gemini/config/skills/interactive-learning-platform/` and `.agents/skills/interactive-learning-platform/`):
    - Adopted the progressive notebook laboratory standard as the gold standard for tutorial content structure and pedagogical depth.
    - Added dedicated section **"4. Content Style & Pedagogical Standards (Progressive Notebooks Standard)"** codifying the 6-stage anatomy: (1) Motivation & Problem Formulation before formulas, (2) Rigorous Step-by-Step Mathematical Derivations with parameter limits ($\tau \to 0$ vs $\tau \to \infty$), (3) Self-Contained Executable PyTorch Cells with numerical epsilons (`eps=1e-20`), `nn.Module` classes, analytical losses, and dual-objective loops, (4) Empirical Diagnostics & Sensitivity Sweeps, (5) Real-World Engineering Alert Callouts (`[!IMPORTANT]`, `[!NOTE]`, `[!WARNING]`, `[!TIP]`), and (6) Academic Citations & Canonical Repositories.
    - Expanded `references/recipe.md` with a copy-pasteable chapter authoring blueprint.
    - Synchronized Knowledge Item `interactive_ml_platform` `metadata.json` and `artifacts/overview.md`.
- **Post-Training Broken Content Fixes**:
  - **Fixed Lexer Crashes in Mermaid Diagrams**: Identified and eliminated reserved pipe characters `|` embedded inside node labels that triggered Mermaid lexer parsing errors in:
    - `rlvr/tutorials/03_grpo_algorithm.md`: Escaped `o_1, ..., o_G ~ π_old(· ; q)` and `- β D_KL(π_θ ∥ π_ref)`.
    - `rlvr/tutorials/07_dpo_preference_optimization.md`: Escaped `+ ∇_θ log π_θ(y_w ; x)`, `- ∇_θ log π_θ(y_l ; x)`, and `π_ref(y ; x)`.
    - `rlvr/tutorials/08_ablations_and_scaling.md`: Sanitized baseline config label and `KL(π_θ ∥ π_ref)`.
    - `rlvr/tutorials/13_data_flywheel_and_decontamination.md`: Replaced pipe separators with slashes in data mixture breakdown.
    - `rlvr/tutorials/16_inference_optimization_quantization_compilation.md`: Replaced `|X|` with `abs(X)` in activation magnitude calculation.
    - `tutorials/18-llm-as-judge-alignment.md`: Replaced pipe separators with slashes in `JudgeVerdict` node.
  - **Enhanced Chapter Hydration Resilience**: Updated `frontend/src/entities/chapter/chapterLoader.ts` to attempt `data.file` first, falling back to `data.id`, ensuring robust chapter loading across local and production setups.
- **Verification**:
  - Verified 0 remaining pipe issues across all diagrams in the repository.
  - Verified all LaTeX formula `$$` blocks across all 18 chapters are balanced.
  - Rebuilt Vite production bundle (`npm run build` in `frontend/`) successfully with exit code 0.

---

### [2026-09-13] - Applied Post-Training Track Style to Interactive Platform Skills
- **Skill Definition Overhaul (`interactive-learning-platform`)**:
  - Updated workspace skill [.agents/skills/interactive-learning-platform/SKILL.md](file:///Users/derekleung/Documents/deepagents/.agents/skills/interactive-learning-platform/SKILL.md) and global counterpart `~/.gemini/config/skills/interactive-learning-platform/SKILL.md`:
    - Codified the **Post-Training Track 7-Pillar Pedagogical Standard**: (1) Title, epigraph & quantitative scope, (2) 核心心智模型與直觀比喻 (Intuitive Mental Models & Metaphors), (3) Mermaid 視覺流架構圖與顯存分配矩陣 (Visual Flow Architecture & Memory Allocation), (4) 嚴密數學推導與演算法步驟 (KaTeX Math & Algorithmic Steps), (5) 自包含生產級代碼單元 (Self-Contained Executable PyTorch / Transformers Cells), (6) 四維遙測監控指標與工業級急救錦囊 (4D Telemetry Signals & Emergency Runbooks), and (7) 🤔 前沿深度思辨與工業界陷阱 (Frontier AI Lab Architecture Defense: Failure Mode & Fix, High-Frequency Q&A).
    - Synchronized the active **18 parameter visual simulation laboratories** catalog matching `labCatalog.ts`.
    - Codified the retirement of `CodeInspector` from `ReaderCanvas` and the `codeLines: []` invariant in chapter metadata.
    - Standardized language policy: fluent Traditional Chinese (繁體中文) explanations with standard international English ML terminology.
- **Reference & Data Model Modernization**:
  - `references/recipe.md`: Updated platform architectural evolution across the 3 epochs (Vanilla prototype -> React 19 / FSD -> Post-Training Track transformation) and provided a complete copy-pasteable 7-pillar chapter authoring blueprint.
  - `references/data-model.md`: Completely modernized from legacy prototype schema to active production TypeScript interfaces (`SiteManifest`, `ChapterData`, `LabSpec`, `MilestoneTutorial`) and ESM glob dynamic loading pipeline.
  - `references/visual-labs.md`: Upgraded catalog from outdated 17 labs to all 18 production simulation laboratories with mathematical formulations, controls, and rendering components.
- **Knowledge Item & Autonomous Documentation Synchronization**:
  - Synchronized Knowledge Item `overview.md` at `<appDataDir>/knowledge/interactive_ml_platform/artifacts/overview.md`.
  - Created directory indexes: [.agents/skills/interactive-learning-platform/INDEX.md](file:///Users/derekleung/Documents/deepagents/.agents/skills/interactive-learning-platform/INDEX.md) and [.agents/skills/interactive-learning-platform/references/INDEX.md](file:///Users/derekleung/Documents/deepagents/.agents/skills/interactive-learning-platform/references/INDEX.md).
  - Updated parent index [.agents/skills/INDEX.md](file:///Users/derekleung/Documents/deepagents/.agents/skills/INDEX.md).
- **Verification**:
  - `npm --prefix frontend run typecheck`: Passed with exit code 0.
  - `npm --prefix frontend run build`: Clean build into `web/dist/` in 7.24s with 0 errors.

---

### [2026-09-13] - Applied Post-Training Track Style to Agent Track (DeepAgents 31 Chapters)
- **Comprehensive 31-Chapter Curriculum Overhaul (`tutorials/`)**:
  - Rewrote and enriched all 31 tutorial chapters (`tutorials/01-harness-engineering-basics.md` through `31-self-improving-eval-loop.md`) matching the Post-Training 7-pillar pedagogical standard:
    - **Language & Tone**: High-fidelity Traditional Chinese (繁體中文) retaining international English technical terminology (Agent Harness, Tool Calling, Subagents, VFS, StateBackend, Context Window, HITL, Rubric, Trajectory, LLM-as-a-Judge, Prompt Caching, etc.).
    - **核心心智模型 (Intuitive Mental Models)**: Integrated vivid engineering metaphors (Microkernel OS, Onion Shield, Airport Checkpoint, Swap Space Paging, Flight Data Recorder, Courtroom Chain of Custody, Aerospace 4-Layer Inspection, Game Save States, Self-Adapting Organisms) with styled Mermaid architectural diagrams.
    - **Frontier Lab Architectural Insights (🤔 前沿深度思辨與工業界陷阱)**: Attached senior AI Agent / MLE architectural playbooks (OpenAI, Anthropic, Cursor, Google) to all chapters, featuring real-world Failure Mode & Fix and high-frequency architecture Q&As.
    - **Clean Navigation**: Replaced fragmented anchors with standardized `## 下一步` cross-chapter links.
- **Frontend Chapter Metadata Cleansing (`frontend/src/entities/chapter/data/deepagents/`)**:
  - Completely eliminated `codeFile` and `codeLines` stub fields from all 31 files (`da01.js` through `da31.js`), eliminating code inspector bloat.
  - Enriched chapter `summary` fields with intuitive mental models and core engineering principles.
  - Standardized chapter titles with bilingual subtitles and regenerated clean `toc` anchors matching markdown headings.
  - Chapter 28 transformed from obsolete line-inspector guide into `可執行案例庫與工程實戰 (Runnable Harness Catalog)`.
- **Manifest & Visual Simulation Labs Extension**:
  - `frontend/src/entities/manifest/deepagentsManifest.ts`: Added `hasVisualizer` mappings for all 31 chapters, updated chapter 28 title/summary, and refined module competencies.
  - `frontend/src/entities/simulation/labCatalog.ts`: Added 7 first-class interactive simulation lab configurations (`agent_harness`, `agentic`, `eval_workbench`, `agent_vfs`, `rubric_eval`, `context_budget`, `pitfalls`) with equations and descriptions.
- **Autonomous Documentation Protocol Synchronization**:
  - Synchronized `tutorials/INDEX.md` with updated chapter 28 catalog description.
  - Synchronized `frontend/src/entities/chapter/data/INDEX.md` and `frontend/src/entities/simulation/INDEX.md`.
  - Recorded synchronization event in `docs/SYSTEM_STATE.md`.
- **Verification**:
  - `npm --prefix frontend run typecheck`: Passed with 0 errors (exit code 0).
  - `npm --prefix frontend run build`: Clean production bundle compiled into `web/dist/` in 7.25s with 0 errors.

---

### [2026-09-13] - Applied Post-Training Track Style to Research RL Track (18 Chapters)
- **Comprehensive 18-Chapter Curriculum Decoupling & Modernization (`rl/tutorials/`)**:
  - Authored all 18 chapters (`rl/tutorials/01_cartpole_physics_and_control.md` through `18_frontier_systems_architecture_alignment.md`) adhering to the progressive 6-stage tutorial anatomy:
    - **Language & Tone**: Rigorous, authentic Traditional Chinese (繁體中文) retaining international standard ML terminology (二階拉格朗日運動方程, 霍夫丁不等式, Bellman 最優算子, 策略梯度定理, GAE, Tanh 高斯重參數化, CQL, PPO-Clip, DPO, GRPO, Freeze Gate, Pass@k, PRM, SWE-bench, VLM KV Cache, STAR 架構決策話術).
    - **核心心智模型 (Intuitive Mental Models)**: High-level intuition, analogies, and detailed Mermaid architectural diagrams before mathematical formulas.
    - **Executable PyTorch Code Blocks**: Production-ready code with typing, assertions, and defensive guards (e.g. `1 - a^2 + 1e-6` Jacobian protection, vectorized GAE recursion, DPO softplus loss, GRPO Z-score advantage).
    - **Frontier AI Lab Architectural Insights (🤔 前沿深度思辨與工業界陷阱)**: 2 deep, senior-level architecture questions per chapter addressing real-world failure modes and architectural tradeoffs (OpenAI, DeepMind, Anthropic, Wayve, Apple).
    - **Scholarly Citations & Canonical Repositories**: Formal academic references concluding each tutorial.
- **FastAPI Backend Hydration (`web/`)**:
  - Created `web/routers/rl_chapters.py` exposing `GET /api/rl/chapters` and `GET /api/rl/chapters/{chapter_id}` with prefix resolution (e.g. `ch01` -> `01_cartpole_physics_and_control.md`).
  - Mounted router in `web/app.py` under `/api/rl/chapters` and registered in self-describing `/api` catalog.
- **Frontend Architecture & Manifest (`frontend/src/`)**:
  - Updated `chapterLoader.ts` to hydrate `siteId === 'rl'` via `/api/rl/chapters/${targetFile}`.
  - Added `file` properties to all 18 chapters in `RL_MANIFEST` (`frontend/src/entities/manifest/rlManifest.ts`).
  - Streamlined `frontend/src/entities/chapter/data/rl/` (`ch01.js` through `ch18.js`): removed legacy inline Simplified Chinese `html`, redundant `codeLines`, and disconnected `qa` dictionaries in favor of lean metadata and clean `toc` anchors.
- **Autonomous Documentation Protocol Synchronization**:
  - Created `rl/tutorials/INDEX.md`.
  - Updated `rl/INDEX.md`, `docs/INDEX.md`, and `docs/API_SURFACE.md`.
  - Recorded execution history in `docs/SYSTEM_STATE.md`.
- **Verification**:
  - `npm --prefix frontend run typecheck`: Passed with 0 errors (exit code 0).
  - `npm --prefix frontend run build`: Clean production bundle compiled in 7.23s into `web/dist/`.
  - FastAPI TestClient: Verified 200 OK on `/`, `/api/rl/chapters`, `/api/rl/chapters/ch01`, `/api/rl/chapters/ch18`.
---

### [2026-09-13] - Free-Tier GitHub Pages Static Deployment & Privacy Hardening
- **Static Raw Markdown Bundling (`frontend/src/entities/chapter/chapterLoader.ts`)**:
  - Incorporated Vite raw glob imports (`import.meta.glob<string>('.../*.md', { query: '?raw', import: 'default' })`) across all three curriculum tracks (`tutorials/`, `rl/tutorials/`, `rlvr/tutorials/`).
  - Added prefix-aware `findMarkdownLoader` resolving target stems and chapter IDs client-side.
  - Preserved `/api/...` fetch fallback for seamless local FastAPI backend development.
  - Enabled 100% autonomous client-side execution for all 67 chapters and 18 interactive simulation labs without requiring a live Python server.
- **Vite Configuration & Base Path (`frontend/vite.config.ts`)**:
  - Configured dynamic base URL: `base: process.env.GITHUB_ACTIONS ? '/post-training/' : (process.env.VITE_BASE_PATH || '/')`.
  - Configured `server.fs.allow: ['..']` granting local Vite dev server access to root tutorial assets.
- **CI/CD GitHub Actions Automation (`.github/workflows/deploy.yml`)**:
  - Added automated GitHub Pages deployment workflow triggered on `push` to `main` and `workflow_dispatch`.
  - Configured Node 20 environment with npm caching, `npm ci`, and `npm run build` in `frontend/`.
  - Attached official `actions/configure-pages@v5`, `actions/upload-pages-artifact@v3` (`web/dist`), and `actions/deploy-pages@v4`.
- **Privacy & Confidential Document Protection (`.gitignore`)**:
  - Added explicit exclusion patterns protecting personal documents (`docs/cv.tex`, `docs/job-descriptions/`, `*cv.tex`, `*cv.pdf`, `*confidential*.tex`).
  - Cleaned up obsolete local directory references in `docs/INDEX.md` and `rl/AGENTS.md`.
---

### [2026-09-13] - Cryptographic SHA-256 Developer Mode & Privacy Gate
- **Cryptographic SHA-256 Authorization Store (`frontend/src/entities/chapter/privacyStore.ts`)**:
  - Implemented `usePrivacyStore` verifying user passphrases via Web Crypto API `crypto.subtle.digest('SHA-256', ...)`.
  - Target hash: `2c9da6721d235446090e2b4276993e0c4a66a7b384d5004e8a32e66b0bbb6c75` (hash of `frontier-mle-2026`).
  - Stored in browser `localStorage` under `deepagents_career_unlocked` with instant one-click `lock()` method.
- **Manifest & Navigation Gating (`frontend/src/entities/manifest/`)**:
  - Added `isCareerGated: true` to `rlvr14` (Post-Training Systems & Triage Playbook) and `ch18` (Frontier Systems Architecture Alignment).
  - Sanitized public manifest descriptions and stage titles (`Stage V · 前沿架構擴展與系統設計對齊 (Frontier Systems Design)`).
  - Filtered out gated chapters and stages in `SidebarNavigation.tsx` when `!isUnlocked`.
- **Milestone Workbench Modal Gating (`frontend/src/entities/milestone/MilestoneTutorialModal.tsx`)**:
  - Completely hides Tabs 2 (架構決策防禦), 3 (系統架構深度思辨), and 4 (工程實踐亮點) when `!isUnlocked`.
  - Hides portfolio technical highlight copy buttons and architectural scope descriptions from the header.
  - Automatically resets active tab to `'pipeline'` (4-Step Pipeline Code) if locked.
- **Developer Unlock Dialog & Secret Trigger (`frontend/src/widgets/privacy/CareerUnlockModal.tsx`)**:
  - Discrete modal titled "Developer Mode / Advanced Settings" mounted in `App.tsx`.
  - Triggers via keyboard shortcut (`Ctrl+Shift+U` / `Cmd+Shift+U`) or by clicking the `v4` brand badge 5 times in 2 seconds.
- **Verification**:
  - TypeScript compilation and Vite production build passed cleanly in 8.06s with 0 errors.

---

### [2026-09-13] - Established Progressive Executable Notebook Standard Across Platform & Upgraded rlvr07
- **Progressive Executable Notebook Standard Integration**:
  - In response to user feedback on single disconnected code snippets, codified the universal **5-Stage Progressive Executable Notebook Standard** for all educational/curriculum code:
    1. **Synthetic Batch Pipeline & Tensors**: Realistic input batches, causal labels, and attention masks.
    2. **Causal Log-Prob Gathering Module**: Precision `torch.gather` extraction with shifted causal alignments.
    3. **Vectorized Loss Engine & Telemetry Signals**: Production loss function with implicit rewards, margins, and WandB metrics.
    4. **Pathological Stress Tests & Failure Mode Simulations**: Concrete multi-step iterations simulating industrial disasters (Likelihood Displacement probability collapse, Verbosity Bias length exploitation) with diagnostic telemetry traces.
    5. **Production Remediation & Comparative Ablation**: Side-by-side verification proving that SFT-anchoring and length-normalized SimPO resolve the failure modes.
  - **Alternating Execution Output Blocks**: Every Python code block is immediately followed by a styled console execution output block (````text` tagged with `[Execution Output / Telemetry Log]`) showing real tensor shapes, loss traces, and diagnostic signals.
- **Flagship Gold Standard Benchmark Upgrade (`rlvr07`)**:
  - `rlvr/tutorials/07_dpo_preference_optimization.md`: Completely overhauled Section 四 from a single 40-line snippet into the 5-stage Progressive Executable Notebook Laboratory.
  - `frontend/src/entities/chapter/data/rlvr/rlvr07.js`: Synchronized TOC anchors, summary, and module metadata.
  - `rlvr/tutorials/INDEX.md`: Updated Chapter 7 row to reflect the 5-stage laboratory.
- **Harness Contracts, Skills & Memory Synchronization**:
  - `AGENTS.md`: Enforced Prime Directive 5: Progressive Executable Notebook Standard (strictly forbidding single isolated snippets).
  - `.agents/rules/auto-index.md`: Added progressive notebook integrity checks to the execution protocol.
  - `.agents/skills/interactive-learning-platform/SKILL.md`: Upgraded Pillar 5 in the 7-Pillar standard and added verification checklist items.
  - `.agents/skills/interactive-learning-platform/references/recipe.md`: Updated Section X.3 with the 5-stage progressive laboratory blueprint and output blocks.
  - `docs/ARCHITECTURE.md`: Registered the progressive notebook presentation standard in the Presentation Layer.
  - Knowledge Item `interactive_ml_platform`: Synchronized `artifacts/overview.md` and `artifacts/features/interactive_inspector.md`.
- **Verification**:
  - Typecheck and production bundle build verified.
- **UI Bloat Elimination in Reader Canvas (`ReaderCanvas.tsx`)**:
  - Eliminated bulky text labels (`換行中` / `不換行` and `複製` / `已複製`) from code block headers, replacing with sleek, minimalist icon controls with tooltips.
  - Implemented specialized `isOutputBlock` detection for terminal/execution outputs:
    - Stripped OS window traffic lights and redundant `TEXT` language badges.
    - Extracted output bracketed headers (e.g. `[Execution Output / Batch Diagnostics]`) into a clean terminal header (`Terminal Output · Batch Diagnostics`).
---

### [2026-09-13] - ReadTheDocs-Grade Sleek Reading Body & Prism Highlighting
- **Prism.js Syntax Highlighting & Pygments Theme (`frontend/src/app/styles/index.css`)**:
  - Integrated `prismjs` for lightweight client-side syntax highlighting across Python, Bash, TypeScript, and JSON.
  - Implemented Pygments-style syntax highlighting tokens matching Sphinx ReadTheDocs in both dark mode (`#0d1117` / `#0b0f19`) and light mode (`#f8fafc`): keywords (coral/red), strings (emerald/green), functions & classes (purple), comments (muted italic slate), numbers & booleans (sky blue), and built-ins (amber).
- **Sleek Minimalist Code & Docked Output Cells (`frontend/src/widgets/reader-canvas/ReaderCanvas.tsx`)**:
  - Eliminated bulky window title bars and static text buttons from code blocks.
  - Added floating hover action pills in the top-right corner (`opacity-0 group-hover:opacity-100`) containing a subtle uppercase language tag, wrap toggle icon, and copy button with checkmark animation.
  - Docked terminal execution outputs directly beneath source code blocks (`-mt-2.5 mb-6`) with a clean `#080d16` terminal header, `#05080f` dark obsidian output canvas, and emerald diagnostic text.
- **Publication-Grade Header & Badge Strip**:
  - Streamlined chapter header to eliminate large embedded cards above the fold.
  - Formatted resources as a sleek, compact horizontal pill strip (Sphinx RTD style): Kaggle milestone playbook trigger (`Trophy`), interactive simulation lab trigger (`FlaskConical`), read time (`Clock`), competency tags, and completion toggle.
- **Interactive In-Page Table of Contents (On this page)**:
  - Added dynamic regex heading extraction (`##` and `###`) from chapter markdown content.
  - Implemented responsive right rail (`hidden xl:block w-60 shrink-0`) with `IntersectionObserver` scrollspy tracking active section headings and enabling smooth scrolling navigation.
- **Documentation & Verification**:
  - Updated `frontend/src/widgets/reader-canvas/INDEX.md` and `frontend/src/app/styles/INDEX.md`.
  - Verified `npm --prefix frontend run typecheck` passed (exit code 0).
  - Verified `npm --prefix frontend run build` compiled clean production assets in 8.30s (exit code 0).

---

### [2026-09-13] - Platform-Wide Mobile Compatibility & Responsive Touch Architecture
- **Responsive Navigation Drawer (`frontend/src/widgets/sidebar-nav/SidebarNavigation.tsx`, `frontend/src/pages/guide/GuidePage.tsx`)**:
  - Converted sidebar navigation on mobile viewports (`< md`) from an in-flow flex column to an off-canvas overlay drawer (`fixed inset-y-0 left-0 z-50 w-[285px] sm:w-[300px] max-w-[85vw]`).
  - Added dimmed backdrop overlay (`fixed inset-0 z-40 bg-black/65 backdrop-blur-xs md:hidden`) to dismiss the drawer when tapping outside.
  - Implemented auto-closing of the sidebar on mobile device initial page load (`window.innerWidth < 768`) and on chapter/milestone navigation clicks.
  - Positioned floating "目錄導航" button with responsive padding (`top-3.5 left-3.5 sm:top-4 sm:left-4`) preventing layout overlaps.
- **Reading Canvas Mobile Ergonomics (`frontend/src/widgets/reader-canvas/ReaderCanvas.tsx`)**:
  - Adjusted main reading container padding to `px-3.5 pt-14 pb-8 sm:px-8 sm:py-8` to maximize screen real estate on 360px–430px devices while ensuring safe top clearance beneath floating controls.
  - Added responsive breadcrumb navigation with text truncation and wrapping (`truncate max-w-[100px] sm:max-w-none`).
  - Chapter title protected with `break-words` and responsive typography scaling (`text-lg sm:text-2xl md:text-[1.65rem]`).
  - Introduced mobile in-page table of contents (`<details className="xl:hidden">`) allowing mobile readers to quickly jump across multi-stage lab sections without horizontal crowding.
  - Converted CodeBlock and OutputBlock floating hover actions to accessible touch controls (`opacity-80 sm:opacity-0 sm:group-hover:opacity-100`).
  - Stacked footer navigation buttons responsively (`flex-col sm:flex-row items-stretch sm:items-center`) with full-width tap targets for mobile thumbs.
- **Data Tables, Math & Diagrams Responsive Containment (`frontend/src/app/styles/index.css`, `frontend/src/shared/lib/MermaidRenderer.tsx`)**:
  - Constrained `.katex-display` with `max-width: 100% !important`, `overflow-x: auto !important`, and `-webkit-overflow-scrolling: touch`.
  - Added `-webkit-overflow-scrolling: touch` to `pre`, `code`, and `table` elements.
  - Set table wrapper min-width (`min-w-[480px]`) and `max-w-full overflow-x-auto` to preserve multi-column comparison readability without squishing columns.
  - Made Mermaid zoom controls permanently accessible on touch devices (`opacity-80 sm:opacity-0 sm:group-hover:opacity-100`) and added smooth horizontal pan scrolling.
- **Dialog & Modal Responsive Adaptation (`frontend/src/shared/ui/Dialog.tsx`, `frontend/src/entities/simulation/`, `frontend/src/entities/milestone/`)**:
  - Updated `DialogContent` default bounds to `w-[calc(100vw-1.5rem)] sm:w-full max-w-lg max-h-[92vh] p-4 sm:p-6`.
  - Upgraded `SimulationModal.tsx` and `MilestoneTutorialModal.tsx` dialog widths to `w-[calc(100vw-1rem)] sm:w-full max-w-4xl p-3.5 sm:p-6`.
  - Refactored simulation modal headers to wrap vertically on mobile (`flex-col sm:flex-row sm:items-center`).
  - Made HUD elements and manual control buttons in `CartPoleSimulator.tsx`, `PpoSimulator.tsx`, `VramSimulator.tsx`, and `GenericInteractiveLab.tsx` responsive to narrow mobile widths.
- **Verification**:
  - `npm --prefix frontend run typecheck`: Passed with 0 errors (exit code 0).
  - `npm --prefix frontend run build`: Compiled production bundle in 7.52s with 0 errors.
  - Confirmed 0 mentions of restricted keywords across the entire codebase.

---

### [2026-09-13] - Decommission Obsolete CodeInspector Widget & Legacy Un-namespaced API Routes
- **Removed Dead CodeInspector Widget (`frontend/src/widgets/code-inspector/`)**:
  - Deleted legacy unreferenced `CodeInspector.tsx` and its local `INDEX.md`. The reader experience fully relies on in-canvas ReadTheDocs/Jupyter syntax-highlighted code blocks (`CodeBlock.tsx` and `OutputBlock.tsx`).
  - Synchronized `frontend/src/widgets/INDEX.md` to reflect active widgets (`reader-canvas`, `sidebar-nav`, `privacy`).
- **Removed Obsolete Un-namespaced Backend Routes (`web/app.py`)**:
  - Stripped duplicate legacy un-prefixed APIRouter mounts (`/api/toy`, `/api/llm`, `/api/offline`, `/api/policy`, `/api/tutorial`, `/api/wiki`).
  - All backend endpoints are strictly structured under modular domain prefixes (`/api/rl/*`, `/api/rlvr/*`, `/api/deepagents/*`, `/api/eval/*`).
- **Verification & Integrity**:
  - Verified `web/app.py` compiles cleanly (`python3 -m py_compile web/app.py`).
  - Verified TypeScript typing with `npm --prefix frontend run typecheck` (exit code 0).
  - Built fresh production web bundle with `npm --prefix frontend run build` in 8.16s (exit code 0).
  - Zero documentation drift and zero mentions of restricted terms.

---

### [2026-09-13] - Frontend CSS Polish, JS Tri-Modal CodeBlock & Dependency Updates
- **Dependency Upgrades (`frontend/package.json`, `frontend/package-lock.json`)**:
  - Updated `lucide-react` to `^1.45.0` and audited 660 packages (0 vulnerabilities).
  - Audited and updated compatible dependencies via `npm update`.
- **CSS Design System & Visual Polish (`frontend/src/app/styles/index.css`)**:
  - Added `.ascii-blueprint-container`, `.ascii-blueprint-canvas`, and `.ascii-blueprint-text` styling with pixel-perfect monospace ligatures and high-contrast dark/light drafting canvas modes.
  - Enhanced `.terminal-telemetry-container` with authentic macOS micro-status dots and emerald terminal output styling.
  - Upgraded table typography and zebra striping contrast under light mode (`[data-theme="light"]`).
- **Tri-Modal Code Block Architecture (`frontend/src/widgets/reader-canvas/ReaderCanvas.tsx`)**:
  - Partitioned code rendering into three specialized modes:
    1. **Architectural Blueprints (`isAsciiDiagram`)**: Renders box-drawing topology maps with `Network` icon, blueprint header, and high-density monospace layout.
    2. **Runtime Telemetry (`isOutputBlock`)**: Renders execution logs with `Terminal` icon, micro-dots, and `Out:` gutter.
    3. **Source Code (`CodeBlock`)**: Renders syntax-highlighted code cells with `Code2` icon, language badge, and `In:` gutter.
- **Chapter Metadata Title Standardization (`frontend/src/entities/chapter/data/rlvr/`)**:
  - Stripped redundant `Chapter XX:` prefixes from `rlvr05.js`, `06.js`, `08.js`, `09.js`, `12.js`, `13.js`, `14.js`, `16.js`, `17.js`, `18.js` to match the clean naming format of `rlvrManifest.ts`.
- **Verification & Zero-Drift**:
  - Updated `frontend/src/app/styles/INDEX.md` and `frontend/src/widgets/reader-canvas/INDEX.md`.
  - Passed `npm --prefix frontend run typecheck` (exit code 0).
  - Compiled fresh production web bundle via `npm --prefix frontend run build` in 7.60s (exit code 0).

---

### [2026-09-13] - Update System Skills, Invariants & Global Harness Instructions
- **Root Harness Contract (`AGENTS.md`)**:
  - Codified Prime Directive 6: **ASCII Architectural & Systems Topology Blueprint Standard** (dense monospaced blueprints with box-drawing characters alongside Mermaid diagrams).
  - Codified Prime Directive 7: **Kaggle Hands-On Execution & End-to-End Verifiability** (100% self-contained, reproducible, fully visible code passing sandbox execution).
  - Codified Prime Directive 8: **Mobile & Touch Ergonomics** (responsive drawers, touch copy/wrap controls, overflow containment).
  - Codified Prime Directive 9: **Strict Decommissioning of Dead Assets** (zero dead widgets or duplicate un-prefixed API routes).
- **Autonomous Documentation Protocol Rule (`.agents/rules/auto-index.md`)**:
  - Integrated ASCII blueprints and Kaggle executability verification triggers.
  - Corrected relative skill resolution to `[.agents/skills/auto-index/SKILL.md](.agents/skills/auto-index/SKILL.md)`.
- **Interactive Learning Platform Skill (`.agents/skills/interactive-learning-platform/SKILL.md`, `INDEX.md`)**:
  - Updated file paths (`frontend/src/app/styles/index.css`, build outDir `web/dist/`).
  - Formalized Tri-Modal Code Block Architecture (Architectural Blueprints, Runtime Telemetry, Source Code).
  - Codified Mobile & Touch Ergonomics Standard and Home Navigation Invariant.
  - Codified decommissioning of legacy `CodeInspector.tsx` and un-namespaced backend routes.
  - Refreshed Quality & Verification Checklist.
- **RL Domain Harness Contract (`rl/AGENTS.md`)**:
  - Updated simulation catalog count (18 labs) and tri-modal code block references.
- **Integrity**: Zero documentation drift across all functional directories.

