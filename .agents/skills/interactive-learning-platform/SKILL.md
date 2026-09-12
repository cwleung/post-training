---
name: interactive-learning-platform
description: >-
  Develop, maintain, test, and extend the DeepAgents interactive learning platform (React 19, Vite, TypeScript, Tailwind CSS v4, Feature-Sliced Design). Covers the 18 interactive parameter visual simulation labs, Kaggle milestone career defense workbench, multi-track curricula (DeepAgents, RL, Post-Training MLE), the Post-Training Track 7-Pillar Pedagogical Standard (Intuitive Mental Models, Mermaid visual flows, self-contained production code, 4D telemetry signals, industrial runbooks, Frontier Lab interview defense), and backend FastAPI hydration.
---

# Interactive Learning Platform Skill

This skill guides you through developing, testing, maintaining, and extending the interactive learning platform hosted in `frontend/`. It embeds the production design system, architectural invariants, and content authoring standards established by the **Post-Training Track**.

---

## External & Background References

- **Pedagogical & Content Gold Standard**: Established by the **Post-Training Track** (`rlvr/tutorials/` & `frontend/src/entities/chapter/data/rlvr/`), incorporating:
  - **UvA Deep Learning Tutorials**: Rigorous mathematical derivations, continuous reparameterizations, and parameter boundary analyses ([UvA SGA Reference](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/sampling/introduction.html)).
  - **Frontier AI Lab Engineering Practices**: Intuitive mental models, Mermaid memory & dataflow graphs, production telemetry signals, industrial emergency triage runbooks, and Frontier AI Lab (OpenAI / Anthropic / DeepMind / Apple / Meta) interview defense playbooks.
- **UI/UX & Documentation Layout Reference**: [UVA Applied Machine Learning Tutorials (ReadTheDocs / Jupyter Book)](https://uva-applied-ml.readthedocs.io/en/latest/notebooks/2_reg_knn_linreg.html).
- [Platform Architecture & Evolution Blueprint](./references/recipe.md)
- [Data Models & TypeScript Entity Interfaces](./references/data-model.md)
- [18 Visual Simulation Labs Catalog & Formulations](./references/visual-labs.md)

---

## 1. System Architecture (`frontend/src/`)

The platform is a **React 19 SPA** powered by **Vite 6**, **TypeScript**, and **Tailwind CSS v4**, organized according to **Feature-Sliced Design (FSD)**:

```
frontend/src/
├── app/                  # Application initialization, root providers, global CSS (Tailwind v4)
│   ├── App.tsx           # QueryClientProvider, theme setup, routing
│   └── app.css           # Modern Tailwind CSS v4 directives & theme variables
├── pages/                # Route compositions
│   └── guide/GuidePage.tsx # Master layout coordinating sidebar, reader canvas, and modal
├── widgets/              # Autonomous composite UI widgets
│   ├── sidebar-nav/      # Track switcher (DeepAgents, RL Track, Post-Train), chapter tree, search
│   └── reader-canvas/    # Markdown renderer, KaTeX math, deep links, milestone cards
├── entities/             # Core domain models and state stores
│   ├── chapter/          # Dynamic ESM glob loader (`chapterLoader.ts`), Zustand `chapterStore.ts`
│   ├── manifest/         # Track manifests (`deepagentsManifest`, `rlManifest`, `rlvrManifest`)
│   ├── simulation/       # 18-lab catalog (`labCatalog.ts`), `SimulationModal`, dedicated simulators
│   └── milestone/        # Kaggle milestone tutorials, `MilestoneTutorialCard`, `MilestoneTutorialModal`
└── shared/               # Reusable UI primitives, utilities, and TypeScript declarations
    ├── types/            # Core domain types: ChapterData, LabSpec, MilestoneTutorial, SiteId
    ├── lib/utils.ts      # Tailwind class merger (`clsx` + `tailwind-merge`)
    └── ui/               # Radix UI primitives: Badge, Button, Slider, Dialog
```

> [!NOTE]
> **CodeInspector Deprecation**: The separate line-by-line `CodeInspector` component has been retired from `ReaderCanvas`. Code implementations are now presented directly within self-contained, syntax-highlighted notebook markdown cells. All chapter metadata records `codeLines: []`.

### Dependency Rules (Feature-Sliced Design)
Layers can strictly only import from layers below them:
`shared` $\leftarrow$ `entities` $\leftarrow$ `widgets` $\leftarrow$ `pages` $\leftarrow$ `app`. Never import upwards or create circular dependencies across slices.

---

## 2. Core Workflows & Commands

All frontend commands should be executed from the `frontend/` directory (or using `npm --prefix frontend`):

```bash
# 1. Start development server (port 5173, proxies /api to FastAPI backend on 8000)
npm --prefix frontend run dev

# 2. Strict TypeScript typechecking
npm --prefix frontend run typecheck

# 3. Production build (typecheck + Vite bundle to frontend/dist/)
npm --prefix frontend run build

# 4. Launch backend and serve frontend assets (from repo root)
python serve.py
```

---

## 3. UI/UX Layout Patterns (ReadTheDocs & Post-Training Standard)

Inspired by ReadTheDocs, Jupyter Book, and modern post-training developer workbenches:

1. **Multi-Track Navigation & Breadcrumbs**:
   - Contextual switcher allows toggling between `DeepAgents`, `RL Track`, and `Post-Train` (`Production Post-Training MLE Handbook`).
   - Breadcrumbs clearly anchor position: `Curriculum Track > Part / Section > Chapter Title`.

2. **Notebook-Style Executable Canvas**:
   - Content is structured as high-density, interactive lecture notes.
   - Self-contained PyTorch/Transformers/PEFT code blocks include line numbers, syntax highlighting, copy buttons, and inline parameter explanations.

3. **In-Page Anchor Navigation & Heading Hierarchy**:
   - H2/H3 headings generate automatic smooth-scroll anchor links (`#section-id`).
   - Sticky table-of-contents highlight currently visible headings during reading.

4. **Interactive Simulation Deep-Links**:
   - In-text badges and launch buttons (`<button data-lab="lab_id">Launch Simulation</button>`) dynamically trigger parameter-tuning modals without disrupting reading flow.

---

## 4. Content Style & Pedagogical Standards: The Post-Training Track 7-Pillar Gold Standard

Every curriculum chapter on this platform must conform to the **7-Pillar Standard** established by the Post-Training Track:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              Post-Training Track 7-Pillar Pedagogical Anatomy                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│  1. 標題與核心名言 (Title, Epigraph & Scope)                                    │
│     • English technical keyword in title (e.g. Training Pipeline & LoRA)        │
│     • Epigraph quoting core engineering philosophy (systems thinking over brute force) │
│     • Concise summary with explicit VRAM numbers, equations, or metric targets  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  2. 核心心智模型與直觀比喻 (Intuitive Mental Model & Metaphors)                 │
│     • Visceral real-world metaphor (e.g. 大廚備料, 開卷考試裁判, 同儕相互激勵)   │
│     • Answers: What physical/system obstacle does this solve? Why do baselines fail? │
├─────────────────────────────────────────────────────────────────────────────────┤
│  3. Mermaid 視覺流架構圖與顯存分配 (Mermaid Visual Architecture & Memory Flows)  │
│     • Structured subgraphs depicting tensor shapes, memory budgets, or pipelines│
│     • Rich `classDef` styles (lora, vram, highlight, danger)                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│  4. 嚴密數學推導與演算法步驟 (Mathematical Derivations & Algorithmic Steps)      │
│     • LaTeX formulations ($...$ and $$...$$) with closed-form derivations       │
│     • Numbered algorithmic steps; boundary analysis (τ → 0, β → ∞, G limits)    │
├─────────────────────────────────────────────────────────────────────────────────┤
│  5. UvA-DLC 漸進式可執行代碼實驗室 (UvA-DLC Progressive Executable Notebook Suite) │
│     • 嚴格杜絕單一孤立代碼片段 (NEVER a single disconnected code snippet)         │
│     • 5-Stage 漸進式實驗室標準 (inspired by UvA DL Tutorial 4 Optimization & Init):│
│       1. 合成數據與批次管道 (Synthetic Batch Pipeline & Tensors)                │
│       2. 核心因果張量前向與對數機率抽取 (Causal Log-Prob Gathering with torch.gather) │
│       3. 向量化損失引擎與即時遙測字典 (Vectorized Loss Engine & WandB Telemetry)  │
│       4. 病態曲率與致命失效邊界復現 (Pathological Stress Tests & Failure Simulations)│
│       5. 工業級急救處方與對比消融實驗 (Production Remediation & Comparative Ablation)│
│     • ReadTheDocs 規範：每段代碼單元後必須緊鄰真實終端輸出區塊 (Execution Output)│
├─────────────────────────────────────────────────────────────────────────────────┤
│  6. 四維遙測監控指標與工業級急救錦囊 (4D Telemetry Signals & Industrial Runbook) │
│     • Telemetry table: reward/mean, reward/std, objective/kl, completion_length │
│     • Healthy trend patterns vs abnormal alarms & root causes                   │
│     • Step-by-step triage runbooks for OOM, gradient spikes, or reward gaming   │
├─────────────────────────────────────────────────────────────────────────────────┤
│  7. 🤔 面試深度思辨與工業界陷阱 (Frontier Lab Interview Defense & Pitfalls)      │
│     • Frontier AI Lab focus (OpenAI / DeepMind / Anthropic / Meta / Apple MLE)  │
│     • Failure Mode & Fix: battle-tested industrial failure modes & solutions    │
│     • High-frequency interview questions with deep first-principles answers     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Content Rules

1. **Intuitive Mental Models Before Equations**:
   - Never introduce an equation in isolation. Always ground it in a physical or operational metaphor:
     - *Data Prep*: **「大廚備料」心智模型** (XML boundaries, left-padding causal masking).
     - *Reward Engineering*: **「開卷考試裁判」心智模型** (Deterministic regex verifiers vs neural RM).
     - *GRPO*: **「同儕相互激勵」心智模型** (Peer relative advantage eliminating Critic network).
     - *LoRA / PEFT*: **「透明描圖紙」心智模型** (Frozen 70B backbone + low-rank adapter).
     - *Pass@k*: **「抽獎不放回」心智模型** (Hypergeometric unbiased combinatorial estimator).
     - *KL Penalty*: **「水庫調洪」心智模型** (Balancing policy exploration vs language collapse).
     - *veRL Distributed*: **「雙輪驅動引擎」心智模型** (Decoupled Rollout & Learner clusters).
     - *SimPO*: **「卸下沉重行囊」心智模型** (Reference-free margin eliminating RM memory).

2. **Mermaid Visual Architecture & Memory Footprints**:
   - Include clear visual diagrams with customized CSS classes:
   ```mermaid
   graph LR
       IN["Input x"] --> BASE["Frozen W_0 (4-bit NF4)"]
       IN --> DOWNA["Down-proj A (d × r)"]
       DOWNA --> UPB["Up-proj B (r × d)"]
       BASE --> SUM["Sum ⊕"]
       UPB --> SUM
       SUM --> OUT["Output h"]
       classDef lora fill:#1a365d,stroke:#3182ce,stroke-width:1.5px,color:#fff;
       class DOWNA,UPB lora;
   ```

3. **UvA-DLC 漸進式代碼實驗室與交替輸出區塊 (UvA-DLC Progressive Notebook & Alternating Outputs)**:
   - 參照 [UvA DL Notebook Tutorial 4 (Optimization & Initialization)](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial4/Optimization_and_Initialization.html) 的黃金規範，**嚴禁在章節中只展示孤立的單一函數或片段**。
   - 代碼小節必須按照 5 階流水線層層推進：
     1. **合成數據與批次管道 (Synthetic Batch Pipeline & Tensors)**：以自包含代碼構建真實形狀的輸入張量、Attention Masks 與因果 Label 遮蔽。
     2. **因果對數機率抽取核心模組 (Causal Log-Prob Gathering with torch.gather)**：實現嚴格自回歸位移切片、詞表索引 gather、遮罩過濾與平均/求和累積。
     3. **向量化損失引擎與即時遙測字典 (Vectorized Loss Engine & Telemetry Signals)**：生產級向量化損失計算，同時輸出標量 Loss 與 WandB 遙測字典。
     4. **病態曲率與致命失效邊界模擬 (Pathological Stress Tests & Failure Simulations)**：主動編寫多步優化循環或極限邊界測試，重現諸如「概率塌陷（Likelihood Displacement）」或「長度作弊陷阱（Verbosity Bias Trap）」等工業現場災難，並打印每步衰竭軌跡。
     5. **工業級急救處方與對比消融實驗 (Production Remediation & Comparative Ablation)**：實現具體修復方案（如 SFT 正則、SimPO 長度歸一化），並在相同病態輸入下進行橫向消融對比，以真實數據證明修復成功。
   - **交替單元標準**：每一個 Python 可執行代碼塊（` ```python `）必須緊鄰其獨立的終端控制台輸出區塊（` ```text ` 標記 `[Execution Output / Telemetry Log]`），呈現真實張量維度、方差、損失收斂軌跡與報警信號。

4. **Four-Dimensional Telemetry Table**:
   - Every training or algorithm chapter must feature a telemetry radar table:
   | 遙測信號 (Telemetry Signal) | 健康趨勢形態 | 異常警報與失效原因 |
   |---|---|---|
   | `reward/mean` | 單調平穩爬升 ($0.2 \to 1.4$) | 停滯在 $0.0$（獎勵稀疏）或瞬間垂直飆至滿分（作弊刷分） |
   | `reward/std` | 保持健康方差 ($0.2 \le \sigma \le 0.7$) | 驟降至 $\approx 0.0$（組內同質化／策略坍塌） |
   | `objective/kl` | 平滑緩步微增 ($0.05 \to 0.8$) | 突破 $> 5.0$（策略嚴重飄移／語言能力破碎） |
   | `completion_length` | 自然緩慢延伸（學會自我檢驗） | 幾十步內頂到上限截斷（死循環作弊） |

5. **Industrial Emergency Runbooks (急救錦囊)**:
   - Provide concrete steps for real-world incidents:
     - **CUDA OOM**: Reduce `max_completion_length`, decrease group size $G$ while increasing `gradient_accumulation_steps`, enable FlashAttention-2, switch to PagedAdamW 8-bit.
     - **Length Explosion**: Inject soft length penalty $R_{\text{len}} = -\lambda \cdot \max(0, L - L_{\text{target}})$ or employ Dr. GRPO token-level normalization.
     - **Gradient Spike**: Clamp `max_grad_norm = 1.0`, enforce FP32 AdamW master weights.

6. **Frontier Lab Interview Defense Section**:
   - Conclude each chapter with a dedicated `🤔 面試深度思辨與工業界陷阱 (Interview Insight & Production Pitfalls)` block:
     - **Failure Mode & Fix**: Describe how production systems fail and how to fix them.
     - **High-Frequency Question & Model Answer**: Deep technical Q&A ready for MLE career defense.

7. **Language & Terminology Standard**:
   - Write conceptual explanations, mental models, and interview insights in fluent **Traditional Chinese (繁體中文)**.
   - Retain standard international English terminology for all machine learning concepts (e.g. *Rollout, Verifiers, KL Divergence, Pass@k, Loss, DPO, SimPO, PRM, GAE, PPO, LoRA, KV-Cache, Roofline Model, Throughput*).

---

## 5. 18 Interactive Visual Simulation Labs

The platform embeds **18 parameter-tuning visual simulation laboratories** in `frontend/src/entities/simulation/labCatalog.ts`:

| Lab ID | Title | Category | Key Concept / Simulator |
| :--- | :--- | :--- | :--- |
| `cartpole` | CartPole 物理動力學與基礎控制 | `control` | `CartPoleSimulator.tsx` (二階拉格朗日物理動力學) |
| `bandit` | 多臂老虎機·ε-Greedy 與 UCB1 | `control` | 累積遺憾界限比較、探索-利用權衡 |
| `dqn` | DQN 經驗回放與 TD-Error 震盪 | `control` | Target 網絡複製週期 $C$、Q 值高估現象 |
| `policy_gradient` | 策略梯度 REINFORCE 與 Baseline | `control` | Value Baseline 方差削減散佈圖 |
| `actor_critic` | Actor-Critic 與 GAE 優勢估計 | `control` | $\text{GAE}(\gamma, \lambda)$ 偏差-方差連續平滑權衡 |
| `continuous` | 連續動作空間高斯策略與熵正則化 | `control` | 溫度係數 $\alpha$、$\tanh$ 動作壓縮變形 |
| `offline` | 離線保守策略 (CQL) 與分佈偏移 | `control` | OOD 動作注入、保守 Q 學習壓制過估 |
| `ppo` | PPO 截斷代理目標 (Clipped Objective) | `alignment` | `PpoSimulator.tsx` (重要性權重比率 $r_t$、優勢 $A_t$、截斷邊界 $\epsilon$) |
| `dpo` | DPO 隱式獎勵邊界 | `alignment` | 溫度 $\beta$、勝者/敗者隱式獎勵推拉梯度 |
| `grpo` | GRPO 群組相對優勢歸一化 (DeepSeek-R1) | `alignment` | `GrpoSimulator.tsx` (群組規模 $G$、零 Critic 網絡標準化) |
| `rlvr_data` | XML 標籤解析與左側填充 (Left-Padding) | `rlvr_systems` | 正則抽取、Left-padding 因果遮蔽矩陣 |
| `rlvr_rewards` | 複合獎勵函數加權平衡器 | `rlvr_systems` | 正確率、格式覆蓋與長度懲罰平衡 (防範 Reward Hacking) |
| `rlvr_lora_vram` | 顯卡 VRAM 預算與 LoRA 顯存分配器 | `rlvr_systems` | `VramSimulator.tsx` (16GB T4 / 24GB 3090 / 80GB A100 顯存與 OOM 預測) |
| `rlvr_passk` | Pass@k 與多數投票 (Majority@k) 組合曲線 | `reasoning` | Chen et al. 無偏組合估計 vs 多數共識投票 |
| `rlvr_scaling` | 推理期思考 Token 擴展定律 | `reasoning` | 思考長度對數擴展 $\alpha \log(\text{Tokens}) + \beta \log(G)$ |
| `rlvr_coldstart` | SFT 冷啟動蒸餾與拒絕採樣 | `rlvr_systems` | 思維鏈密度閾值 $\tau$、高質量推理軌跡採樣篩選 |
| `rlvr_distributed` | 3D-HybridEngine (veRL + vLLM) 計算器 | `rlvr_systems` | GPU 叢集規模、Tensor Parallelism TP、PagedAttention 加速比 |
| `rlvr_simpo` | SimPO (無參考模型) vs 經典 DPO 邊界對比 | `alignment` | 目標邊界 $\gamma$、消除 Reference Model 與長度偏見 |

---

## 6. Kaggle Milestone & Interview Defense System

Located in `frontend/src/entities/milestone/`, this system turns theoretical lessons into portfolio-grade engineering assets:

1. **4-Step Kaggle Practice Tutorial**:
   - **Step 1: Environment & Setup**: GPU accelerator configs (T4 $\times$ 2 / P100 / A100), Pip packages.
   - **Step 2: Core Pipeline Implementation**: Production-grade, copy-pasteable PyTorch / HuggingFace pipeline.
   - **Step 3: Verification & Evaluation**: Deterministic benchmark scripts, loss curves, and Pass@k evaluations.
   - **Step 4: Interview Defense & STAR Playbook**:
     - *Situation*: Real-world problem context and operational requirements.
     - *Task*: Engineering constraints, latency limits, and VRAM budgets.
     - *Action*: Technical decisions (e.g. why GRPO instead of PPO, why left-padding, why LoRA $r=16$).
     - *Result*: Quantifiable metrics, GPU cost savings, and Pass@k improvements.
2. **One-Click Career Portfolio Generator**:
   - Automatically renders formatted, high-impact resume bullet points aligned with top-tier AI lab JD requirements (OpenAI, DeepMind, Anthropic, Apple).
3. **UI Integration**:
   - `MilestoneTutorialCard`: Embedded at the conclusion of each curriculum module in `ReaderCanvas`.
   - `MilestoneTutorialModal`: Fullscreen interactive workbench accessible via sidebar pills or card click.

---

## 7. Data Loading & Hydration Architecture

Chapters are dynamically loaded in `frontend/src/entities/chapter/chapterLoader.ts`:

1. **Dynamic ESM Glob Import**:
   `import.meta.glob<{ default: ChapterData }>('./data/*/*.js')` loads chapter payloads on demand with zero initial bundle overhead.
2. **Backend API Hydration**:
   If live tutorial updates or markdown files are served from FastAPI (`serve.py`), `chapterLoader` transparently queries:
   - `/api/deepagents/tutorials/{id}`
   - `/api/rlvr/chapters/{id}`
   and dynamically populates `data.markdownContent`.
3. **Metadata Invariant**:
   Chapter data files (`frontend/src/entities/chapter/data/*/*.js`) define `id`, `num`, `title`, `file`, `hasVisualizer`, `readTime`, `summary`, `toc`, and `codeLines: []`.
4. **Zustand Chapter Store**:
   `useChapterStore` tracks the active `siteId`, `currentChapterId`, search filter, sidebar visibility, and milestone modal state.

---

## 8. Quality & Verification Checklist

Before finalizing any changes to the interactive platform:

- [ ] **Typecheck**: `npm --prefix frontend run typecheck` exits with code 0.
- [ ] **Build Check**: `npm --prefix frontend run build` completes successfully.
- [ ] **FSD Layer Boundaries**: No widget or page code is imported into `shared` or `entities`.
- [ ] **7-Pillar Content Conformance**:
  - [ ] Chapter opens with clear title, epigraph, and quantitative scope.
  - [ ] Features an intuitive mental model with real-world metaphors.
  - [ ] Includes a styled Mermaid visual architecture / memory allocation graph.
  - [ ] Provides rigorous KaTeX formulas and step-by-step mathematical derivations.
  - [ ] **UvA-DLC 5-Stage Progressive Code Conformance**:
    - [ ] Strictly zero isolated single snippets; code is an end-to-end 5-stage laboratory (Data Setup $\to$ Causal Gathering $\to$ Vectorized Engine $\to$ Pathological Simulation $\to$ Remediation & Ablation).
    - [ ] Alternating Notebook Blocks: Every Python code block is immediately followed by a dedicated console execution output block (`[Execution Output / Telemetry Log]`).
    - [ ] Reproduces real industrial failure modes (e.g. Likelihood Displacement, Verbosity Bias) with concrete iteration traces.
  - [ ] Features a 4D telemetry signals table and an industrial emergency triage runbook.
  - [ ] Concludes with Frontier Lab Interview Defense (Failure Mode & Fix + High-Frequency Q&A).
  - [ ] Uses fluent Traditional Chinese with standard English machine learning keywords.
- [ ] **Lab Functionality**: New or modified labs in `labCatalog.ts` render cleanly within `SimulationModal`.
- [ ] **No CodeInspector Regressions**: Code is embedded directly in markdown; chapter metadata maintains `codeLines: []`.
- [ ] **Milestone Consistency**: Parts with `milestone` metadata have corresponding tutorial definitions in `milestoneTutorials.ts` or fallback gracefully.
- [ ] **Responsive & Theme**: Verified in both Dark and Light modes; sidebar and canvas adapt cleanly across desktop and mobile screens.
