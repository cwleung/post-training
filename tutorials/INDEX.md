# tutorials

> Auto-maintained by Agent. Do not edit manually.

## Purpose
Comprehensive 31-chapter curriculum and reference syllabus covering AI Agent Harness Engineering, multi-agent delegation, runtime rubrics, and production evaluation loops.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `01-harness-engineering-basics.md` | File | Harness architecture vs runtime, prompt inflation, and 4 capability pillars |
| `02-the-deep-agents-stack.md` | File | Middleware stack layering order, lean vs full stack, merge semantics |
| `03-custom-middleware.md` | File | Writing custom `AgentMiddleware`, `wrap_tool_call`, and stack surgery |
| `04-backends-and-permissions.md` | File | Sandbox VFS, state/store backends, and `FilesystemPermission` safety |
| `05-harness-profiles.md` | File | Model-specific defaults, tool exclusion, and GP subagent controls |
| `06-subagents-and-delegation.md` | File | `task` tool, declarative subagent orchestration, and context isolation |
| `07-context-management.md` | File | Dynamic skills, episodic memory, offloading, and runtime caching |
| `08-human-in-the-loop.md` | File | `interrupt_on`, human-in-the-loop approvals, and state resumption |
| `09-putting-it-together.md` | File | Production research agent assembly combining all harness components |
| `10-middleware-deep-dive.md` | File | Async pipeline flow control, command writing, and `jump_to` semantics |
| `11-trajectory-evaluation.md` | File | Deterministic trajectory assertions and AgentEvals matching paradigms |
| `12-rubric-grading.md` | File | Runtime rubric middleware, judge subagents, and self-critique loops |
| `13-end-to-end-benchmarking.md` | File | VFS side-effect assertions, LangSmith batch runs, and pytest testing |
| `14-event-streaming.md` | File | Real-time typed projection streaming, SSE adapters, and telemetry spans |
| `15-multi-agent-failure-taxonomy.md` | File | MAST 14 failure modes, delegation cycle detection, and `pass@k` |
| `16-security-red-teaming.md` | File | Indirect prompt injection, tool poisoning, and adversarial red teaming |
| `17-eval-as-rl-environment.md` | File | Deterministic verifier design, difficulty calibration, and RL env models |
| `18-llm-as-judge-alignment.md` | File | Cohen's Kappa judge calibration, G-Eval alignment, and GRPO reward weights |
| `19-eval-infrastructure.md` | File | Live dataset flywheels, evaluation-driven development (EDD), and CI gates |
| `20-harness-as-context.md` | File | Model/Harness/Skill triad, context compression, and tool design rules |
| `21-benchmark-integrity.md` | File | Decontamination, benchmark saturation, and cost as a first-class metric |
| `22-observability-five-surfaces.md` | File | 5-surface observability (Output/Trace/Memory/Environment/Mechanistic) |
| `23-error-analysis.md` | File | Root cause analysis, axial coding, and error taxonomy classification |
| `24-multi-turn-evaluation.md` | File | Long-horizon multi-turn evaluations, drift dampening, and session isolation |
| `25-rag-evaluation.md` | File | RAG triad, claim-level faithfulness, and retriever vs generator separation |
| `26-production-monitoring.md` | File | KL divergence drift detection, evaluation flywheels, and circuit breakers |
| `27-cost-latency-metrics.md` | File | Token economics, Pareto frontiers, and AgentHealthDashboard metrics |
| `28-runnable-examples.md` | File | Runnable catalog of 22 core Python harness modules with architecture patterns |
| `29-tool-call-evaluation-stack.md` | File | 4-layer tool call evaluation stack (Selection, Extraction, Digest, Recovery) |
| `30-checkpoint-replay.md` | File | Deterministic checkpoint replay, snapshotting, and time-travel debugging |
| `31-self-improving-eval-loop.md` | File | Generate-Critique-Revise loops, Reflexion, DSPy MIPROv2 auto-compilation |

## Invariants & Rules
- All tutorial chapters are sequentially numbered and self-contained.
- File references in `web/routers/deepagents.py` depend on the exact IDs and titles listed here.

---

## Curriculum Syllabus & Milestone Modules

### Module I — 執行框架工程與中介軟體基礎（Ch 01–05）
> **🎯 里程碑**：建立沙箱 VFS 虛擬檔案系統與自訂中介軟體攔截器  
> **💼 對齊崗位**：Agent Platform Engineer (LangChain / Anthropic Tooling)

### Module II — 代理構建、委派與人機協作（Ch 06–10）
> **🎯 里程碑**：實作具備狀態隔離的子代理委派調度與人機介入 (HITL) 中斷恢復  
> **💼 對齊崗位**：Multi-Agent Systems Engineer (OpenAI / Cursor)

### Module III — 多輪軌跡評估、Rubrics 與安全防護（Ch 11–16）
> **🎯 里程碑**：建立 CI/CD 軌跡斷言、動態 Rubrics 評判與 Prompt 注入紅隊防禦  
> **💼 對齊崗位**：AI Safety & Evaluation Engineer (DeepEval / Anthropic)

### Module IV — 評估即 RL 環境、裁判對齊與基準誠信（Ch 17–21）
> **🎯 里程碑**：設計確定性 Verifiers、Kappa 校準的 G-Eval 裁判與防污染黃金測試集  
> **💼 對齊崗位**：Evaluation & Benchmarks MLE (Meta FAIR / OpenAI)

### Module V — 全鏈路可觀測性、長任務與生產經濟學（Ch 22–27）
> **🎯 里程碑**：整合 OTel 五維追蹤、RAG 三角驗收與 Token 成本/延遲 Pareto 最優化  
> **💼 對齊崗位**：Production GenAI Systems Engineer (Apple / Datadog)

### Module VI — 前沿自改進迴圈、確定性除錯與實戰庫（Ch 28–31）
> **🎯 里程碑**：達成 Reflexion + DSPy MIPROv2 自編譯全閉環與可執行工程庫  
> **💼 對齊崗位**：Senior / Lead Agent Architect (Anthropic / OpenAI)
