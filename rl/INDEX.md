# rl

> Auto-maintained by Agent. Do not edit manually.

## Purpose
Core reinforcement learning and in-context policy optimization engine, implementing GRPO algorithms, adaptive policy feedback loops, Kairos append-only memory logging, and offline training pipelines.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `AGENTS.md` | File | Domain-specific harness contract and research invariants |
| `IN_CONTEXT_POLICY_SPEC.md` | File | Formal specification for in-context policy transitions (T1–T5 / F1–F7) |
| `README.md` | File | Comprehensive overview of the RL research and teaching codebase |
| `__init__.py` | File | Module package initialization and sys.path bootstrap for internal scripts |
| `demo_log_to_grpo.py` | File | Pipeline converting Kairos interaction logs to GRPO training samples |
| `grpo_llm_char.py` | File | Character-level autoregressive LLM trained with GRPO and reward shaping |
| `grpo_offline_resample.py` | File | Offline dataset rejection sampling and advantage filtering |
| `grpo_play_core.py` | File | Shared environment mechanics, state transitions, and reward functions for FastAPI router and live demos |
| `grpo_toy.py` | File | Pedagogical, minimal implementation of Group Relative Policy Optimization |
| `kairos_log.py` | File | Append-only dual-view trajectory logging and memory management |
| `memory_selector.py` | File | Dynamic selection and compression of historical interaction traces |
| `memory_selector_demo.py` | File | Demonstration script for memory retrieval and summarization strategies |
| `policy_feedback.py` | File | In-context policy adaptation, feedback parsing, and anomaly detection |
| `policy_feedback_demo.py` | File | Interactive simulation demo of iterative system prompt self-refinement |
| `training_pipeline.py` | File | End-to-end training orchestrator connecting buffers, models, and optimizers |
| `tutorials/` | Dir | [tutorials/INDEX.md](tutorials/INDEX.md) - 18-chapter full-stack RL & alignment curriculum in Traditional Chinese |
| `JD/` | Dir | [JD/INDEX.md](JD/INDEX.md) - Target job descriptions and gap-analysis matrix |
| `paper/` | Dir | [paper/INDEX.md](paper/INDEX.md) - Research paper draft, LaTeX sources, and standalone code |
| `requirement/` | Dir | [requirement/INDEX.md](requirement/INDEX.md) - Benchmark requirements and skill level specifications |
| `docs/` | Dir | [docs/INDEX.md](docs/INDEX.md) - Curriculum vitae and LaTeX documentation |

## Invariants & Rules
- Never fabricate digests, benchmark scores, or citations (Rule 0 in `AGENTS.md`).
- Core RL scripts must be runnable standalone via Python CLI without requiring FastAPI or frontend dependencies.
- Model checkpoints and evaluation logs must be written to `checkpoints/` and `logs/`, never committed to source directories.
