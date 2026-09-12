# interactive-learning-platform (.agents/skills/interactive-learning-platform)

> Auto-maintained by Agent. Do not edit manually.

## Purpose
Domain workflow skill for developing, maintaining, testing, and authoring content for the DeepAgents interactive learning platform (`frontend/`), enforcing the Post-Training Track 7-Pillar pedagogical standard, 18 visual simulation labs, and FSD architecture.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `SKILL.md` | File | Main skill definition: system architecture, workflows, 7-pillar pedagogical rules, 18 visual labs, Kaggle defense, and verification checklist |
| `references/` | Dir | [references/INDEX.md](references/INDEX.md) - Deep architectural evolution blueprints, active data models, and visual lab specifications |

## Invariants & Rules
- All new curriculum content must adhere to the Post-Training Track 7-Pillar standard (Intuitive Mental Model, Mermaid architecture, KaTeX math, production code cells, 4D telemetry signals, emergency runbooks, Frontier Lab interview defense).
- Code blocks belong directly in self-contained markdown cells; do not introduce external CodeInspector line annotations (`codeLines: []`).
- Simulation labs in `references/visual-labs.md` must stay synchronized with `frontend/src/entities/simulation/labCatalog.ts`.
