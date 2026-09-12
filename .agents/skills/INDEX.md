# skills (.agents/skills)

> Auto-maintained by Agent. Do not edit manually.

## Purpose
Workflow skills repository for Antigravity agents within the workspace customization root (`.agents/skills/`). Houses specialized domain workflows and procedural guidelines.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `auto-index/` | Dir | [auto-index/SKILL.md](auto-index/SKILL.md) - Autonomous Documentation Protocol skill to generate and maintain `INDEX.md` and `docs/` suite |
| `excalidraw-skill/` | Dir | [excalidraw-skill/SKILL.md](excalidraw-skill/SKILL.md) - Programmatic canvas toolkit for diagramming, visual layout, and scene generation |
| `interactive-learning-platform/` | Dir | [interactive-learning-platform/INDEX.md](interactive-learning-platform/INDEX.md) - Post-Training Track 7-Pillar pedagogical standard, 18 visual simulation labs, and FSD architecture guidelines |
| `rubric-evaluation/` | Dir | [rubric-evaluation/SKILL.md](rubric-evaluation/SKILL.md) - Systematic multi-dimensional rubric evaluation, auditing, and self-improving workflow |

## Invariants & Rules
- All skills must contain a valid `SKILL.md` with standard YAML frontmatter (`name`, `description`).
- Only active, supported domain workflow skills are kept here; obsolete legacy route mutation skills (`add_data`, `add_jd`, `add_resource`) have been decommissioned.
- Changes to skill definitions must maintain zero documentation drift across local indexes.
