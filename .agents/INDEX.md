# .agents

> Auto-maintained by Agent. Do not edit manually.

## Purpose
Root directory for repository-scoped Antigravity customizations, harboring persistent rules and local workflow skills.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `rules/` | Dir | [rules/INDEX.md](rules/INDEX.md) - Continuous agent execution rules (e.g., `auto-index.md` zero-drift invariant) |
| `skills/` | Dir | [skills/INDEX.md](skills/INDEX.md) - Active domain workflow skills (`auto-index`, `excalidraw-skill`, `interactive-learning-platform`, `rubric-evaluation`) |

## Invariants & Rules
- Recognized by Antigravity IDE as the primary workspace customization root (`.agents/`).
- Only active, repository-relevant skills are kept in `skills/`; obsolete legacy page-manipulation skills are retired.
- All additions, modifications, or deletions of rules and skills must preserve zero documentation drift.

