# entities (frontend/src/entities)

> Auto-maintained by Agent. Do not edit manually.

## Purpose
Encapsulates domain models, data fetching state, and business logic for core entities including learning chapters, curriculum manifests, milestone badges, and interactive simulation runners.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `chapter/` | Dir | [chapter/INDEX.md](chapter/INDEX.md) - Chapter data models, dynamic loader, and reading progress state |
| `manifest/` | Dir | [manifest/INDEX.md](manifest/INDEX.md) - Curriculum index schemas, reading track outlines, and topic hierarchies |
| `milestone/` | Dir | [milestone/INDEX.md](milestone/INDEX.md) - Module completion milestones, career alignments, and celebration modals |
| `simulation/` | Dir | [simulation/INDEX.md](simulation/INDEX.md) - 17 interactive visual simulation labs, controllers, and modal canvases |

## Invariants & Rules
- Entities may only depend on `shared/` and must remain decoupled from higher-level widgets and pages.
