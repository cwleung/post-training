# shared (frontend/src/shared)

> Auto-maintained by Agent. Do not edit manually.

## Purpose
Reusable UI primitives, styling tokens, utilities, and TypeScript type declarations shared across all layers.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `lib/` | Dir | [lib/INDEX.md](lib/INDEX.md) - General utilities, class merge helpers, and Mermaid diagram renderers |
| `types/` | Dir | [types/INDEX.md](types/INDEX.md) - Shared TypeScript interfaces, curriculum schemas, and common data contracts |
| `ui/` | Dir | [ui/INDEX.md](ui/INDEX.md) - Design system primitives: buttons, inputs, dialogs, cards, badges, and sliders |

## Invariants & Rules
- `shared/` must never import from higher-level slices (`entities`, `widgets`, `pages`, `app`).
