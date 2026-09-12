# app (frontend/src/app)

> Auto-maintained by Agent. Do not edit manually.

## Purpose
Initializes global application layout, context providers, router entry, and core stylesheets.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `App.tsx` | File | Top-level React component orchestrating navigation and shell layout |
| `styles/` | Dir | [styles/INDEX.md](styles/INDEX.md) - Global styles, Tailwind CSS v4 design tokens, and theme palettes |

## Invariants & Rules
- May import from all lower slices (`pages`, `widgets`, `entities`, `shared`).
- Avoid storing component-specific logic directly in `App.tsx`.
