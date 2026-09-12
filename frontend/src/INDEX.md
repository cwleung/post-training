# src (frontend/src)

> Auto-maintained by Agent. Do not edit manually.

## Purpose
Root application source code structured according to Feature-Sliced Design (FSD), containing layers for app initialization, pages, composite widgets, domain entities, and shared primitives.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `main.tsx` | File | DOM root mounting and application bootstrap |
| `vite-env.d.ts` | File | TypeScript client environment type definitions |
| `app/` | Dir | [app/INDEX.md](app/INDEX.md) - Global application providers, routing, and stylesheet setup |
| `pages/` | Dir | [pages/INDEX.md](pages/INDEX.md) - Top-level page compositions and route views |
| `widgets/` | Dir | [widgets/INDEX.md](widgets/INDEX.md) - Self-contained composite UI widgets (inspectors, canvas, navigation) |
| `entities/` | Dir | [entities/INDEX.md](entities/INDEX.md) - Core domain models and state slices (chapter, manifest, simulation) |
| `shared/` | Dir | [shared/INDEX.md](shared/INDEX.md) - Reusable UI primitives, utilities, and TypeScript types |

## Invariants & Rules
- Adhere strictly to Feature-Sliced Design dependency rules: lower layers cannot import from upper layers (`shared` <- `entities` <- `widgets` <- `pages` <- `app`).
