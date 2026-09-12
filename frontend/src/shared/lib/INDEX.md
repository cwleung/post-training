# lib (frontend/src/shared/lib)

> Auto-maintained by Agent. Do not edit manually.

## Purpose
Utility helpers, class name merger, and diagram rendering components.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `MermaidRenderer.tsx` | TSX Component | Dynamic Mermaid diagram renderer with dark mode alignment, automatic lexical syntax sanitizer (subgraphs & pipes), and DOM cleanup |
| `utils.ts` | TS Module | Tailwind class name merging (`cn`) helper |

## Invariants & Rules
- Shared utility functions must be side-effect free.
