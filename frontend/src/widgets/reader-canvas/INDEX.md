# reader-canvas (frontend/src/widgets/reader-canvas)

> Auto-maintained by Agent. Do not edit manually.

## Purpose
Markdown reader canvas parsing KaTeX formulas, callouts, and embedded interactive simulation components.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `ReaderCanvas.tsx` | TSX Component | Central reading canvas with custom markdown component overrides, equation rendering, and simulation triggers |

## Invariants & Rules
- Intercepts internal markdown links and routes them via `useChapterStore`.
- Embeds interactive visual labs seamlessly inside curriculum content.
