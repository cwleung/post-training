# code-inspector (frontend/src/widgets/code-inspector)

> Auto-maintained by Agent. Do not edit manually.

## Purpose
ReadTheDocs-style unified code wrap and inspection widget featuring line number gutters, text wrap toggles, one-click clipboard copying, and inline expandable annotations.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `CodeInspector.tsx` | TSX Component | Interactive code inspector rendering line-by-line syntax-highlighted code and annotations |

## Invariants & Rules
- Synchronizes selected code line with `useChapterStore.selectedCodeLine`.
- Supports dark and light styling cleanly using CSS variables.
