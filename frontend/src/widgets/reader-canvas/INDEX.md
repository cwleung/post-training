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
- Renders source code blocks with ReadTheDocs / Jupyter minimal style: Prism syntax highlighting for Python/Bash/TS/JSON, clean rounded frames, `In:` gutter, language badge, and subtle floating action pills (wrap toggle and copy).
- Renders ASCII architectural maps and systems topology diagrams as dedicated Blueprint canvases with crisp monospaced alignment, network icons, and blueprint framing.
- Renders execution outputs as sleek docked terminal panels (`Out:` gutter, micro-status indicators, emerald output) attached flush directly beneath the preceding code cell.
- Features a dynamic, scrollspy-enabled "On this page" right rail on large viewports (`xl:block`) reflecting chapter subheadings.
- Streamlines publication header with compact badge pills for Kaggle milestone playbooks, simulation labs, read time, and competency tags.
