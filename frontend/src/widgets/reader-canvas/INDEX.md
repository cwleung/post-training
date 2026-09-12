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
- Renders source code blocks with UvA DLC ReadTheDocs minimal style: Prism syntax highlighting for Python/Bash/TS/JSON, clean rounded frames, and subtle floating hover action pills in top-right (no bulky text buttons or window title bars).
- Renders execution outputs as sleek docked terminal panels (`Terminal Output`) attached flush directly beneath the preceding code cell without OS chrome.
- Features a dynamic, scrollspy-enabled "On this page" right rail on large viewports (`xl:block`) reflecting chapter subheadings.
- Streamlines publication header with compact badge pills for Kaggle milestone playbooks, simulation labs, read time, and competency tags.
