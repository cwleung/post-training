# widgets (frontend/src/widgets)

> Auto-maintained by Agent. Do not edit manually.

## Purpose
Composite, autonomous UI blocks providing multi-component features such as interactive line-by-line code inspection, markdown rendering canvas, and curriculum sidebar navigation.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `code-inspector/` | Dir | [code-inspector/INDEX.md](code-inspector/INDEX.md) - ReadTheDocs-style code wrap component with line gutter, wrap toggle, copy button, and annotations |
| `reader-canvas/` | Dir | [reader-canvas/INDEX.md](reader-canvas/INDEX.md) - Markdown reader rendering KaTeX equations, callouts, and embedded lab simulation canvases |
| `sidebar-nav/` | Dir | [sidebar-nav/INDEX.md](sidebar-nav/INDEX.md) - Collapsible sidebar tree navigation with search filter and completion tracking |
| `privacy/` | Dir | [privacy/INDEX.md](privacy/INDEX.md) - Developer mode and career authorization console dialog |

## Invariants & Rules
- Widgets may compose multiple entities and shared components, but must not import from sibling widgets or parent pages.
