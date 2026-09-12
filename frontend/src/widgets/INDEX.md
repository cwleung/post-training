# widgets (frontend/src/widgets)

> Auto-maintained by Agent. Do not edit manually.

## Purpose
Composite, autonomous UI blocks providing multi-component features such as the markdown rendering canvas with embedded lab simulations, curriculum sidebar navigation, and privacy/career console dialogs.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `reader-canvas/` | Dir | [reader-canvas/INDEX.md](reader-canvas/INDEX.md) - Markdown reader rendering KaTeX equations, callouts, and embedded lab simulation canvases |
| `sidebar-nav/` | Dir | [sidebar-nav/INDEX.md](sidebar-nav/INDEX.md) - Collapsible sidebar tree navigation with search filter and completion tracking |
| `privacy/` | Dir | [privacy/INDEX.md](privacy/INDEX.md) - Developer mode and career authorization console dialog |

## Invariants & Rules
- Widgets may compose multiple entities and shared components, but must not import from sibling widgets or parent pages.
