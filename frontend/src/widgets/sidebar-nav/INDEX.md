# sidebar-nav (frontend/src/widgets/sidebar-nav)

> Auto-maintained by Agent. Do not edit manually.

## Purpose
Clean, sleek, and minimalist navigation sidebar for the DeepAgents, RL, and RLVR curriculum. Provides floating segmented track switching, instant keyboard-driven search (`/`), labs-only filter, milestone highlights, and responsive collapse/expand controls without clutter or progress distractions.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `SidebarNavigation.tsx` | TSX Component | `SidebarNavigation`: Main navigation sidebar with track switcher, search bar, labs filter, part tree accordion, and collapse controls |

## Invariants & Rules
- Must preserve active track selection across DeepAgents, RL, and RLVR.
- Must synchronize current chapter and completion state with `useChapterStore`.
- Supports dark and light themes seamlessly without hardcoded unstyled containers.
- Must provide quick keyboard navigation (`/` or `⌘K` to search, `Esc` to clear).
