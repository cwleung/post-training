# guide (frontend/src/pages/guide)

> Auto-maintained by Agent. Do not edit manually.

## Purpose
Main interactive guide page view assembling the sidebar navigation, reader canvas, code inspector, and simulation modals with responsive collapse/expand management.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `GuidePage.tsx` | TSX Component | `GuidePage`: Page shell combining `SidebarNavigation`, `ReaderCanvas`, and floating sidebar toggle controls |

## Invariants & Rules
- Controls sidebar expandability and floating open button when navigation is collapsed.
- Coordinates simulation modal launch events between reader canvas and chapter store.
