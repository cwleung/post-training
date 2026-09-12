# styles (frontend/src/app/styles)

> Auto-maintained by Agent. Do not edit manually.

## Purpose
Core global styling, Tailwind CSS v4 design token declarations, typography themes, and dark/light mode palette variables.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `index.css` | CSS Stylesheet | Global CSS entrypoint: `@theme` token definitions, `:root` (dark) and `[data-theme="light"]` CSS variables, `@tailwindcss/typography` styling overrides, scrollbar styling, callout styles, and KaTeX contrast rules |

## Invariants & Rules
- All Tailwind CSS v4 semantic tokens (`--color-background`, `--color-foreground`, `--color-card`, `--color-sidebar`, etc.) must be declared within the `@theme` block.
- Both dark and light themes must be fully defined with accessible contrast ratios.
- Do not import component-specific styles here; keep them scoped to their respective components or utility classes.
