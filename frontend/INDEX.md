# frontend

> Auto-maintained by Agent. Do not edit manually.

## Purpose
Vite, React, and TypeScript Single Page Application providing interactive learning guides, code inspectors, and RL simulation canvases.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `index.html` | File | Single Page Application entrypoint HTML document |
| `package.json` | File | Node.js project manifest, dependencies, and build scripts |
| `package-lock.json` | File | Locked npm dependency tree |
| `tsconfig.json` | File | TypeScript compiler configuration and path aliases |
| `tsconfig.tsbuildinfo` | File | TypeScript incremental compilation cache |
| `vite.config.ts` | File | Vite build configuration, plugins, and proxy rules |
| `src/` | Dir | [src/INDEX.md](src/INDEX.md) - Application source code organized by Feature-Sliced Design |

## Invariants & Rules
- Exclude `node_modules/` and build artifacts (`dist/`) from indexing.
- Code should follow TypeScript strict mode standards.
