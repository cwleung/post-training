# chapter (frontend/src/entities/chapter)

> Auto-maintained by Agent. Do not edit manually.

## Purpose
Chapter data models, dynamic chapter loader, and persistent reading progress state slice.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `chapterLoader.ts` | TS Module | Lazy-loads chapter data dynamically for `deepagents`, `rl`, and `rlvr` tracks with bundled static raw markdown support for GitHub Pages and API fallback |
| `chapterStore.ts` | TS Module | Zustand persistent state store for active track, chapter ID, completion status, and theme |
| `privacyStore.ts` | TS Module | Zustand persistent store for developer authorization and career content privacy lock |
| `data/` | Dir | Static data modules containing chapter text, quizzes, and code definitions |

## Invariants & Rules
- Persists user progress (`doneChapters`) to browser `localStorage`.
- Loads chapter content asynchronously to keep bundle size lightweight.
