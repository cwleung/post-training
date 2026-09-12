# manifest (frontend/src/entities/manifest)

> Auto-maintained by Agent. Do not edit manually.

## Purpose
Curriculum manifests, reading track outlines, part divisions, and chapter search indexes.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `deepagentsManifest.ts` | TS Module | 31-chapter curriculum structure for Agent Harness Engineering |
| `rlManifest.ts` | TS Module | 18-chapter curriculum structure for Reinforcement Learning & GRPO |
| `rlvrManifest.ts` | TS Module | 18-chapter curriculum structure for Verifiable Rewards & Post-Training MLE |
| `index.ts` | TS Module | Manifest lookup helpers, search indexers, and track aggregation |

## Invariants & Rules
- Manifest chapter IDs must correspond to valid entries in `chapterLoader.ts`.
