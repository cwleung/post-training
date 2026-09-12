# milestone (frontend/src/entities/milestone)

> Auto-maintained by Agent. Do not edit manually.

## Purpose
Entities, data models, and interactive tutorial components for Milestone practice guides. Converts curriculum milestone goals into hands-on step-by-step Kaggle practice tutorials, system architecture defense playbooks, and engineering portfolio technical highlights.

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `milestoneTutorials.ts` | TS Module | Complete registry of 16 comprehensive Kaggle milestone tutorials across all tracks (Post-Training 1-5, RL 1-5, DeepAgents 1-6), hardware specs, expected runtimes, 4-step runnable Python pipelines, STAR playbooks, multi-variant technical highlights, and frontier lab architectural Q&A |
| `MilestoneTutorialCard.tsx` | TSX Component | In-canvas interactive milestone card displaying hardware/runtime badges, architectural scope, quick STAR highlights preview, in-card runnable code runner, and fullscreen workbench trigger |
| `MilestoneTutorialModal.tsx` | TSX Component | Fullscreen 4-tab interactive workbench: 4-Step Pipeline Code with continuous End-to-End full view (all 4 steps visibly rendered sequentially), unified consolidated Python script card, quick-jump anchors, per-step copy buttons, Architecture Decisions Workbench, Frontier Lab Systems Deep Dive Q&A, and Technical Highlights Generator |
| `index.ts` | TS Module | Public barrel export for milestone entities |

## Invariants & Rules
- Each milestone must provide concrete, self-contained, runnable Kaggle Python code and structured objectives across 4 standardized steps (Environment/Data -> Core Algorithm -> Evaluation/Metrics -> Production/Defense).
- Must provide hardware specifications (GPU VRAM budget, CUDA), runtime estimates, structured STAR playbooks (S/T/A/R), multi-variant technical highlights (Standard STAR, Metric-Driven, Systems & Infra), and Tier-1 AI lab architectural Q&As.
- Seamlessly accessible from both `SidebarNavigation` milestone links and `ReaderCanvas` embedded cards.

