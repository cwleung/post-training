# AGENTS.md — instructions for the agent working in this repo

> This file is the **harness contract**: global rules and invariants that are
> always loaded. Triggered procedures live as Skills in `.agents/skills/`
> and are loaded on demand — that lazy-load model is itself the
> in-context-control distinction this repo teaches.

This repo is a teaching/demo site for **agentic harness × self-improving-agent
RL**, built around a real in-context policy-learning pipeline. The web frontend
is a 10-page SPA; the backend is FastAPI. **Do not break either.**

---

## 0. The single most important rule

**Never fabricate.** If you cannot read a source, say so. If you cannot verify a
link resolves, do not add it. A portfolio site that demonstrates rigor must not
ship invented digests, dead links, or claims attributed to sources you didn't
read. This rule overrides "be helpful" every time. (See §5 — the failure modes
are all things I have actually done wrong in earlier sessions and fixed.)

---

## 1. Repo layout (what lives where)

```
deepagents/
├── AGENTS.md                     ← Global harness contract
├── serve.py                      ← Unified development server launcher (FastAPI on :8000)
├── rl/                           ← Reinforcement Learning & In-Context Policy subsystem
│   ├── AGENTS.md                 ← YOU ARE HERE. RL-specific invariants.
│   ├── policy_feedback.py        ← Part 2 core: FeedbackEvent / compact / detect_anomalies
│   ├── policy_feedback_demo.py   ← 7 scenario factories + render_agents_md + fake_llm_rewrite
│   ├── kairos_log.py             ← KairosLogMemory (append-only log, dual-view format)
│   ├── IN_CONTEXT_POLICY_SPEC.md ← T1–T5 / F1–F7 spec
│   ├── grpo_toy.py / grpo_llm_char.py / grpo_play_core.py ← Part 1 core teaching algorithms
│   ├── training_pipeline.py      ← Production training loop scaffolding
│   ├── memory_selector.py        ← Context-budget-aware memory retrieval and compaction
│   └── tutorials/                ← 18-part full-stack RL & alignment curriculum in Traditional Chinese
├── web/
│   ├── app.py                    ← Unified FastAPI backend; mounts /assets and serves web/dist/index.html
│   ├── routers/                  ← Modular routers (deepagents, eval, rlvr, rl: toy/llm/offline/policy/wiki)
│   ├── schemas.py                ← Pydantic validation schemas
│   ├── state.py                  ← Runtime session state stores
│   └── dist/                     ← Production React 19 SPA build bundle
└── frontend/                     ← React 19 + Vite + TypeScript interactive learning application
    └── src/
        ├── entities/manifest/    ← Curriculum metadata (deepagents, rl, rlvr)
        ├── entities/chapter/     ← Chapter data models, loaders, and progress stores
        ├── entities/simulation/  ← 18 visual interactive simulation labs
        ├── widgets/sidebar-nav/  ← 3-track navigation sidebar with instant search and progress tracking
        └── widgets/reader-canvas/← Markdown & KaTeX reader with tri-modal code block renderer
```

### Unified Presentation Layer
The legacy static prototype has been consolidated into the unified 3-track interactive platform:
1. **`deepagents`**: 31 tutorial chapters covering Agent Harness Engineering and Evaluation.
2. **`rl`**: 18 chapters with 5-stage progressive code laboratories, KaTeX derivations, and inline labs.
3. **`rlvr`**: 18 chapters covering Post-Training MLE, verifiable rewards, and Kaggle showcases.

---

## 2. Invariants & Rules

- **Zero-Drift Documentation**: All file changes must be reflected in the directory `INDEX.md` and global `docs/` in the same turn.
- **Backend Stability**: The FastAPI backend (`serve.py`, `web/app.py`) must remain compatible with the React frontend and core CLI runners.
- **Frontend Code Quality**: Any UI changes in `frontend/` must pass `npm run typecheck` and `npm run build`.

---

## 3. Frontend Invariants & Design System

The unified web application is built with React 19, TypeScript, and Tailwind CSS v4 in `frontend/`:
- **Theme Variables**: Theme tokens are defined in `frontend/src/app/styles/index.css`. Use CSS custom properties (`--bg-primary`, `--bg-secondary`, `--text-primary`, `--accent-cyan`) across dark and light modes.
- **Component Architecture**: Follow Feature-Sliced Design (`shared/` -> `entities/` -> `widgets/` -> `pages/` -> `app/`).
- **Markdown & Math Rendering**: Content is rendered via `ReactMarkdown` with `remark-math` and `rehype-katex`. Internal chapter links are automatically normalized and intercepted by `MarkdownLink` in `ReaderCanvas.tsx`.
- **Interactive Simulation Labs**: All 17 simulation sandboxes are defined in `frontend/src/entities/simulation/` and mapped in `labCatalog.ts`.

---

## 4. Verification Checklist (Run After EVERY Change)

```bash
# 1. Frontend TypeScript typecheck
cd frontend && npm run typecheck

# 2. Frontend production build validation
npm run build

# 3. Backend import & server verification
python3 -c "import web.app; print('FastAPI app loaded successfully')"
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/

# 4. In-Context Policy simulation regression test
python3 -m rl.policy_feedback_demo

# 5. Core GRPO algorithm convergence test
python3 -m rl.grpo_toy
```

If any check fails, **fix before declaring done.** State plainly in your summary which checks passed and which (if any) you couldn't run.

---

## 7. Tone & honesty in summaries

- Tell the user what you verified vs. what you couldn't.
- If a source is unreadable, say "I couldn't read X because Y" — then offer
  options (resolve the redirect, user pastes content, skip it).
- Never claim "done and verified" if you skipped a check. Say "skipped: <reason>".
- The portfolio's whole pitch is *rigor and knowing boundaries*. The agent's
  behavior must embody that, not undermine it.
