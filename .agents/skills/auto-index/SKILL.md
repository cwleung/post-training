---
name: auto-index
description: Autonomous Documentation Protocol skill to generate, synchronize, and maintain self-describing INDEX.md files in every functional subdirectory and maintain the global docs/ suite with zero documentation drift.
---

# Autonomous Documentation Protocol (`auto-index`)

This skill governs the generation, synchronization, and maintenance of distributed `INDEX.md` files across all functional subdirectories and maintains the central `docs/` repository suite.

## Audience & Invariants

> [!IMPORTANT]
> **Agent-Only Readership**: All files under `docs/` and all distributed `INDEX.md` files are strictly **internal harness contracts and machine memory designed to be read and maintained by AI agents**. They are not human-facing prose or end-user marketing material.
> - Keep descriptions concise, factual, and dense.
> - Prioritize structured tables, machine-parseable lists, and exact relative markdown file links.
> - Never fabricate APIs, benchmarks, or file paths.

---

## Core Directives

### 1. Zero-Drift Mandate
Any file creation, deletion, rename, or functional update requires updating documentation in the exact same response. A task is incomplete if code changes exist without corresponding documentation updates.

### 2. Distributed Subdirectory Index Protocol
Every functional directory and subdirectory in the project (excluding transient, build, and runtime output folders) must contain its own self-describing `INDEX.md`.

### 3. Dynamic Topic Documentation in `docs/`
Agents are **fully authorized and encouraged to create new topic-specific markdown files under `docs/`** (e.g., `docs/<TOPIC>.md`) whenever:
- A new architectural subsystem or engine is designed.
- A new research methodology, training protocol, or evaluation workflow is established.
- A complex integration (e.g., external APIs, new RL algorithms, security policies) requires dedicated reference documentation.

**Topic Document Lifecycle**:
1. Name the file with upper snake/kebab case: `docs/<TOPIC_NAME>.md` (e.g., `docs/EVAL_PROTOCOL.md`, `docs/BENCHMARK_SPEC.md`).
2. Add the standard banner: `> Auto-maintained by Agent. Machine-readable harness memory.`
3. **Immediately register** the new topic document in `docs/INDEX.md` under the Core System Documentation table with its purpose and scope.
4. Record the addition in `docs/SYSTEM_STATE.md`.

---

## Subdirectory `INDEX.md` Schema

Every local `INDEX.md` must strictly follow this exact concise format:

```markdown
# <Directory Name>

> Auto-maintained by Agent. Do not edit manually.

## Purpose
<1-2 sentences explaining the single responsibility of this directory.>

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `example.py` | File | Brief summary of primary responsibilities, classes, or exports |
| `subfolder/` | Dir | [subfolder/INDEX.md](subfolder/INDEX.md) - Summary of subfolder responsibility |

## Invariants & Rules
- <Critical constraints, dependencies, conventions, or rules specific to this directory>
```

### Filtering & Exclusion Invariants
- **Do NOT index**:
  - Version control: `.git/`
  - Python virtualenvs: `.venv/`, `env/`, `venv/`
  - JavaScript dependencies: `node_modules/`
  - Caches & compiled binaries: `__pycache__/`, `*.pyc`, `.pytest_cache/`, `dist/`, `.vite/`
  - Dynamic runtime outputs & model checkpoints: `checkpoints/*`, `logs/*` (transient experiment runs)
- **Do index**:
  - All source code, config, documentation, and asset directories under active development (`rl/`, `rlvr/`, `web/`, `frontend/`, `tutorials/`, `.agents/rules/`, `.agents/skills/`).

---

## Directory Maintenance Triggers

Whenever a task touches any folder:

1. **File Added/Removed/Renamed**:
   - Update that directory's `INDEX.md` table.
   - If a new functional directory is introduced, update the parent directory's `INDEX.md` and `docs/INDEX.md`.
2. **Behavior/Exports Changed**:
   - Update the "Responsibility / Exports" cell in that directory's `INDEX.md`.
3. **New Directory Created**:
   - Immediately generate a new `INDEX.md` inside it following the required schema.
   - Add the new directory link to its parent directory's `INDEX.md` `Contents` table.
4. **New Topic or Subsystem Introduced**:
   - Create `docs/<TOPIC>.md` and register it in `docs/INDEX.md`.
5. **Directory Deleted**:
   - Remove its reference from the parent directory's `INDEX.md` and `docs/INDEX.md`.

---

## 4-Step Continuous Subdirectory & Global Doc Sync Workflow

Whenever any file is edited, created, or deleted during a task:

### Step 1: Map the Modified Tree
1. List all modified, created, or deleted file paths.
2. Extract all distinct parent directories for those paths.
3. Filter out excluded directories (`.git`, `.venv`, `node_modules`, `checkpoints`, `logs`, `__pycache__`).

### Step 2: Sync Local Subdirectory Indexes
For each affected directory:
1. Check if `<dir>/INDEX.md` exists. If missing, create it using the standard schema.
2. Reconcile the `Contents` table:
   - Add newly created files or subfolders.
   - Remove entries for deleted files.
   - Update export/responsibility notes for modified files.
3. If a subfolder was created or altered, ensure its parent directory links to `<dir>/<subfolder>/INDEX.md`.

### Step 3: Sync Global Project Docs
- **`docs/INDEX.md`**: Update the top-level tree, directory list, and topic documents table.
- **`docs/API_SURFACE.md`**: Update altered API endpoints, CLI scripts, or Gradio interfaces.
- **`docs/STORAGE.md`**: Update if data structures, checkpoint formats, memory logs, or schemas changed.
- **`docs/ARCHITECTURE.md`**: Update if module boundaries, dependency graphs, or data flows shifted.
- **`docs/<TOPIC>.md`**: Create or update relevant topic-specific documentation files as needed.

### Step 4: Record State in `docs/SYSTEM_STATE.md`
Append an entry to `docs/SYSTEM_STATE.md`:
```markdown
### [YYYY-MM-DD] - <Task Name / Description>
- **Updated Subdirectory Indexes**: `<list of dir/INDEX.md paths>`
- **Updated Global Docs**: `<list of docs/ paths>`
- **Summary**: Brief description of sync actions performed.
```

---

## Bootstrap Scan Workflow (Full Repo Onboarding)

When initializing documentation on an unindexed repository:
1. Traverse the repository root to enumerate all functional non-ignored directories.
2. For every folder and subfolder, generate a local `INDEX.md` complying with the standard schema.
3. Ensure parent `INDEX.md` files link to child directory `INDEX.md` files.
4. Populate `docs/INDEX.md` as the master catalog.
5. Record initial state in `docs/SYSTEM_STATE.md`.
