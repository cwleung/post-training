# Autonomous Documentation Protocol

You are solely responsible for creating and maintaining all project documentation and directory indexes across the codebase.

## 0. Readership & Format Invariant (Agent-Only)
- All files under `docs/` and all distributed `INDEX.md` files are **strictly internal harness contracts and machine memory designed to be read and maintained by AI agents**.
- They are NOT marketing materials or human-facing prose. Keep descriptions dense, structured, factual, and token-efficient.
- Prefer tables, bullet points, and exact relative links. Never fabricate links, signatures, or endpoints.

## 1. Zero-Drift Rule
- Any file creation, deletion, rename, or functional update requires updating documentation in the exact same response.
- A task is incomplete if code changes exist without corresponding documentation updates.

## 2. Distributed Subdirectory Index Protocol
Every functional directory and subdirectory in the project (excluding transient/build outputs: `.git`, `.venv`, `node_modules`, `checkpoints`, `logs`, and `__pycache__`) must contain its own self-describing `INDEX.md`.

### Subdirectory `INDEX.md` Structure
Every local `INDEX.md` must follow this exact concise schema:

```markdown
# <Directory Name>

> Auto-maintained by Agent. Do not edit manually.

## Purpose
<1-2 sentences explaining the single responsibility of this directory.>

## Contents
| File / Subdir | Type | Responsibility / Exports |
|---|---|---|
| `example.py` | File | Handles token validation and auth guards |
| `utils/` | Dir | Shared string and date formatters |

## Invariants & Rules
- <Critical constraints or rules specific to this directory>
```

## 3. Dynamic Topic Documentation under `docs/`
- Agents are explicitly empowered to create new topic-specific markdown files under `docs/` (e.g. `docs/<TOPIC>.md`) whenever a new system subsystem, training/eval protocol, research track, or API contract is introduced.
- Whenever a new topic file is created or modified:
  1. Add header: `> Auto-maintained by Agent. Machine-readable harness memory.`
  2. Register the document in `docs/INDEX.md` under Core System Documentation.
  3. Log the creation in `docs/SYSTEM_STATE.md`.

## 4. Directory Maintenance Triggers
Whenever a task touches a folder:
1. **File Added/Removed/Renamed**: Update that directory's `INDEX.md` table and `docs/INDEX.md`.
2. **Behavior/Exports Changed**: Update the "Responsibility / Exports" cell in that directory's `INDEX.md`.
3. **New Directory Created**: Immediately generate a new `INDEX.md` inside it and link it in the parent directory's `INDEX.md`.
4. **New Topic Introduced**: Create `docs/<TOPIC>.md` and register it in `docs/INDEX.md`.
5. **Directory Deleted**: Remove its references from parent `INDEX.md` files.

## 5. Execution Protocol on Every Prompt
Before finalizing any response that touches the codebase:
1. Identify all touched directories.
2. Review and patch the local `INDEX.md` in each touched directory.
3. Update global docs in `docs/` (`ARCHITECTURE.md`, `API_SURFACE.md`, `STORAGE.md`, `INDEX.md`, or relevant `docs/<TOPIC>.md`).
4. Enforce Progressive Executable Notebook Laboratory Standard: Ensure any tutorial/curriculum code additions or edits strictly conform to the 5-stage progressive pipeline with alternating execution output blocks (never single isolated snippets).
5. Append an entry to `docs/SYSTEM_STATE.md`.

Refer to the `auto-index` skill (`~/.gemini/config/skills/auto-index/SKILL.md`) for complete workflow guidance.
