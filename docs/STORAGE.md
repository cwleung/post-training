# Data Storage & Persistence Specifications (`docs/STORAGE.md`)

> Auto-maintained by Agent. Do not edit manually.

This document specifies the persistence layers, data structures, state models, and storage invariants across DeepAgents.

---

## 1. Storage Layers Overview

| Layer / Directory | Format | Persistence Model | Purpose |
|---|---|---|---|
| `checkpoints/` | PyTorch (`.pt`), safetensors, JSON | Disk directory per session | Model weights, optimizer states, policy checkpoints |
| `logs/` | Text, JSONL | Append-only files | Training trajectories, evaluation logs, stdout |
| `eval_runs/` | JSON / YAML | File per run | Benchmark outputs, rubric grades, optimization metrics |
| `kairos_log.py` | JSONL + in-memory | Append-only dual-view | Episodic and semantic trajectory logs for agent policy adaptation |
| `web/state.py` | In-memory + JSON | Server lifecycle | FastAPI active connection state and run statistics |

---

## 2. In-Context Policy Feedback Data Model

Implemented in `rl/policy_feedback.py`:

```python
class FeedbackEvent(BaseModel):
    event_id: str                # Unique event identifier (UUID)
    timestamp: float             # Epoch timestamp
    category: str                # E.g., 'hallucination', 'format_violation', 'tool_misuse'
    severity: str                # 'low', 'medium', 'high', 'critical'
    trigger_text: str            # Snippet of model response that caused feedback
    feedback_text: str           # User or judge corrective feedback
    proposed_rule: str           # Invariant or rule derived from feedback
    applied: bool                # Whether rule has been incorporated into in-context prompt
```

### Kairos Dual-View Format (`rl/kairos_log.py`)
- **View 1 (Raw Trace)**: Comprehensive JSONL records capturing full prompts, completions, tool invocations, and timing metrics.
- **View 2 (In-Context Digest)**: Dynamically compacted summary of active policy rules and recent error anomalies, injected directly into model system context.

---

## 3. Checkpoint Directory Convention

Model checkpoints are stored under `checkpoints/<session_id>/`:

```
checkpoints/
└── s<timestamp>/
    ├── model.pt           # Model weights
    ├── optimizer.pt       # Optimizer state
    ├── config.json        # Hyperparameters and architecture config
    └── metrics.json       # Best loss, reward, or validation accuracy
```

---

## 4. Storage Invariants & Exclusion Rules

1. **Transient Exclusion**:
   - `checkpoints/` and `logs/` contain ephemeral runtime artifacts. They are ignored by Git (`.gitignore`) and excluded from distributed `INDEX.md` indexing.
2. **Schema Compatibility**:
   - Pydantic models in `web/schemas.py` and `rl/policy_feedback.py` must validate all persisted JSON representations.
3. **Zero-Drift**:
   - Any change to state schemas or storage paths requires immediate reconciliation of this document.
