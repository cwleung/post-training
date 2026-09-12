# In-Context Policy Learning — Implementation Spec

A spec for another AI / engineer to implement in-context policy learning for an
LLM copilot (the "L4" autonomy tier). **Functional requirements** (what the
system must do) and **theoretical requirements** (why, with the constraints that
rule out the "obvious" approach). No code, no UI — just the contract.

---

## 0. The problem, in one paragraph

We have an LLM copilot that helps users build things (dashboards, etc.) across
many sessions. Users give implicit feedback — they **confirm** a plan, **accept**
an edit, **undo** an edit, **abandon** a plan, or just get a quick **answer**.
We want the copilot to *learn* from this feedback and get better over time. The
naive approach is "fine-tune / RLHF it" — but read the theoretical requirements
first; that approach is ruled out by the structure of the problem.

---

## 1. Theoretical requirements (the constraints that shape everything)

### T1 — You cannot use GRPO / group-baseline RL here. Why not.

GRPO computes a group baseline: the average reward over **G exchangeable samples
of the same prompt**. That requires exchangeability. In this copilot:

- A single user request can be answered in **three different generation modes**
  (see T2), each producing a structurally different kind of response.
- `L4_confirm` (a plan was approved), `L3_undo` (an edit was rolled back), and
  `L2_answer` (a quick text answer) are **not exchangeable** — they are outcomes
  of different generation processes, at different autonomy levels.
- Averaging reward across these different intents produces an *advantage* with no
  statistical meaning. **Increasing N does not fix this** — N=10, N=100, same
  problem. The non-exchangeability is structural, not a sample-size issue.

> **Requirement:** Do NOT pool rewards across intent levels to compute a single
> advantage. All statistics must be **stratified by intent level**. This is the
> single most important constraint.

### T2 — Three generation modes, three different feedback signals

The copilot operates in exactly three modes, each with its own feedback signal
and reward. These are **not interchangeable**:

| Mode | What it does | Feedback signal | Reward |
|---|---|---|---|
| Plan Mode (L4) | Ask clarifying questions, propose a plan, get explicit confirmation, then build | `L4_confirm` / `L4_abandon` | +1.0 / −0.5 |
| Direct Mode (L3) | Skip the dialog, build immediately (with a progress bar) | `L3_accept` / `L3_undo` | +0.5 / −1.0 |
| Answer Mode (L2) | No artifact, just give an analytical answer | `L2_answer` / `L2_answer_cq` (answer with a clarifying question) | +0.3 / +0.5 |

Note the reward shaping: **`L3_undo` (−1.0) is punished harder than `L4_abandon`
(−0.5)**. Undo means "you did the wrong thing and now must be rolled back";
abandon means "the user never committed." The former has higher cleanup cost, so
the negative reward is larger. This is intentional shaping, not arbitrary.

### T3 — Feedback is logged, not trained on

We never run gradient updates against the production model. The model's
**weights do not move**. Instead, feedback is recorded in an append-only log,
aggregated into human-readable **rules**, and those rules are injected into the
model's context (the system prompt / `AGENTS.md`) for future sessions. This is
"in-context" policy learning: **behavior changes, weights stay frozen.**

> **Requirement:** The implementation must not depend on a training loop,
> optimizer, or weight access. It is pure logging + statistics + prompt
> composition.

### T4 — Learning has a hard safety boundary

Whatever is learned must **never** weaken the safety guardrails. The system
prompt has three sections:

1. `## System Constraints` — hard rules (must use framework's controllers, cannot
   access data sources outside the subscription). **Immutable. Learning never
   touches this.**
2. `## User Preferences` — user-level preferences. **Immutable to the learner.**
3. `## Learned Behavior Rules` — KEEP / AVOID rules extracted from feedback.
   **This is the ONLY section the learner may rewrite.**

> **Requirement:** The learned-rules writer must be unable to modify sections 1
> and 2. Prove it by construction: the renderer takes the guardrails as fixed
> input and only emits the Learned section anew.

### T5 — The learner must refuse to learn from hacked feedback (reward hacking)

If the model discovers it can rack up `+1.0` rewards by, e.g., repeatedly
confirming trivially similar plans ("build dashboard", "build dashboard", ...),
the win_rate looks great but the model is gaming the metric, not learning. The
system must **detect this and freeze** rather than encode the degenerate behavior
into rules.

---

## 2. Functional requirements (what to build)

The system is a pipeline of pure-ish functions. Each takes the feedback log and
produces a derived artifact.

### Data model

**FeedbackEvent** — one row in the log:

- `signal`: one of the six signals in T2 (`L4_confirm`, `L4_abandon`,
  `L3_accept`, `L3_undo`, `L2_answer`, `L2_answer_cq`).
- `query`: the user's request text (needed for diversity checks, T5).
- `session_id`: which session it came from (needed for per-session stats).
- `policy_version`: the version of the learned rules that was *active when this
  event happened* (`v0`, `v1`, …). Critical for off-policy handling (F5).
- (derived) `reward`: looked up from the fixed reward table (T2).
- (derived) `intent_level`: looked up from the signal→level map (T2: L4/L3/L2).

The log is **append-only**. Never rewrite history.

### F1 — `stratified_stats(events)`

Group events by `intent_level` (L4 / L3 / L2). For each level, compute:

- `n`: count
- `mean_reward`: average of rewards in that level
- `win_rate`: fraction of events with reward > 0
- `undo_rate`: fraction of events that are `L3_undo`

> Constraint from T1: **do not produce an overall / pooled win_rate.** Only
> per-level. If the consumer wants a headline number, they must pick a level.

### F2 — `detect_anomalies(events)` (the reward-hacking guard)

Compute two flags. Both must be cheap and use only the log:

- **`win_rate_spike`**: look at the L4 win_rate **per session**. If the most
  recent session's win_rate jumps by more than **+0.25** over the previous
  session's, flag it. (This catches the "suddenly everything is getting
  confirmed" pattern.) Requires ≥ 2 sessions; otherwise no flag.
- **`diversity_collapse`**: among the recent **positive-reward** events, compute
  the token-level diversity of the `query` field. If it drops below **0.50**
  (configurable), flag it. ("All queries are the same string" → near-zero
  diversity.)

> The detail string produced (e.g. `win_rate 0.80→1.00 (Δ=+0.20); recent-positive-
> query token diversity=0.20`) must be human-readable and state the numbers, so
> an operator can audit why a freeze was triggered.

### F3 — `compact(events)` (the rule extractor)

This is the core. Produce a list of **rules**, each:

- `kind`: `KEEP` (from mostly-positive evidence) or `AVOID` (from mostly-negative).
- `intent_level`: which level the evidence came from.
- `pattern`: a short, human-readable description of the behavior (e.g. "L4_confirm
  behavior is landing well").
- `evidence`: the proof — `{n, win_rate, undo_rate}` for the stratum that produced
  the rule. **This field is mandatory and never stripped**, even if an LLM later
  rewrites the `pattern` into nicer prose. Rules without evidence are forbidden
  (T5 / auditability).
- `source`: `heuristic` (the deterministic pass) or `llm` (a refinement pass).

**Extraction thresholds (do not over-learn):**
- Require a **minimum of 3 events** (`min_n=3`) in a stratum before extracting any
  rule from it. Below that, emit nothing. "Better to not learn than to learn
  wrong."
- A stratum becomes `KEEP` if its `win_rate ≥ 0.5` (and n ≥ 3); `AVOID` if
  `win_rate ≤ 0.3` (and n ≥ 3).

**Optional LLM refinement pass:** given `use_llm=true`, take the heuristic rules
and ask an LLM to rewrite each `pattern` into natural-language guidance suitable
for a system prompt. Store the result as `rules_refined`. **The `evidence` dict
must survive this pass unchanged** — the LLM touches prose, never the proof.

### F4 — `render_agents_md(rules, old_version)`

Take the extracted rules and produce an `AGENTS.md` with the three sections from
T4:

1. `## System Constraints` — fixed guardrails (passed in, never generated).
2. `## User Preferences` — fixed (passed in, never generated).
3. `## Learned Behavior Rules` — **generated from `rules`**. Bump the
   `policy_version` (`v0` → `v1` → …) whenever this section changes.

> Constraint from T4: the function signature must make it impossible to mutate
> sections 1–2 from `rules`. Take them as separate, explicit parameters.

### F5 — Off-policy handling

Each event carries the `policy_version` that was active when it occurred. If the
log contains **mixed versions** (some `v0`, some `v1`), the events from the older
version are **off-policy** relative to the current rules: they were generated
under a different policy, so their accept/undo does not directly measure the
current policy's quality.

- **Detect it:** when ≥ 2 distinct `policy_version` values appear in the log,
  emit an off-policy note.
- **What we do NOT do:** importance sampling (IS). Be honest: precise IS is
  infeasible here (we don't have action probabilities). Instead, **flag and
  down-weight** — a heuristic. State this explicitly in the output so nobody
  mistakes it for a principled estimator.

### F6 — Freeze on anomaly (the closed loop)

`compact()` and `render_agents_md()` must check the anomaly flags (F2) **before
writing rules**. If `win_rate_spike` or `diversity_collapse` is set, the system
must **freeze**: do not commit the new rules to `AGENTS.md`, and surface the
anomaly to the operator. Committing hacked behavior into the system prompt is the
one outcome to prevent.

### F7 — The full closed loop

```
user confirm/undo  →  append FeedbackEvent to log
                  →  stratified_stats(events)            [F1]
                  →  detect_anomalies(events)            [F2]
                  →  compact(events)  → rules            [F3]
                  →  if no anomaly: render_agents_md     [F4, F6]
                  →  inject into next session's prompt
```

The next session's copilot runs under the updated `## Learned Behavior Rules`.
**Weights never change; only the prompt does.** This is the entire mechanism.

---

## 3. Honest non-goals (things you might expect, but are explicitly out of scope)

- **No gradient training.** Not RLHF, not DPO, not GRPO. (See T1, T3.)
- **No pooled/cross-level advantage.** Statistics are per-level only. (T1.)
- **No importance sampling.** Off-policy is handled by detection + down-weighting,
  not IS. (F5.) State this limitation plainly in the output.
- **No learned reward model.** The reward table (T2) is fixed, hand-shaped.
- **No mutation of safety guardrails.** The Learned section is the only writable
  surface. (T4.)
- **No silent learning.** Every rule carries evidence; every freeze carries a
  human-readable reason. (T5, F3.)

---

## 4. Acceptance criteria (how to know it's correct)

1. Given a log of all-`L4_confirm` events with near-identical queries,
   `detect_anomalies` sets `diversity_collapse`, and the loop **freezes** (no new
   rules written). ✓ T5/F6.
2. Given a log with < 3 events in every level, `compact` returns **zero rules**.
   ✓ F3 min_n.
3. Given mixed `policy_version` values, the output includes an **off-policy note**
   and does not claim to do importance sampling. ✓ F5.
4. `render_agents_md` output always contains the unchanged `## System Constraints`
   section, regardless of input rules. ✓ T4.
5. Every emitted rule has a non-empty `evidence` dict, even after the LLM
   refinement pass. ✓ F3.
6. There is no code path that produces a single pooled win_rate across L4/L3/L2.
   ✓ T1.
7. No function in the pipeline reads or writes model weights. ✓ T3.
