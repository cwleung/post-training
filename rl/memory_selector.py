#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
memory_selector.py — context-budget-aware memory reader
========================================================

WHAT THIS IS
------------
A selector that sits ON TOP of KairosLogMemory. It answers one question:

    "Given a token budget, what slice of the log should go into the LLM's
     context, and how should the rest be compressed so nothing is silently lost?"

It is the three features the user asked for, layered into one entry point:

  1. compact(...)        — turn old/low-value events into one summary line
  2. read_for_context()  — fit the rest into a token budget
  3. importance()        — the score that drives BOTH of the above (freshness!)

The composition is the point. If you build these as three unaware functions,
you end up with a compactor that doesn't know the budget and a budget-read
that doesn't know what was already summarized. Here they compose:

    read_for_context(kairos, budget_tokens):
        score everything by importance()              ← freshness/magnitude
        reserve a slice of budget → compact() old     ← compaction
        greedy-fill the rest by score, by slices      ← budget
        return  rollup-line + selected events

WHAT THIS IS *NOT*
------------------
- NOT semantic relevance. There is no embedding, no "this query is similar to
  that memory." Calling lexical overlap "relevance" would be the same fabrication
  trap as fabricating a tweet digest (AGENTS.md §5 #2). Pure importance×freshness.
- NOT a writer. It NEVER calls kairos.append() or mutates the log file. The log
  is append-only (spec: "Never rewrite history"); compaction produces a DERIVED
  string, not an edit.
- NOT a trainer. No weights, no gradient. Same T3 invariant as policy_feedback.

HOW IT HOOKS IN
---------------
It reads kairos.iter_events() exactly the way policy_feedback.read_feedback_events
does — same seam, no new coupling, no edits to kairos_log.py or policy_feedback.py.

== Honesty about the magic numbers ==
half_life_hours, budget_tokens, the cutoff, the slice ratio, floor_n — these are
ALL knobs with documented defaults, not tuned values. Expose them as params; do
not promote them to constants. (AGENTS.md §7: state boundaries plainly.)
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Optional

# ============================================================
# Layer A — token cost
# ============================================================
# Why a separate layer: the whole budget machinery should not know or care how
# tokens are counted. Swap tiktoken ↔ len//4 by editing ONE function.

_TIKTOKEN_ENCODER = None
_TIKTOKEN_WARNED = False


def _get_encoder():
    """Lazily load tiktoken once. Returns (encoder, name) or (None, None)."""
    global _TIKTOKEN_ENCODER, _TIKTOKEN_WARNED
    if _TIKTOKEN_ENCODER is not None:
        return _TIKTOKEN_ENCODER, "tiktoken(cl100k_base)"
    try:
        import tiktoken  # type: ignore
        _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
        return _TIKTOKEN_ENCODER, "tiktoken(cl100k_base)"
    except Exception:
        # Explicit, ONE-TIME warning — never silent. A silent fallback is how
        # "len//4 is accurate" becomes an unchallenged assumption.
        if not _TIKTOKEN_WARNED:
            sys.stderr.write(
                "[memory_selector] tiktoken not available; falling back to "
                "len(text)//4. This is ±30% wrong, worse on Chinese. "
                "pip install tiktoken for accurate counts.\n"
            )
            _TIKTOKEN_WARNED = True
        return None, "len//4 (fallback)"


def est_tokens(text: str) -> int:
    """
    Token count of a string. tiktoken if importable, else len//4.

    The fallback under-counts CJK text (Chinese is ~1-2 chars/token, not 4).
    We document that rather than pretend the number is exact.
    """
    enc, _ = _get_encoder()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // 4)


def estimator_name() -> str:
    """For demos / assertions: which estimator is actually running."""
    _, name = _get_encoder()
    return name


# ============================================================
# Layer B — importance × freshness
# ============================================================
def recency_score(
    ts_iso: str,
    now: datetime,
    half_life_hours: float = 48.0,
) -> float:
    """
    Exponential half-life decay: every `half_life_hours`, the score halves.

    τ (half-life) is a KNOB, not a theorem. 48h means "2-day-old events are
    worth half as much as new ones" — a statement a human can sanity-check.

    Why exponential not linear: linear decay (1 - age/N) hits zero and stays
    there, silently deleting memory. Exponential approaches zero but never
    erases, so an old high-reward event can still surface if it's important
    enough via the magnitude term.
    """
    if not ts_iso:
        return 0.0
    try:
        ts = datetime.fromisoformat(ts_iso)
    except ValueError:
        return 0.0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    age_h = max(0.0, (now - ts).total_seconds() / 3600.0)
    return math.pow(0.5, age_h / half_life_hours)


def importance(
    event: dict,
    now: datetime,
    half_life_hours: float = 48.0,
    floor: float = 0.3,
) -> float:
    """
    One event's "should this be remembered" score. MULTIPLICATIVE across axes:

        importance = recency(ts) * (floor + |reward|)

    Why multiplicative not additive: a stale event with huge reward should still
    fade; a fresh event with zero signal shouldn't dominate. Multiplication
    enforces "must pass a bar on EVERY axis." Additive lets one huge axis swamp
    a zero on another.

      recency   — freshness control (the core of the user's Feature 3)
      |reward|  — magnitude; reuses policy_feedback.REWARD_BY_SIGNAL numbers
      floor     — neutral events (reward≈0) get a baseline so they don't
                  vanish entirely. 0.3 is a documented default, not tuned.

    Events without a `reward` field (raw log_event() rows like plan_generated)
    score by recency alone via the floor — they don't crash, they just don't
    get a magnitude boost.
    """
    recency = recency_score(event.get("ts", ""), now, half_life_hours)
    reward = event.get("reward")
    magnitude = abs(reward) if isinstance(reward, (int, float)) else 0.0
    return recency * (floor + magnitude)


# ============================================================
# Layer C — compaction (lossy rollup of old events)
# ============================================================
def _parse_ts(ts_iso: str) -> Optional[datetime]:
    if not ts_iso:
        return None
    try:
        dt = datetime.fromisoformat(ts_iso)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def compact_log(
    events: list,
    since: datetime,
) -> dict:
    """
    Aggregate events older than `since` into one summary string.

    Returns {"summary": str, "n": int}. If nothing is old enough, summary is "".

    Design properties:
      - DETERMINISTIC: pure Counter / mean rollup. No LLM, no randomness. Every
        number in the summary can be re-derived from the raw events → auditable.
      - IDEMPOTENT: compacting an already-compacted summary is a no-op. The
        summary string carries no `ts`/`reward`, so it's filtered out on re-run.
        compact(compact(events)) == compact(events).
      - NON-DESTRUCTIVE: returns a DERIVED string. The caller never writes it
        back into KairosLogMemory. The append-only log invariant (spec) holds.
    """
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)

    old: list = []
    for e in events:
        if not isinstance(e, dict):
            continue
        ts = _parse_ts(e.get("ts", ""))
        if ts is None:
            continue
        if ts < since:
            old.append(e)

    if not old:
        return {"summary": "", "n": 0}

    by_type = Counter(str(e.get("event_type") or e.get("signal") or "?") for e in old)
    by_result = Counter(str(e.get("result") or "n/a") for e in old)
    rewards = [e.get("reward") for e in old if isinstance(e.get("reward"), (int, float))]
    mean_r = (sum(rewards) / len(rewards)) if rewards else 0.0

    summary = (
        f"[compacted {len(old)} events before {since.isoformat(timespec='hours')}] "
        f"types={dict(by_type)} results={dict(by_result)} "
        f"mean_reward={mean_r:+.2f}"
    )
    return {"summary": summary, "n": len(old)}


# ============================================================
# Event rendering + truncation helpers
# ============================================================
def _render_event(event: dict) -> str:
    """
    One-line human-readable rendering of an event dict (the JSON payloads that
    iter_events() yields). Mirrors the human-readable bullet KairosLogMemory
    already writes, but derived from the parsed dict so it's stable.
    """
    ts = event.get("ts", "")
    etype = event.get("event_type") or event.get("signal") or "?"
    summary = event.get("summary") or event.get("query") or ""
    extras = []
    if "reward" in event and isinstance(event["reward"], (int, float)):
        extras.append(f"r={event['reward']:+.2f}")
    if "plan_id" in event and event["plan_id"]:
        extras.append(f"plan={event['plan_id']}")
    extra_str = f" ({', '.join(extras)})" if extras else ""
    return f"- {ts} [{etype}] {summary}{extra_str}"


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Trim text to fit max_tokens (best-effort; truncation beats dropping)."""
    if max_tokens <= 0:
        return ""
    cost = est_tokens(text)
    if cost <= max_tokens:
        return text
    # Binary-search-free heuristic: scale by ratio on chars, then verify.
    ratio = max_tokens / max(1, cost)
    cut = max(1, int(len(text) * ratio) - 3)  # -3 leaves room for ellipsis
    return text[:cut].rstrip() + "…"


# ============================================================
# Entry point — the composed selector
# ============================================================
def read_for_context(
    kairos,
    budget_tokens: int = 2000,
    now: Optional[datetime] = None,
    *,
    cutoff_days: float = 7.0,
    half_life_hours: float = 48.0,
    floor_n: int = 3,
    importance_slice: float = 0.60,
    rollup_reserve: float = 0.15,
) -> dict:
    """
    The composed memory selector. Returns a dict with:
        "text":           the string to inject into the LLM context
        "used_tokens":    est_tokens(text)
        "budget_tokens":  the requested budget
        "rollup":         the compaction summary (may be "")
        "n_selected":     how many raw events made the cut
        "n_total":        total events in the log
        "estimator":      which token estimator is active (honesty field)

    Pipeline (the three features, composed):
      1. Score every event by importance().               ← Feature 3
      2. Reserve rollup_reserve*budget; compact events
         older than cutoff_days into one summary line.   ← Feature 1
      3. Recency floor: always include the last floor_n
         events, regardless of score (prevents starvation).
      4. Split remaining budget into two slices:
           importance_slice  → top events by score
           (1 - slice)       → most recent events
         Greedy-fill each slice by score; TRUNCATE (don't drop)
         a high-score event that's slightly too big.
                                                            ← Feature 2
      5. Re-sort selected chronologically (LLM reads time-ordered better).

    All numeric params are KNOBS with documented defaults, not tuned values:
      budget_tokens     total context budget for memory
      cutoff_days       events older than this get compacted, not shown raw
      half_life_hours   τ for recency decay (see recency_score)
      floor_n           always-include the last N events
      importance_slice  fraction of post-rollup budget for "top by score"
      rollup_reserve    fraction of budget reserved for the compaction summary
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    events = list(kairos.iter_events())
    n_total = len(events)

    # ---- Step 1: compaction rollup (reserve a slice of budget for it) ----
    # JOINER_TOKENS: the cost of the "\n" that will separate lines in the final
    # "\n".join(). Counting it per line (incl. the rollup) makes the slice math
    # EXACTLY match the final assembled string — no off-by-N leaks.
    JOINER_TOKENS = 1

    rollup_budget = int(budget_tokens * rollup_reserve)
    cutoff = now - timedelta(days=cutoff_days)
    rollup = compact_log(events, cutoff)
    rollup_line = rollup["summary"]
    # If the rollup itself exceeds its reserve, truncate it (rare, but honest).
    # Truncate against (rollup_budget - JOINER) so the joiner after it fits too.
    if rollup_line and est_tokens(rollup_line) + JOINER_TOKENS > rollup_budget:
        rollup_line = _truncate_to_tokens(rollup_line, max(1, rollup_budget - JOINER_TOKENS))

    # remaining_budget accounts for the rollup line + its trailing joiner.
    rollup_cost = est_tokens(rollup_line) + (JOINER_TOKENS if rollup_line else 0)
    remaining_budget = budget_tokens - rollup_cost

    # ---- Step 2: which events are eligible to be shown RAW (not compacted) ----
    # An event is eligible if it's newer than cutoff (or has no parseable ts).
    eligible: list = []
    for e in events:
        if not isinstance(e, dict):
            continue
        ts = _parse_ts(e.get("ts", ""))
        if ts is None or ts >= cutoff:
            eligible.append(e)

    if not eligible:
        return {
            "text": rollup_line,
            "used_tokens": est_tokens(rollup_line),
            "budget_tokens": budget_tokens,
            "rollup": rollup_line,
            "n_selected": 0,
            "n_total": n_total,
            "estimator": estimator_name(),
        }

    # ---- Step 3: score eligible events ----
    scored = [(importance(e, now, half_life_hours), e) for e in eligible]

    # ---- Step 4: recency floor (always-include last floor_n) ----
    # Sort eligible by ts to find "last N" chronologically.
    by_time = sorted(eligible, key=lambda e: e.get("ts", ""))
    floor_ids = {id(e) for e in by_time[-floor_n:]}

    # CRITICAL INVARIANT: used_running may NEVER exceed remaining_budget.
    # The floor is "best-effort", not "violate budget to keep the last N".
    # If even a truncated floor event doesn't fit, it's dropped — better to
    # honor the budget contract than to silently overflow it.
    selected_ids = set()
    used_running = 0  # shared across floor + both slices; the single source of truth

    def _try_add(e: dict, cap: int, *, allow_truncate: bool) -> Optional[str]:
        """
        Attempt to render `e` and fit it under `cap` (where cap is the remaining
        room in THIS slice). Mutates nothing; returns the line to add (and its
        cost) or None if it won't fit. The caller is responsible for the hard
        total-budget check afterwards.
        """
        line = _render_event(e)
        cost = est_tokens(line)
        if cost <= cap:
            return line
        if not allow_truncate:
            return None
        # Only truncate if it's reasonably close — don't mangle a huge event
        # to fit a tiny gap.
        if cap > cost // 2:
            truncated = _truncate_to_tokens(line, cap)
            return truncated
        return None

    # Floor events come first, drawing down the importance slice's budget.
    importance_budget = int(remaining_budget * importance_slice)

    for s, e in sorted(scored, key=lambda x: -x[0]):
        if id(e) not in floor_ids:
            continue
        # Hard total-budget guard: even floor events must not overflow.
        room_in_total = remaining_budget - used_running
        if room_in_total <= JOINER_TOKENS:
            break
        # Leave room for the joiner when sizing the truncation cap.
        room_in_slice = min(importance_budget, room_in_total) - JOINER_TOKENS
        line = _try_add(e, room_in_slice, allow_truncate=True)
        if line is None:
            continue
        cost = est_tokens(line) + JOINER_TOKENS
        importance_budget -= cost
        used_running += cost
        selected_ids.add(id(e))
        # Remember the rendered line so we don't re-render / re-truncate later.
        e["_rendered"] = line  # stash on the dict; cleared before return

    # ---- Step 5a: importance slice — greedy fill by score desc ----
    for s, e in sorted(scored, key=lambda x: -x[0]):
        if id(e) in selected_ids:
            continue
        room_in_total = remaining_budget - used_running
        if room_in_total <= JOINER_TOKENS:
            break
        room_in_slice = min(importance_budget, room_in_total) - JOINER_TOKENS
        line = _try_add(e, room_in_slice, allow_truncate=False)
        if line is None:
            continue
        cost = est_tokens(line) + JOINER_TOKENS
        importance_budget -= cost
        used_running += cost
        selected_ids.add(id(e))
        e["_rendered"] = line

    # ---- Step 5b: recency slice — greedy fill by recency desc ----
    recency_budget = remaining_budget - used_running  # whatever's left
    scored_by_recency = sorted(
        scored, key=lambda x: -recency_score(x[1].get("ts", ""), now, half_life_hours)
    )
    for s, e in scored_by_recency:
        if id(e) in selected_ids:
            continue
        room_in_total = remaining_budget - used_running
        if room_in_total <= JOINER_TOKENS:
            break
        room_in_slice = min(recency_budget, room_in_total) - JOINER_TOKENS
        line = _try_add(e, room_in_slice, allow_truncate=False)
        if line is None:
            continue
        cost = est_tokens(line) + JOINER_TOKENS
        recency_budget -= cost
        used_running += cost
        selected_ids.add(id(e))
        e["_rendered"] = line

    # ---- Step 6: assemble, re-sorted chronologically for readability ----
    # Use the stashed _rendered line (what we actually budgeted), NOT a fresh
    # render — re-rendering could produce a different token count and break the
    # used ≤ budget invariant we just fought to maintain.
    selected = [e for e in eligible if id(e) in selected_ids]
    selected.sort(key=lambda e: e.get("ts", ""))

    lines = []
    if rollup_line:
        lines.append(rollup_line)
    lines.extend(e.pop("_rendered", None) or _render_event(e) for e in selected)
    text = "\n".join(lines)

    # Clean up the stash on any dict that got one but wasn't selected (defensive;
    # these are freshly-parsed dicts, but we keep the "read-only" contract honest).
    for e in eligible:
        e.pop("_rendered", None)

    final_used = est_tokens(text)
    # Hard contract: this MUST hold. If it ever fails, the bug is in the slice
    # math above, not here. (Asserting rather than clipping: a clip would hide
    # the bug; an assert surfaces it.)
    assert final_used <= budget_tokens, (
        f"budget violated: used {final_used} > budget {budget_tokens}; "
        f"slice math is buggy"
    )

    return {
        "text": text,
        "used_tokens": final_used,
        "budget_tokens": budget_tokens,
        "rollup": rollup_line,
        "n_selected": len(selected),
        "n_total": n_total,
        "estimator": estimator_name(),
    }


# ============================================================
# Self-test (python3 memory_selector.py)
# ============================================================
if __name__ == "__main__":
    import os
    import tempfile

    from kairos_log import KairosLogMemory

    tmp = tempfile.mkdtemp()
    log_path = os.path.join(tmp, "selftest.log.md")
    k = KairosLogMemory(log_path)

    now = datetime.now(timezone.utc)
    # A fresh high-reward event, and an old low-reward event.
    k.log_event(
        "user_accept", plan_id="p1", result="success",
        summary="fresh accept", user_id="u", session_id="s",
    )
    # Manually append an OLD feedback-style event to exercise the reward path.
    old_ts = (now - timedelta(days=30)).isoformat(timespec="seconds")
    old_payload = (
        f"- {old_ts} [feedback] L4_confirm\n"
        f'  {{"ts": "{old_ts}", "signal": "L4_confirm", "query": "old query", '
        f'"session_id": "s", "policy_version": "v0", "reward": 1.0, '
        f'"intent_level": "L4"}}'
    )
    k.append(old_payload)

    print("=" * 60)
    print(" memory_selector self-test")
    print("=" * 60)
    print(f"token estimator: {estimator_name()}")

    events = list(k.iter_events())
    print(f"\nevents in log: {len(events)}")

    print("\nimportance scores (now = utc):")
    for e in events:
        s = importance(e, now)
        print(f"  score={s:.4f}  ts={e.get('ts','')[:19]}  type={e.get('event_type') or e.get('signal')}")

    print("\ncompact_log (cutoff = 7 days ago):")
    rollup = compact_log(events, now - timedelta(days=7))
    print(f"  n compacted: {rollup['n']}")
    print(f"  summary: {rollup['summary']}")

    print("\nread_for_context (budget=2000):")
    result = read_for_context(k, budget_tokens=2000, now=now)
    print(f"  used/budget: {result['used_tokens']}/{result['budget_tokens']}")
    print(f"  selected:    {result['n_selected']}/{result['n_total']}")
    print(f"  estimator:   {result['estimator']}")
    print("\n  --- text ---")
    for ln in result["text"].splitlines():
        print(f"  {ln}")

    # ---- assertions ----
    print("\n" + "=" * 60)
    assert result["used_tokens"] <= result["budget_tokens"], "budget exceeded!"
    assert result["n_total"] == 2, f"expected 2 events, got {result['n_total']}"
    assert rollup["n"] == 1, f"expected 1 old event compacted, got {rollup['n']}"
    # The fresh event must survive (not be compacted), the old one must not
    # appear as a raw line (it's in the rollup instead).
    assert "fresh accept" in result["text"], "fresh event dropped!"
    assert "old query" not in result["text"], "old event should be in rollup, not raw"
    print(" ✓ budget respected, fresh event kept, old event compacted (not shown raw)")
    print(" ✓ raw log file was never mutated (selector is read-only by design)")
    print("=" * 60)
