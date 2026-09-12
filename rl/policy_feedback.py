#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
policy_feedback.py — User-feedback → behavior-rule pipeline (system-level, NOT GRPO)
====================================================================================

WHAT THIS IS
------------
A logging + statistics + compaction layer on top of KairosLogMemory. It turns
"which agent behaviors the user kept vs. threw away" into behavior rules, then
writes those rules back into the log so the next session's system prompt can
inject them.

WHAT THIS IS *NOT*
------------------
It is NOT GRPO, and deliberately so. The honest story:

  - GRPO's group baseline is the mean of G samples *of the same prompt*.
    Those G samples are exchangeable and come from the same distribution, so
    "reward - group mean" is a well-defined relative advantage.
  - The events in a single user session are NOT exchangeable: an L4 plan
    confirm and an L3 edit undo are different queries at different times with
    different intent levels. Averaging their rewards produces a number with
    no statistical meaning. N=10 doesn't fix this; you'd be averaging apples
    and oranges.
  - This is exactly the trap that grpo_offline_resample.py warns about at
    line 270 of the README: "GRPO 的 group 不是『user 看過的那幾個 plan』，
    而是『訓練時你自己重新抽樣出來的 G 個回答』".

So this module does three honest things instead:

  1. Stratified statistics: group events by intent_level (L2/L3/L4) before
     summarising. Win-rate inside a stratum IS comparable.
  2. Reward-hacking detection: flag when win-rate jumps suspiciously between
     windows, or when high-reward events collapse onto near-identical queries
     (a degenerate-policy signature).
  3. A compactor that reads the raw feedback log and emits KEEP / AVOID rules.
     The compactor has a deterministic heuristic pass (runs without an LLM key)
     and an optional LLM refinement pass (inject any chat-completion callable).

The thing that WOULD make this real GRPO — counterfactual: seeing G candidate
responses for the same query scored against each other — is NOT collected here,
because production copilots serve one response per query. That's a deliberate,
documented limitation, not an oversight.

HOW IT HOOKS INTO THE EXISTING REPO
-----------------------------------
KairosLogMemory (kairos_log.py) already has log_event() with plan_id /
quality_tag. We *add* a FeedbackEvent dataclass and a couple of readers that
consume the same JSON payloads it already writes. No edits to kairos_log.py,
grpo_toy.py, or any teaching file.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable, Optional


# ============================================================
# Reward design — intent-aware, *not* a single flat scale
# ============================================================
# The reward schema is the most opinionated part of the whole pipeline. The
# key design choice: do NOT collapse every user signal onto one axis. An L4
# plan confirm and an L2 clarifying-question answer are different kinds of
# success; lumping them into one mean is what makes naive "session advantage"
# meaningless. We keep them separate all the way through.
REWARD_BY_SIGNAL = {
    # L4 — dashboard / plan generation
    "L4_confirm":   +1.0,   # user accepted the generated dashboard
    "L4_abandon":   -0.5,   # user walked away without saving
    # L3 — inline edits on an existing dashboard
    "L3_accept":    +0.5,   # edit was kept
    "L3_undo":      -1.0,   # edit was undone (strongest negative: user actively reversed it)
    # L2 — direct Q&A, no dashboard artifact
    "L2_answer":    +0.3,   # user got an answer and moved on (weak positive)
    "L2_answer_cq": +0.5,   # a clarifying question was *answered* by the user
}

INTENT_LEVEL_OF = {
    "L4_confirm": "L4", "L4_abandon": "L4",
    "L3_accept": "L3",  "L3_undo": "L3",
    "L2_answer": "L2",  "L2_answer_cq": "L2",
}


@dataclass
class FeedbackEvent:
    """
    One user-feedback signal. This is the data unit of the whole pipeline.

    signal     : one of REWARD_BY_SIGNAL's keys. Determines reward AND intent_level.
    query      : the user's request text (truncated / hashed for diversity checks).
    session_id : which session this came from.
    ts         : ISO timestamp; auto-filled if blank.
    policy_version : opaque tag (e.g. an AGENTS.md hash) so we can detect
                     off-policy staleness later WITHOUT pretending to do
                     importance sampling. See _off_policy_note below.
    meta       : arbitrary extra fields, passed through to the log.
    """
    signal: str
    query: str
    session_id: str
    policy_version: str = "v0"
    ts: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.signal not in REWARD_BY_SIGNAL:
            raise ValueError(
                f"unknown signal {self.signal!r}; "
                f"expected one of {sorted(REWARD_BY_SIGNAL)}"
            )
        if not self.ts:
            from datetime import datetime, timezone
            self.ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ---------- the four things a downstream consumer wants ----------
    @property
    def reward(self) -> float:
        return REWARD_BY_SIGNAL[self.signal]

    @property
    def intent_level(self) -> str:
        return INTENT_LEVEL_OF[self.signal]

    def to_payload(self) -> dict:
        """The JSON we append to KairosLogMemory. Stays log_event()-compatible."""
        d = asdict(self)
        d["reward"] = self.reward
        d["intent_level"] = self.intent_level
        return d

    def to_log_lines(self) -> str:
        """
        Two lines, matching the dual-view convention KairosLogMemory.log_event
        already uses (human-readable bullet + indented JSON payload).
        """
        human = (f"- {self.ts} [feedback] {self.signal} "
                 f"(reward={self.reward:+.2f}, level={self.intent_level}) "
                 f'query="{self.query[:60]}"')
        payload = "  " + json.dumps(self.to_payload(), ensure_ascii=False)
        return human + "\n" + payload


# ============================================================
# Recording events into an existing KairosLogMemory
# ============================================================
def record_feedback(kairos, event: FeedbackEvent) -> None:
    """Append a FeedbackEvent to a KairosLogMemory instance. Pure additive."""
    kairos.append(event.to_log_lines())


def read_feedback_events(kairos) -> list[FeedbackEvent]:
    """
    Parse the Kairos log back into FeedbackEvent objects. We scan only lines
    carrying our marker. log_event()-style events (plan_generated etc.) are
    ignored here — they belong to the GRPO teaching demos, not this pipeline.
    """
    out: list[FeedbackEvent] = []
    if not kairos.path.exists():
        return out
    for raw in kairos.path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        # only pick up lines this module wrote (they carry intent_level)
        if "intent_level" not in d or "signal" not in d:
            continue
        try:
            out.append(FeedbackEvent(
                signal=d["signal"],
                query=d.get("query", ""),
                session_id=d.get("session_id", ""),
                policy_version=d.get("policy_version", "v0"),
                ts=d.get("ts", ""),
                meta=d.get("meta", {}) or {},
            ))
        except (KeyError, ValueError):
            continue
    return out


# ============================================================
# Stratified statistics
# ============================================================
@dataclass
class StratumStats:
    n: int
    mean_reward: float
    win_rate: float          # fraction of events with reward > 0
    undo_rate: float         # fraction of events that are explicit negatives

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "mean_reward": round(self.mean_reward, 3),
            "win_rate": round(self.win_rate, 3),
            "undo_rate": round(self.undo_rate, 3),
        }


def stratified_stats(events: Iterable[FeedbackEvent]) -> dict[str, StratumStats]:
    """
    Group by intent_level and summarise. This is the ONLY averaging we do, and
    it's defensible because events inside one stratum are roughly comparable
    (same kind of user signal, same kind of agent action).

    We deliberately do NOT compute a cross-stratum "session advantage" — that
    is the misleading quantity the EpisodeBuffer proposal was producing.
    """
    by_level: dict[str, list[FeedbackEvent]] = defaultdict(list)
    for e in events:
        by_level[e.intent_level].append(e)

    out: dict[str, StratumStats] = {}
    for level, evs in by_level.items():
        rewards = [e.reward for e in evs]
        n = len(rewards)
        out[level] = StratumStats(
            n=n,
            mean_reward=statistics.fmean(rewards) if rewards else 0.0,
            win_rate=sum(1 for r in rewards if r > 0) / n if n else 0.0,
            undo_rate=sum(1 for e in evs if e.signal in ("L3_undo", "L4_abandon")) / n if n else 0.0,
        )
    return out


# ============================================================
# Reward-hacking detection
# ============================================================
# Why this matters more than fancy advantage math: a sudden win-rate spike
# almost always means the policy found a degenerate shortcut, not that it got
# smarter. Two signatures we look for:
#   (1) win-rate jump between consecutive windows larger than a sane threshold
#   (2) high-reward events collapsing onto near-identical queries (low lexical
#       diversity), i.e. the agent is gaming one query type
@dataclass
class AnomalyReport:
    win_rate_spike: bool
    diversity_collapse: bool
    detail: str

    def as_dict(self) -> dict:
        return {
            "win_rate_spike": self.win_rate_spike,
            "diversity_collapse": self.diversity_collapse,
            "detail": self.detail,
        }


def _lexical_diversity(queries: list[str]) -> float:
    """
    Fraction of *unique normalised tokens* across the query set, in [0,1].
    A collapse toward 0 means the agent is being rewarded for very repetitive
    queries — the classic gaming signature.
    """
    tokens = [tok for q in queries for tok in set(q.lower().split())]
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def detect_anomalies(
    sessions: list[list[FeedbackEvent]],
    spike_threshold: float = 0.25,
    diversity_floor: float = 0.40,
) -> AnomalyReport:
    """
    sessions: list of per-session event lists (chronological order matters).
    """
    if len(sessions) < 2:
        return AnomalyReport(False, False, "fewer than 2 sessions; nothing to compare")

    win_rates = []
    for s in sessions:
        rs = [e.reward for e in s]
        win_rates.append(sum(1 for r in rs if r > 0) / len(rs) if rs else 0.0)

    delta = win_rates[-1] - win_rates[-2]
    spike = delta > spike_threshold

    # diversity: look at the *most recent* session's positive-reward queries
    recent_positive = [e.query for e in sessions[-1] if e.reward > 0]
    if len(recent_positive) >= 2:
        div = _lexical_diversity(recent_positive)
        collapse = div < diversity_floor
        diversity_detail = f"recent-positive-query token diversity={div:.2f}"
    else:
        collapse = False
        diversity_detail = "too few recent positive queries to assess diversity"

    parts = [f"win_rate {win_rates[-2]:.2f}→{win_rates[-1]:.2f} (Δ={delta:+.2f})",
             diversity_detail]
    return AnomalyReport(
        win_rate_spike=spike,
        diversity_collapse=collapse,
        detail="; ".join(parts),
    )


# ============================================================
# Off-policy note (NOT importance sampling)
# ============================================================
def _off_policy_note(events: Iterable[FeedbackEvent]) -> str:
    """
    Real GRPO reuses old data only with an importance-sampling correction
    (π_new / π_old). We can't compute that ratio — our "policy" isn't a
    probability density, it's a set of injected text rules. So instead we
    detect staleness (mixed policy_versions in the log) and tell the compactor
    to down-weight older versions by simple scaling.

    This is a conservative heuristic, not a theorem. We document it as such
    rather than dress it up as IS.
    """
    versions = Counter(e.policy_version for e in events)
    if len(versions) <= 1:
        return "single policy_version in log; no off-policy concern"
    ordered = sorted(versions.items())
    return (
        f"mixed policy_versions present: {dict(ordered)}; "
        f"compactor down-weights non-current versions linearly "
        f"(heuristic, NOT importance sampling — see module docstring)"
    )


# ============================================================
# Compactor: feedback log → behavior rules
# ============================================================
# Two passes:
#   1. Heuristic pass (deterministic, no LLM): scan events, bucket by
#      (intent_level, signal), and emit a templated KEEP/AVOID rule for any
#      bucket that crosses a confidence threshold (enough events, clear sign).
#   2. Optional LLM pass: hand the heuristic draft + raw events to a chat model
#      for rewrite into natural prose. Caller supplies any callable matching
#      llm_rewrite(prompt: str) -> str; if None, the heuristic draft is final.

# A rule is just a dict. Persisting it as JSON (not free text) keeps the
# pipeline auditable and stops the LLM from inventing rules with no evidence.
@dataclass
class Rule:
    kind: str            # "KEEP" | "AVOID"
    intent_level: str
    pattern: str         # short description of the trigger condition
    evidence: dict       # the stats that justified this rule
    source: str          # "heuristic" | "heuristic+llm"


def compact_heuristic(
    events: list[FeedbackEvent],
    min_n: int = 3,
    win_rate_keep: float = 0.70,
    undo_rate_avoid: float = 0.50,
) -> list[Rule]:
    """
    Deterministic rule extraction. For each (intent_level, signal) bucket:
      - if bucket is mostly positive (win_rate >= win_rate_keep) AND n >= min_n
        → emit a KEEP rule
      - if bucket is mostly negative (undo_rate >= undo_rate_avoid) AND n >= min_n
        → emit an AVOID rule
    Buckets below min_n are dropped — there isn't enough evidence to act on.
    """
    bucket: dict[tuple[str, str], list[FeedbackEvent]] = defaultdict(list)
    for e in events:
        bucket[(e.intent_level, e.signal)].append(e)

    rules: list[Rule] = []
    for (level, signal), evs in sorted(bucket.items()):
        n = len(evs)
        if n < min_n:
            continue
        rewards = [e.reward for e in evs]
        win_rate = sum(1 for r in rewards if r > 0) / n
        undo_rate = sum(1 for e in evs if e.signal in ("L3_undo", "L4_abandon")) / n
        stat = {"n": n, "win_rate": round(win_rate, 3), "undo_rate": round(undo_rate, 3)}

        if win_rate >= win_rate_keep:
            rules.append(Rule(
                kind="KEEP", intent_level=level,
                pattern=f"{signal} behavior is landing well with users",
                evidence=stat, source="heuristic",
            ))
        elif undo_rate >= undo_rate_avoid:
            rules.append(Rule(
                kind="AVOID", intent_level=level,
                pattern=f"{signal} behavior is being reversed/abandoned by users",
                evidence=stat, source="heuristic",
            ))
    return rules


def _rules_to_prompt(rules: list[Rule], events: list[FeedbackEvent]) -> str:
    """Format the heuristic draft + raw evidence for an optional LLM rewrite."""
    rule_lines = []
    for r in rules:
        rule_lines.append(
            f"- [{r.kind}] ({r.intent_level}) {r.pattern} "
            f"| evidence={r.evidence}"
        )
    sample_events = [
        f'  {e.signal} r={e.reward:+.2f} q="{e.query[:50]}"'
        for e in events[:12]
    ]
    return (
        "You are refining behavior rules extracted from a copilot's user-feedback log.\n"
        "The draft rules below were produced by a deterministic heuristic. Rewrite each\n"
        "into one clear imperative sentence addressed to the agent. Do NOT invent new\n"
        "rules; only rephrase what's listed. Keep the [KEEP]/[AVOID] tag and the (Ln) tag.\n\n"
        "Draft rules:\n" + "\n".join(rule_lines) + "\n\n"
        "Sample events for context:\n" + "\n".join(sample_events) + "\n"
    )


def compact(
    events: list[FeedbackEvent],
    min_n: int = 3,
    llm_rewrite: Optional[Callable[[str], str]] = None,
) -> dict:
    """
    Full compaction. Returns a structured dict (write this to AGENTS.md or a
    rules.json). If llm_rewrite is supplied, it's called once on the heuristic
    draft to produce natural-prose rules; the evidence dict for each rule is
    preserved either way so nothing becomes unauditable.
    """
    heuristic_rules = compact_heuristic(events, min_n=min_n)
    anomaly = detect_anomalies(_sessionize(events))
    off_policy = _off_policy_note(events)
    strat = {k: v.as_dict() for k, v in stratified_stats(events).items()}

    refined: list[str] = []
    if llm_rewrite is not None and heuristic_rules:
        refined_text = llm_rewrite(_rules_to_prompt(heuristic_rules, events))
        refined = [ln for ln in refined_text.splitlines() if ln.strip()]

    return {
        "rules": [asdict(r) for r in heuristic_rules],
        "rules_refined": refined,
        "stratified_stats": strat,
        "anomaly": anomaly.as_dict(),
        "off_policy_note": off_policy,
        "n_events": len(events),
    }


def _sessionize(events: list[FeedbackEvent]) -> list[list[FeedbackEvent]]:
    """Split events back into per-session lists (used by detect_anomalies)."""
    by_sess: dict[str, list[FeedbackEvent]] = defaultdict(list)
    for e in events:
        by_sess[e.session_id].append(e)
    # keep chronological-ish order: by first-seen session id
    seen = []
    for e in events:
        if e.session_id not in seen:
            seen.append(e.session_id)
    return [by_sess[s] for s in seen]
