#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
web/schemas.py — pydantic request/response models
=================================================

每個 model 對齊一個底層函式的回傳形狀（已在 planning 階段確認）。
型別註解用 Python 3.9 相容寫法（List/Optional 而非 list/|）。
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ============================================================
# 通用
# ============================================================
class ErrorOut(BaseModel):
    error: str
    detail: Optional[str] = None


# ============================================================
# Part 1 · Toy（PolicyState）
# ============================================================
class ToyCreateIn(BaseModel):
    theme: str = Field("dashboard", description="'dashboard' 或 'math'")


class ToyQueryOut(BaseModel):
    q: str
    correct: Optional[int] = None          # math 有，dashboard 沒有


class ToyCreateOut(BaseModel):
    session_id: str
    theme: str
    name: str
    action_label: str
    queries: List[ToyQueryOut]
    action_descriptions: Optional[Dict[str, str]] = None   # dashboard 才有
    auto_label: bool
    probs: List[float]                      # 初始（uniform）某 query 的機率


class ToySampleIn(BaseModel):
    session_id: str
    query_idx: int = 0


class ToySampleOut(BaseModel):
    group: List[int]                        # G 個 action
    query_idx: int


class ToyStepIn(BaseModel):
    session_id: str
    query_idx: int = 0
    group: List[int]
    accepted_idx: int = Field(..., description="0..G-1，-1 = 全拒絕")


class ToyStepOut(BaseModel):
    """逐字對應 PolicyState.apply_user_choice 的回傳 dict。"""
    group: List[int]
    rewards: List[float]
    baseline: float
    advantages: List[float]
    prob_before: List[float]
    prob_after: List[float]
    accepted_action: Optional[int] = None


class ToyResetIn(BaseModel):
    session_id: str
    theme: str = "dashboard"


# ============================================================
# Part 1+1.5 · LLM char-level（TrainingPipeline）
# ============================================================
class LLMMetricsOut(BaseModel):
    round: int
    policy_version: int
    rolling_mean_reward: Optional[float] = None
    frozen: bool
    hacking_detail: str
    checkpoints: int
    beta_kl: float
    kl_implemented: bool


class LLMCreateOut(BaseModel):
    session_id: str
    state: LLMMetricsOut


class LLMStateOut(BaseModel):
    metrics: LLMMetricsOut
    checkpoints: List[Dict[str, Any]]


class LLMGenerateIn(BaseModel):
    session_id: str
    prompt: str = Field("1+1=", description="vocab: <>+= 0123456789")


class LLMCandidate(BaseModel):
    label: str                              # A/B/C/D
    text: str                               # 生成字串（prompt + generated）
    gen_readable: str                       # 把 EOS/空白用符號 render
    logp: float


class LLMGenerateOut(BaseModel):
    prompt: str
    candidates: List[LLMCandidate]


class LLMChoiceIn(BaseModel):
    session_id: str


class LLMResultRow(BaseModel):
    candidate: str                          # A/B/C/D
    reward: float
    advantage: float
    logp_before: float
    logp_after: float
    ratio: float
    clipped: bool
    accepted: bool


class LLMResultOut(BaseModel):
    result_table: List[LLMResultRow]
    baseline: float
    accepted_idx: int
    state: LLMMetricsOut
    note: Optional[str] = None              # 例：frozen 時的說明


class LLMRewardCurveOut(BaseModel):
    rounds: List[int]
    mean_rewards: List[float]
    baselines: List[float]
    rolling: List[Optional[float]]


class LLMPolicyPrefOut(BaseModel):
    """policy 對幾個 context 的 top 轉移機率（給前端畫 bar）。"""
    contexts: List[str]
    series: List[Dict[str, Any]]            # [{label, values:[top5 prob]}...]
    uniform: float
    round: int


class LLMOfflineTrainIn(BaseModel):
    session_id: str
    epochs: int = 1


class LLMOfflineTrainOut(BaseModel):
    iters: List[Dict[str, Any]]
    initial_version: int
    new_version: int
    frozen: bool
    n_samples: int
    off_policy_versions: List[int]


class LLMResetIn(BaseModel):
    session_id: str


# ============================================================
# Part 1 · Offline（Kairos→dataset + resample）
# ============================================================
class OfflineCollectIn(BaseModel):
    group_size: int = 6


class DatasetGroupOut(BaseModel):
    query_id: str
    plan_ids: List[str]
    rewards: List[float]
    advantage: List[float]
    baseline: float
    std: float


class OfflineDatasetOut(BaseModel):
    groups: List[DatasetGroupOut]
    n_events: int


class OfflineResampleOut(BaseModel):
    """重跑 grpo_offline_resample.main() 的邏輯，回 JSON。"""
    log_rows: List[Dict[str, Any]]          # Phase 0：每個 query 1 個 plan
    n_queries: int
    correct_map: Dict[str, int]
    iters: List[Dict[str, Any]]             # [{iter, avg_reward}]
    final_policy: List[Dict[str, Any]]      # [{query, correct, probs, best, ok}]
    n_correct: int


class OfflineLogOut(BaseModel):
    human_view: str
    events: List[Dict[str, Any]]


# ============================================================
# Part 2 · Policy（in-context policy learning）
# ============================================================
class RuleOut(BaseModel):
    kind: str                               # KEEP | AVOID
    intent_level: str
    pattern: str
    evidence: Dict[str, Any]
    source: str


class CompactOut(BaseModel):
    """逐字對應 policy_feedback.compact() 的回傳 dict。"""
    rules: List[RuleOut]
    rules_refined: List[str]
    stratified_stats: Dict[str, Dict[str, Any]]
    anomaly: Dict[str, Any]
    off_policy_note: str
    n_events: int


class ScenarioOut(BaseModel):
    n: int
    key: str
    title: str
    what: str                               # 演什麼
    sessions_used: List[str]                # 用到哪些 factory
    events: List[Dict[str, Any]]            # 每個 event 的 to_payload()
    compact: CompactOut
    agents_md: Optional[str] = None         # S7 才有
    new_version: Optional[str] = None       # S7 才有


class ScenariosIndexOut(BaseModel):
    scenarios: List[Dict[str, Any]]         # [{n,title,what}]


class FeedbackEventIn(BaseModel):
    """給自由 compact 用。"""
    signal: str
    query: str = ""
    session_id: str = "custom"
    policy_version: str = "v0"


class CompactIn(BaseModel):
    events: List[FeedbackEventIn]
    use_llm: bool = False


class AgentsMdIn(BaseModel):
    rules: List[RuleOut]
    old_version: str = "v0"


class AgentsMdOut(BaseModel):
    markdown: str
    new_version: str


# ============================================================
# Tutorial · 從零實作 GRPO（單步 tracer）
# ============================================================
class TutorialSetupIn(BaseModel):
    n_actions: int = Field(6, ge=2, le=10, description="action 數 K（toy 預設 6）")
    group_size: int = Field(8, ge=2, le=16, description="group 大小 G")
    lr: float = Field(1.0, description="學習率")
    eps_clip: float = Field(0.2, description="PPO clip 範圍 ε")
    inner_epochs: int = Field(4, description="同一 group 重用幾次")
    correct_action: int = Field(2, description="哪個 action 答對（reward=1）")
    seed: int = 0


class TutorialSetupOut(BaseModel):
    session_id: str
    k: int
    group_size: int
    lr: float
    eps_clip: float
    inner_epochs: int
    correct_action: int
    pi: List[float]                          # 初始 uniform 機率


# ============================================================
# Wiki · Query & Compound Engine + Wiki Linter
# ============================================================
class WikiQueryIn(BaseModel):
    query: str = Field(..., description="Query asked against the wiki/knowledge base")
    format: str = Field("markdown", description="'markdown', 'table', 'slides', 'chart', 'canvas'")
    auto_search: bool = True


class WikiCitation(BaseModel):
    id: str
    title: str
    type: str  # 'chapter', 'paper', 'kb_node', 'policy_spec'
    url: Optional[str] = None
    excerpt: str


class WikiQueryOut(BaseModel):
    query: str
    format: str
    title: str
    content: str
    data: Optional[Dict[str, Any]] = None   # Structured data for chart / table / canvas
    citations: List[WikiCitation]
    related_queries: List[str] = []


class WikiFileIn(BaseModel):
    title: str = Field(..., description="Node or page title to file into the wiki")
    domain_id: str = Field("synthesis", description="Target domain / category")
    def_text: str = Field(..., description="Core definition or synthesized summary")
    insight: str = Field(..., description="Key technical insight or takeaway")
    format: str = Field("markdown", description="'markdown', 'table', 'slides', 'chart', 'canvas'")
    content: str = Field("", description="Full body or formatted payload")
    sources: List[str] = Field(default_factory=list, description="Citations / paper / chapter keys")
    related: List[str] = Field(default_factory=list, description="Related domain IDs")
    repo: str = Field("", description="Repository mapping pointer")
    tags: List[str] = Field(default_factory=list, description="Search tags")


class WikiFileOut(BaseModel):
    status: str
    node_id: str
    title: str
    filed_at: str
    total_nodes: int


class WikiLintIssue(BaseModel):
    id: str
    type: str       # 'contradiction', 'stale_claim', 'orphan', 'missing_concept', 'missing_crossref', 'data_gap'
    severity: str   # 'critical', 'warning', 'info'
    title: str
    description: str
    evidence: str
    suggested_action: str
    auto_fixable: bool = False
    fix_payload: Optional[Dict[str, Any]] = None


class WikiLintOut(BaseModel):
    health_score: int
    issues_count: int
    issues: List[WikiLintIssue]
    summary: Dict[str, Any]
    recommended_questions: List[str]
    candidate_sources: List[Dict[str, str]]


class WikiLintFixIn(BaseModel):
    issue_id: str
    action: str = "apply_fix"
    payload: Optional[Dict[str, Any]] = None


class WikiLintFixOut(BaseModel):
    success: bool
    message: str
    fixed_issue_id: str
    new_health_score: int

