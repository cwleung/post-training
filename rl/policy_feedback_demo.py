#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
policy_feedback_demo.py — 路線 A 完整閉環 demo（in-context policy learning）
==============================================================================

這個 demo 把 policy_feedback.py 串成一條**看得見的閉環**，跑給你看：

    user feedback
        │  FeedbackEvent
        ▼
    KairosLogMemory  ← append-only daily log
        │
        ▼
    read_feedback_events()  ──→  stratified_stats()    （分層 win_rate）
                            ──→  detect_anomalies()     （reward-hacking 偵測）
                            ──→  compact()              （提取 KEEP/AVOID rules）
                                                            │
                                                            ▼
                                              寫回 AGENTS.md 的 ## Learned Behavior Rules
                                                            │
                                                            ▼
                                              下一個 session 的 system prompt 自動注入
                                              → 行為改變，但 LLM 權重完全不動

== 為什麼要這個 demo ==
policy_feedback.py 的每個函式都能單獨跑，但**閉環本身**才是面試時最能講的東西：
「我把 user feedback 變成 behavior rule，再變回下個 session 的 system prompt，
全程沒有 weight update」。這個 demo 就是把這句話變成可執行的證據。

== 它刻意不是 GRPO ==
policy_feedback.py 的 docstring 已經講清楚為什麼不做 within-session advantage。
這個 demo 對應 README 裡的「路線 A」。要看真 GRPO，跑 grpo_toy.py /
demo_log_to_grpo.py / grpo_offline_resample.py。

== 執行 ==
    python3 policy_feedback_demo.py

不依賴 numpy、不依賴任何 LLM API key。compactor 的 LLM pass 在這個 demo
裡用一個 deterministic 的 fake-LLM 示範接線方式；換成真的 chat-completion
callable 就能直接用。

== 五個場景 ==
    [S1]  正常 traffic       — 混合 L4/L3/L2，沒有異常
    [S2]  純學好的 agent     — L4_confirm 為主，產生 KEEP rules
    [S3]  純學壞的 agent     — L3_undo 為主，產生 AVOID rules
    [S4]  reward-hacking     — win_rate 跳升 + query 多樣性崩塌 → 觸發異常
    [S5]  跨 policy_version  — log 裡有新舊兩版，off-policy note 偵測到
    [S6]  LLM 接線示範       — compact() 接受一個 callable，輸出自然語句 rules
    [S7]  寫回 AGENTS.md     — 把 rules 真的 render 進 markdown 的三層結構
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from textwrap import dedent

from kairos_log import KairosLogMemory
from policy_feedback import (
    FeedbackEvent,
    record_feedback,
    read_feedback_events,
    stratified_stats,
    detect_anomalies,
    compact,
    _sessionize,
)


# ============================================================
# 模擬資料工廠 — 五種「traffic 形狀」
# ============================================================
# 每個 factory 回傳一個 session 的 FeedbackEvent list。
# 這些不是亂編的：它們對應 README 裡列的真實 user signal，
# 數值直接來自 policy_feedback.REWARD_BY_SIGNAL。

def session_normal(sid: str = "s1") -> list[FeedbackEvent]:
    """混合三個 intent level 的健康 traffic。"""
    return [
        FeedbackEvent("L4_confirm",   "build a Q3 sales dashboard",      sid, "v1"),
        FeedbackEvent("L2_answer",    "what is absorption rate",         sid, "v1"),
        FeedbackEvent("L3_undo",      "change the chart color",          sid, "v1"),
        FeedbackEvent("L4_confirm",   "build a pricing trends dashboard", sid, "v1"),
        FeedbackEvent("L2_answer_cq", "which markets do you mean",       sid, "v1"),
    ]


def session_learning_well(sid: str = "s2") -> list[FeedbackEvent]:
    """L4 plan 一直被接受 → 會觸發 KEEP rule。"""
    return [
        FeedbackEvent("L4_confirm", "build a starts-by-month dashboard", sid, "v1"),
        FeedbackEvent("L4_confirm", "build a closings scorecard",        sid, "v1"),
        FeedbackEvent("L4_confirm", "build a VDL inventory view",        sid, "v1"),
        FeedbackEvent("L2_answer",  "top 5 builders in dallas",          sid, "v1"),
    ]


def session_learning_bad(sid: str = "s3") -> list[FeedbackEvent]:
    """L3 edit 一直被 undo → 會觸發 AVOID rule。"""
    return [
        FeedbackEvent("L3_undo", "recolor without asking",  sid, "v1"),
        FeedbackEvent("L3_undo", "reorder columns silently", sid, "v1"),
        FeedbackEvent("L3_undo", "swap the x/y axes",        sid, "v1"),
        FeedbackEvent("L4_confirm", "build a dashboard",     sid, "v1"),
    ]


def session_hacking(sid: str = "s4") -> list[FeedbackEvent]:
    """
    Reward-hacking signature: 高 reward + query 幾乎一模一樣。
    模擬 agent 學到「永遠先 propose plan，無視 query 內容」的捷徑。
    win_rate 會很高，但 lexical diversity 會崩塌。
    """
    return [
        FeedbackEvent("L4_confirm", "build dashboard",    sid, "v1"),
        FeedbackEvent("L4_confirm", "build dashboard",    sid, "v1"),
        FeedbackEvent("L4_confirm", "build dashboard",    sid, "v1"),
        FeedbackEvent("L4_confirm", "build dashboard",    sid, "v1"),
        FeedbackEvent("L4_confirm", "build dashboard",    sid, "v1"),
    ]


def session_old_policy(sid: str = "s5") -> list[FeedbackEvent]:
    """policy_version=v0 的舊 events，跟新版本混在一起。"""
    return [
        FeedbackEvent("L4_confirm", "build austin market overview", sid, "v0"),
        FeedbackEvent("L2_answer",  "absorption rate phoenix",      sid, "v0"),
    ]


# ============================================================
# 共用工具
# ============================================================

def banner(title: str, char: str = "=", width: int = 70) -> None:
    print()
    print(char * width)
    print(title)
    print(char * width)


def record_sessions(kairos: KairosLogMemory, sessions: list[list[FeedbackEvent]]) -> None:
    for sess in sessions:
        for e in sess:
            record_feedback(kairos, e)


# ============================================================
# 場景 S1 — 正常 traffic 的分層統計
# ============================================================

def run_s1_normal_traffic(tmpdir: Path) -> None:
    banner("[S1] 正常 traffic — 分層 win_rate，無異常")

    kairos = KairosLogMemory(tmpdir / "s1.log.md")
    sessions = [session_normal("s1"), session_learning_well("s1b")]
    record_sessions(kairos, sessions)

    events = read_feedback_events(kairos)
    print(f"\n讀回 {len(events)} 個 feedback events")
    print(f"\nlog 檔內容（前 6 行）：")
    for line in kairos.read().splitlines()[:6]:
        print(f"  {line}")

    print(f"\n分層統計（不垮層平均）：")
    for level, stat in sorted(stratified_stats(events).items()):
        print(f"  {level}: {stat.as_dict()}")

    anomaly = detect_anomalies(_sessionize(events))
    print(f"\n異常偵測：{anomaly.as_dict()}")
    print("\n→ L4 win_rate=1.0（兩個 confirm），L3 undo_rate=1.0（一個 undo）；")
    print("  沒有跨 session 的 win_rate 跳升，沒有 diversity 崩塌 → 正常。")


# ============================================================
# 場景 S2 — 學得好 → KEEP rules
# ============================================================

def run_s2_keep_rules(tmpdir: Path) -> None:
    banner("[S2] agent 學得好 — compact() 產生 KEEP rules")

    kairos = KairosLogMemory(tmpdir / "s2.log.md")
    record_sessions(kairos, [
        session_learning_well("s2a"),
        session_learning_well("s2b"),
    ])

    events = read_feedback_events(kairos)
    result = compact(events, min_n=3)

    print(f"\n讀回 {result['n_events']} 個 events")
    print(f"\n分層統計：")
    for level, stat in sorted(result["stratified_stats"].items()):
        print(f"  {level}: {stat}")

    print(f"\n提取出的 rules（heuristic pass）：")
    if not result["rules"]:
        print("  (無 — 樣本數不足 min_n=3)")
    for r in result["rules"]:
        tag = r["kind"]
        print(f"  [{tag}] ({r['intent_level']}) {r['pattern']}")
        print(f"         evidence={r['evidence']}")

    print("\n→ L4_confirm bucket 的 win_rate 達 1.0 且 n>=3 → 自動產生 KEEP rule。")
    print("  rule 自帶 evidence dict，可被稽核，不是 LLM hallucination。")


# ============================================================
# 場景 S3 — 學得壞 → AVOID rules
# ============================================================

def run_s3_avoid_rules(tmpdir: Path) -> None:
    banner("[S3] agent 學得壞 — compact() 產生 AVOID rules")

    kairos = KairosLogMemory(tmpdir / "s3.log.md")
    record_sessions(kairos, [
        session_learning_bad("s3a"),
        session_learning_bad("s3b"),
    ])

    events = read_feedback_events(kairos)
    result = compact(events, min_n=3)

    print(f"\n分層統計：")
    for level, stat in sorted(result["stratified_stats"].items()):
        print(f"  {level}: {stat}")

    print(f"\n提取出的 rules：")
    for r in result["rules"]:
        tag = r["kind"]
        print(f"  [{tag}] ({r['intent_level']}) {r['pattern']}")
        print(f"         evidence={r['evidence']}")

    print("\n→ L3_undo bucket 的 undo_rate 達 1.0 且 n>=3 → 自動產生 AVOID rule。")
    print("  這條 rule 下個 session 會變成 system prompt 裡的「不要默默改東西」。")


# ============================================================
# 場景 S4 — reward hacking 偵測（最重要的 demo）
# ============================================================

def run_s4_hacking(tmpdir: Path) -> None:
    banner("[S4] reward-hacking 偵測 — win_rate 跳升 + query 多樣性崩塌")

    kairos = KairosLogMemory(tmpdir / "s4.log.md")
    # 上一個 session：正常混合，win_rate 中等
    # 這個 session：全部 confirm，但 query 幾乎一模一樣
    record_sessions(kairos, [
        session_normal("s4_prev"),
        session_hacking("s4_now"),
    ])

    events = read_feedback_events(kairos)
    result = compact(events, min_n=3)

    print(f"\n分層統計：")
    for level, stat in sorted(result["stratified_stats"].items()):
        print(f"  {level}: {stat}")

    anomaly = result["anomaly"]
    print(f"\n異常偵測：")
    print(f"  win_rate_spike     = {anomaly['win_rate_spike']}")
    print(f"  diversity_collapse = {anomaly['diversity_collapse']}")
    print(f"  detail             = {anomaly['detail']}")

    if anomaly["win_rate_spike"] or anomaly["diversity_collapse"]:
        print("\n→ ⚠️  觸發 reward-hacking 警示！")
        print("  解讀：win_rate 飆高，但 query 幾乎一模一樣 → agent 在走捷徑，")
        print("        不是真的變聰明。下一個 /dream 應該 freeze，不落地新 rules。")
    else:
        print("\n→ 沒觸發（如果出現這行，閾值要調）。")


# ============================================================
# 場景 S5 — 跨 policy_version（off-policy note）
# ============================================================

def run_s5_offpolicy(tmpdir: Path) -> None:
    banner("[S5] off-policy 偵測 — log 裡混合新舊 policy_version")

    kairos = KairosLogMemory(tmpdir / "s5.log.md")
    record_sessions(kairos, [
        session_old_policy("s5_old"),     # v0
        session_normal("s5_new"),         # v1
    ])

    events = read_feedback_events(kairos)
    result = compact(events, min_n=3)

    print(f"\noff-policy note：")
    print(f"  {result['off_policy_note']}")

    print("\n→ 我們不做 importance sampling（沒有機率密度可比）。")
    print("  改用誠實做法：偵測到版本混合時，提示 compactor 對舊版本線性降權。")
    print("  這是 heuristic，不是定理；docstring 裡有寫清楚。")


# ============================================================
# 場景 S6 — LLM 接線示範
# ============================================================

def fake_llm_rewrite(prompt: str) -> str:
    """
    假的 LLM callable，示範 compact() 的 LLM 接線。
    真實場景換成 OpenAI/Anthropic 的 chat-completion 即可，簽名一致：
        llm_rewrite(prompt: str) -> str
    """
    # 從 prompt 裡把 heuristic draft 抓出來，改寫成自然語句
    out_lines = []
    for line in prompt.splitlines():
        line = line.strip()
        if not line.startswith("- ["):
            continue
        # 簡單改寫：[KEEP] (L4) ... → "When handling L4 requests, keep doing ..."
        if "[KEEP]" in line:
            body = line.split(")", 1)[1].split("|")[0].strip()
            level = line.split("(", 1)[1].split(")")[0]
            out_lines.append(f"When handling {level} requests, keep: {body}.")
        elif "[AVOID]" in line:
            body = line.split(")", 1)[1].split("|")[0].strip()
            level = line.split("(", 1)[1].split(")")[0]
            out_lines.append(f"When handling {level} requests, avoid: {body}.")
    return "\n".join(out_lines) if out_lines else "(no rules to refine)"


def run_s6_llm_wiring(tmpdir: Path) -> None:
    banner("[S6] LLM 接線示範 — compact() 接受 callable，輸出自然語句")

    kairos = KairosLogMemory(tmpdir / "s6.log.md")
    record_sessions(kairos, [
        session_learning_well("s6a"),
        session_learning_well("s6b"),
        session_learning_bad("s6c"),
    ])

    events = read_feedback_events(kairos)
    result = compact(events, min_n=3, llm_rewrite=fake_llm_rewrite)

    print(f"\nheuristic draft rules（機器可讀，自帶 evidence）：")
    for r in result["rules"]:
        print(f"  {r}")

    print(f"\nLLM 改寫後（自然語句，可直接放 system prompt）：")
    for line in result["rules_refined"]:
        print(f"  {line}")

    print("\n→ evidence dict 不管有沒有接 LLM 都會保留 → rules 永遠可被稽核。")
    print("  換真 LLM 只要傳一個 callable，不需要改 compact() 的程式碼。")


# ============================================================
# 場景 S7 — 寫回 AGENTS.md（三層結構）
# ============================================================

AGENTS_MD_TEMPLATE = dedent("""\
    # AGENTS.md

    <!-- policy_version: v0 -->

    ## System Constraints
    - 必須使用 framework 內的 controller/visualization
    - 不能存取 subscription 外的 data source

    ## User Preferences
    - 偏好簡潔回答
    - 喜歡深色圖表

    ## Learned Behavior Rules
    (none yet)
    """)


def render_agents_md(rules: list[dict], old_version: str = "v0") -> tuple[str, str]:
    """
    把 compact() 的 rules render 進 AGENTS.md 的三層結構。
    回傳 (new_markdown, new_version)。

    關鍵：System Constraints 和 User Preferences 原樣不動，
    只動 Learned Behavior Rules 區塊 + bump policy_version。
    """
    keep = [r for r in rules if r["kind"] == "KEEP"]
    avoid = [r for r in rules if r["kind"] == "AVOID"]

    new_version = f"v{int(old_version.lstrip('v')) + 1}" if rules else old_version

    learned_lines = ["## Learned Behavior Rules"]
    if not rules:
        learned_lines.append("_(no rules yet — accumulate events first)_")
    else:
        learned_lines.append(f"<!-- policy_version: {new_version}; "
                             f"source: policy_feedback.compact() -->")
        if keep:
            learned_lines.append("")
            learned_lines.append("### KEEP")
            for r in keep:
                learned_lines.append(
                    f"- [{r['intent_level']}] {r['pattern']} "
                    f"(evidence: win_rate={r['evidence']['win_rate']}, "
                    f"n={r['evidence']['n']})"
                )
        if avoid:
            learned_lines.append("")
            learned_lines.append("### AVOID")
            for r in avoid:
                learned_lines.append(
                    f"- [{r['intent_level']}] {r['pattern']} "
                    f"(evidence: undo_rate={r['evidence']['undo_rate']}, "
                    f"n={r['evidence']['n']})"
                )

    # 重組：guardrail + preference 原樣，只換 learned
    new_md = "\n".join([
        "# AGENTS.md",
        "",
        f"<!-- policy_version: {new_version} -->",
        "",
        "## System Constraints",
        "- 必須使用 framework 內的 controller/visualization",
        "- 不能存取 subscription 外的 data source",
        "",
        "## User Preferences",
        "- 偏好簡潔回答",
        "- 喜歡深色圖表",
        "",
    ] + learned_lines) + "\n"
    return new_md, new_version


def run_s7_write_agents_md(tmpdir: Path) -> None:
    banner("[S7] 寫回 AGENTS.md — 三層結構，guardrail 不被動到")

    agents_md = tmpdir / "AGENTS.md"
    agents_md.write_text(AGENTS_MD_TEMPLATE)

    kairos = KairosLogMemory(tmpdir / "s7.log.md")
    record_sessions(kairos, [
        session_learning_well("s7a"),
        session_learning_well("s7b"),
        session_learning_bad("s7c"),
        session_learning_bad("s7d"),
    ])

    events = read_feedback_events(kairos)
    result = compact(events, min_n=3)

    new_md, new_version = render_agents_md(result["rules"], old_version="v0")
    agents_md.write_text(new_md)

    print(f"\n落地後的 AGENTS.md（policy_version: v0 → {new_version}）：")
    print(new_md)

    # 斷言：guardrail 從頭到尾沒被動到
    assert "必須使用 framework" in new_md, "guardrail 被動到了！"
    assert "不能存取 subscription" in new_md, "guardrail 被動到了！"
    assert "偏好簡潔回答" in new_md, "user preference 被動到了！"
    assert new_version == "v1", "version 應該 bump 到 v1"
    print("→ ✓ guardrail 和 user preference 原樣保留，只有 Learned 區塊被更新。")
    print("→ ✓ policy_version 從 v0 → v1，下個 session 可以偵測 off-policy data。")


# ============================================================
# 閉環總覽 — 全部串起來的視覺化
# ============================================================

def run_overview() -> None:
    banner("路線 A 完整閉環總覽", char="=")
    print("""
    user 按了 confirm / undo / abandon
        │
        │  record_feedback()
        ▼
    KairosLogMemory (append-only daily .md)
        │
        │  read_feedback_events()
        ▼
    ┌─────────────────────────────────────────────────┐
    │  stratified_stats()   ← 分層 win_rate（不垮層）  │
    │  detect_anomalies()   ← reward-hacking 偵測      │
    │  compact()            ← KEEP/AVOID rules         │
    └─────────────────────────────────────────────────┘
        │
        │  rules 帶 evidence dict（可稽核）
        ▼
    AGENTS.md ## Learned Behavior Rules
    + policy_version bump（off-policy 追蹤）
        │
        │  下個 session 的 ClaudeCodeMemoryMiddleware
        ▼
    system prompt 注入 learned rules
        │
        ▼
    LLM 行為改變 —— 但權重完全不動，沒有 weight update

    == 對照：真 GRPO（grpo_toy.py / demo_log_to_grpo.py）==
    GRPO 多了：G-sample 抽樣 → group baseline → token-level ratio → clip loss → 梯度更新
    路線 A 沒這些，因為 production copilot 每 query 只服務一個 response，
    沒有 G 個候選可比 → 刻意不做 advantage estimation，改用 in-context rules。
    """)


# ============================================================
# main
# ============================================================

def main() -> None:
    run_overview()

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        run_s1_normal_traffic(tmpdir)
        run_s2_keep_rules(tmpdir)
        run_s3_avoid_rules(tmpdir)
        run_s4_hacking(tmpdir)
        run_s5_offpolicy(tmpdir)
        run_s6_llm_wiring(tmpdir)
        run_s7_write_agents_md(tmpdir)

    banner("全部 7 個場景跑完")
    print("""
    重點回顧：
      [S1] 分層 win_rate 是唯一可辯護的平均方式
      [S2/S3] heuristic compactor 自帶 evidence，不靠 LLM 就能跑
      [S4] reward-hacking 偵測 = freeze 機制的觸發條件
      [S5] off-policy 用誠實的版本追蹤，不裝作會 importance sampling
      [S6] LLM 是可插換的 callable，evidence 永遠保留
      [S7] AGENTS.md 三層分離，guardrail 不會被學掉

    面試時可以這樣講：
      「我設計了一個 in-context policy learning pipeline。User feedback 被記成
       FeedbackEvent，按 intent_level 分層算 win_rate（不垮層平均——這是我從
       早期 within-session advantage 的坑學到的）。Compactor 從分層統計提取
       KEEP/AVOID rules，帶 evidence dict，寫進 AGENTS.md 的 Learned 區塊，
       下個 session 自動注入 system prompt。

       為了防 reward hacking，我偵測 win_rate 跳升 + query 多樣性崩塌，
       觸發時 freeze /dream，policy_version 不動。

       整套是 group-relative formulation inspired by GRPO，但我刻意不做
       weight update——production copilot 每 query 只服務一個 response，
       沒有 G 個候選可比，做 advantage estimation 統計上不成立。
       升級到真 RL（DPO/GRPO）的觸發條件我都寫進 README 了。」

    要看真 GRPO 的話：
      python3 grpo_toy.py             # on-policy 玩具，每步印數字
      python3 demo_log_to_grpo.py     # log → dataset → update 完整流程
      python3 grpo_offline_resample.py # 解「group 不是 log 重播」的觀念
    """)


if __name__ == "__main__":
    main()
