#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
memory_selector_demo.py — context-budget memory reader, end-to-end demo
=======================================================================

這個 demo 把 memory_selector.py 串成一條**看得見的閉環**，跑給你看：

    log 裡有 N 個 events（新舊混雜、reward 高低不一）
        │
        │  importance()  ← freshness(指數衰減) × magnitude(|reward|)
        ▼
    排分數：愈新 + reward 愈大 → 愈該被記得
        │
        │  compact_log() ← 把 cutoff 之前的舊事件壓成一行 summary
        ▼
    read_for_context(budget_tokens)
        │   ├─ 預留 rollup 額度 → 舊事件變成 summary 一行
        │   ├─ recency floor   → 最後 floor_n 個事件無論分數一律保留
        │   ├─ importance 分片 → 60% 預算給「分數最高」
        │   └─ recency 分片    → 40% 預算給「最新」
        ▼
    一條塞進 token budget 的字串，給 LLM 的 system prompt 用

== 為什麼要這個 demo ==
memory_selector.py 的每個函式都能單獨跑，但**三層的組合**才是面試時最能
講的東西：「我把 user feedback log 變成一個 budget-aware 的 context 注入器：
舊的事件被 compact 成 summary，剩下的事件依 importance×freshness 排序，
greedy 塞進 token budget，全程不刪 log、不動 weight」。

== 它跟 policy_feedback_demo.py 的關係 ==
policy_feedback_demo.py 是「feedback → rules」的路線 A。
這個 demo 是「log → context slice」的**記憶讀取層**，跟路線 A 正交：
路線 A 關心「學到什麼 rule」，這裡關心「當 context 有限時該把什麼放進去」。
兩者讀的是同一個 KairosLogMemory。

== 執行 ==
    python3 memory_selector_demo.py

零依賴（tiktoken 有就用、沒有就 fallback 並明確警告）。
六個場景對應 README 裡列的 memory 三功能：compact / budget / freshness。

== 六個場景 ==
    [M1] importance×freshness  — 看分數隨時間指數衰減
    [M2] budget 強制           — log 是預算的 3 倍，看誰被擠掉
    [M3] recency floor         — 最後幾個事件低分但強制保留（防餓死）
    [M4] log compaction        — 200 個事件壓成一行，log 檔不變
    [M5] 完整閉環              — 真實混合 log 跑 read_for_context，印 scorecard
    [M6] estimator 誠實性      — tiktoken vs len//4，明講現在用哪個
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from kairos_log import KairosLogMemory
from memory_selector import (
    compact_log,
    est_tokens,
    estimator_name,
    importance,
    read_for_context,
    recency_score,
)


# ============================================================
# 共用工具
# ============================================================

def banner(title: str, char: str = "=", width: int = 70) -> None:
    print()
    print(char * width)
    print(title)
    print(char * width)


NOW = datetime.now(timezone.utc)


def _feedback_payload(ts, signal, query, reward, level, sid="s1", pver="v1"):
    """直接 append 一個 feedback-style JSON payload（帶 reward/intent_level）。"""
    return (
        f"- {ts} [feedback] {signal}\n"
        f'  {{"ts": "{ts}", "signal": "{signal}", "query": "{query}", '
        f'"session_id": "{sid}", "policy_version": "{pver}", '
        f'"reward": {reward}, "intent_level": "{level}"}}'
    )


# ============================================================
# M1 — importance × freshness
# ============================================================
def run_m1_importance_freshness(tmpdir: Path) -> None:
    banner("[M1] importance×freshness — 分數隨時間指數衰減")

    print(f"\nrecency 半衰期 = 48 小時（每 48h 分數減半）：")
    print(f"{'年齡':>10} | {'recency':>8} | 意義")
    print("-" * 50)
    for hours, label in [(0, "剛發生"), (24, "1 天前"), (48, "2 天前 (=τ)"),
                         (96, "4 天前"), (168, "1 週前")]:
        r = recency_score((NOW - timedelta(hours=hours)).isoformat(), NOW)
        print(f"{hours:>8}h | {r:>8.4f} | {label}")

    print(f"\nimportance = recency × (0.3 + |reward|)：")
    print(f"固定 reward=+1.0，只變年齡：")
    events = [
        ("fresh, +1.0",       NOW - timedelta(hours=2),   1.0),
        ("1 day, +1.0",       NOW - timedelta(days=1),    1.0),
        ("1 week, +1.0",      NOW - timedelta(days=7),    1.0),
        ("1 month, +1.0",     NOW - timedelta(days=30),   1.0),
    ]
    for label, ts, reward in events:
        e = {"ts": ts.isoformat(timespec="seconds"), "reward": reward}
        s = importance(e, NOW)
        bar = "█" * int(s * 20)
        print(f"  {label:<18} score={s:.4f} {bar}")

    print(f"\n固定年齡=新，只變 reward（magnitude 軸）：")
    for label, reward in [("reward=0 (neutral)", 0.0), ("reward=+0.5", 0.5),
                          ("reward=+1.0", 1.0), ("reward=-1.0", -1.0)]:
        e = {"ts": NOW.isoformat(timespec="seconds"), "reward": reward}
        s = importance(e, NOW)
        bar = "█" * int(s * 20)
        print(f"  {label:<20} score={s:.4f} {bar}")

    print("\n→ 兩軸相乘：極舊的高 reward 事件仍會淡出（recency→0），")
    print("  極新的零訊號事件不會獨大（floor=0.3）。這就是「freshness control」。")


# ============================================================
# M2 — budget 強制
# ============================================================
def run_m2_budget_enforcement(tmpdir: Path) -> None:
    banner("[M2] budget 強制 — log 是預算 3 倍時，誰被擠掉？")

    kairos = KairosLogMemory(tmpdir / "m2.log.md")
    # 造 20 個事件：年齡 + reward 都不同，總 token 約 3 倍於預算
    for i in range(20):
        age_hours = i * 6                      # 0h, 6h, 12h, ... 逐個變舊
        reward = 1.0 if i % 3 == 0 else 0.0    # 每 3 個一個高 reward
        ts = (NOW - timedelta(hours=age_hours)).isoformat(timespec="seconds")
        kairos.append(_feedback_payload(
            ts, "L4_confirm" if reward else "L2_answer",
            f"query number {i} with some extra words to grow token cost",
            reward, "L4" if reward else "L2",
        ))

    events = list(kairos.iter_events())
    total_tokens = sum(est_tokens(str(e)) for e in events)
    print(f"\nlog 有 {len(events)} 個事件，總 token ≈ {total_tokens}")

    for budget in [2000, 400, 150]:
        result = read_for_context(kairos, budget_tokens=budget, now=NOW,
                                  cutoff_days=365)   # 關掉 compact，純看 budget
        print(f"\n  budget={budget}: used={result['used_tokens']}, "
              f"selected={result['n_selected']}/{result['n_total']}")
        # 印前 3 行看塞了什麼
        for ln in result["text"].splitlines()[:3]:
            print(f"    {ln[:80]}")

    print("\n→ 預算變小時，舊 + 低分事件先被擠掉（依 importance 排序，不是行序）。")
    print("  重點：永不超預算（used ≤ budget 是硬斷言）。")


# ============================================================
# M3 — recency floor（防餓死）
# ============================================================
def run_m3_recency_floor(tmpdir: Path) -> None:
    banner("[M3] recency floor — 最新事件無論分數都保留")

    kairos = KairosLogMemory(tmpdir / "m3.log.md")
    # 造幾個「舊但高 reward」的事件 —— 它們分數會很高
    for i in range(5):
        ts = (NOW - timedelta(days=10, hours=i)).isoformat(timespec="seconds")
        kairos.append(_feedback_payload(ts, "L4_confirm", f"old high-value {i}",
                                        1.0, "L4"))
    # 最後 3 個事件：很新但 reward=0（neutral）—— 純 top-K 會把它們餓死
    for i in range(3):
        ts = (NOW - timedelta(minutes=5 * (i + 1))).isoformat(timespec="seconds")
        kairos.append(_feedback_payload(ts, "L2_answer", f"recent neutral {i}",
                                        0.0, "L2"))

    events = list(kairos.iter_events())
    print(f"\nlog: 5 個舊高 reward + 3 個新 neutral")

    print(f"\n極小預算（budget=80，幾乎塞不下幾條）：")
    result = read_for_context(kairos, budget_tokens=80, now=NOW,
                              cutoff_days=365, floor_n=3)
    print(f"  selected={result['n_selected']}/{result['n_total']}, "
          f"used={result['used_tokens']}/{result['budget_tokens']}")
    for ln in result["text"].splitlines():
        print(f"    {ln}")

    has_recent = "recent neutral" in result["text"]
    print(f"\n→ 最新 3 個 neutral 事件是否被保留？ {'✓ 是' if has_recent else '✗ 否（floor 失效）'}")
    print("  解讀：純 top-K 會餓死「最新但低分」的事件；floor 保證即時上文不丟。")


# ============================================================
# M4 — log compaction
# ============================================================
def run_m4_compaction(tmpdir: Path) -> None:
    banner("[M4] log compaction — 200 個舊事件壓成一行，log 檔不變")

    kairos = KairosLogMemory(tmpdir / "m4.log.md")
    # 200 個 30 天前的事件 + 3 個今天的事件
    for i in range(200):
        ts = (NOW - timedelta(days=30, seconds=i)).isoformat(timespec="seconds")
        kairos.append(_feedback_payload(
            ts, "L4_confirm" if i % 2 == 0 else "L3_undo",
            f"old query {i}", 1.0 if i % 2 == 0 else -1.0,
            "L4" if i % 2 == 0 else "L3",
        ))
    for i in range(3):
        ts = (NOW - timedelta(hours=i)).isoformat(timespec="seconds")
        kairos.append(_feedback_payload(ts, "L4_confirm", f"fresh query {i}",
                                        1.0, "L4"))

    # 記住 compaction 前的 log 檔內容（證明 selector 不會改它）
    before = kairos.path.read_text(encoding="utf-8")
    before_lines = before.count("\n")

    events = list(kairos.iter_events())
    rollup = compact_log(events, NOW - timedelta(days=7))

    after = kairos.path.read_text(encoding="utf-8")
    after_lines = after.count("\n")

    print(f"\nlog 有 {len(events)} 個事件（{before_lines} 行）")
    print(f"\ncompact 後的 summary（1 行）：")
    print(f"  {rollup['summary']}")
    print(f"\n  壓縮比：{rollup['n']} 個事件 → {est_tokens(rollup['summary'])} tokens 的 1 行")

    print(f"\n  log 檔行數：compaction 前={before_lines}，後={after_lines}")
    print(f"  log 檔內容：{'完全不變 ✓' if before == after else '✗ 被改動了！'}")

    assert before == after, "compact_log mutated the log file! (it must be read-only)"
    assert rollup["n"] == 200, f"expected 200 compacted, got {rollup['n']}"
    print("\n→ ✓ compaction 產生 derived summary，永不寫回 append-only log。")
    print("  這就是 spec 的『Never rewrite history』—— selector 是讀取層，不是寫入層。")


# ============================================================
# M5 — 完整閉環
# ============================================================
def run_m5_full_loop(tmpdir: Path) -> None:
    banner("[M5] 完整閉環 — read_for_context 端到端 + token scorecard")

    kairos = KairosLogMemory(tmpdir / "m5.log.md")
    # 真實混合：舊的高 reward、新的混合、一些 neutral
    mix = [
        (NOW - timedelta(days=20), "L4_confirm", 1.0, "L4", "build quarterly dashboard"),
        (NOW - timedelta(days=15), "L3_undo",   -1.0, "L3", "recolor silently"),
        (NOW - timedelta(days=12), "L4_confirm", 1.0, "L4", "build pricing view"),
        (NOW - timedelta(days=5),  "L2_answer",  0.3, "L2", "what is absorption rate"),
        (NOW - timedelta(days=2),  "L4_confirm", 1.0, "L4", "build closings scorecard"),
        (NOW - timedelta(hours=8), "L3_accept",  0.5, "L3", "add YoY column"),
        (NOW - timedelta(hours=2), "L2_answer",  0.3, "L2", "top 5 builders"),
        (NOW - timedelta(minutes=30), "L4_confirm", 1.0, "L4", "build VDL inventory"),
    ]
    for ts, sig, reward, level, query in mix:
        kairos.append(_feedback_payload(
            ts.isoformat(timespec="seconds"), sig, query, reward, level,
        ))

    print(f"\nlog: {len(mix)} 個事件，新舊混合（20 天前 → 30 分鐘前）")
    print(f"\n跑 read_for_context(budget=2000)：\n")

    result = read_for_context(kairos, budget_tokens=2000, now=NOW)
    print(result["text"])

    print(f"\n{'─' * 50}")
    print(f"  TOKEN SCORECARD")
    print(f"{'─' * 50}")
    print(f"  estimator:     {result['estimator']}")
    print(f"  used / budget: {result['used_tokens']} / {result['budget_tokens']}  "
          f"({result['used_tokens']/result['budget_tokens']*100:.0f}%)")
    print(f"  selected:      {result['n_selected']} / {result['n_total']} events")
    rollup_tokens = est_tokens(result['rollup']) if result['rollup'] else 0
    print(f"  rollup:        {rollup_tokens} tokens "
          f"({'有' if result['rollup'] else '無'})")
    print(f"\n→ 輸出結構：[rollup 一行] + [被選中的 raw events]。")
    print("  舊事件被 compact、新事件依 importance×freshness 排序塞進預算。")
    print("  這就是『context-budget-aware memory reader』—— 可以直接講的面試 visual。")


# ============================================================
# M6 — estimator 誠實性
# ============================================================
def run_m6_estimator_honesty(tmpdir: Path) -> None:
    banner("[M6] estimator 誠實性 — tiktoken vs len//4，明講用哪個")

    name = estimator_name()
    print(f"\n目前活躍的 estimator：{name}")

    sample_en = "build a dashboard with quarterly sales breakdown"
    sample_zh = "建一個季度銷售明細的儀表板"
    print(f"\n  英文樣本 ({len(sample_en)} chars): '{sample_en}'")
    print(f"    {name}: {est_tokens(sample_en)} tokens")
    print(f"\n  中文樣本 ({len(sample_zh)} chars): '{sample_zh}'")
    print(f"    {name}: {est_tokens(sample_zh)} tokens")

    if "fallback" in name:
        print(f"\n→ ⚠️  目前用 len//4 fallback。中文會被低估（中文約 1-2 char/token，")
        print(f"    不是 4 char/token）。要準的話：pip install tiktoken")
        print(f"    換裝後重跑這個 demo，estimator 會自動切換成 tiktoken，")
        print(f"    不用改任何程式碼。")
    else:
        print(f"\n→ ✓ 用 tiktoken 精確計 token。")


# ============================================================
# main
# ============================================================

def main() -> None:
    print("=" * 70)
    print(" memory_selector — context-budget-aware memory reader")
    print("=" * 70)
    print("""
    記憶讀取層（跟 policy_feedback 的『學 rule』正交）：
      importance()      ← freshness(指數衰減) × magnitude(|reward|)
      compact_log()     ← 舊事件 → 一行 summary（derived，不寫回 log）
      read_for_context()← 三層組合：compact + budget + freshness

    跟 policy_feedback_demo.py 的差別：
      policy_feedback  = feedback → KEEP/AVOID rules（路線 A，學習層）
      memory_selector  = log → context slice（記憶讀取層）
      兩者讀同一個 KairosLogMemory，職責不重疊。
    """)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        run_m1_importance_freshness(tmpdir)
        run_m2_budget_enforcement(tmpdir)
        run_m3_recency_floor(tmpdir)
        run_m4_compaction(tmpdir)
        run_m5_full_loop(tmpdir)
        run_m6_estimator_honesty(tmpdir)

    banner("全部 6 個場景跑完")
    print("""
    重點回顧：
      [M1] importance = recency × (floor+|reward|)；兩軸相乘，任一為零就淡出
      [M2] budget 是硬約束；超預算時依分數（不是行序）擠掉舊低分事件
      [M3] recency floor 防止「最新但低分」的事件被 top-K 餓死
      [M4] compaction 產生 derived summary，log 檔永不變（append-only 不破）
      [M5] 完整閉環 + token scorecard —— 面試時直接講的 visual
      [M6] estimator 誠實：tiktoken 有就用，沒有就明講用 fallback

    跟你現有 memory 系統的關係：
      - 完全 additive：沒改 kairos_log.py / policy_feedback.py / 任何 web 檔
      - 讀的是同一個 iter_events()，跟 policy_feedback 同一個 seam
      - 沒有 weight update、沒有 GRPO、沒有 importance sampling

    三個功能的對應：
      add compact function   → compact_log() + read_for_context 的 rollup 步驟
      add budget context size → read_for_context(budget_tokens) + est_tokens
      add freshness control  → importance() = recency_score() × magnitude

    面試時可以這樣講：
      「我在 KairosLogMemory 上加了一個 context-budget-aware reader。
       每個事件依 importance = freshness(指數半衰期) × |reward| 打分；
       舊事件 compact 成一行 summary；剩下的事件依分數 + recency floor
       greedy 塞進 token budget，truncate-not-drop。

       全程只讀 log、不寫 log（append-only 不破）、不動 weight。
       沒有 semantic relevance（那需要 embedding，我刻意不做，避免
       把 lexical overlap 包裝成 relevance —— 那是 fabrication）。」
    """)


if __name__ == "__main__":
    main()
