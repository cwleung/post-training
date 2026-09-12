#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KairosLogMemory + log_event() helper  +  log → GRPO dataset
===========================================================

這個檔案做兩件事：

1. 一個最小但忠於你 memory 設計的 KairosLogMemory：
   - append-only 檔案（每行一個事件）
   - read() 給 LLM 看（人類可讀）
   - 新的 log_event(**fields) helper：把「reward 訊號」結構化存下來

2. 一組把 log 轉成 GRPO dataset 的函式：
   - group_log_by_plan(): 依 plan_id 把事件聚成 group
   - reward_from_quality(): quality_tag → 數值 reward
   - to_grpo_dataset(): 還原成 GRPO 要的 (group, rewards, advantages)

== 為什麼 log_event 是 GRPO 的命根子 ==
GRPO 的 reward + group key 全住在這裡：
   - plan_id  → 決定哪些 candidate 算同一組（group）
   - quality_tag → 決定每個 candidate 的 reward
   - event_type → 區分 plan_generated / user_accept / user_reject / undo

user 按 undo 或說「這不對」的當下，這個訊號是「會消失」的 ——
如果那一刻沒記下來，事後再也補不回來。
所以 instrument 的第一優先永遠是 Kairos 的 log_event，不是 Conversation 的 meta。
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

# ============================================================
# 1. KairosLogMemory（最小版，介面對齊你的 memory tier）
# ============================================================
class KairosLogMemory:
    """append-only 的事件日誌。read() 給 LLM，log_event() 給 RL pipeline。"""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    # ---------- 原本就有的低階寫入 ----------
    def append(self, line: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")

    # ---------- 原本就有的低階讀取（給 LLM / middleware 注入用）----------
    def read(self, max_lines: int = 200) -> str:
        """人類可讀的視圖：給 middleware 注入 system prompt 用。"""
        if not self.path.exists():
            return "(no kairos events yet)"
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[-max_lines:]) if lines else "(no kairos events yet)"

    # ========================================================
    # 2. 新的 helper：log_event()
    #    這是 GRPO reward 訊號的唯一正規棲息地。
    # ========================================================
    def log_event(
        self,
        event_type: str,
        *,
        plan_id: Optional[str] = None,
        result: Optional[str] = None,
        quality_tag: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        summary: str = "",
    ) -> dict:
        """
        結構化地記一個事件。雙視圖寫入：
          - 第 1 行：人類可讀的 bullet（給 read() / LLM）
          - 第 2 行：JSON payload（給 RL pipeline parse）

        參數（對應你之前設計的 schema）：
          event_type : "plan_generated" | "dashboard_built"
                       | "modification_applied" | "user_accept"
                       | "user_reject" | "undo" | "error"
          plan_id    : 把「plan → build → feedback」串成同一個 group 的 key
          result     : "success" | "partial" | "failed" | None
          quality_tag: "user_accepted" | "user_revised" | "user_rejected" | None
                       ↑ 這是 GRPO reward 的主要來源
          summary    : 人類可讀的一句話
        """
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        payload = {
            "ts": ts,
            "event_type": event_type,
            "plan_id": plan_id,
            "result": result,
            "quality_tag": quality_tag,
            "user_id": user_id,
            "session_id": session_id,
            "summary": summary,
        }
        # 第 1 行：給人讀；第 2 行：給機器讀（前面縮排 2 格，視覺上從屬於第 1 行）
        line = f"- {ts} [{event_type}] {summary}"
        if plan_id:
            line += f"  (plan={plan_id})"
        line += f"\n  {json.dumps(payload, ensure_ascii=False)}"
        self.append(line)
        return payload  # 回傳方便測試 / orchestrator 紀錄

    # ---------- 給 RL pipeline 用的結構化讀取 ----------
    def iter_events(self):
        """把整個 log parse 成一條 dict 串流。parse 失敗的行直接跳過。"""
        if not self.path.exists():
            return
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw.startswith("{"):
                continue  # 跳過人類可讀那行
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


# ============================================================
# 3. log → GRPO dataset 的轉換函式
# ============================================================

# quality_tag → reward 的對照表（可調，這只是合理的預設）
QUALITY_TO_REWARD = {
    "user_accepted": 1.0,
    "user_revised": 0.3,   # 有被採用但需要改 → 弱正向
    "user_rejected": -1.0,
    None: None,            # 沒有 user 訊號 → 之後用 fallback
}


def reward_from_event(evt: dict, fallback_no_feedback: float = 0.1) -> Optional[float]:
    """
    把一個 event 的 (quality_tag, event_type, result) 換成數值 reward。

    優先序：
      1. 有 quality_tag → 直接對表（這是最強的訊號）
      2. event_type == "undo" → 視為拒絕 (-1.0)
      3. event_type == "user_reject" → -1.0
      4. event_type == "user_accept" → +1.0
      5. 只有 build 成功但沒 user feedback → fallback_no_feedback（弱正向）
      6. 完全沒資訊 → None
    """
    qt = evt.get("quality_tag")
    if qt in QUALITY_TO_REWARD and QUALITY_TO_REWARD[qt] is not None:
        return QUALITY_TO_REWARD[qt]

    et = evt.get("event_type")
    if et in ("undo", "user_reject"):
        return -1.0
    if et == "user_accept":
        return 1.0
    if et in ("dashboard_built", "modification_applied") and evt.get("result") == "success":
        return fallback_no_feedback
    return None


def group_log_by_plan(events):
    """
    把事件依 plan_id 聚起來。
    回傳 {plan_id: [events...]}，沒 plan_id 的丟進 None 鍵（之後可單獨處理）。
    """
    groups: dict = {}
    for e in events:
        pid = e.get("plan_id")
        groups.setdefault(pid, []).append(e)
    return groups


def _final_reward_for_plan(evts, fallback_no_feedback: float = 0.1):
    """
    一個 plan 的 reward = 它收到的「最終 user 評價」。

    優先序（從最強訊號到最弱）：
      1. 該 plan 有 user_accept / user_reject / undo 事件 → 取那條的 reward
      2. 只有 build/modify 成功但沒 user feedback → fallback_no_feedback
      3. 完全沒資訊 → None

    重點：一個 plan 只會對應到「一個 reward」，
    不會因為它有多條 event 就產生多個 candidate。
    """
    # 倒著看，取最後一條帶有 user 評價（accept/reject/undo）的事件
    for e in reversed(evts):
        et = e.get("event_type")
        if et in ("user_accept", "user_reject", "undo"):
            return reward_from_event(e, fallback_no_feedback)
    # 沒有 user 評價 → 看 build/modify 有沒有成功
    for e in reversed(evts):
        et = e.get("event_type")
        if et in ("dashboard_built", "modification_applied") and e.get("result") == "success":
            return fallback_no_feedback
    return None


def to_grpo_dataset(kairos: KairosLogMemory, fallback_no_feedback: float = 0.1):
    """
    把整個 Kairos log 轉成 GRPO 要的 dataset。

    語意（重要）：
      - 一個 plan_id = 一個 candidate（不是一條 event = 一個 candidate）。
      - 每個 candidate 的 reward = 它收到的「最終 user 評價」。
      - GRPO 的 group = 同一個 user query 下的多個 plan（候選方案）。
        如果你的 log 裡 query_id 跟 plan_id 是 1:1（每個 query 只有一個 plan），
        那這個 group 就只有 1 個 candidate，advantage 會是 0（學不到東西）。
        要有 GRPO 效果，同一個 query 必須對應「多個 plan」（A/B、重做、n>1 sampling）。

    回傳 list of group，每個 group 是 dict:
      {
        "plan_id": ...,          # 這個 group 的 key（= user query）
        "candidates": [ {plan_id, events, reward}, ... ],
        "rewards":   [float, ...],
      }

    注意：這裡先用 plan_id 當 group key 示範。
    真實場景你應該用 query_id（同一個 user query）當 group key，
    把該 query 下所有 plan 收進同一個 group。
    demo_log_to_grpo.py 示範了「以 query 為 group key」的正確版本。
    """
    events = list(kairos.iter_events())
    by_plan = group_log_by_plan(events)

    dataset = []
    for pid, evts in by_plan.items():
        if pid is None:
            continue
        r = _final_reward_for_plan(evts, fallback_no_feedback)
        if r is None:
            continue
        dataset.append({
            "plan_id": pid,
            "candidates": [{"plan_id": pid, "events": evts, "reward": r}],
            "rewards": [r],
        })
    return dataset


def group_relative_advantage(rewards, eps: float = 1e-4):
    """
    GRPO 的核心：group 內 baseline + 標準化 advantage。
    A_i = (r_i - mean(r)) / (std(r) + eps)

    這跟 grpo_toy.py 裡 grpo_update() 算 advantage 那段完全一樣，
    只是把來源從「現採樣」換成「從 log 還原」。
    """
    import numpy as np
    r = np.asarray(rewards, dtype=float)
    mean = r.mean()
    std = r.std()
    adv = (r - mean) / (std + eps)
    return adv, mean, std


# ============================================================
# 4. 一個超小的 self-test（直接 python kairos_log.py 就會跑）
# ============================================================
if __name__ == "__main__":
    import tempfile, os

    tmp = tempfile.mkdtemp()
    log_path = os.path.join(tmp, "kairos.md")
    k = KairosLogMemory(log_path)

    print("=" * 60)
    print(" 模擬 orchestrator 的三個 log_event 呼叫點")
    print("=" * 60)

    # --- 情境 A：一個 plan 被接受 ---
    k.log_event("plan_generated", plan_id="p1", summary="Build Austin dashboard v1",
                user_id="alice", session_id="s1")
    k.log_event("dashboard_built", plan_id="p1", result="success",
                summary="Austin dashboard built OK", user_id="alice", session_id="s1")
    k.log_event("user_accept", plan_id="p1", quality_tag="user_accepted",
                summary="alice said looks good", user_id="alice", session_id="s1")

    # --- 情境 B：另一個 plan 被拒絕（同一個 query，重做）---
    k.log_event("plan_generated", plan_id="p2", summary="Build Austin dashboard v2 (alt)",
                user_id="alice", session_id="s1")
    k.log_event("dashboard_built", plan_id="p2", result="success",
                summary="Austin dashboard v2 built OK", user_id="alice", session_id="s1")
    k.log_event("undo", plan_id="p2", quality_tag="user_rejected",
                summary="alice hit undo", user_id="alice", session_id="s1")

    # --- 情境 C：只 build 成功，還沒 user feedback ---
    k.log_event("plan_generated", plan_id="p3", summary="Add YoY section",
                user_id="alice", session_id="s2")
    k.log_event("modification_applied", plan_id="p3", result="success",
                summary="YoY section added", user_id="alice", session_id="s2")

    print("\n--- read() 的視圖（給 LLM 看的）---")
    print(k.read())

    print("\n--- 轉成 GRPO dataset（一個 plan = 一個 candidate）---")
    ds = to_grpo_dataset(k)
    print("  語意提醒：group 應該以 query 為 key（同一 query 下多個 plan）。")
    print("           這裡先以 plan_id 示範，所以每個 group 只有 1 個 candidate，")
    print("           advantage 會是 0（單一 candidate 無相對差異，學不到東西）。")
    print("           正確的「以 query 為 group key」版本請見 demo_log_to_grpo.py。")
    for g in ds:
        print(f"\n  plan_id={g['plan_id']}")
        print(f"    reward = {g['rewards']}   (這個 plan 的最終 user 評價)")

    print("\n" + "=" * 60)
    print(" ✓ log_event 寫入 OK，log → GRPO dataset 還原 OK")
    print(" ✓ 注意：要讓 GRPO 真的學到東西，必須有「同一 query 下多個 candidate」。")
    print("   完整的正確流程請跑 demo_log_to_grpo.py")
    print("=" * 60)
