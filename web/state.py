#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
web/state.py — in-memory session stores
=======================================

互動 demo 的 policy state 是 mutable（theta + rng），不能跨 request 共享，
否則兩個 tab 會互踩同一份 theta。這裡用 dict[sid -> state] 做隔離。

三個 store：
  TOY_SESSIONS   : sid -> PolicyState          （Part 1 toy）
  LLM_SESSIONS   : sid -> TrainingPipeline     （Part 1+1.5 LLM）
  OFFLINE_STATE  : 全域唯一，存 KairosLogMemory + theta + rng（offline demo 是單跑的）

重啟即清 — 教學工具不需要持久化 DB。
"""

import threading
from collections import defaultdict
from typing import Any, Dict

# session_id -> 物件。存取都走下面的 *_lock 保護。
TOY_SESSIONS: Dict[str, Any] = {}
LLM_SESSIONS: Dict[str, Any] = {}

# Part 4 / offline 用單一全域狀態（不是互動式，每個 endpoint 重建）。
OFFLINE_STATE: Dict[str, Any] = {}

_LOCK = threading.Lock()


def new_sid() -> str:
    """簡單的 session id（不用 uuid，避免多餘依賴）。"""
    import time
    return f"s{int(time.time() * 1000)}"


def with_lock():
    """回傳模組級 lock（給需要在 endpoint 外做複合操作時用）。"""
    return _LOCK
