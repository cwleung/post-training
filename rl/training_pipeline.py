#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training_pipeline.py — production-grade 訓練 pipeline 的工程外殼
=================================================================

這個模組把 toy GRPO（grpo_llm_char.py 的 grpo_update_llm）包進一個有結構的
production 訓練 pipeline。Policy 還是 1-gram toy，但圍繞它的工程要像「真的在跑訓練」：

  - TrainingConfig        : 所有超參數集中（不再散落）
  - CheckpointManager     : policy save/load + version 目錄
  - RewardLogger          : 每輪 metric 寫 CSV（TensorBoard 風格但零依賴）
  - KairosRunLogger       : 訓練事件寫進 append-only log（重啟可讀回）
  - PolicyVersionTracker  : 每次 checkpoint 後 bump version
  - RewardHackingGuard    : 偵測 win_rate spike + query 多樣性崩塌 → freeze
  - TrainingPipeline      : 組合上面所有東西，對外只露 step() / metrics()

== 跟真實 production 的關係 ==
這些元件不是裝飾 —— 每一個都對應一個真實的 production 需求：
  - checkpoint   : 訓練崩了要能從上個版本恢復
  - reward log   : 要能畫 learning curve、debug 訓練是否收斂
  - kairos log   : offline 訓練的資料來源（你 policy_feedback.py 的 _off_policy_note 就靠它）
  - version      : 不同 policy version 的 log 不能混著訓練（off-policy 偏差）
  - hacking guard: DeepSeek R1 / 任何 RL 都會踩 reward hacking，要能 freeze

== 刻意「不做」的事（誠實標出）==
  - KL penalty / reference policy : 方向 A，不在這版。config 留 beta_kl placeholder。
  - 真 transformer policy          : 方向 C，不在這版。
  - 多 user / session isolation    : 單一 process-global policy（demo 用足夠）。
"""

import csv
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np

from grpo_llm_char import (
    CharPolicy, V, seq_logprob, grpo_update_llm, I2V, V2I, decode,
)
from kairos_log import KairosLogMemory
from policy_feedback import detect_anomalies, FeedbackEvent


# ============================================================
# 1. TrainingConfig — 所有超參數集中
# ============================================================
@dataclass
class TrainingConfig:
    """所有超參數 + 路徑集中。production 用 yaml/hydra，這裡用 dataclass。"""
    # GRPO 超參數
    group_size: int = 4
    lr: float = 1.0
    eps_clip: float = 0.2
    inner_epochs: int = 4
    max_len: int = 8

    # ★ KL penalty placeholder ★
    # 真實 GRPO 會加 β·KL(π_θ‖π_ref)。這版「不實作」（方向 A 才做）。
    # 留這個欄位是為了讓 config 跟真實 production 對齊，並在 checkpoint 裡留紀錄。
    beta_kl: float = 0.0
    kl_implemented: bool = False   # 旗標：UI / log 會標明「KL 沒做」

    # checkpoint / log 路徑
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    kairos_path: str = "logs/kairos.md"
    reward_csv: str = "logs/reward_log.csv"

    # checkpoint 間隔（每 N 輪存一次）
    checkpoint_every: int = 5

    # reward-hacking 偵測
    hacking_spike_threshold: float = 0.30   # 兩 session 間 win_rate 跳升
    hacking_diversity_floor: float = 0.40   # query 多樣性低於此 → collapse
    hacking_session_size: int = 5           # 每 N 輪切一個 session

    # seed
    seed: int = 0

    def ensure_dirs(self):
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. PolicyVersionTracker — 每次 checkpoint 後 bump
# ============================================================
class PolicyVersionTracker:
    """
    Production 的關鍵：offline log 來自「不同 policy version」，不能混著訓練。
    （你 policy_feedback.py:313 的 _off_policy_note 就偵測這個。）
    """
    def __init__(self, start: int = 0):
        self._version = start

    def bump(self):
        self._version += 1
        return self._version

    @property
    def current(self) -> int:
        return self._version


# ============================================================
# 3. CheckpointManager — policy save/load
# ============================================================
class CheckpointManager:
    """
    policy save/load。每個 checkpoint = .npz（theta）+ sidecar JSON（metadata）。
    版本目錄：checkpoint_dir/policy_v{N}.npz + policy_v{N}.json
    """
    def __init__(self, config: TrainingConfig, version_tracker: PolicyVersionTracker):
        self.config = config
        self.version_tracker = version_tracker
        config.ensure_dirs()

    def save(self, policy: CharPolicy, round_num: int,
             metrics: Dict[str, Any]) -> Path:
        """存 policy.theta + metadata。bump version。回傳 .npz 路徑。"""
        version = self.version_tracker.bump()
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        npz_path = Path(self.config.checkpoint_dir) / f"policy_v{version}.npz"
        json_path = Path(self.config.checkpoint_dir) / f"policy_v{version}.json"

        # theta 存 npz
        np.savez(npz_path, theta=policy.theta)
        # metadata 存 sidecar json
        meta = {
            "policy_version": version,
            "round_num": round_num,
            "ts": ts,
            "config": {k: v for k, v in asdict(self.config).items()
                       if not isinstance(v, (Path,))},
            "metrics": metrics,
        }
        json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return npz_path

    def load_latest(self) -> Optional[Path]:
        """回傳最新 checkpoint 的 .npz 路徑，沒有就 None。"""
        ckpt_dir = Path(self.config.checkpoint_dir)
        if not ckpt_dir.exists():
            return None
        npzs = sorted(ckpt_dir.glob("policy_v*.npz"),
                      key=lambda p: int(p.stem.split("_v")[1]))
        return npzs[-1] if npzs else None

    def load_into(self, policy: CharPolicy, npz_path: Path) -> dict:
        """把 .npz 的 theta 載入 policy，回傳 sidecar metadata。"""
        data = np.load(npz_path)
        policy.theta = data["theta"]
        json_path = npz_path.with_suffix(".json")
        meta = json.loads(json_path.read_text())
        # 同步 version tracker
        self.version_tracker._version = meta.get("policy_version", 0)
        return meta

    def list_checkpoints(self) -> List[dict]:
        """列出所有 checkpoint 的 metadata（給 UI 顯示用）。"""
        ckpt_dir = Path(self.config.checkpoint_dir)
        out = []
        for jp in sorted(ckpt_dir.glob("policy_v*.json"),
                         key=lambda p: int(p.stem.split("_v")[1])):
            out.append(json.loads(jp.read_text()))
        return out


# ============================================================
# 4. RewardLogger — 每輪 metric 寫 CSV
# ============================================================
class RewardLogger:
    """
    每輪 append 一行 CSV。TensorBoard 風格但零依賴。
    欄位：round, policy_version, prompt, accepted_idx, mean_reward,
          baseline, std, accepted_logp_before, accepted_logp_after
    """
    FIELDS = ["round", "policy_version", "prompt", "accepted_idx",
              "mean_reward", "baseline", "std",
              "accepted_logp_before", "accepted_logp_after"]

    def __init__(self, config: TrainingConfig, version_tracker: PolicyVersionTracker):
        self.path = Path(config.reward_csv)
        self.version_tracker = version_tracker
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def log(self, round_num: int, prompt: str, accepted_idx: int,
            mean_reward: float, baseline: float, std: float,
            accepted_logp_before: Optional[float],
            accepted_logp_after: Optional[float]):
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow({
                "round": round_num,
                "policy_version": self.version_tracker.current,
                "prompt": prompt,
                "accepted_idx": accepted_idx,
                "mean_reward": f"{mean_reward:.4f}",
                "baseline": f"{baseline:.4f}",
                "std": f"{std:.4f}",
                "accepted_logp_before": (f"{accepted_logp_before:.4f}"
                                         if accepted_logp_before is not None else ""),
                "accepted_logp_after": (f"{accepted_logp_after:.4f}"
                                        if accepted_logp_after is not None else ""),
            })

    def read_all(self) -> List[dict]:
        """讀回所有列（給 reward curve 畫圖用）。"""
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def rolling_mean_reward(self, window: int = 5) -> Optional[float]:
        """最近 N 輪的平均 reward（給 UI 顯示用）。"""
        rows = self.read_all()
        if not rows:
            return None
        recent = rows[-window:]
        return float(np.mean([float(r["mean_reward"]) for r in recent]))


# ============================================================
# 5. KairosRunLogger — 訓練事件寫進 append-only log
# ============================================================
class KairosRunLogger:
    """
    包裝 KairosLogMemory，把每輪 GRPO 更新寫成 grpo_step 事件。
    這讓訓練歷史持久化 —— 離線訓練（如 offline-train API）就是讀這個 log 還原訓練資料。
    """
    def __init__(self, config: TrainingConfig, version_tracker: PolicyVersionTracker):
        self.kairos = KairosLogMemory(config.kairos_path)
        self.version_tracker = version_tracker

    def log_step(self, round_num: int, prompt: str, accepted_idx: int,
                 rewards: List[float], baseline: float, std: float,
                 advantages: List[float], group_seqs: List[List[int]]):
        quality = ("user_accepted" if accepted_idx >= 0 else "user_rejected")
        data_payload = {
            "prompt": prompt,
            "group_seqs": group_seqs,
            "accepted_idx": accepted_idx
        }
        data_str = json.dumps(data_payload, ensure_ascii=False)
        summary = (f"round={round_num} prompt={prompt!r} "
                   f"accepted={'ABCDE'[accepted_idx] if accepted_idx>=0 else 'none'} "
                   f"mean_reward={np.mean(rewards):+.2f} baseline={baseline:+.2f} "
                   f"data={data_str}")
        self.kairos.log_event(
            event_type="grpo_step",
            plan_id=f"round_{round_num}",
            result="success",
            quality_tag=quality,
            user_id="trainer",
            session_id=f"policy_v{self.version_tracker.current}",
            summary=summary,
        )


# ============================================================
# 6. RewardHackingGuard — win_rate spike + diversity collapse → freeze
# ============================================================
@dataclass
class HackingReport:
    win_rate_spike: bool = False
    diversity_collapse: bool = False
    detail: str = ""
    @property
    def should_freeze(self) -> bool:
        return self.win_rate_spike or self.diversity_collapse


class RewardHackingGuard:
    """
    累積每輪的 reward 成「session」單位（每 N 輪一個）。
    偵測 win_rate spike + query 多樣性崩塌 → freeze（對應 production 的 /dream freeze）。

    直接使用 policy_feedback.detect_anomalies 進行偵測。
    """
    def __init__(self, config: TrainingConfig):
        self.spike_threshold = config.hacking_spike_threshold
        self.diversity_floor = config.hacking_diversity_floor
        self.session_size = config.hacking_session_size
        self.current_session: List[tuple] = []   # [(prompt, reward), ...]
        self.past_sessions: List[List[tuple]] = []
        self.last_report: HackingReport = HackingReport()

    def record(self, prompt: str, reward: float):
        """記一輪。reward > 0 視為 win。"""
        self.current_session.append((prompt, reward))
        if len(self.current_session) >= self.session_size:
            self.past_sessions.append(self.current_session)
            self.current_session = []
            self._recompute()

    def _recompute(self):
        if len(self.past_sessions) < 2:
            self.last_report = HackingReport(detail="fewer than 2 sessions")
            return
        
        # Convert past sessions to FeedbackEvent format for detect_anomalies
        sessions_events = []
        for sess_idx, session in enumerate(self.past_sessions):
            sess_evs = []
            for prompt, r in session:
                # Map positive reward (> 0) to "L4_confirm", else to "L4_abandon"
                signal = "L4_confirm" if r > 0 else "L4_abandon"
                sess_evs.append(FeedbackEvent(
                    signal=signal,
                    query=prompt,
                    session_id=f"sess_{sess_idx}",
                ))
            sessions_events.append(sess_evs)
            
        report = detect_anomalies(
            sessions_events,
            spike_threshold=self.spike_threshold,
            diversity_floor=self.diversity_floor,
        )
        self.last_report = HackingReport(
            win_rate_spike=report.win_rate_spike,
            diversity_collapse=report.diversity_collapse,
            detail=report.detail,
        )

    @property
    def frozen(self) -> bool:
        return self.last_report.should_freeze


# ============================================================
# 7. TrainingPipeline — 組合所有東西，對外只露 step() / metrics()
# ============================================================
class TrainingPipeline:
    """
    單一入口：step(prompt, group_seqs, accepted_idx)。
    內部做：GRPO 更新 + reward log + kairos log + checkpoint + anomaly check + version bump。
    """
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.config.ensure_dirs()
        self.policy = CharPolicy()
        self.rng = np.random.default_rng(self.config.seed)
        self.version_tracker = PolicyVersionTracker()
        self.ckpt = CheckpointManager(self.config, self.version_tracker)
        self.reward_logger = RewardLogger(self.config, self.version_tracker)
        self.kairos_logger = KairosRunLogger(self.config, self.version_tracker)
        self.guard = RewardHackingGuard(self.config)
        self.round_num = 0

    # ---- 載入最新 checkpoint（重啟時用）----
    def resume_from_latest(self) -> bool:
        latest = self.ckpt.load_latest()
        if latest is None:
            return False
        meta = self.ckpt.load_into(self.policy, latest)
        self.round_num = meta.get("round_num", 0)
        return True

    # ---- 核心：單一 GRPO 步驟 + 所有 side-effect ----
    def step(self, prompt: str, prompt_ids: list, group_seqs: list,
             accepted_idx: int) -> dict:
        """
        prompt_ids  : 編碼後的 prompt（含 BOS）
        group_seqs  : G 段候選序列（prompt + generated）
        accepted_idx: user 接受第幾個，-1 = 全拒絕

        回傳 metrics dict（給 UI / offline script 用）。
        """
        if self.guard.frozen:
            return {"frozen": True, "detail": self.guard.last_report.detail}

        prompt_len = len(prompt_ids)
        G = len(group_seqs)

        # 1. 形成 reward
        rewards = np.full(G, -1.0)
        if accepted_idx >= 0:
            rewards[accepted_idx] = 1.0

        # 2. 記住更新前 logp（給 reward log 用）
        logp_before = [seq_logprob(self.policy, s, prompt_len) for s in group_seqs]

        # 3. GRPO 更新（核心，不動 grpo_update_llm）
        baseline, std, advantages, _ = grpo_update_llm(
            self.policy, prompt_ids, "", group_seqs, rewards,
            lr=self.config.lr, eps_clip=self.config.eps_clip,
            inner_epochs=self.config.inner_epochs, rng=self.rng,
        )

        # 4. 記住更新後 logp
        logp_after = [seq_logprob(self.policy, s, prompt_len) for s in group_seqs]
        self.round_num += 1

        acc_lp_before = logp_before[accepted_idx] if accepted_idx >= 0 else None
        acc_lp_after = logp_after[accepted_idx] if accepted_idx >= 0 else None
        mean_reward = float(np.mean(rewards))

        # 5. reward CSV log
        self.reward_logger.log(
            self.round_num, prompt, accepted_idx,
            mean_reward, baseline, std, acc_lp_before, acc_lp_after)

        # 6. kairos log（持久化訓練事件，offline script 讀這個）
        self.kairos_logger.log_step(
            self.round_num, prompt, accepted_idx,
            rewards.tolist(), baseline, std, advantages.tolist(),
            group_seqs=group_seqs)

        # 7. reward-hacking guard
        # 注意：傳給 guard 的是「這輪有沒有正向 reward」（user 接受 = win），
        # 不是 mean_reward（accept 時 mean=-0.5，仍是負的）。
        win_signal = 1.0 if accepted_idx >= 0 else -1.0
        self.guard.record(prompt, win_signal)

        # 8. checkpoint（每 N 輪）
        ckpt_path = None
        if self.round_num % self.config.checkpoint_every == 0:
            metrics = self.metrics()
            ckpt_path = self.ckpt.save(self.policy, self.round_num, metrics)

        return {
            "frozen": self.guard.frozen,
            "round": self.round_num,
            "policy_version": self.version_tracker.current,
            "mean_reward": mean_reward,
            "baseline": baseline,
            "std": std,
            "advantages": advantages.tolist(),
            "logp_before": logp_before,
            "logp_after": logp_after,
            "accepted_idx": accepted_idx,
            "checkpoint_path": str(ckpt_path) if ckpt_path else None,
            "hacking_report": {
                "should_freeze": self.guard.frozen,
                "detail": self.guard.last_report.detail,
            },
        }

    # ---- 給 UI 的 metrics 彙總 ----
    def metrics(self) -> dict:
        return {
            "round": self.round_num,
            "policy_version": self.version_tracker.current,
            "rolling_mean_reward": self.reward_logger.rolling_mean_reward(),
            "frozen": self.guard.frozen,
            "hacking_detail": self.guard.last_report.detail,
            "checkpoints": len(self.ckpt.list_checkpoints()),
            "beta_kl": self.config.beta_kl,
            "kl_implemented": self.config.kl_implemented,
        }

    # ---- 重置（UI 的 reset 按鈕用）----
    def reset(self):
        self.policy = CharPolicy()
        self.rng = np.random.default_rng(self.config.seed)
        self.round_num = 0
        self.version_tracker = PolicyVersionTracker()
        self.guard = RewardHackingGuard(self.config)
        # 注意：不刪 checkpoint / log —— 那是歷史紀錄，production 不會清。


# ============================================================
# self-test
# ============================================================
if __name__ == "__main__":
    import tempfile, shutil
    from grpo_llm_char import encode, generate

    tmp = tempfile.mkdtemp()
    cfg = TrainingConfig(
        checkpoint_dir=os.path.join(tmp, "ckpts"),
        log_dir=os.path.join(tmp, "logs"),
        kairos_path=os.path.join(tmp, "logs/kairos.md"),
        reward_csv=os.path.join(tmp, "logs/reward.csv"),
        checkpoint_every=2,   # 測試用：每 2 輪存
        hacking_session_size=3,  # 測試用：每 3 輪一個 session
    )
    p = TrainingPipeline(cfg)

    print("=" * 60)
    print(" TrainingPipeline self-test")
    print("=" * 60)

    prompt = "1+1="
    prompt_ids = [V2I["<"]] + encode(prompt)

    # 跑 3 輪，全接受第一個
    for rnd in range(3):
        group = [generate(p.policy, prompt_ids, 8, p.rng) for _ in range(4)]
        res = p.step(prompt, prompt_ids, group, accepted_idx=0)
        print(f"\n[round {res['round']}] mean_reward={res['mean_reward']:+.2f} "
              f"version=v{res['policy_version']} "
              f"ckpt={'yes' if res['checkpoint_path'] else 'no'}")

    print("\n--- metrics() ---")
    for k, v in p.metrics().items():
        print(f"  {k}: {v}")

    print("\n--- reward_log.csv 內容 ---")
    for row in p.reward_logger.read_all():
        print(f"  {row}")

    print("\n--- kairos.md 內容（前 5 行）---")
    print("\n".join(p.kairos_logger.kairos.read().splitlines()[:5]))

    print("\n--- checkpoint 列表 ---")
    for c in p.ckpt.list_checkpoints():
        print(f"  v{c['policy_version']} round={c['round_num']}")

    # 測 resume
    print("\n--- 測 resume：新建 pipeline，load 最新 checkpoint ---")
    p2 = TrainingPipeline(cfg)
    ok = p2.resume_from_latest()
    print(f"  resume 成功: {ok}")
    print(f"  resume 後 round_num={p2.round_num} version=v{p2.version_tracker.current}")

    # 測 reward-hacking freeze
    print("\n--- 測 reward-hacking freeze（連續高 reward + 低多樣性）---")
    cfg2 = TrainingConfig(
        checkpoint_dir=os.path.join(tmp, "ckpts2"),
        log_dir=os.path.join(tmp, "logs2"),
        kairos_path=os.path.join(tmp, "logs2/kairos.md"),
        reward_csv=os.path.join(tmp, "logs2/reward.csv"),
        hacking_session_size=3,
        hacking_spike_threshold=0.1,   # 很低的 threshold，容易觸發
    )
    p3 = TrainingPipeline(cfg2)
    # 製造：前 3 輪全 reject（win_rate=0），後 3 輪全 accept（win_rate=1）→ spike
    for rnd in range(6):
        group = [generate(p3.policy, prompt_ids, 8, p3.rng) for _ in range(4)]
        acc = -1 if rnd < 3 else 0   # 前3拒絕，後3接受
        res = p3.step(prompt, prompt_ids, group, accepted_idx=acc)
        frozen = res.get("frozen", False)
        hack = res.get("hacking_report", {})
        print(f"  round {res['round']} accepted={acc} frozen={frozen} "
              f"hack={hack.get('detail','')[:40]}")
    print(f"\n  最終 frozen={p3.guard.frozen}")

    shutil.rmtree(tmp)
    print("\n✓ pipeline self-test OK")
