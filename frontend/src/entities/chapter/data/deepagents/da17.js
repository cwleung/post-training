// deepagents/da17.js
export default {
  "id": "da17",
  "num": "17",
  "title": "評估即 RL 環境：Verifier 設計與可驗證獎勵 (Eval as RL Environment)",
  "icon": "🎯",
  "file": "17-eval-as-rl-environment",
  "hasVisualizer": "eval_workbench",
  "readTime": "25 min",
  "summary": "Agent Gymnasium 心智模型：確定性驗證器、正則獎勵回報、難度梯度與反思信號。",
  "toc": [
    {
        "level": 2,
        "text": "核心心智模型：Agent 的強化學習 Gymnasium (Gymnasium Environment & Reward Signal)",
        "anchor": "核心心智模型agent-的強化學習-gymnasium-gymnasium-environment-reward-signal"
    },
    {
        "level": 2,
        "text": "1. 評估⇄能力⇄RL 三角循環模型",
        "anchor": "1-評估能力rl-三角循環模型"
    },
    {
        "level": 2,
        "text": "2. 兩類評估範式的本質差異",
        "anchor": "2-兩類評估範式的本質差異"
    },
    {
        "level": 2,
        "text": "3. RL 環境分解模型：E = {T, H, V, S, C}",
        "anchor": "3-rl-環境分解模型e-t-h-v-s-c"
    },
    {
        "level": 2,
        "text": "4. 難度校準（Difficulty Calibration）：Goldilocks 法則",
        "anchor": "4-難度校準difficulty-calibrationgoldilocks-法則"
    },
    {
        "level": 2,
        "text": "5. Verifier 設計三原則（來自 Harbor Framework 實踐）",
        "anchor": "5-verifier-設計三原則來自-harbor-framework-實踐"
    },
    {
        "level": 2,
        "text": "6. 完整驗證器示範（Pure Python Verifier）",
        "anchor": "6-完整驗證器示範pure-python-verifier"
    },
    {
        "level": 2,
        "text": "7. GRPO 的 2026 變體全景與業界共識",
        "anchor": "7-grpo-的-2026-變體全景與業界共識"
    },
    {
        "level": 2,
        "text": "8. RSI 架構全景：評估（Verifier）是遞歸自我改進的核心瓶頸",
        "anchor": "8-rsi-架構全景評估verifier是遞歸自我改進的核心瓶頸"
    },
    {
        "level": 2,
        "text": "🤔 架構深度思辨與工業界陷阱 (Architectural Insight & Production Pitfalls)",
        "anchor": "架構深度思辨與工業界陷阱-architectural-insight-production-pitfalls"
    },
    {
        "level": 2,
        "text": "練習",
        "anchor": "練習"
    }
]
};
