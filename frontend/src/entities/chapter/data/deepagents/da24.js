// deepagents/da24.js
export default {
  "id": "da24",
  "num": "24",
  "title": "多輪次與長任務評估 (Multi-Turn Evaluation & Long Horizon)",
  "icon": "⏱️",
  "file": "24-multi-turn-evaluation",
  "hasVisualizer": "eval_workbench",
  "readTime": "22 min",
  "summary": "遠洋航偏校準心智模型：METR 時間視野評估、目標漂移檢測與累積雪崩效應抑制。",
  "toc": [
    {
        "level": 2,
        "text": "核心心智模型：遠洋航行的羅盤與航偏校正 (Ocean Navigation & Drift Dampening)",
        "anchor": "核心心智模型遠洋航行的羅盤與航偏校正-ocean-navigation-drift-dampening"
    },
    {
        "level": 2,
        "text": "1. 為什麼多輪次評估根本不同",
        "anchor": "1-為什麼多輪次評估根本不同"
    },
    {
        "level": 2,
        "text": "2. 成功率 vs. 人類時間：METR 長任務量尺",
        "anchor": "2-成功率-vs-人類時間metr-長任務量尺"
    },
    {
        "level": 2,
        "text": "3. 長任務評估的三個核心指標",
        "anchor": "3-長任務評估的三個核心指標"
    },
    {
        "level": 2,
        "text": "4. 多輪次評估的隔離原則",
        "anchor": "4-多輪次評估的隔離原則"
    },
    {
        "level": 2,
        "text": "5. τ²-Bench 的雙控制架構：真實多輪次評估",
        "anchor": "5-τ²-bench-的雙控制架構真實多輪次評估"
    },
    {
        "level": 2,
        "text": "6. 多輪次評估的成本管理",
        "anchor": "6-多輪次評估的成本管理"
    },
    {
        "level": 2,
        "text": "7. 長任務評估的 Session-level vs. Turn-level 指標",
        "anchor": "7-長任務評估的-session-level-vs-turn-level-指標"
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
