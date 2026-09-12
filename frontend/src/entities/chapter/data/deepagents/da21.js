// deepagents/da21.js
export default {
  "id": "da21",
  "num": "21",
  "title": "基準測試完整性：污染、飽和與排行榜博弈 (Benchmark Integrity)",
  "icon": "📊",
  "file": "21-benchmark-integrity",
  "hasVisualizer": "eval_workbench",
  "readTime": "22 min",
  "summary": "防弊考場心智模型：MinHash 13-gram 去污染、基準飽和對抗與 Goodhart 定律規避。",
  "toc": [
    {
        "level": 2,
        "text": "核心心智模型：防弊考場與性價比天平 (Anti-Cheat Testing & Pareto Frontier)",
        "anchor": "核心心智模型防弊考場與性價比天平-anti-cheat-testing-pareto-frontier"
    },
    {
        "level": 2,
        "text": "1. 基準測試 vs. 評估：根本性的區別",
        "anchor": "1-基準測試-vs-評估根本性的區別"
    },
    {
        "level": 2,
        "text": "2. 基準測試的三大失效模式",
        "anchor": "2-基準測試的三大失效模式"
    },
    {
        "level": 2,
        "text": "3. 代理時代的基準測試特有問題（8 層失效模型）",
        "anchor": "3-代理時代的基準測試特有問題8-層失效模型"
    },
    {
        "level": 2,
        "text": "4. 自動化基準評分的不可信賴性",
        "anchor": "4-自動化基準評分的不可信賴性"
    },
    {
        "level": 2,
        "text": "5. 對抗污染的評估設計原則",
        "anchor": "5-對抗污染的評估設計原則"
    },
    {
        "level": 2,
        "text": "6. 代理評估的 Holistic Leaderboard（HAL）方法",
        "anchor": "6-代理評估的-holistic-leaderboardhal方法"
    },
    {
        "level": 2,
        "text": "7. 你應該如何使用公開基準",
        "anchor": "7-你應該如何使用公開基準"
    },
    {
        "level": 2,
        "text": "8. 新一代抗污染基準設計（值得追蹤的方向）",
        "anchor": "8-新一代抗污染基準設計值得追蹤的方向"
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
