// deepagents/da25.js
export default {
  "id": "da25",
  "num": "25",
  "title": "RAG 評估：檢索增強代理的三角驗收框架 (RAG Triad Evaluation)",
  "icon": "📐",
  "file": "25-rag-evaluation",
  "hasVisualizer": "eval_workbench",
  "readTime": "22 min",
  "summary": "法庭證據鏈心智模型：脈絡相關性、紮實度與回答相關性三權分立之原子命題驗收。",
  "toc": [
    {
        "level": 2,
        "text": "核心心智模型：法庭證據鏈審查 (Courtroom Chain of Custody & Claim-Level Verification)",
        "anchor": "核心心智模型法庭證據鏈審查-courtroom-chain-of-custody-claim-level-verification"
    },
    {
        "level": 2,
        "text": "1. 為什麼 RAG 評估需要獨立框架",
        "anchor": "1-為什麼-rag-評估需要獨立框架"
    },
    {
        "level": 2,
        "text": "2. RAG 三角（RAG Triad）：三個核心評估指標",
        "anchor": "2-rag-三角rag-triad三個核心評估指標"
    },
    {
        "level": 2,
        "text": "3. 宣告式評估（Claim-level Evaluation）：RAGChecker 方法",
        "anchor": "3-宣告式評估claim-level-evaluationragchecker-方法"
    },
    {
        "level": 2,
        "text": "4. 檢索器 vs. 生成器錯誤的分離診斷",
        "anchor": "4-檢索器-vs-生成器錯誤的分離診斷"
    },
    {
        "level": 2,
        "text": "5. Deep Agents RAG 評估 Middleware",
        "anchor": "5-deep-agents-rag-評估-middleware"
    },
    {
        "level": 2,
        "text": "6. ARES 方法：置信區間而非點估計",
        "anchor": "6-ares-方法置信區間而非點估計"
    },
    {
        "level": 2,
        "text": "7. RAG 評估的 CI 閘門配置",
        "anchor": "7-rag-評估的-ci-閘門配置"
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
