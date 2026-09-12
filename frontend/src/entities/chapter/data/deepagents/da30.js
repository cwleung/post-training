// deepagents/da30.js
export default {
  "id": "da30",
  "num": "30",
  "title": "檢查點回放與確定性除錯 (Checkpoint Replay)",
  "icon": "⏪",
  "file": "30-checkpoint-replay",
  "hasVisualizer": "eval_workbench",
  "readTime": "24 min",
  "summary": "時光機重演心智模型：LangGraph Checkpoint 狀態重放、Mock 副作用與確定性時空回溯。",
  "toc": [
    {
        "level": 2,
        "text": "核心心智模型：平行時空遊戲存檔點 (Game Save States & Time-Travel Debugging)",
        "anchor": "核心心智模型平行時空遊戲存檔點-game-save-states-time-travel-debugging"
    },
    {
        "level": 2,
        "text": "30.1 檢查點重放資料結構設計",
        "anchor": "301-檢查點重放資料結構設計"
    },
    {
        "level": 2,
        "text": "1. 為什麼代理需要 Checkpoint Replay",
        "anchor": "1-為什麼代理需要-checkpoint-replay"
    },
    {
        "level": 2,
        "text": "2. 檢查點資料結構設計",
        "anchor": "2-檢查點資料結構設計"
    },
    {
        "level": 2,
        "text": "3. 檢查點管理器",
        "anchor": "3-檢查點管理器"
    },
    {
        "level": 2,
        "text": "4. Replay 引擎：確定性回放",
        "anchor": "4-replay-引擎確定性回放"
    },
    {
        "level": 2,
        "text": "5. 確定性 Regression Test 模式",
        "anchor": "5-確定性-regression-test-模式"
    },
    {
        "level": 2,
        "text": "6. 自動檢查點注入中介軟體",
        "anchor": "6-自動檢查點注入中介軟體"
    },
    {
        "level": 2,
        "text": "7. 檢查點回放在評估工程中的五大應用場景",
        "anchor": "7-檢查點回放在評估工程中的五大應用場景"
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
