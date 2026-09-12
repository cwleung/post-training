// deepagents/da03.js
export default {
  "id": "da03",
  "num": "03",
  "title": "撰寫自訂中介軟體 (Custom Middleware)",
  "icon": "🔌",
  "file": "03-custom-middleware",
  "hasVisualizer": "agentic",
  "readTime": "18 min",
  "summary": "安檢閘門心智模型：AgentMiddleware 攔截器生命週期、狀態變更與 wrap_tool_call 過濾。",
  "toc": [
    {
        "level": 2,
        "text": "核心心智模型：機場安檢閘門與封包過濾器",
        "anchor": "核心心智模型機場安檢閘門與封包過濾器"
    },
    {
        "level": 2,
        "text": "3.1 風格 1：@wrap_tool_call——輕量級工具呼叫攔截",
        "anchor": "31-風格-1wrap-tool-call輕量級工具呼叫攔截"
    },
    {
        "level": 2,
        "text": "3.2 風格 2：AgentMiddleware 子類別——完整生命週期掛鉤",
        "anchor": "32-風格-2agentmiddleware-子類別完整生命週期掛鉤"
    },
    {
        "level": 2,
        "text": "3.3 中介軟體內部狀態變更與上下文傳遞",
        "anchor": "33-中介軟體內部狀態變更與上下文傳遞"
    },
    {
        "level": 2,
        "text": "🤔 架構深度思辨與工業界陷阱 (Architectural Insight & Production Pitfalls)",
        "anchor": "架構深度思辨與工業界陷阱-architectural-insight-production-pitfalls"
    },
    {
        "level": 2,
        "text": "下一步",
        "anchor": "下一步"
    }
]
};
