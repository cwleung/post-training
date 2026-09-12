// deepagents/da04.js
export default {
  "id": "da04",
  "num": "04",
  "title": "後端與權限機制 (Backends & Permissions)",
  "icon": "🛡️",
  "file": "04-backends-and-permissions",
  "hasVisualizer": "agent_vfs",
  "readTime": "18 min",
  "summary": "容器沙箱心智模型：StateBackend 狀態、StoreBackend 與 Path-level ACL 權限阻斷防禦。",
  "toc": [
    {
        "level": 2,
        "text": "核心心智模型：容器沙箱與 POSIX 檔案描述符",
        "anchor": "核心心智模型容器沙箱與-posix-檔案描述符"
    },
    {
        "level": 2,
        "text": "4.1 內建儲存後端體系全景",
        "anchor": "41-內建儲存後端體系全景"
    },
    {
        "level": 2,
        "text": "4.2 關鍵實踐：CompositeBackend 混合路由",
        "anchor": "42-關鍵實踐compositebackend-混合路由"
    },
    {
        "level": 2,
        "text": "4.3 權限守衛（Permissions）與路徑級 ACL",
        "anchor": "43-權限守衛permissions與路徑級-acl"
    },
    {
        "level": 2,
        "text": "4.4 沙盒後端矩陣 (Docker / Modal / E2B)",
        "anchor": "44-沙盒後端矩陣-docker-modal-e2b"
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
