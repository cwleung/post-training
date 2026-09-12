// deepagents/da26.js
export default {
  "id": "da26",
  "num": "26",
  "title": "生產監控與分佈偏移偵測 (Production Monitoring & Drift)",
  "icon": "🚨",
  "file": "26-production-monitoring",
  "hasVisualizer": "eval_workbench",
  "readTime": "24 min",
  "summary": "電網保險絲心智模型：KL 散度輸入偏移監測、異常軌跡即時熔斷與生產資料回流。",
  "toc": [
    {
        "level": 2,
        "text": "核心心智模型：發電廠過載保護開關 (Power Grid Fuse & Circuit Breakers)",
        "anchor": "核心心智模型發電廠過載保護開關-power-grid-fuse-circuit-breakers"
    },
    {
        "level": 2,
        "text": "1. 生產監控的三個核心問題",
        "anchor": "1-生產監控的三個核心問題"
    },
    {
        "level": 2,
        "text": "2. 生產評估採樣架構",
        "anchor": "2-生產評估採樣架構"
    },
    {
        "level": 2,
        "text": "3. 分佈偏移偵測（Distribution Shift Detection）",
        "anchor": "3-分佈偏移偵測distribution-shift-detection"
    },
    {
        "level": 2,
        "text": "4. Whatbroke：代理 Trace 的確定性 diff 工具",
        "anchor": "4-whatbroke代理-trace-的確定性-diff-工具"
    },
    {
        "level": 2,
        "text": "5. 評估飛輪（Eval Flywheel）：把生產信號轉化為評估改善",
        "anchor": "5-評估飛輪eval-flywheel把生產信號轉化為評估改善"
    },
    {
        "level": 2,
        "text": "6. 告警設計：不要讓告警成為噪音",
        "anchor": "6-告警設計不要讓告警成為噪音"
    },
    {
        "level": 2,
        "text": "7. 生產監控的 Deployment Simulation 防禦",
        "anchor": "7-生產監控的-deployment-simulation-防禦"
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
