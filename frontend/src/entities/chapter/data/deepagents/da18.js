// deepagents/da18.js
export default {
  "id": "da18",
  "num": "18",
  "title": "Agent-as-Judge、裁判校準與 RLVR 接入 (LLM-as-a-Judge Calibration)",
  "icon": "⚖️",
  "file": "18-llm-as-judge-alignment",
  "hasVisualizer": "rubric_eval",
  "readTime": "26 min",
  "summary": "陪審團校準心智模型：Auto-CoT 生成、對數機率期望值與 Cohen's Kappa 裁判一致性。",
  "toc": [
    {
        "level": 2,
        "text": "核心心智模型：陪審團共識與裁判偏差校準 (Jury Consensus & Position Bias Correction)",
        "anchor": "核心心智模型陪審團共識與裁判偏差校準-jury-consensus-position-bias-correction"
    },
    {
        "level": 2,
        "text": "1. LLM 裁判的三大系統性偏差",
        "anchor": "1-llm-裁判的三大系統性偏差"
    },
    {
        "level": 2,
        "text": "2. 校準方程：Cohen's Kappa 對齊度",
        "anchor": "2-校準方程cohens-kappa-對齊度"
    },
    {
        "level": 2,
        "text": "3. 裁判設計七大最佳實踐",
        "anchor": "3-裁判設計七大最佳實踐"
    },
    {
        "level": 2,
        "text": "4. Agent-as-Judge 的起源與三階段演化",
        "anchor": "4-agent-as-judge-的起源與三階段演化"
    },
    {
        "level": 2,
        "text": "5. Agent-as-Judge：從單一 LLM 裁判升級為多代理判官",
        "anchor": "5-agent-as-judge從單一-llm-裁判升級為多代理判官"
    },
    {
        "level": 2,
        "text": "6. 接入 RLVR：裁判在訓練迴圈中的角色",
        "anchor": "6-接入-rlvr裁判在訓練迴圈中的角色"
    },
    {
        "level": 2,
        "text": "7. Reward Hacking 在訓練迴圈的特有風險",
        "anchor": "7-reward-hacking-在訓練迴圈的特有風險"
    },
    {
        "level": 2,
        "text": "8. SOTA 量化基準與 Rubric-as-Reward 演進脈絡（2026）",
        "anchor": "8-sota-量化基準與-rubric-as-reward-演進脈絡2026"
    },
    {
        "level": 2,
        "text": "9. 裁判校準工作流程（完整流程）",
        "anchor": "9-裁判校準工作流程完整流程"
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
