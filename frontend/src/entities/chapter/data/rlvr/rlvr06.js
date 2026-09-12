// rlvr/rlvr06.js
export default {
  "id": "rlvr06",
  "num": "06",
  "title": "Agentic 多輪強化學習 (Agentic RLVR & Multi-Turn Verification)",
  "icon": "🤖",
  "file": "06_agentic_rlvr",
  "hasVisualizer": "agent_sandbox",
  "readTime": "26 min",
  "summary": "Post-Training 實戰：單輪 vs 多輪智能體閉環、環境沙箱隔離、Observation 軌跡損失掩碼生死線、步數懲罰與信用分配。",
  "toc": [
    {
      "level": 2,
      "text": "核心心智模型：單輪數學推理 vs. 多輪智能體閉環",
      "anchor": "核心心智模型單輪數學推理-vs-多輪智能體閉環"
    },
    {
      "level": 2,
      "text": "6.1 Agentic RLVR 四大基石",
      "anchor": "61-agentic-rlvr-四大基石"
    },
    {
      "level": 2,
      "text": "6.2 軌跡損失掩碼 (Loss Masking)：遮蔽環境觀察的生死線",
      "anchor": "62-軌跡損失掩碼-loss-masking遮蔽環境觀察的生死線"
    },
    {
      "level": 2,
      "text": "6.3 多輪獎勵塑形：防刷分步數懲罰 (Step Efficiency Penalty)",
      "anchor": "63-多輪獎勵塑形防刷分步數懲罰-step-efficiency-penalty"
    },
    {
      "level": 2,
      "text": "6.4 信用分配（Credit Assignment）：結果監督 vs. 過程監督 (PRM)",
      "anchor": "64-信用分配credit-assignment結果監督-vs-過程監督-prm"
    },
    {
      "level": 2,
      "text": "🤔 架構深度思辨與工業界陷阱",
      "anchor": "-架構深度思辨與工業界陷阱-architectural-insight--production-pitfalls"
    }
  ]
};
