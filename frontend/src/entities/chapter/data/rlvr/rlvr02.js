// rlvr/rlvr02.js
export default {
  "id": "rlvr02",
  "num": "02",
  "title": "獎勵工程與驗證器 (Reward Engineering & Verifiers)",
  "icon": "🎯",
  "file": "02_rewards",
  "hasVisualizer": "rlvr_rewards",
  "readTime": "18 min",
  "summary": "Post-Training 實戰：確定性 Verifiers、格式規範度評分、多目標獎勵排程退火與 Goodhart's Law 作弊防禦全景。",
  "toc": [
    {
      "level": 2,
      "text": "核心心智模型：RLHF 主觀裁判 vs. RLVR 確定性驗證器",
      "anchor": "核心心智模型rlhf-主觀裁判-vs-rlvr-確定性驗證器"
    },
    {
      "level": 2,
      "text": "2.1 驗證器在 GRPO 中的核心作用",
      "anchor": "21-驗證器在-grpo-中的核心作用"
    },
    {
      "level": 2,
      "text": "2.2 實作正確性驗證器 (Correctness Verifier)",
      "anchor": "22-實作正確性驗證器-correctness-verifier"
    },
    {
      "level": 2,
      "text": "2.3 實作格式結構獎勵 (Format Reward)",
      "anchor": "23-實作格式結構獎勵-format-reward"
    },
    {
      "level": 2,
      "text": "2.4 多獎勵組合與權重退火調度 (Weight Decay Schedule)",
      "anchor": "24-多獎勵組合與權重退火調度-weight-decay-schedule"
    },
    {
      "level": 2,
      "text": "2.5 常見作弊模式 (Reward Hacking) 與防禦矩陣",
      "anchor": "25-常見作弊模式-reward-hacking-與防禦矩陣"
    },
    {
      "level": 2,
      "text": "🤔 架構深度思辨與工業界陷阱",
      "anchor": "-架構深度思辨與工業界陷阱-architectural-insight--production-pitfalls"
    }
  ]
};
