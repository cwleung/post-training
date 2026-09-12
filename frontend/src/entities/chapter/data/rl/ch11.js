// rl/ch11.js
export default {
  "id": "ch11",
  "num": "11",
  "title": "三階段對齊·RLHF 獎勵建模與邊界 (3-Stage RLHF Pipeline)",
  "icon": "🛡️",
  "file": "11_rlhf_3stage_pipeline",
  "hasVisualizer": "rlhf",
  "readTime": "22 min",
  "summary": "SFT → Reward Modeling → PPO 完整流程與 KL 散度約束邊界。",
  "toc": [
    {
      "level": 2,
      "text": "核心心智模型：經典 3 階段對齊流水線",
      "anchor": "核心心智模型經典-3-階段對齊流水線"
    },
    {
      "level": 2,
      "text": "11.1 獎勵模型 (Reward Model) 的訓練目標與 Pairwise Ranking 損失",
      "anchor": "111-獎勵模型-reward-model-的訓練目標與-pairwise-ranking-損失"
    },
    {
      "level": 2,
      "text": "11.2 為什麼必須加入 KL 散度約束？古德哈特定律與策略漂移",
      "anchor": "112-為什麼必須加入-kl-散度約束古德哈特定律與策略漂移"
    },
    {
      "level": 2,
      "text": "11.3 可執行的 PyTorch 獎勵模型 Pairwise 訓練模組",
      "anchor": "113-可執行的-pytorch-獎勵模型-pairwise-訓練模組"
    },
    {
      "level": 2,
      "text": "11.4 RLHF 核心階段關鍵工程指標監控清單",
      "anchor": "114-rlhf-核心階段關鍵工程指標監控清單"
    },
    {
      "level": 2,
      "text": "🤔 架構深度思辨與工業界陷阱",
      "anchor": "-架構深度思辨與工業界陷阱-architectural-insight--production-pitfalls"
    },
    {
      "level": 2,
      "text": "參考文獻與經典論文",
      "anchor": "參考文獻與經典論文"
    }
  ]
};
