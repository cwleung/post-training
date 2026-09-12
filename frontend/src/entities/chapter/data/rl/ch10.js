// rl/ch10.js
export default {
  "id": "ch10",
  "num": "10",
  "title": "群體相對策略優化·GRPO 組間歸一化 (DeepSeek R1 GRPO)",
  "icon": "👥",
  "file": "10_grpo_group_relative",
  "hasVisualizer": "grpo",
  "readTime": "25 min",
  "summary": "DeepSeek R1 核心算法：無需 Critic 的組內 Z-Score 優勢歸一化與 KL 懲罰。",
  "toc": [
    {
      "level": 2,
      "text": "核心心智模型：徹底廢除 Critic 網絡的輕量化革命",
      "anchor": "核心心智模型徹底廢除-critic-網絡的輕量化革命"
    },
    {
      "level": 2,
      "text": "10.1 GRPO 目標函數與組內相對優勢公式",
      "anchor": "101-grpo-目標函數與組內相對優勢公式"
    },
    {
      "level": 2,
      "text": "10.2 可執行的 PyTorch 向量化 GRPO 損失實現",
      "anchor": "102-可執行的-pytorch-向量化-grpo-損失實現"
    },
    {
      "level": 2,
      "text": "10.3 PPO vs GRPO 在大模型後訓練中的終極對比",
      "anchor": "103-ppo-vs-grpo-在大模型後訓練中的終極對比"
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
