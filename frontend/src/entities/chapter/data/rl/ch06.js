// rl/ch06.js
export default {
  "id": "ch06",
  "num": "06",
  "title": "連續動作控制·Soft Actor-Critic (SAC & MaxEnt RL)",
  "icon": "🏎️",
  "file": "06_sac_continuous_control",
  "hasVisualizer": "continuous",
  "readTime": "20 min",
  "summary": "最大熵強化學習框架、Tanh 高斯動作重參數化技巧與自動溫度調節。",
  "toc": [
    {
      "level": 2,
      "text": "核心心智模型：最大熵強化學習框架 (MaxEnt RL)",
      "anchor": "核心心智模型最大熵強化學習框架-maxent-rl"
    },
    {
      "level": 2,
      "text": "6.1 Tanh 高斯重參數化與雅可比修正公式",
      "anchor": "61-tanh-高斯重參數化與雅可比修正公式"
    },
    {
      "level": 2,
      "text": "6.2 雙 Q 網絡與目標計算公式",
      "anchor": "62-雙-q-網絡與目標計算公式"
    },
    {
      "level": 2,
      "text": "6.3 可執行的 PyTorch Squashed Gaussian Actor 實現",
      "anchor": "63-可執行的-pytorch-squashed-gaussian-actor-實現"
    },
    {
      "level": 2,
      "text": "6.4 連續控制經典演算法多維橫向比較",
      "anchor": "64-連續控制經典演算法多維橫向比較"
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
