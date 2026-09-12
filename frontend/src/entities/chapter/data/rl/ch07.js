// rl/ch07.js
export default {
  "id": "ch07",
  "num": "07",
  "title": "離線強化學習·Conservative Q-Learning (CQL Offline RL)",
  "icon": "💾",
  "file": "07_cql_offline_rl",
  "hasVisualizer": "offline",
  "readTime": "22 min",
  "summary": "分佈偏移 (Distribution Shift)、分佈外 (OOD) 動作懲罰與保守價值估計。",
  "toc": [
    {
      "level": 2,
      "text": "核心心智模型：分佈偏移與分佈外動作 (OOD) 的價值膨脹",
      "anchor": "核心心智模型分佈偏移與分佈外動作-ood-的價值膨脹"
    },
    {
      "level": 2,
      "text": "7.1 保守價值估計 (CQL) 的數學形式與下界保證",
      "anchor": "71-保守價值估計-cql-的數學形式與下界保證"
    },
    {
      "level": 2,
      "text": "7.2 可執行的 PyTorch 離散/連續動作 CQL 損失計算",
      "anchor": "72-可執行的-pytorch-離散連續動作-cql-損失計算"
    },
    {
      "level": 2,
      "text": "7.3 離線強化學習核心範式全景對比",
      "anchor": "73-離線強化學習核心範式全景對比"
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
