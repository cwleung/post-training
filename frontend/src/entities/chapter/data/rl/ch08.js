// rl/ch08.js
export default {
  "id": "ch08",
  "num": "08",
  "title": "近端策略優化·PPO 截斷目標與重要性採樣 (PPO Clipped Objective)",
  "icon": "📐",
  "file": "08_ppo_clipped_objective",
  "hasVisualizer": "ppo",
  "readTime": "22 min",
  "summary": "PPO-Clip 目標函數、重要性採樣比率剪裁與信任域約束。",
  "toc": [
    {
      "level": 2,
      "text": "核心心智模型：信任域與重要性採樣比率",
      "anchor": "核心心智模型信任域與重要性採樣比率"
    },
    {
      "level": 2,
      "text": "8.1 截斷代理目標函數 (Clipped Surrogate Objective) 的幾何構造",
      "anchor": "81-截斷代理目標函數-clipped-surrogate-objective-的幾何構造"
    },
    {
      "level": 2,
      "text": "8.2 完整 PPO 聯合損失函數",
      "anchor": "82-完整-ppo-聯合損失函數"
    },
    {
      "level": 2,
      "text": "8.3 可執行的 PyTorch 向量化 PPO 損失計算模組",
      "anchor": "83-可執行的-pytorch-向量化-ppo-損失計算模組"
    },
    {
      "level": 2,
      "text": "8.4 PPO 與前代架構多維度特性對比",
      "anchor": "84-ppo-與前代架構多維度特性對比"
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
