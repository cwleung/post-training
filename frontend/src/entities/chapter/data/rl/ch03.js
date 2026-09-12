// rl/ch03.js
export default {
  "id": "ch03",
  "num": "03",
  "title": "深度 Q 網絡·Replay Buffer 與 Target Network (DQN & Stability)",
  "icon": "🧠",
  "file": "03_dqn_replay_target",
  "hasVisualizer": "dqn",
  "readTime": "18 min",
  "summary": "經驗回放機制、目標網絡同步週期、Bellman 算子收縮性與 TD-Error 穩定性。",
  "toc": [
    {
      "level": 2,
      "text": "核心心智模型：馴服「致命三要素 (Deadly Triad)」",
      "anchor": "核心心智模型馴服致命三要素-deadly-triad"
    },
    {
      "level": 2,
      "text": "3.1 貝爾曼最優方程與目標網絡 (Target Network) 穩定性",
      "anchor": "31-貝爾曼最優方程與目標網絡-target-network-穩定性"
    },
    {
      "level": 2,
      "text": "3.2 Double DQN：消除 Q 值的系統性高估 (Overestimation Bias)",
      "anchor": "32-double-dqn消除-q-值的系統性高估-overestimation-bias"
    },
    {
      "level": 2,
      "text": "3.3 可執行的 PyTorch DQN / Double DQN 核心模組",
      "anchor": "33-可執行的-pytorch-dqn--double-dqn-核心模組"
    },
    {
      "level": 2,
      "text": "3.4 DQN 演進家族關鍵技術規格對比",
      "anchor": "34-dqn-演進家族關鍵技術規格對比"
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
