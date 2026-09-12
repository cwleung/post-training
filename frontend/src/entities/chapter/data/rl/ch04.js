// rl/ch04.js
export default {
  "id": "ch04",
  "num": "04",
  "title": "策略梯度·REINFORCE 與方差縮減 (Policy Gradient Theorem)",
  "icon": "📈",
  "file": "04_policy_gradient_reinforce",
  "hasVisualizer": "policy_gradient",
  "readTime": "18 min",
  "summary": "似然率梯度定理 (Policy Gradient Theorem)、對數導數技巧與 Baseline 方差削減。",
  "toc": [
    {
      "level": 2,
      "text": "核心心智模型：似然率技巧與信用加權",
      "anchor": "核心心智模型似然率技巧與信用加權"
    },
    {
      "level": 2,
      "text": "4.1 策略梯度定理 (Policy Gradient Theorem) 的嚴密推導",
      "anchor": "41-策略梯度定理-policy-gradient-theorem-的嚴密推導"
    },
    {
      "level": 2,
      "text": "4.2 Baseline 零偏差減方差數學證明",
      "anchor": "42-baseline-零偏差減方差數學證明"
    },
    {
      "level": 2,
      "text": "4.3 可執行的 PyTorch REINFORCE 帶 Baseline 演算法模組",
      "anchor": "43-可執行的-pytorch-reinforce-帶-baseline-演算法模組"
    },
    {
      "level": 2,
      "text": "4.4 REINFORCE 與其他範式核心特性對比",
      "anchor": "44-reinforce-與其他範式核心特性對比"
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
