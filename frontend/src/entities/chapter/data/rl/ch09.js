// rl/ch09.js
export default {
  "id": "ch09",
  "num": "09",
  "title": "直接偏好優化·DPO 隱式獎勵與對齊 (Direct Preference Optimization)",
  "icon": "⚖️",
  "file": "09_dpo_implicit_reward",
  "hasVisualizer": "dpo",
  "readTime": "20 min",
  "summary": "Bradley-Terry 偏好模型推導、隱式獎勵替換與解析閉式解。",
  "toc": [
    {
      "level": 2,
      "text": "核心心智模型：從顯式獎勵建模到隱式代數閉式解",
      "anchor": "核心心智模型從顯式獎勵建模到隱式代數閉式解"
    },
    {
      "level": 2,
      "text": "9.1 布拉德利-特里 (Bradley-Terry) 偏好模型與解析代換",
      "anchor": "91-布拉德利-特里-bradley-terry-偏好模型與解析代換"
    },
    {
      "level": 2,
      "text": "9.2 DPO 目標函數與梯度推導",
      "anchor": "92-dpo-目標函數與梯度推導"
    },
    {
      "level": 2,
      "text": "9.3 可執行的 PyTorch DPO 損失計算模組",
      "anchor": "93-可執行的-pytorch-dpo-損失計算模組"
    },
    {
      "level": 2,
      "text": "9.4 DPO 與現代偏好對齊演算法全景對比",
      "anchor": "94-dpo-與現代偏好對齊演算法全景對比"
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
