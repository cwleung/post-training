// rl/ch05.js
export default {
  "id": "ch05",
  "num": "05",
  "title": "Actor-Critic 架構與廣義優勢估計 (Actor-Critic & GAE)",
  "icon": "🎭",
  "file": "05_actor_critic_gae",
  "hasVisualizer": "actor_critic",
  "readTime": "18 min",
  "summary": "策略與價值雙網絡協同、自舉 TD 誤差與廣義優勢估計 GAE(γ, λ) 數學證明。",
  "toc": [
    {
      "level": 2,
      "text": "核心心智模型：策略與價值的雙網絡交響",
      "anchor": "核心心智模型策略與價值的雙網絡交響"
    },
    {
      "level": 2,
      "text": "5.1 時序差分殘差 (TD Error) 作為優勢估計的無偏性",
      "anchor": "51-時序差分殘差-td-error-作為優勢估計的無偏性"
    },
    {
      "level": 2,
      "text": "5.2 廣義優勢估計 GAE(γ, λ) 的數學證明",
      "anchor": "52-廣義優勢估計-gaeγ-λ-的數學證明"
    },
    {
      "level": 2,
      "text": "5.3 高效向量化 GAE 遞推計算實現 (PyTorch)",
      "anchor": "53-高效向量化-gae-遞推計算實現-pytorch"
    },
    {
      "level": 2,
      "text": "5.4 各階優勢估計方法綜合評估矩陣",
      "anchor": "54-各階優勢估計方法綜合評估矩陣"
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
