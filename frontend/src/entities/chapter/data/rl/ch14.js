// rl/ch14.js
export default {
  "id": "ch14",
  "num": "14",
  "title": "過程獎勵模型·PRM 逐步推理與束搜索剪枝 (Process Reward Models)",
  "icon": "🌲",
  "file": "14_prm_step_supervision_search",
  "hasVisualizer": "prm",
  "readTime": "24 min",
  "summary": "Step-level Reward Modeling、Beam Search 狀態樹展開與死枝剪枝。",
  "toc": [
    {
      "level": 2,
      "text": "核心心智模型：結果監督 (ORM) vs. 過程監督 (PRM)",
      "anchor": "核心心智模型結果監督-orm-vs-過程監督-prm"
    },
    {
      "level": 2,
      "text": "14.1 過程標註難題與 Math-Shepherd 自動合成技術",
      "anchor": "141-過程標註難題與-math-shepherd-自動合成技術"
    },
    {
      "level": 2,
      "text": "14.2 束搜索 (Beam Search) 與死枝剪枝算法推導",
      "anchor": "142-束搜索-beam-search-與死枝剪枝算法推導"
    },
    {
      "level": 2,
      "text": "14.3 可執行的 PyTorch PRM 步驟評分器與束搜索控制器",
      "anchor": "143-可執行的-pytorch-prm-步驟評分器與束搜索控制器"
    },
    {
      "level": 2,
      "text": "14.4 ORM vs PRM 多維工程特性對比",
      "anchor": "144-orm-vs-prm-多維工程特性對比"
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
