// rlvr/rlvr05.js
export default {
  "id": "rlvr05",
  "num": "05",
  "title": "評估基準與組合數學 (Evaluation Benchmarking & Pass@k)",
  "icon": "📈",
  "file": "05_evaluation",
  "hasVisualizer": "rlvr_eval",
  "readTime": "24 min",
  "summary": "Post-Training 實戰：Pass@1 貪婪 vs Pass@k 探索天花板、超幾何無偏估計量、Bootstrap 置信區間、作弊警報與 TPCS 推理經濟學。",
  "toc": [
    {
      "level": 2,
      "text": "核心心智模型：Pass@1 直覺回答 vs. Pass@k 潛力天花板",
      "anchor": "核心心智模型pass1-直覺回答-vs-passk-潛力天花板"
    },
    {
      "level": 2,
      "text": "5.1 評估策略：訓練前後的基準對照 (Before vs. After)",
      "anchor": "51-評估策略訓練前後的基準對照-before-vs-after"
    },
    {
      "level": 2,
      "text": "5.2 無偏 Pass@k 估計器的組合數學推導",
      "anchor": "52-無偏-passk-估計器的組合數學推導"
    },
    {
      "level": 2,
      "text": "5.3 統計顯著性與 Bootstrap 95% 置信區間",
      "anchor": "53-統計顯著性與-bootstrap-95-置信區間"
    },
    {
      "level": 2,
      "text": "5.4 作弊警報：識別 Reward Hacking 與策略坍塌",
      "anchor": "54-作弊警報識別-reward-hacking-與策略坍塌"
    },
    {
      "level": 2,
      "text": "5.5 推理成本經濟學：每正確解 Token 成本 (TPCS)",
      "anchor": "55-推理成本經濟學每正確解-token-成本-tpcs"
    },
    {
      "level": 2,
      "text": "5.6 失效模式與迭代診斷決策樹",
      "anchor": "56-失效模式與迭代診斷決策樹"
    },
    {
      "level": 2,
      "text": "🤔 架構深度思辨與工業界陷阱",
      "anchor": "-架構深度思辨與工業界陷阱-architectural-insight--production-pitfalls"
    }
  ]
};
