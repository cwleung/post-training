// rl/ch13.js
export default {
  "id": "ch13",
  "num": "13",
  "title": "測試期擴展·Test-Time Scaling 與 Pass@K (Inference Compute)",
  "icon": "📈",
  "file": "13_test_time_compute_passk",
  "hasVisualizer": "reasoning",
  "readTime": "20 min",
  "summary": "Inference-Time Compute 擴展定律、Thinking Tokens 與 Pass@K 組合數學。",
  "toc": [
    {
      "level": 2,
      "text": "核心心智模型：從預訓練擴展轉向測試期思考擴展",
      "anchor": "核心心智模型從預訓練擴展轉向測試期思考擴展"
    },
    {
      "level": 2,
      "text": "13.1 Chen et al. (2021) Pass@k 無偏組合數估計推導",
      "anchor": "131-chen-et-al-2021-passk-無偏組合數估計推導"
    },
    {
      "level": 2,
      "text": "13.2 思考 Token（Thinking Tokens）的對數縮放定律",
      "anchor": "132-思考-tokenthinking-tokens的對數縮放定律"
    },
    {
      "level": 2,
      "text": "13.3 可執行的 Python/PyTorch Pass@k 無偏估計防溢出實現",
      "anchor": "133-可執行的-pythonpytorch-passk-無偏估計防溢出實現"
    },
    {
      "level": 2,
      "text": "13.4 測試期計算擴展三大範式橫向對比",
      "anchor": "134-測試期計算擴展三大範式橫向對比"
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
