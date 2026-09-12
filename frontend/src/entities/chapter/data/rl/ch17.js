// rl/ch17.js
export default {
  "id": "ch17",
  "num": "17",
  "title": "多模態視覺 Agent·VLM RL 與 KV Cache 內存建模 (Multimodal RL)",
  "icon": "👁️",
  "file": "17_vlm_multimodal_rl_kvcache",
  "hasVisualizer": "vlm",
  "readTime": "20 min",
  "summary": "Visual Tokens 編碼切片、多模態長上下文 KV Cache 顯存佔用建模。",
  "toc": [
    {
      "level": 2,
      "text": "核心心智模型：視覺網格切片與長上下文顯存挑戰",
      "anchor": "核心心智模型視覺網格切片與長上下文顯存挑戰"
    },
    {
      "level": 2,
      "text": "17.1 KV Cache 顯存消耗的精確數學解析模型",
      "anchor": "171-kv-cache-顯存消耗的精確數學解析模型"
    },
    {
      "level": 2,
      "text": "17.2 可執行的 Python VLM KV Cache 顯存與 Token 開銷計算器",
      "anchor": "172-可執行的-python-vlm-kv-cache-顯存與-token-開銷計算器"
    },
    {
      "level": 2,
      "text": "17.3 視覺 Agent 強化學習獎勵設計範式",
      "anchor": "173-視覺-agent-強化學習獎勵設計範式"
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
