// rlvr/rlvr01.js
export default {
  "id": "rlvr01",
  "num": "01",
  "title": "資料準備與格式化 (Data Preparation for RLVR)",
  "icon": "📊",
  "file": "01_data",
  "hasVisualizer": "rlvr_data",
  "readTime": "18 min",
  "summary": "Post-Training 實戰：SFT vs RLVR 核心心智模型、GSM8K 資料解構、XML 標籤邊界與 Left-Padding 自回歸批次生成直覺解析。",
  "toc": [
    {
      "level": 2,
      "text": "核心心智模型：SFT 模仿 vs. RLVR 探索",
      "anchor": "核心心智模型sft-模仿-vs-rlvr-探索"
    },
    {
      "level": 2,
      "text": "1.1 理想 RLVR 數據的四大特質",
      "anchor": "11-理想-rlvr-數據的四大特質"
    },
    {
      "level": 2,
      "text": "1.2 GSM8K — 基準訓練資料集解構",
      "anchor": "12-gsm8k--基準訓練資料集解構"
    },
    {
      "level": 2,
      "text": "1.3 資料清洗與 Chat Template 格式化管線",
      "anchor": "13-資料清洗與-chat-template-格式化管線"
    },
    {
      "level": 2,
      "text": "1.4 自回歸 Rollout 與 Left-Padding 直覺深度解析",
      "anchor": "14-自回歸-rollout-與-left-padding-直覺深度解析"
    },
    {
      "level": 2,
      "text": "1.5 資料品質檢驗與完整性防禦",
      "anchor": "15-資料品質檢驗與完整性防禦"
    },
    {
      "level": 2,
      "text": "1.6 常見 RLVR 數據集全景對比",
      "anchor": "16-常見-rlvr-數據集全景對比"
    },
    {
      "level": 2,
      "text": "🤔 架構深度思辨與工業界陷阱",
      "anchor": "-架構深度思辨與工業界陷阱-architectural-insight--production-pitfalls"
    }
  ]
};
