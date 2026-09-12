// rlvr/rlvr04.js
export default {
  "id": "rlvr04",
  "num": "04",
  "title": "輕量訓練管線 (Training Pipeline & Stabilization)",
  "icon": "⚡",
  "file": "04_training",
  "hasVisualizer": "rlvr_training",
  "readTime": "22 min",
  "summary": "Post-Training 實戰：16GB 單卡顯存預算精算、LoRA 低秩分解、GRPOConfig 訓練排程與四維遙測監控指標。",
  "toc": [
    {
      "level": 2,
      "text": "核心心智模型：16GB 單卡如何撬動大模型強化學習？",
      "anchor": "核心心智模型16gb-單卡如何撬動大模型強化學習"
    },
    {
      "level": 2,
      "text": "4.1 硬體規格與模型選型矩陣",
      "anchor": "41-硬體規格與模型選型矩陣"
    },
    {
      "level": 2,
      "text": "4.2 模型載入與 PEFT LoRA 配置",
      "anchor": "42-模型載入與-peft-lora-配置"
    },
    {
      "level": 2,
      "text": "4.3 設定 GRPO 訓練參數 (GRPOConfig)",
      "anchor": "43-設定-grpo-訓練參數-grpoconfig"
    },
    {
      "level": 2,
      "text": "4.4 訓練啟動與四維遙測監控 (Telemetry Signals)",
      "anchor": "44-訓練啟動與四維遙測監控-telemetry-signals"
    },
    {
      "level": 2,
      "text": "4.5 顯存爆炸 (OOM) 工業級急救錦囊",
      "anchor": "45-顯存爆炸-oom-工業級急救錦囊"
    },
    {
      "level": 2,
      "text": "🤔 架構深度思辨與工業界陷阱",
      "anchor": "-架構深度思辨與工業界陷阱-architectural-insight--production-pitfalls"
    }
  ]
};
