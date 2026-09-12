// rlvr/rlvr16.js
export default {
  "id": "rlvr16",
  "num": "16",
  "title": "大模型推論極限優化 — 量化 (GPTQ/AWQ)、投機解碼、KV-Cache 與硬體編譯",
  "icon": "🚀",
  "file": "16_inference_optimization_quantization_compilation",
  "hasVisualizer": "rlvr_scaling",
  "readTime": "32 min",
  "summary": "Roofline 模型物理邊界、AWQ 激活感知 4-bit 量化、投機解碼無損拒絕採樣數學證明、自適應步長熔斷調度器與 MLA/PagedAttention 內存革新",
  "toc": [
    {
      "level": 2,
      "text": "一、工業背景與技術演進 (Background & Architectural Evolution)",
      "anchor": "一工業背景與技術演進-background--architectural-evolution"
    },
    {
      "level": 2,
      "text": "二、架構決策樹與 Trade-off 對比 (Architectural Decision Framework)",
      "anchor": "二架構決策樹與-trade-off-對比-architectural-decision-framework"
    },
    {
      "level": 2,
      "text": "三、系統心智模型與邊界直覺 (Systems Mechanics & Mathematical Formulations)",
      "anchor": "三系統心智模型與邊界直覺-systems-mechanics--mathematical-formulations"
    },
    {
      "level": 2,
      "text": "四、漸進式可執行代碼實驗室 (Interactive Notebook Lab)",
      "anchor": "四漸進式可執行代碼實驗室-interactive-notebook-lab"
    },
    {
      "level": 2,
      "text": "五、工業級現場急救手冊與四維遙測監控雷達 (Production Runbook & Telemetry Radar)",
      "anchor": "五工業級現場急救手冊與四維遙測監控雷達-production-runbook--telemetry-radar"
    },
    {
      "level": 2,
      "text": "六、前沿系統架構深度思辨與極限設計 (Frontier Architecture & Whiteboard Defense)",
      "anchor": "六前沿系統架構深度思辨與極限設計-frontier-architecture--whiteboard-defense"
    },
    {
      "level": 2,
      "text": "本章小結與學習路徑 (Summary & Roadmap)",
      "anchor": "本章小結與學習路徑-summary--roadmap"
    }
  ],
  "codeLines": []
};
