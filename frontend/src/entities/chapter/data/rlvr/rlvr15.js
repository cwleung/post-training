// rlvr/rlvr15.js
export default {
  "id": "rlvr15",
  "num": "15",
  "title": "LoRA, QLoRA 與參數高效後訓練 (PEFT & VRAM Arithmetic)",
  "icon": "⚡",
  "file": "15_lora_qlora_peft",
  "hasVisualizer": "rlvr_lora_vram",
  "tag": "🔥 Core Hotspot",
  "readTime": "24 min",
  "summary": "Post-Training 核心考點：全參微調顯存牆、透明描圖紙心智模型、NF4 資訊理論等分位數證明、雙重量化 DQ、漸進式 5 階代碼實驗室（4-bit 合併精度漂移復現、70B 單卡心算）、4D 遙測雷達與 Frontier Lab 面試手撕。",
  "toc": [
    {
      "level": 2,
      "text": "一、工業背景與技術演進：全參數微調的顯存之牆與 PEFT 革命",
      "anchor": "一工業背景與技術演進全參數微調的顯存之牆與-peft-革命"
    },
    {
      "level": 2,
      "text": "二、架構決策樹與 Trade-off 對比",
      "anchor": "二架構決策樹與-trade-off-對比"
    },
    {
      "level": 2,
      "text": "三、系統心智模型與邊界直覺 (Systems Mechanics & Boundary Intuition)",
      "anchor": "三系統心智模型與邊界直覺-systems-mechanics--boundary-intuition"
    },
    {
      "level": 2,
      "text": "四、漸進式可執行代碼實驗室：NF4 量化模擬、LoRA 前向與 70B 顯存精算 (Interactive Notebook Lab)",
      "anchor": "四漸進式可執行代碼實驗室nf4-量化模擬lora-前向與-70b-顯存精算-interactive-notebook-lab"
    },
    {
      "level": 2,
      "text": "五、工業級現場急救手冊與四維遙測監控雷達 (Runbook & 4D Telemetry Radar)",
      "anchor": "五工業級現場急救手冊與四維遙測監控雷達-runbook--4d-telemetry-radar"
    },
    {
      "level": 2,
      "text": "六、前沿系統架構深度思辨與極限設計 (Frontier Architecture Scenarios & Whiteboard Defense)",
      "anchor": "六前沿系統架構深度思辨與極限設計-frontier-architecture-scenarios--whiteboard-defense"
    },
    {
      "level": 2,
      "text": "本章小結與學習路徑",
      "anchor": "本章小結與學習路徑"
    }
  ]
};
