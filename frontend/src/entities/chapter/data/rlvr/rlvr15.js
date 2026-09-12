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
  "summary": "Post-Training 核心考點：全參微調顯存牆、LoRA 低秩矩陣分解、NF4 資訊理論推導、雙重量化 DQ、All-Linear 掛載策略、單卡 70B 顯存心算與 Frontier Lab 面試真題手撕。",
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
      "text": "四、代碼剖析、實時遙測巡檢與失效急救",
      "anchor": "四代碼剖析實時遙測巡檢與失效急救"
    },
    {
      "level": 2,
      "text": "五、前沿系統架構深度思辨與極限設計 (Frontier Architecture Scenarios & Whiteboard Defense)",
      "anchor": "五前沿系統架構深度思辨與極限設計-frontier-architecture-scenarios--whiteboard-defense"
    },
    {
      "level": 2,
      "text": "本章小結與學習路徑",
      "anchor": "本章小結與學習路徑"
    }
  ]
};
