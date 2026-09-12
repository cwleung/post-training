// rlvr/rlvr10.js
export default {
  "id": "rlvr10",
  "num": "10",
  "title": "分佈式後訓練系統 — veRL、vLLM 與 3D-HybridEngine (Distributed Systems & Scale)",
  "icon": "🌐",
  "file": "10_distributed_systems_verl_vllm",
  "hasVisualizer": "rlvr_distributed",
  "tag": "🔥 Core Hotspot",
  "readTime": "25 min",
  "summary": "Post-Training 核心考點：雙模態系統瓶頸、雙離合變速箱心智模型、veRL + vLLM 3D-HybridEngine 動態重分片 (<800ms)、漸進式 5 階代碼實驗室（木桶短板與 NCCL 死鎖復現、FP8 KV 消融）、4D 遙測雷達與 Frontier Lab 面試手撕。",
  "toc": [
    {
      "level": 2,
      "text": "一、工業背景與技術演進：雙模態負載矛盾與推訓解耦革命",
      "anchor": "一工業背景與技術演進雙模態負載矛盾與推訓解耦革命"
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
      "text": "四、漸進式可執行代碼實驗室：3D-HybridEngine 顯存精算、動態重分片與 NCCL 異常模擬 (Interactive Notebook Lab)",
      "anchor": "四漸進式可執行代碼實驗室3d-hybridengine-顯存精算動態重分片與-nccl-異常模擬-interactive-notebook-lab"
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
