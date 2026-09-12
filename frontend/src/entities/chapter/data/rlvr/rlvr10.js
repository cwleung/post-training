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
  "summary": "Post-Training 核心考點：雙模態系統瓶頸、veRL + vLLM 3D-HybridEngine 推訓解耦、動態重分片 (<800ms)、KV-Cache 顯存精算、NCCL 死鎖排查與 Frontier Lab 面試真題手撕。",
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
