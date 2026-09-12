import type { SiteManifest } from '@/shared/types';

export const RLVR_MANIFEST: SiteManifest = {
  id: 'rlvr',
  name: 'Production Post-Training MLE Handbook',
  shortName: 'Post-Training',
  icon: '🚀',
  description: '前沿 AI 實驗室 Post-Training ML Engineer 核心工程實踐：4 大核心維度、veRL 分佈式、SimPO、PRMs、LoRA 顯存精算與 64x H100 系統設計架構手冊。',
  parts: [
    {
      id: 'rlvr_pillar1',
      label: 'Pillar I · 基礎強化學習、數據工程與驗證器 (Data, Rewards & RLVR Foundations)',
      icon: '🎯',
      description: 'GSM8K 數據清洗、確定性正則驗證器、GRPO 零 Critic 演算法推導、LoRA 訓練管線與無偏 Pass@k 統計指標',
      milestone: '精通 GSM8K 數據清洗、確定性驗證器、GRPO 零 Critic 顯存架構與無偏 Pass@k',
      jobTarget: 'Post-Training RL & Preference Optimization MLE (OpenAI / Anthropic)',
      chapters: [
        {
          id: 'rlvr01',
          file: '01_data',
          num: '01',
          title: '資料準備與格式化 (Data Preparation & Formatting)',
          icon: '📊',
          hasVisualizer: 'rlvr_data',
          tag: 'GSM8K Data',
          readTime: '15 min',
          summary: 'GSM8K 載入清洗、XML 標籤驗證、Token 邊界與 Prompt Schema 設計',
          competencies: ['Dataset Hygiene', 'Left-Padding Tokenization', 'Chat Templates']
        },
        {
          id: 'rlvr02',
          file: '02_rewards',
          num: '02',
          title: '獎勵工程與驗證器 (Reward Engineering & Verifiers)',
          icon: '🎯',
          hasVisualizer: 'rlvr_rewards',
          tag: 'Verifiers',
          readTime: '18 min',
          summary: '確定性 Verifiers、格式懲罰、多目標獎勵組合與 Reward Hacking 防護',
          competencies: ['Deterministic Verifiers', 'Reward Shaping', 'Hacking Mitigation']
        },
        {
          id: 'rlvr03',
          file: '03_grpo_algorithm',
          num: '03',
          title: 'GRPO 演算法推導 (The GRPO Algorithm & Dr. GRPO)',
          icon: '👥',
          hasVisualizer: 'grpo',
          tag: '🔥 Core Hotspot',
          readTime: '25 min',
          summary: '從 PPO 到 GRPO 的工業演進、零 Critic 顯存架構、同儕標準化優勢與邊界行為直覺',
          competencies: ['GRPO Formulation', 'Z-Score Normalization', 'KL Divergence Penalty', 'Critic-Free Scaling']
        },
        {
          id: 'rlvr04',
          file: '04_training',
          num: '04',
          title: '訓練流水線與 LoRA 優化 (Training Pipeline & LoRA)',
          icon: '⚡',
          hasVisualizer: 'rlvr_lora_vram',
          tag: 'LoRA & VRAM',
          readTime: '20 min',
          summary: 'Unsloth + LoRA + GRPOTrainer 整合，GPU 顯存極限優化實戰',
          competencies: ['PEFT / LoRA', 'Unsloth Engine', 'VRAM Profiling']
        },
        {
          id: 'rlvr05',
          file: '05_evaluation',
          num: '05',
          title: '評估與統計指標 (Evaluation & Statistical Metrics)',
          icon: '📈',
          hasVisualizer: 'rlvr_passk',
          tag: 'Pass@k Metrics',
          readTime: '22 min',
          summary: '無偏 Pass@k、Majority@k、Bootstrap 置信區間與資料集污染檢測',
          competencies: ['Unbiased Pass@k', 'Majority@k', 'Bootstrap Confidence Intervals']
        }
      ]
    },
    {
      id: 'rlvr_pillar2',
      label: 'Pillar II · 偏好優化、多輪代理與冷啟動蒸餾 (Preference Optimization & Reasoning Bootstrap)',
      icon: '⚖️',
      description: 'Agentic 多輪探索、DPO 隱式獎勵推拉力學、消融實驗與擴展規律、SFT Cold-Start 長思維鏈合成蒸餾',
      milestone: '掌握 Agentic 多輪驗證、DPO 偏好優化、超參數擴展消融與冷啟動思維鏈合成',
      jobTarget: 'Reasoning Data Flywheel & Post-Training MLE (xAI / DeepMind)',
      chapters: [
        {
          id: 'rlvr06',
          file: '06_agentic_rlvr',
          num: '06',
          title: 'Agentic RLVR 與多輪 Rollouts (Multi-Turn Rollouts)',
          icon: '🤖',
          hasVisualizer: 'agentic',
          tag: 'Agent RLVR',
          readTime: '26 min',
          summary: '工具呼叫、沙箱程式碼執行環境、多輪軌跡中間信用分配',
          competencies: ['Multi-Turn Rollouts', 'Tool Call Sandboxes', 'Credit Assignment']
        },
        {
          id: 'rlvr07',
          file: '07_dpo_preference_optimization',
          num: '07',
          title: 'DPO 與偏好優化 (DPO Preference Optimization)',
          icon: '⚖️',
          hasVisualizer: 'dpo',
          tag: '🔥 Core Hotspot',
          readTime: '22 min',
          summary: 'Bradley-Terry 偏好模型演進、隱式獎勵閉式解、DPO vs PPO 決策樹與長度偏見失效排查',
          competencies: ['DPO Objective', 'Implicit Reward Dynamics', 'Offline Preference', 'Length Bias Mitigation']
        },
        {
          id: 'rlvr08',
          file: '08_ablations_and_scaling',
          num: '08',
          title: '消融實驗與擴展分析 (Ablations & Scaling Analysis)',
          icon: '🔬',
          hasVisualizer: 'rlvr_scaling',
          tag: 'Scaling Curves',
          readTime: '20 min',
          summary: 'Beta 系數、Group 大小 G、學習率敏感度分析與性能擴展曲線',
          competencies: ['Hyperparameter Sensitivity', 'Group Size Ablations', 'Compute Curves']
        },
        {
          id: 'rlvr09',
          file: '09_sft_cold_start',
          num: '09',
          title: 'SFT Cold-Start 階段 (SFT Cold-Start Stage)',
          icon: '🧊',
          hasVisualizer: 'rlvr_coldstart',
          tag: '🔥 Core Hotspot',
          readTime: '22 min',
          summary: '推理 Cold-Start 困境、長思維鏈合成蒸餾、格式標籤約束與防範模式坍塌',
          competencies: ['Cold-Start Bootstrap', 'Synthetic CoT Generation', 'Model Distillation', 'Mode Collapse Guard']
        }
      ]
    },
    {
      id: 'rlvr_pillar3',
      label: 'Pillar III · 分佈式系統、現代對齊與過程監督 (Distributed Scale, Modern Preference & PRMs)',
      icon: '⚡',
      description: 'veRL/vLLM 3D-HybridEngine 解耦架構、免參考模型 SimPO/ReMax、步驟級 PRM 樹搜索與 13-gram 基準去污染',
      milestone: '掌握 3D-HybridEngine 動態重分片、SimPO 顯存減半、步驟級 PRM 與測試集嚴格去污染',
      jobTarget: 'Distributed Post-Training Systems & Acceleration MLE (DeepSeek / Apple)',
      chapters: [
        {
          id: 'rlvr10',
          file: '10_distributed_systems_verl_vllm',
          num: '10',
          title: '分佈式系統 — veRL、vLLM 與 3D-HybridEngine',
          icon: '🌐',
          hasVisualizer: 'rlvr_distributed',
          tag: '🔥 Core Hotspot',
          readTime: '25 min',
          summary: '解耦 Rollout/Learner 集群、動態權重重新分片 (<800ms)、KV Cache 顯存數學與 NCCL 掛起排查',
          competencies: ['veRL Architecture', '3D-HybridEngine', 'vLLM PagedAttention Integration', 'NCCL Triage']
        },
        {
          id: 'rlvr11',
          file: '11_modern_preference_simpo_remax_kto',
          num: '11',
          title: '現代偏好優化 — SimPO、ReMax 與 KTO',
          icon: '💡',
          hasVisualizer: 'rlvr_simpo',
          tag: '🔥 Core Hotspot',
          readTime: '24 min',
          summary: '無參考模型 SimPO (節省 50% 顯存)、目標邊界 γ 直覺、無 Critic 的 ReMax 與 KTO 遙測對齊',
          competencies: ['Reference-Free SimPO', 'Length Normalization', 'ReMax Baseline', 'VRAM Halving']
        },
        {
          id: 'rlvr12',
          file: '12_process_supervision_and_test_time_compute',
          num: '12',
          title: '過程監督與測試期計算 — PRMs & MCTS',
          icon: '🌲',
          hasVisualizer: 'prm',
          tag: 'PRM & Search',
          readTime: '26 min',
          summary: '步驟級 PRM、Math-Shepherd 自動標籤、Beam Search 與 MCTS 樹剪枝',
          competencies: ['Step-Level PRM', 'Math-Shepherd', 'Beam Search Pruning']
        },
        {
          id: 'rlvr13',
          file: '13_data_flywheel_and_decontamination',
          num: '13',
          title: '工業級數據飛輪與去污染 (Data Flywheel & Decontamination)',
          icon: '🔄',
          hasVisualizer: 'pitfalls',
          tag: 'Data Flywheel',
          readTime: '22 min',
          summary: 'Magpie 自生成無 Prompt 數據、MinHash 去重、13-gram 基準測試嚴格去污染',
          competencies: ['Magpie Engine', 'MinHash LSH', 'Benchmark Decontamination']
        }
      ]
    },
    {
      id: 'rlvr_pillar4',
      label: 'Pillar IV · 線上事故排查、推論極限與安全對齊 (Triage Playbook, Systems Limits & Safety)',
      icon: '🛡️',
      description: 'Frontier Labs 64x H100 叢集架構、線上事故急診 Playbook、LoRA/QLoRA 顯存精算、推論量化與憲政 AI 紅隊',
      milestone: '通關 64x H100 系統設計、線上事故秒級止血、NF4 雙重量化與 3 層紅隊安全防線',
      jobTarget: 'Staff Post-Training Engineer & Alignment Architect (Meta / Frontier Labs)',
      chapters: [
        {
          id: 'rlvr14',
          file: '14_post_training_systems_and_triage_playbook',
          num: '14',
          title: 'Post-Training 分散式架構與線上事故排查 Playbook',
          icon: '🛡️',
          hasVisualizer: 'swe_rl',
          tag: '🔥 Core Hotspot',
          readTime: '30 min',
          summary: 'Frontier AI Labs 64x H100 分散式系統設計、叢集故障排查與生產事故急診',
          competencies: ['64x H100 Cluster Sizing', 'Fault Troubleshooting', 'Distributed Resharding', 'Production Loss Defense'],
          isCareerGated: true
        },
        {
          id: 'rlvr15',
          file: '15_lora_qlora_peft',
          num: '15',
          title: 'LoRA, QLoRA 與參數高效後訓練 (LoRA & QLoRA PEFT)',
          icon: '⚡',
          hasVisualizer: 'rlvr_lora_vram',
          tag: '🔥 Core Hotspot',
          readTime: '24 min',
          summary: '全量微調顯存牆、NF4 資訊理論直覺、雙重量化 DQ、分頁優化器與 70B 顯存精確公式',
          competencies: ['NF4 Information Theory', 'Double Quantization', 'VRAM Arithmetic', 'Zero-Slack Budgeting']
        },
        {
          id: 'rlvr16',
          file: '16_inference_optimization_quantization_compilation',
          num: '16',
          title: '推論極限優化 — 量化 (GPTQ/AWQ)、投機解碼與硬體編譯',
          icon: '🚀',
          hasVisualizer: 'rlvr_scaling',
          tag: 'Inference Opt',
          readTime: '26 min',
          summary: 'Roofline 模型、GPTQ/AWQ 4-bit 量化、投機解碼無損驗證定理、PagedAttention 與 Apple MLX 編譯',
          competencies: ['AWQ Activation Awareness', 'Speculative Decoding Lossless Proof', 'Apple MLX UMA Compilation']
        },
        {
          id: 'rlvr17',
          file: '17_llm_evaluation_benchmarking_prompt_optimization',
          num: '17',
          title: '評估基準與提示優化 — 準確率、延遲、安全與成本',
          icon: '📊',
          hasVisualizer: 'eval_workbench',
          tag: 'Evaluation 4D',
          readTime: '25 min',
          summary: '四維度平衡矩陣、G-Eval 機率加權、LLM-as-a-Judge 偏差消除、DSPy MIPROv2 提示編譯',
          competencies: ['4D Balance Matrix', 'G-Eval Continuous Scoring', 'DSPy Prompt Optimization']
        },
        {
          id: 'rlvr18',
          file: '18_alignment_safety_red_teaming',
          num: '18',
          title: 'RL 對齊、憲政 AI 與紅隊對抗防禦 (Safety & Red-Teaming)',
          icon: '🛡️',
          hasVisualizer: 'pitfalls',
          tag: 'Safety Alignment',
          readTime: '28 min',
          summary: 'PPO vs DPO vs GRPO 對比、Anthropic 憲政 AI、自動化紅隊 GCG/TAP 攻擊與 3 層架構安全邊界',
          competencies: ['Constitutional AI', 'Automated Red-Teaming (GCG/TAP)', 'Three-Layer Security Boundary']
        }
      ]
    }
  ],
  totalChapters: 18
};
