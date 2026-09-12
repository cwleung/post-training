import type { SiteManifest } from '@/shared/types';

export const RL_MANIFEST: SiteManifest = {
  id: 'rl',
  name: 'Reinforcement Learning & Alignment',
  shortName: 'Research RL',
  icon: '🧠',
  description: '全棧強化學習與大模型後訓練交互式學習平台：18 完整教學章節、17 實時參數仿真實驗室。',
  parts: [
    {
      id: 'rl_stage1',
      label: 'Stage I · 經典強化學習與價值/策略基石',
      icon: '🕹️',
      description: '從 MDP、動態規劃到無模型強化學習、連續控制與離線保守策略',
      milestone: '推導 MDP 物理動力學、方差縮減定理與最大熵連續控制',
      jobTarget: 'Robotics / RL Researcher (DeepMind / Wayve)',
      chapters: [
        {
          id: 'ch01',
          file: '01_cartpole_physics_and_control',
          num: '01',
          title: 'CartPole 物理動力學與基礎控制 (CartPole Dynamics & Control)',
          icon: '🛒',
          hasVisualizer: 'cartpole',
          tag: 'MDP & Control',
          readTime: '15 min',
          summary: '倒立擺物理平衡動力學推導、拉格朗日運動微分方程與狀態空間離散化控制',
          competencies: ['MDP', 'Lagrangian Dynamics', 'Classical Control']
        },
        {
          id: 'ch02',
          file: '02_bandit_exploration_ucb',
          num: '02',
          title: '探索與利用·多臂老虎機與 UCB (Bandit Exploration & UCB1)',
          icon: '🎰',
          hasVisualizer: 'bandit',
          tag: 'Exploration',
          readTime: '15 min',
          summary: 'ε-Greedy、Hoeffding 不等式、Upper Confidence Bound (UCB1) 與累積遺憾界限推導',
          competencies: ['Exploration-Exploitation', 'UCB1', 'Regret Bound']
        },
        {
          id: 'ch03',
          file: '03_dqn_replay_target',
          num: '03',
          title: '深度 Q 網絡·Replay Buffer 與 Target Network (DQN & Stability)',
          icon: '🧠',
          hasVisualizer: 'dqn',
          tag: 'Value-Based',
          readTime: '18 min',
          summary: '經驗回放機制、目標網絡同步週期、Bellman 算子收縮性與 TD-Error 穩定性',
          competencies: ['DQN', 'Replay Buffer', 'TD-Error Stability']
        },
        {
          id: 'ch04',
          file: '04_policy_gradient_reinforce',
          num: '04',
          title: '策略梯度·REINFORCE 與方差縮減 (Policy Gradient Theorem)',
          icon: '📈',
          hasVisualizer: 'policy_gradient',
          tag: 'Policy Gradient',
          readTime: '18 min',
          summary: '似然率梯度定理 (Policy Gradient Theorem)、對數導數技巧與 Baseline 方差削減',
          competencies: ['Likelihood Ratio', 'Baseline Variance Reduction', 'REINFORCE']
        },
        {
          id: 'ch05',
          file: '05_actor_critic_gae',
          num: '05',
          title: 'Actor-Critic 架構與廣義優勢估計 (Actor-Critic & GAE)',
          icon: '🎭',
          hasVisualizer: 'actor_critic',
          tag: 'Actor-Critic',
          readTime: '18 min',
          summary: '策略與價值雙網絡協同、自舉 TD 誤差與廣義優勢估計 GAE(γ, λ) 數學證明',
          competencies: ['Actor-Critic', 'GAE', 'Advantage Estimation']
        },
        {
          id: 'ch06',
          file: '06_sac_continuous_control',
          num: '06',
          title: '連續動作控制·Soft Actor-Critic (SAC & MaxEnt RL)',
          icon: '🏎️',
          hasVisualizer: 'continuous',
          tag: 'Continuous Control',
          readTime: '20 min',
          summary: '最大熵強化學習框架、Tanh 高斯動作重參數化技巧與自動溫度調節',
          competencies: ['Maximum Entropy RL', 'Reparameterization Trick', 'SAC']
        },
        {
          id: 'ch07',
          file: '07_cql_offline_rl',
          num: '07',
          title: '離線強化學習·Conservative Q-Learning (CQL Offline RL)',
          icon: '💾',
          hasVisualizer: 'offline',
          tag: 'Offline RL',
          readTime: '22 min',
          summary: '分佈偏移 (Distribution Shift)、分佈外 (OOD) 動作懲罰與保守價值估計',
          competencies: ['Offline RL', 'Distribution Shift', 'CQL']
        }
      ]
    },
    {
      id: 'rl_stage2',
      label: 'Stage II · 大模型強化學習後訓練與對齊',
      icon: '⚡',
      description: '大語言模型 Post-Training 核心對齊算法、隱式偏好與獎勵工程',
      milestone: '掌握 PPO Clip 信任域、DPO 隱式對齊、GRPO 組歸一化與 Freeze Gate 守衛',
      jobTarget: 'Post-Training Alignment Engineer (Anthropic / OpenAI)',
      chapters: [
        {
          id: 'ch08',
          file: '08_ppo_clipped_objective',
          num: '08',
          title: '近端策略優化·PPO 截斷目標與重要性採樣 (PPO Clipped Objective)',
          icon: '📐',
          hasVisualizer: 'ppo',
          tag: 'PPO & Alignment',
          readTime: '22 min',
          summary: 'PPO-Clip 目標函數、重要性採樣比率剪裁與信任域約束',
          competencies: ['PPO-Clip', 'Importance Sampling', 'Trust Region']
        },
        {
          id: 'ch09',
          file: '09_dpo_implicit_reward',
          num: '09',
          title: '直接偏好優化·DPO 隱式獎勵與對齊 (Direct Preference Optimization)',
          icon: '⚖️',
          hasVisualizer: 'dpo',
          tag: 'DPO Alignment',
          readTime: '20 min',
          summary: 'Bradley-Terry 偏好模型推導、隱式獎勵替換與解析閉式解',
          competencies: ['DPO', 'Bradley-Terry Model', 'Implicit Reward']
        },
        {
          id: 'ch10',
          file: '10_grpo_group_relative',
          num: '10',
          title: '群體相對策略優化·GRPO 組間歸一化 (DeepSeek R1 GRPO)',
          icon: '👥',
          hasVisualizer: 'grpo',
          tag: 'DeepSeek R1',
          readTime: '25 min',
          summary: 'DeepSeek R1 核心算法：無需 Critic 的組內 Z-Score 優勢歸一化與 KL 懲罰',
          competencies: ['GRPO', 'Group Normalization', 'DeepSeek R1 Architecture']
        },
        {
          id: 'ch11',
          file: '11_rlhf_3stage_pipeline',
          num: '11',
          title: '三階段對齊·RLHF 獎勵建模與邊界 (3-Stage RLHF Pipeline)',
          icon: '🛡️',
          hasVisualizer: 'rlhf',
          tag: 'RLHF 3-Stage',
          readTime: '22 min',
          summary: 'SFT → Reward Modeling → PPO 完整流程與 KL 散度約束邊界',
          competencies: ['RLHF Pipeline', 'Reward Modeling', 'KL Divergence Boundary']
        },
        {
          id: 'ch12',
          file: '12_reward_hacking_pitfalls',
          num: '12',
          title: '獎勵黑客與失敗模式·Reward Hacking & Pitfalls',
          icon: '🚨',
          hasVisualizer: 'pitfalls',
          tag: 'Safety & Guardrails',
          readTime: '18 min',
          summary: 'Goodhart 定律、長度欺騙膨脹、多樣性崩塌與 Freeze Gate 守衛',
          competencies: ['Goodhart Law', 'Length Bias', 'Freeze Gatekeeper']
        }
      ]
    },
    {
      id: 'rl_stage3',
      label: 'Stage III · 測試期計算擴展與過程引導',
      icon: '🔮',
      description: 'Test-Time Compute、CoT 思考鏈與過程獎勵模型 (PRM) 狀態樹剪枝',
      milestone: '實踐 Thinking Tokens 擴展組合數學與 PRM 束搜索剪枝機制',
      jobTarget: 'Reasoning & Search Research Scientist (OpenAI o1 / DeepSeek)',
      chapters: [
        {
          id: 'ch13',
          file: '13_test_time_compute_passk',
          num: '13',
          title: '測試期擴展·Test-Time Scaling 與 Pass@K (Inference Compute)',
          icon: '📈',
          hasVisualizer: 'reasoning',
          tag: 'Test-Time Scaling',
          readTime: '20 min',
          summary: 'Inference-Time Compute 擴展定律、Thinking Tokens 與 Pass@K 組合數學',
          competencies: ['Test-Time Scaling', 'Pass@k Combinatorics', 'Compute Budget']
        },
        {
          id: 'ch14',
          file: '14_prm_step_supervision_search',
          num: '14',
          title: '過程獎勵模型·PRM 逐步推理與束搜索剪枝 (Process Reward Models)',
          icon: '🌲',
          hasVisualizer: 'prm',
          tag: 'PRM & Search',
          readTime: '24 min',
          summary: 'Step-level Reward Modeling、Beam Search 狀態樹展開與死枝剪枝',
          competencies: ['Process Supervision', 'Beam Search Pruning', 'Math-Shepherd']
        }
      ]
    },
    {
      id: 'rl_stage4',
      label: 'Stage IV · Agentic 強化學習與真實環境交互',
      icon: '🤖',
      description: '多輪軌跡信用分配、軟體工程沙箱 SWE-RL 與多模態視覺 Agent',
      milestone: '搭建 Fast-Loop 經驗提煉、SWE-bench 沙箱補丁懲罰與 VLM KV 顯存建模',
      jobTarget: 'Agentic AI / Coding Agent Researcher (Cognition / Anthropic)',
      chapters: [
        {
          id: 'ch15',
          file: '15_in_context_policy_harness',
          num: '15',
          title: '上下文策略學習·In-Context Policy & Harness (Agent Policy)',
          icon: '🔄',
          hasVisualizer: 'agentic',
          tag: 'Harness Engineering',
          readTime: '25 min',
          summary: 'Era of Experience: Fast-loop harness 經驗提煉與 KEEP/AVOID 注入',
          competencies: ['In-Context Policy', 'Experience Distillation', 'Prompt Feedback Loop']
        },
        {
          id: 'ch16',
          file: '16_swe_rl_sandbox_patch',
          num: '16',
          title: '軟體工程 Agent·SWE-RL 沙箱與補丁懲罰 (SWE-Bench RL)',
          icon: '💻',
          hasVisualizer: 'swe_rl',
          tag: 'SWE-Bench RL',
          readTime: '22 min',
          summary: 'SWE-bench 自動化測試回報、程式碼修改行數懲罰與沙箱環境評估',
          competencies: ['SWE-Bench Sandbox', 'Patch Penalties', 'Code Generation RL']
        },
        {
          id: 'ch17',
          file: '17_vlm_multimodal_rl_kvcache',
          num: '17',
          title: '多模態視覺 Agent·VLM RL 與 KV Cache 內存建模 (Multimodal RL)',
          icon: '👁️',
          hasVisualizer: 'vlm',
          tag: 'Multimodal RL',
          readTime: '20 min',
          summary: 'Visual Tokens 編碼切片、多模態長上下文 KV Cache 顯存佔用建模',
          competencies: ['Vision-Language RL', 'Token Slicing', 'KV-Cache Memory Modeling']
        }
      ]
    },
    {
      id: 'rl_stage5',
      label: 'Stage V · 前沿架構擴展與系統設計對齊 (Frontier Systems Design)',
      icon: '🎯',
      description: 'Frontier Lab 系統架構規格對齊、L4/L5 路線圖與架構決策防禦',
      milestone: '完成前沿實驗室大模型架構規格逐條對齊與系統設計防禦',
      jobTarget: 'Senior / Staff Machine Learning Engineer (Frontier Labs)',
      chapters: [
        {
          id: 'ch18',
          file: '18_frontier_systems_architecture_alignment',
          num: '18',
          title: '前沿架構演進·系統架構路線圖與前沿標準對齊 (Frontier Systems Architecture)',
          icon: '🏛️',
          hasVisualizer: 'agentic',
          tag: 'Systems & Architecture',
          readTime: '30 min',
          summary: '前沿實驗室大模型分散式系統架構規格、擴展邊界與設計權衡對齊',
          competencies: ['Systems Architecture', 'Distributed Scaling Limits', 'Architecture Defense'],
          isCareerGated: true
        }
      ]
    }
  ],
  totalChapters: 18
};
