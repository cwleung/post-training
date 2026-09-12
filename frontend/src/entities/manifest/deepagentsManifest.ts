import type { SiteManifest } from '@/shared/types';

export const DEEPAGENTS_MANIFEST: SiteManifest = {
  id: 'deepagents',
  name: 'DeepAgents Harness & Evaluation',
  shortName: 'DeepAgents',
  icon: '🤖',
  description: '代理執行框架工程、中介軟體設計、長任務上下文管理與全生命週期評估體系。',
  parts: [
    {
      id: 'da_module1',
      label: 'Module I · 執行框架工程與中介軟體基礎',
      icon: '🏗️',
      description: '執行框架四大能力支柱、VFS 虛擬檔案系統與自訂攔截器生命週期',
      milestone: '建立沙箱 VFS 虛擬檔案系統與自訂中介軟體攔截器',
      jobTarget: 'Agent Platform Engineer (LangChain / Anthropic Tooling)',
      chapters: [
        {
          id: 'da01',
          file: '01-harness-engineering-basics',
          num: '01',
          title: '執行框架工程基礎 (Harness Engineering Basics)',
          icon: '🧩',
          hasVisualizer: 'agent_harness',
          tag: 'Core Concept',
          readTime: '16 min',
          summary: '微核心心智模型：Prompt 膨脹瓶頸、四大支柱與 create_deep_agent 設計表面直覺解析。',
          competencies: ['Harness Architecture', 'Prompt Inflation', 'Four Pillars']
        },
        {
          id: 'da02',
          file: '02-the-deep-agents-stack',
          num: '02',
          title: 'Deep Agents 堆疊 (The Deep Agents Stack)',
          icon: '🥞',
          hasVisualizer: 'agentic',
          tag: 'Architecture',
          readTime: '16 min',
          summary: '洋蔥防護盾心智模型：中介軟體順序語意、PatchToolCalls 與 Prompt Cache 凍結邊界。',
          competencies: ['Stack Composition', 'LangGraph Runtime', 'Layering Principles']
        },
        {
          id: 'da03',
          file: '03-custom-middleware',
          num: '03',
          title: '撰寫自訂中介軟體 (Custom Middleware)',
          icon: '🔌',
          hasVisualizer: 'agentic',
          tag: 'Extensibility',
          readTime: '18 min',
          summary: '安檢閘門心智模型：AgentMiddleware 攔截器生命週期、狀態變更與 wrap_tool_call 過濾。',
          competencies: ['AgentMiddleware', 'Tool Call Hooks', 'State Interception']
        },
        {
          id: 'da04',
          file: '04-backends-and-permissions',
          num: '04',
          title: '後端與權限機制 (Backends & Permissions)',
          icon: '🛡️',
          hasVisualizer: 'agent_vfs',
          tag: 'Security & VFS',
          readTime: '18 min',
          summary: '容器沙箱心智模型：StateBackend 狀態、StoreBackend 與 Path-level ACL 權限阻斷防禦。',
          competencies: ['VFS', 'StateBackend', 'Permission Guardrails']
        },
        {
          id: 'da05',
          file: '05-harness-profiles',
          num: '05',
          title: '執行框架設定檔 (Harness Profiles)',
          icon: '🎛️',
          hasVisualizer: 'agentic',
          tag: 'Profiles',
          readTime: '16 min',
          summary: '自適應變頻心智模型：模型專屬設定檔、Tool Exclusion 過濾與生產態隔離策略。',
          competencies: ['Environment Configuration', 'Tool Exclusion', 'Profile Switching']
        }
      ]
    },
    {
      id: 'da_module2',
      label: 'Module II · 代理構建、委派與人機協作',
      icon: '👥',
      description: '子代理委派調度、長文本上下文預算控制與人工介入中斷機制',
      milestone: '實作具備狀態隔離的子代理委派調度與人機介入 (HITL) 中斷恢復',
      jobTarget: 'Multi-Agent Systems Engineer (OpenAI / Cursor)',
      chapters: [
        {
          id: 'da06',
          file: '06-subagents-and-delegation',
          num: '06',
          title: '子代理與委派 (Subagents & Delegation)',
          icon: '👥',
          hasVisualizer: 'agentic',
          tag: 'Multi-Agent',
          readTime: '20 min',
          summary: '企業分工心智模型：task 工具調度、上下文資訊防火牆與專職子代理結果匯聚。',
          competencies: ['Subagent Delegation', 'Context Isolation', 'Task Dispatching']
        },
        {
          id: 'da07',
          file: '07-context-management',
          num: '07',
          title: '上下文管理 (Context Management)',
          icon: '📚',
          hasVisualizer: 'context_budget',
          tag: 'Context Budget',
          readTime: '22 min',
          summary: '記憶體分頁交換心智模型：漸進式 Skills、長文本自動轉存 VFS (>20k tokens) 與對話摘要。',
          competencies: ['Skill Injection', 'Context Offloading', 'Prompt Cache']
        },
        {
          id: 'da08',
          file: '08-human-in-the-loop',
          num: '08',
          title: '方向掌控：人工介入 (Human-in-the-Loop)',
          icon: '✋',
          hasVisualizer: 'agentic',
          tag: 'HITL Control',
          readTime: '18 min',
          summary: '駕駛艙警報心智模型：高風險操作 interrupt_on、檢查點回滾、手動批准與狀態覆寫。',
          competencies: ['HITL Control', 'Interruption Handling', 'State Overrides']
        },
        {
          id: 'da09',
          file: '09-putting-it-together',
          num: '09',
          title: '整合應用 (Putting It Together)',
          icon: '🚀',
          hasVisualizer: 'eval_workbench',
          tag: 'Full Assembly',
          readTime: '25 min',
          summary: '交響樂團總指揮心智模型：端到端真實研究 Agent：VFS + Skills + 子代理 + HITL 全拓撲。',
          competencies: ['Full Agent Assembly', 'End-to-End Orchestration']
        },
        {
          id: 'da10',
          file: '10-middleware-deep-dive',
          num: '10',
          title: '中介軟體深入探討 (Middleware Deep Dive)',
          icon: '🔍',
          hasVisualizer: 'agentic',
          tag: 'Internals',
          readTime: '18 min',
          summary: '程式化路由器心智模型：jump_to 指令跳轉、非同步事件管線調度與 Command 狀態變異。',
          competencies: ['Async Pipeline', 'Flow Redirection', 'Fault Tolerance']
        }
      ]
    },
    {
      id: 'da_module3',
      label: 'Module III · 多輪軌跡評估、Rubrics 與安全防護',
      icon: '🔐',
      description: '多輪軌跡比對、動態評判量表、MAST 崩潰模式分類與紅隊對抗評估',
      milestone: '建立 CI/CD 軌跡斷言、動態 Rubrics 評判與 Prompt 注入紅隊防禦',
      jobTarget: 'AI Safety & Evaluation Engineer (DeepEval / Anthropic)',
      chapters: [
        {
          id: 'da11',
          file: '11-trajectory-evaluation',
          num: '11',
          title: '軌跡評估 (Trajectory Evaluation)',
          icon: '📈',
          hasVisualizer: 'eval_workbench',
          tag: 'Trajectory Eval',
          readTime: '20 min',
          summary: '黑盒子記錄儀心智模型：工具呼叫序列驗證、步驟效率評級與確定性狀態轉移斷言。',
          competencies: ['Trajectory Matching', 'Step Efficiency', 'AgentEvals']
        },
        {
          id: 'da12',
          file: '12-rubric-grading',
          num: '12',
          title: '執行期 Rubric 評判 (Runtime Rubric Grading)',
          icon: '📋',
          hasVisualizer: 'rubric_eval',
          tag: 'Rubrics',
          readTime: '22 min',
          summary: '大學聯考規準心智模型：動態評分量表注入、執行期自我反思評定與防護門阻斷。',
          competencies: ['Rubric Grading', 'Judge Subagents', 'Self-Critique']
        },
        {
          id: 'da13',
          file: '13-end-to-end-benchmarking',
          num: '13',
          title: '端到端評估與基準測試 (End-to-End Benchmarking & CI/CD)',
          icon: '🧪',
          hasVisualizer: 'eval_workbench',
          tag: 'Benchmarking',
          readTime: '20 min',
          summary: '實體碰撞測試心智模型：沙箱 VFS 副作用驗收、自動化測試套件與 CI/CD 阻斷管線。',
          competencies: ['CI/CD Gatekeepers', 'Regression Testing', 'Side-Effect Validation']
        },
        {
          id: 'da14',
          file: '14-event-streaming',
          num: '14',
          title: '即時事件串流與可觀測性 (Event Streaming & Observability)',
          icon: '🌊',
          hasVisualizer: 'eval_workbench',
          tag: 'Streaming',
          readTime: '18 min',
          summary: '即時心電圖心智模型：SSE 串流協議、Typed Projections 與 OpenTelemetry Spans 整合。',
          competencies: ['SSE Streaming', 'Typed Projections', 'Telemetry Spans']
        },
        {
          id: 'da15',
          file: '15-multi-agent-failure-taxonomy',
          num: '15',
          title: '多代理故障分類學與統計評估 (MAST & Statistical Reliability)',
          icon: '⚠️',
          hasVisualizer: 'pitfalls',
          tag: 'Failure Taxonomy',
          readTime: '24 min',
          summary: '故障樹心智模型：級聯失敗、委派死鎖等 MAST 14 類崩潰模式與 pass@k 統計可靠度。',
          competencies: ['MAST Taxonomy', 'Deadlock Detection', 'pass@k vs pass^k']
        },
        {
          id: 'da16',
          file: '16-security-red-teaming',
          num: '16',
          title: '安全紅隊與對抗性評估 (Agent Red-Teaming & Safety Evals)',
          icon: '🎯',
          hasVisualizer: 'pitfalls',
          tag: 'Red Teaming',
          readTime: '22 min',
          summary: '特洛伊木馬心智模型：間接提示詞注入、工具毒化攻擊與雙層 LLM 隔離審查架構。',
          competencies: ['Indirect Prompt Injection', 'Tool Poisoning', 'Attack Success Rate']
        }
      ]
    },
    {
      id: 'da_module4',
      label: 'Module IV · 評估即 RL 環境、裁判對齊與基準誠信',
      icon: '⚖️',
      description: '確定性 Verifiers、Agent-as-a-Judge 統計校準、評估基礎設施與防污染',
      milestone: '設計確定性 Verifiers、Kappa 校準的 G-Eval 裁判與防污染黃金測試集',
      jobTarget: 'Evaluation & Benchmarks MLE (Meta FAIR / OpenAI)',
      chapters: [
        {
          id: 'da17',
          file: '17-eval-as-rl-environment',
          num: '17',
          title: '評估即 RL 環境：Verifier 設計與可驗證獎勵 (Eval as RL Environment)',
          icon: '🎯',
          hasVisualizer: 'eval_workbench',
          tag: 'RLVR & Verifier',
          readTime: '25 min',
          summary: 'Agent Gymnasium 心智模型：確定性驗證器、正則獎勵回報、難度梯度與反思信號。',
          competencies: ['Verifier Engineering', 'Deterministic Checkers', 'RL Decomposition']
        },
        {
          id: 'da18',
          file: '18-llm-as-judge-alignment',
          num: '18',
          title: 'Agent-as-Judge、裁判校準與 RLVR 接入 (LLM-as-a-Judge Calibration)',
          icon: '⚖️',
          hasVisualizer: 'rubric_eval',
          tag: 'G-Eval & Judge',
          readTime: '26 min',
          summary: '陪審團校準心智模型：Auto-CoT 生成、對數機率期望值與 Cohen\'s Kappa 裁判一致性。',
          competencies: ['G-Eval', 'Judge Calibration', 'Cohen Kappa Agreement']
        },
        {
          id: 'da19',
          file: '19-eval-infrastructure',
          num: '19',
          title: '評估基礎設施：資料集、CI 流水線與評估驅動開發 (Eval Infrastructure & EDD)',
          icon: '🏗️',
          hasVisualizer: 'eval_workbench',
          tag: 'Infrastructure',
          readTime: '22 min',
          summary: '晶圓廠良率心智模型：Golden Datasets 版本控制、評估驅動開發 (EDD) 與回歸門檻。',
          competencies: ['Eval Infrastructure', 'EDD', 'Golden Dataset Hygiene']
        },
        {
          id: 'da20',
          file: '20-harness-as-context',
          num: '20',
          title: '執行框架工程：上下文即是代理 (Harness-as-Context)',
          icon: '🧠',
          hasVisualizer: 'context_budget',
          tag: 'Context Harness',
          readTime: '24 min',
          summary: '極簡儀表板心智模型：Model/Harness/Skill 三元分解、工具接口人機工學與提示詞壓縮。',
          competencies: ['Context Engineering', 'Triad Architecture', 'Tool Interface Ergonomics']
        },
        {
          id: 'da21',
          file: '21-benchmark-integrity',
          num: '21',
          title: '基準測試完整性：污染、飽和與排行榜博弈 (Benchmark Integrity)',
          icon: '📊',
          hasVisualizer: 'eval_workbench',
          tag: 'Integrity',
          readTime: '22 min',
          summary: '防弊考場心智模型：MinHash 13-gram 去污染、基準飽和對抗與 Goodhart 定律規避。',
          competencies: ['Decontamination', 'Benchmark Saturation', 'Goodhart Avoidance']
        }
      ]
    },
    {
      id: 'da_module5',
      label: 'Module V · 全鏈路可觀測性、長任務與生產經濟學',
      icon: '🔭',
      description: 'OTel 五維追蹤、根本原因錯誤分析、RAG 三角驗收與 Token 成本優化',
      milestone: '整合 OTel 五維追蹤、RAG 三角驗收與 Token 成本/延遲 Pareto 最優化',
      jobTarget: 'Production GenAI Systems Engineer (Apple / Datadog)',
      chapters: [
        {
          id: 'da22',
          file: '22-observability-five-surfaces',
          num: '22',
          title: '可觀測性：五個評估面與 OTel 代理追蹤 (Five Observability Surfaces)',
          icon: '📡',
          hasVisualizer: 'eval_workbench',
          tag: 'Observability',
          readTime: '24 min',
          summary: '全景斷層掃描心智模型：輸出/軌跡/記憶/環境/機理五層無死角監控與 OTel 跨進程整合。',
          competencies: ['Five Observability Surfaces', 'OpenTelemetry GenAI Spans']
        },
        {
          id: 'da23',
          file: '23-error-analysis',
          num: '23',
          title: '錯誤分析：最高 ROI 的代理改善活動 (Error Analysis & RCA)',
          icon: '🔍',
          hasVisualizer: 'pitfalls',
          tag: 'Error Analysis',
          readTime: '22 min',
          summary: '空難解碼心智模型：軸向編碼 (Axial Coding)、根因分類樹與失敗模式 Pareto 優先級。',
          competencies: ['Root Cause Analysis', 'Error Taxonomy', 'Pareto Prioritization']
        },
        {
          id: 'da24',
          file: '24-multi-turn-evaluation',
          num: '24',
          title: '多輪次與長任務評估 (Multi-Turn Evaluation & Long Horizon)',
          icon: '⏱️',
          hasVisualizer: 'eval_workbench',
          tag: 'Long Horizon',
          readTime: '22 min',
          summary: '遠洋航偏校準心智模型：METR 時間視野評估、目標漂移檢測與累積雪崩效應抑制。',
          competencies: ['Long-Horizon Evals', 'METR Horizons', 'Drift Dampening']
        },
        {
          id: 'da25',
          file: '25-rag-evaluation',
          num: '25',
          title: 'RAG 評估：檢索增強代理的三角驗收框架 (RAG Triad Evaluation)',
          icon: '📐',
          hasVisualizer: 'eval_workbench',
          tag: 'RAG Triad',
          readTime: '22 min',
          summary: '法庭證據鏈心智模型：脈絡相關性、紮實度與回答相關性三權分立之原子命題驗收。',
          competencies: ['RAG Triad', 'Faithfulness Evaluation', 'Retriever vs Generator Breakdown']
        },
        {
          id: 'da26',
          file: '26-production-monitoring',
          num: '26',
          title: '生產監控與分佈偏移偵測 (Production Monitoring & Drift)',
          icon: '🚨',
          hasVisualizer: 'eval_workbench',
          tag: 'Monitoring',
          readTime: '24 min',
          summary: '電網保險絲心智模型：KL 散度輸入偏移監測、異常軌跡即時熔斷與生產資料回流。',
          competencies: ['Distribution Drift', 'Circuit Breakers', 'Live Trace Diffing']
        },
        {
          id: 'da27',
          file: '27-cost-latency-metrics',
          num: '27',
          title: '成本與延遲：代理評估的一級指標 (Cost & Latency Economics)',
          icon: '💰',
          hasVisualizer: 'eval_workbench',
          tag: 'Cost & Latency',
          readTime: '20 min',
          summary: '賽車輕量化心智模型：Token 經濟學、TTFT 延遲分解、Prompt Cache 與 Pareto 最優前緣。',
          competencies: ['Cost Economics', 'Pareto Frontier', 'Latency Budgeting']
        }
      ]
    },
    {
      id: 'da_module6',
      label: 'Module VI · 前沿自改進迴圈、確定性除錯與實戰庫',
      icon: '🔁',
      description: '工具呼叫四層評估棧、LangGraph 確定性時空回放、DSPy MIPROv2 自編譯與可執行工程範例',
      milestone: '達成 Reflexion + DSPy MIPROv2 自編譯全閉環與可執行工程庫',
      jobTarget: 'Senior / Lead Agent Architect (Anthropic / OpenAI)',
      chapters: [
        {
          id: 'da28',
          file: '28-runnable-examples',
          num: '28',
          title: '可執行案例庫與工程實戰 (Runnable Harness Catalog)',
          icon: '💻',
          hasVisualizer: 'agentic',
          tag: 'Harness Modules',
          readTime: '18 min',
          summary: '工程工具箱心智模型：22 個核心 Python Harness 模組圖鑑、架構封裝與可測試性實踐。',
          competencies: ['Harness Architecture Modules', 'Dependency Injection', 'Testable Agent Patterns']
        },
        {
          id: 'da29',
          file: '29-tool-call-evaluation-stack',
          num: '29',
          title: '工具呼叫四層評估棧 (Tool Call Evaluation Stack)',
          icon: '🪜',
          hasVisualizer: 'eval_workbench',
          tag: 'Tool Evaluation',
          readTime: '25 min',
          summary: '航太檢驗心智模型：選擇層、抽取層、消化層與恢復層之四層穿透驗收架構。',
          competencies: ['Four-Layer Tool Stack', 'Schema Conformance', 'Execution Sandbox']
        },
        {
          id: 'da30',
          file: '30-checkpoint-replay',
          num: '30',
          title: '檢查點回放與確定性除錯 (Checkpoint Replay)',
          icon: '⏪',
          hasVisualizer: 'eval_workbench',
          tag: 'Replay Debugging',
          readTime: '24 min',
          summary: '時光機重演心智模型：LangGraph Checkpoint 狀態重放、Mock 副作用與確定性時空回溯。',
          competencies: ['Checkpoint Replay', 'Deterministic Debugging', 'Time-Travel Testing']
        },
        {
          id: 'da31',
          file: '31-self-improving-eval-loop',
          num: '31',
          title: '自改進代理評估迴圈 (Self-Improving Eval Loop & DSPy)',
          icon: '🧬',
          hasVisualizer: 'eval_workbench',
          tag: 'DSPy & MIPROv2',
          readTime: '28 min',
          summary: '自適應有機體心智模型：線上反思修正 (Reflexion) 與離線提示自動編譯 (DSPy MIPROv2)。',
          competencies: ['Reflexion Loops', 'DSPy Compilation', 'MIPROv2 Auto-Optimization']
        }
      ]
    }
  ],
  totalChapters: 31
};
