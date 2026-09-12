# Data Models & TypeScript Entity Interfaces Reference

This document outlines the active data structures, TypeScript interfaces, and loading pipelines powering the DeepAgents interactive learning platform in `frontend/`.

---

## 1. Directory Structure

```
frontend/src/entities/
├── chapter/
│   ├── chapterLoader.ts         # Dynamic ESM glob loader + FastAPI hydration fallback
│   ├── chapterStore.ts          # Zustand store (active site, chapter, search, sidebar)
│   └── data/
│       ├── deepagents/          # deepagents01.js ... deepagents31.js
│       ├── rl/                  # rl01.js ... rl18.js
│       └── rlvr/                # rlvr01.js ... rlvr18.js (Post-Training Track)
├── manifest/
│   ├── index.ts                 # MANIFESTS dictionary & getManifest(siteId)
│   ├── deepagentsManifest.ts    # DeepAgents 31-chapter curriculum manifest
│   ├── rlManifest.ts            # RL track manifest
│   └── rlvrManifest.ts          # Production Post-Training MLE Handbook manifest
├── simulation/
│   ├── labCatalog.ts            # 18-lab catalog specification (LabSpec)
│   ├── SimulationModal.tsx      # Unified full-screen simulator dialog
│   └── simulators/              # Dedicated visual parameter simulation components
└── milestone/
    ├── milestoneTutorials.ts    # Kaggle practice, STAR playbook, resume generators
    ├── MilestoneTutorialCard.tsx# Module completion card in ReaderCanvas
    └── MilestoneTutorialModal.tsx # Interactive career defense modal
```

---

## 2. Core TypeScript Interfaces (`frontend/src/shared/types/index.ts`)

### Curriculum & Chapter Schemas

```typescript
export type SiteId = 'deepagents' | 'rl' | 'rlvr';

export interface ChapterTocItem {
  level: number;
  text: string;
  anchor: string;
}

export interface CodeLineItem {
  line: number;
  text: string;
  title?: string;
  note?: string;
}

export interface ChapterSummary {
  id: string;
  num: string;
  title: string;
  file?: string;
  icon?: string;
  tag?: string;
  readTime?: string;
  summary?: string;
  competencies?: string[];
  hasVisualizer?: string;
  codeFile?: string;
}

export interface ChapterData extends ChapterSummary {
  file: string;
  html?: string;
  toc?: ChapterTocItem[];
  codeLines?: CodeLineItem[];
  markdownContent?: string;
}

export interface ManifestPart {
  id: string;
  label: string;
  icon?: string;
  description?: string;
  milestone?: string;
  jobTarget?: string;
  chapters: ChapterSummary[];
}

export interface SiteManifest {
  id: SiteId;
  name: string;
  shortName: string;
  description: string;
  icon: string;
  totalChapters: number;
  parts: ManifestPart[];
}
```

> [!IMPORTANT]
> **CodeLines Invariant**: Following the Post-Training Track standard, all chapter metadata files set `codeLines: []`. Code blocks are rendered cleanly inside notebook-style markdown code cells within the reading canvas.

---

## 3. Chapter Data Payloads (`frontend/src/entities/chapter/data/rlvr/rlvrXX.js`)

Each chapter metadata module exports a default `ChapterData` object:

```javascript
// rlvr/rlvr04.js
export default {
  "id": "rlvr04",
  "num": "04",
  "title": "輕量訓練管線 (Training Pipeline & Stabilization)",
  "icon": "⚡",
  "file": "04_training",
  "hasVisualizer": "rlvr_lora_vram",
  "readTime": "22 min",
  "summary": "Post-Training 實戰：16GB 單卡顯存預算精算、LoRA 低秩分解、GRPOConfig 訓練排程與四維遙測監控指標。",
  "codeLines": [],
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
      "text": "🤔 面試深度思辨與工業界陷阱",
      "anchor": "-面試深度思辨與工業界陷阱-interview-insight--production-pitfalls"
    }
  ]
};
```

---

## 4. Visual Lab Schema (`frontend/src/entities/simulation/labCatalog.ts`)

```typescript
export interface LabSpec {
  id: string;
  title: string;
  category: 'control' | 'alignment' | 'rlvr_systems' | 'reasoning';
  icon: string;
  chapterId: string;
  equation: string;
  description: string;
}
```

---

## 5. Kaggle Milestone Schema (`frontend/src/entities/milestone/milestoneTutorials.ts`)

```typescript
export interface MilestoneTutorial {
  id: string;
  partId: string;
  title: string;
  badge: string;
  track: string;
  kaggleContext: string;
  recommendedHardware: string;
  starPlaybook: {
    situation: string;
    task: string;
    action: string;
    result: string;
  };
  resumeBullets: string[];
  steps: Array<{
    stepNumber: number;
    title: string;
    objective: string;
    explanation: string;
    codeSnippet: string;
  }>;
}
```

---

## 6. Dynamic Loading Pipeline (`chapterLoader.ts`)

1. **Glob Discovery**:
   `const modules = import.meta.glob<{ default: ChapterData }>('./data/*/*.js');` loads modules dynamically.
2. **Hydration Fallback**:
   If the chapter has an external markdown tutorial file (`file: '04_training'`), the loader queries the FastAPI endpoint (`/api/rlvr/chapters/{file}` or `/api/deepagents/tutorials/{file}`) to inject fresh markdown into `data.markdownContent`.
