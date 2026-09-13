// frontend/src/entities/milestone/MilestoneTutorialModal.tsx
import React, { useState, useEffect } from 'react';
import {
  Trophy,
  Copy,
  Check,
  ChevronRight,
  Briefcase,
  Terminal,
  FileCode,
  FileText,
  HelpCircle,
  Sparkles,
  Cpu,
  Clock,
  Target,
  Award,
  ShieldCheck,
  CheckCircle2,
  Layers,
  ListOrdered,
  WrapText,
} from 'lucide-react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/shared/ui/Dialog';
import { Badge } from '@/shared/ui/Badge';
import { cn } from '@/shared/lib/utils';
import { getMilestoneTutorial, type MilestoneTutorial } from './milestoneTutorials';
import { usePrivacyStore } from '@/entities/chapter/privacyStore';
import type { ManifestPart } from '@/shared/types';

interface MilestoneTutorialModalProps {
  part: ManifestPart | null;
  isOpen: boolean;
  onClose: () => void;
}

type TabKey = 'pipeline' | 'star' | 'interview' | 'resume';
type ViewMode = 'all' | 'single' | 'script';

export const MilestoneTutorialModal: React.FC<MilestoneTutorialModalProps> = ({
  part,
  isOpen,
  onClose,
}) => {
  const { isUnlocked } = usePrivacyStore();
  const [activeTab, setActiveTab] = useState<TabKey>('pipeline');
  const [viewMode, setViewMode] = useState<ViewMode>('all');
  const [activeStep, setActiveStep] = useState<number>(0);
  const [copiedStepIdx, setCopiedStepIdx] = useState<number | null>(null);
  const [copiedAllCode, setCopiedAllCode] = useState<boolean>(false);
  const [copiedBulletIdx, setCopiedBulletIdx] = useState<number | null>(null);
  const [copiedStar, setCopiedStar] = useState<boolean>(false);
  const [isCodeWrapped, setIsCodeWrapped] = useState<boolean>(false);

  useEffect(() => {
    if (!isUnlocked && activeTab !== 'pipeline') {
      setActiveTab('pipeline');
    }
  }, [isUnlocked, activeTab]);

  if (!part || !isOpen) return null;

  const tutorial: MilestoneTutorial = getMilestoneTutorial(part);
  const currentStep = tutorial.steps[activeStep] || tutorial.steps[0];

  const getFullPipelineScript = () => {
    return [
      `# ==============================================================================`,
      `# ${tutorial.title}`,
      `# Kaggle Notebook Showcase: ${tutorial.kaggleNotebook}`,
      `# Target Hardware: ${tutorial.hardwareRequirements}`,
      `# Expected Runtime: ${tutorial.expectedRuntime}`,
      `# ==============================================================================\n`,
      ...tutorial.steps.map((s) => (
        `# ------------------------------------------------------------------------------\n` +
        `# Step ${s.stepNumber}: ${s.title} [${s.badge}]\n` +
        `# Objective: ${s.objective}\n` +
        `# ------------------------------------------------------------------------------\n` +
        `${s.codeSnippet}\n`
      )),
      `# ------------------------------------------------------------------------------`,
      `# Pipeline Execution Complete!`,
      `# Milestone achieved: ${part.milestone || tutorial.title}`,
      `# ==============================================================================`
    ].join('\n');
  };

  const handleCopyCode = (code?: string, idx?: number) => {
    if (!code) return;
    navigator.clipboard.writeText(code);
    const stepIdx = idx !== undefined ? idx : activeStep;
    setCopiedStepIdx(stepIdx);
    setTimeout(() => setCopiedStepIdx(null), 2000);
  };

  const handleCopyAllPipelineCode = () => {
    const fullScript = getFullPipelineScript();
    navigator.clipboard.writeText(fullScript);
    setCopiedAllCode(true);
    setTimeout(() => setCopiedAllCode(false), 2200);
  };

  const scrollToAnchor = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const handleCopyBullet = (bullet: string, idx: number) => {
    navigator.clipboard.writeText(bullet);
    setCopiedBulletIdx(idx);
    setTimeout(() => setCopiedBulletIdx(null), 2000);
  };

  const handleCopyStarPlaybook = () => {
    const starText = [
      `【${tutorial.title} · STAR 答辯架構】`,
      `目標崗位：${tutorial.targetRole}`,
      ``,
      `[S - Situation 背景與挑戰]`,
      tutorial.starPlaybook.situation,
      ``,
      `[T - Task 目標與約束]`,
      tutorial.starPlaybook.task,
      ``,
      `[A - Action 工程解法]`,
      tutorial.starPlaybook.action,
      ``,
      `[R - Result 量化成果]`,
      tutorial.starPlaybook.result,
    ].join('\n');

    navigator.clipboard.writeText(starText);
    setCopiedStar(true);
    setTimeout(() => setCopiedStar(false), 2000);
  };

  const bulletVariants = tutorial.portfolioBulletVariants || [
    { label: '標準工程成果摘要 (STAR)', bullet: tutorial.portfolioBullet },
  ];

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="w-[calc(100vw-1rem)] sm:w-full max-w-4xl max-h-[92vh] overflow-y-auto overflow-x-hidden min-w-0 max-w-full flex flex-col p-3.5 sm:p-6 bg-card/95 backdrop-blur-md border border-border shadow-2xl rounded-2xl text-foreground">
        {/* Modal Header */}
        <DialogHeader className="pb-4 border-b border-border min-w-0 max-w-full">
          <div className="flex flex-wrap items-start justify-between gap-3 min-w-0">
            <div className="flex items-start gap-3 min-w-0 flex-1">
              <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-gradient-to-tr from-amber-500 via-orange-500 to-amber-600 text-white shadow-lg shadow-amber-500/25">
                <Trophy className="h-5 w-5" />
              </div>
              <div className="space-y-1 min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2 min-w-0">
                  <DialogTitle className="text-base sm:text-lg font-bold text-foreground font-title leading-snug break-words">
                    {tutorial.title}
                  </DialogTitle>
                  <Badge variant="amber" className="text-[10px] px-1.5 py-0.5 shrink-0">
                    {tutorial.badge}
                  </Badge>
                </div>
                <DialogDescription className="text-xs text-muted-foreground flex flex-wrap items-center gap-2 min-w-0">
                  {isUnlocked && (
                    <>
                      <span className="inline-flex items-center gap-1 text-primary font-medium truncate">
                        <Briefcase className="h-3.5 w-3.5 shrink-0" />
                        <span className="truncate">架構定位：{tutorial.targetRole}</span>
                      </span>
                      <span className="text-muted-foreground/60">•</span>
                    </>
                  )}
                  <span className="inline-flex items-center gap-1 text-amber-500 dark:text-amber-300/90 shrink-0">
                    <Cpu className="h-3 w-3" />
                    <span>{tutorial.hardwareRequirements}</span>
                  </span>
                  <span className="text-muted-foreground/60">•</span>
                  <span className="inline-flex items-center gap-1 text-emerald-500 dark:text-emerald-300/90 shrink-0">
                    <Clock className="h-3 w-3" />
                    <span>{tutorial.expectedRuntime}</span>
                  </span>
                </DialogDescription>
              </div>
            </div>

            {/* Header Action Buttons */}
            <div className="flex items-center gap-2 shrink-0">
              <button
                type="button"
                onClick={handleCopyAllPipelineCode}
                className="flex items-center gap-1.5 rounded-lg border border-amber-500/40 bg-amber-500/10 px-2.5 py-1 text-xs font-semibold text-amber-500 dark:text-amber-300 hover:bg-amber-500/20 transition-all cursor-pointer shadow-sm shrink-0"
                title="一鍵複製完整 4 步驟 Python 實戰腳本"
              >
                {copiedAllCode ? (
                  <>
                    <Check className="h-3.5 w-3.5 text-emerald-500" />
                    <span className="text-emerald-500 font-bold">已複製 4 步驟腳本</span>
                  </>
                ) : (
                  <>
                    <Copy className="h-3.5 w-3.5" />
                    <span>一鍵複製端到端腳本</span>
                  </>
                )}
              </button>

              {isUnlocked && (
                <button
                  type="button"
                  onClick={() => handleCopyBullet(tutorial.portfolioBullet, -1)}
                  className="flex items-center gap-1.5 rounded-lg border border-primary/30 bg-primary/10 px-2.5 py-1 text-xs font-medium text-primary hover:bg-primary/20 transition-all cursor-pointer shrink-0"
                  title="複製工程實踐亮點 (Technical Highlight)"
                >
                  {copiedBulletIdx === -1 ? (
                    <>
                      <Check className="h-3.5 w-3.5 text-emerald-500" />
                      <span className="text-emerald-500">已複製</span>
                    </>
                  ) : (
                    <>
                      <FileText className="h-3.5 w-3.5" />
                      <span>複製工程成果</span>
                    </>
                  )}
                </button>
              )}
            </div>
          </div>

          <p className="mt-2.5 text-xs text-muted-foreground leading-relaxed break-words">
            {tutorial.overview}
          </p>
        </DialogHeader>

        {/* 4-Tab Navigation Header (Filtered when locked) */}
        <div className="flex items-center gap-2 overflow-x-auto pb-1 pt-2 border-b border-border/60 scrollbar-none min-w-0 max-w-full">
          <button
            type="button"
            onClick={() => setActiveTab('pipeline')}
            className={cn(
              'flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all shrink-0 cursor-pointer',
              activeTab === 'pipeline'
                ? 'bg-primary text-primary-foreground shadow-md shadow-primary/25 font-bold'
                : 'text-muted-foreground hover:text-foreground hover:bg-muted'
            )}
          >
            <Terminal className="h-3.5 w-3.5" />
            <span>4 步驟實戰代碼 (端到端全景)</span>
            <Badge variant="outline" className="ml-1 text-[9px] px-1 py-0 border-current">
              4 Steps
            </Badge>
          </button>

          {isUnlocked && (
            <>
              <button
                type="button"
                onClick={() => setActiveTab('star')}
                className={cn(
                  'flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all shrink-0 cursor-pointer',
                  activeTab === 'star'
                    ? 'bg-amber-500 text-slate-950 shadow-md shadow-amber-500/20 font-bold'
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                )}
              >
                <Target className="h-3.5 w-3.5" />
                <span>架構決策防禦 (Architecture Decisions)</span>
              </button>

              <button
                type="button"
                onClick={() => setActiveTab('interview')}
                className={cn(
                  'flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all shrink-0 cursor-pointer',
                  activeTab === 'interview'
                    ? 'bg-amber-500 text-slate-950 shadow-md shadow-amber-500/20 font-bold'
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                )}
              >
                <HelpCircle className="h-3.5 w-3.5" />
                <span>系統架構深度思辨 (Systems Deep Dive)</span>
                <span className="text-[10px] opacity-75 font-mono">
                  ({tutorial.interviewQA?.length || 0})
                </span>
              </button>

              <button
                type="button"
                onClick={() => setActiveTab('resume')}
                className={cn(
                  'flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all shrink-0 cursor-pointer',
                  activeTab === 'resume'
                    ? 'bg-amber-500 text-slate-950 shadow-md shadow-amber-500/20 font-bold'
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                )}
              >
                <Award className="h-3.5 w-3.5" />
                <span>工程實踐亮點 (Technical Highlights)</span>
              </button>
            </>
          )}
        </div>

        {/* Tab 1: 4-Step Pipeline Code (End-to-End Visible & Copyable) */}
        {activeTab === 'pipeline' && (
          <div className="space-y-4 pt-3 min-w-0 max-w-full">
            {/* View Mode Switcher & Quick Navigation Bar */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 p-2.5 rounded-xl border border-border bg-card/70 backdrop-blur-xs min-w-0">
              {/* View Mode Switcher */}
              <div className="flex items-center gap-1 bg-muted/60 p-1 rounded-lg shrink-0 overflow-x-auto">
                <button
                  type="button"
                  onClick={() => setViewMode('all')}
                  className={cn(
                    'flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold transition-all cursor-pointer shrink-0',
                    viewMode === 'all'
                      ? 'bg-amber-500 text-slate-950 shadow-xs font-bold'
                      : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                  )}
                  title="連續展示全部 4 步驟實戰代碼與完整端到端腳本"
                >
                  <Layers className="h-3.5 w-3.5" />
                  <span>端到端連續全景</span>
                </button>
                <button
                  type="button"
                  onClick={() => setViewMode('single')}
                  className={cn(
                    'flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold transition-all cursor-pointer shrink-0',
                    viewMode === 'single'
                      ? 'bg-amber-500 text-slate-950 shadow-xs font-bold'
                      : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                  )}
                  title="以單一步驟分頁聚焦"
                >
                  <ListOrdered className="h-3.5 w-3.5" />
                  <span>單步聚焦</span>
                </button>
                <button
                  type="button"
                  onClick={() => setViewMode('script')}
                  className={cn(
                    'flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold transition-all cursor-pointer shrink-0',
                    viewMode === 'script'
                      ? 'bg-amber-500 text-slate-950 shadow-xs font-bold'
                      : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                  )}
                  title="僅展示完整 Python 腳本"
                >
                  <FileCode className="h-3.5 w-3.5" />
                  <span>完整腳本</span>
                </button>
              </div>

              {/* Quick Jump Anchors (Shown in 'all' view mode) */}
              {viewMode === 'all' && (
                <div className="flex items-center gap-1.5 overflow-x-auto scrollbar-none text-xs min-w-0">
                  <span className="text-[11px] text-muted-foreground shrink-0 font-mono hidden sm:inline">
                    快速導航：
                  </span>
                  {tutorial.steps.map((s) => (
                    <button
                      key={s.stepNumber}
                      type="button"
                      onClick={() => scrollToAnchor(`step-anchor-${s.stepNumber}`)}
                      className="px-2 py-0.5 rounded-md border border-border/70 bg-muted/40 hover:bg-muted text-[11px] text-muted-foreground hover:text-foreground font-mono transition-colors shrink-0 cursor-pointer"
                    >
                      步驟 {s.stepNumber}
                    </button>
                  ))}
                  <button
                    type="button"
                    onClick={() => scrollToAnchor('full-pipeline-anchor')}
                    className="px-2 py-0.5 rounded-md border border-amber-500/40 bg-amber-500/10 hover:bg-amber-500/20 text-[11px] text-amber-500 dark:text-amber-300 font-semibold transition-colors shrink-0 cursor-pointer flex items-center gap-1"
                  >
                    <Terminal className="h-3 w-3" />
                    <span>完整腳本</span>
                  </button>
                </div>
              )}
            </div>

            {/* MODE 1: ALL STEPS CONTINUOUS (DEFAULT - All steps visible and copyable) */}
            {viewMode === 'all' && (
              <div className="space-y-6 min-w-0 max-w-full">
                {tutorial.steps.map((step, idx) => (
                  <div
                    key={step.stepNumber}
                    id={`step-anchor-${step.stepNumber}`}
                    className="scroll-mt-4 rounded-xl border border-border bg-card/60 p-4 space-y-3.5 relative transition-all shadow-xs min-w-0 max-w-full overflow-hidden"
                  >
                    {/* Step Header */}
                    <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border/60 pb-2.5 min-w-0">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-amber-500 text-slate-950 font-mono text-xs font-bold shadow-xs shrink-0">
                          {step.stepNumber}
                        </span>
                        <div className="min-w-0">
                          <h4 className="text-sm font-bold text-foreground flex items-center gap-2 min-w-0">
                            <span className="truncate">步驟 {step.stepNumber} · {step.title}</span>
                          </h4>
                        </div>
                      </div>
                      <div className="flex items-center gap-2 shrink-0">
                        <Badge variant="outline" className="text-[10px] font-mono">
                          {step.badge}
                        </Badge>
                        <button
                          type="button"
                          onClick={() => handleCopyCode(step.codeSnippet, idx)}
                          className="flex items-center gap-1.5 text-xs text-amber-500 dark:text-amber-300 hover:text-amber-400 bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/30 px-2.5 py-1 rounded-lg transition-all cursor-pointer font-medium"
                          title={`複製步驟 ${step.stepNumber} 代碼`}
                        >
                          {copiedStepIdx === idx ? (
                            <>
                              <Check className="h-3.5 w-3.5 text-emerald-400" />
                              <span className="text-emerald-400 font-semibold">已複製步驟 {step.stepNumber}</span>
                            </>
                          ) : (
                            <>
                              <Copy className="h-3.5 w-3.5" />
                              <span>複製步驟 {step.stepNumber} 代碼</span>
                            </>
                          )}
                        </button>
                      </div>
                    </div>

                    {/* Objective Banner */}
                    <div className="rounded-lg border border-amber-500/25 bg-amber-500/10 px-3 py-2 text-xs text-amber-600 dark:text-amber-200/90 flex items-start gap-2 min-w-0">
                      <Target className="h-4 w-4 text-amber-500 dark:text-amber-400 shrink-0 mt-0.5" />
                      <div className="min-w-0 break-words">
                        <strong className="text-amber-600 dark:text-amber-300 font-bold">實戰目標：</strong>{' '}
                        {step.objective}
                      </div>
                    </div>

                    {/* Code Snippet Box */}
                    {step.codeSnippet && (
                      <div className="rounded-xl border border-border bg-[#090d16] overflow-hidden shadow-inner min-w-0 max-w-full">
                        <div className="flex items-center justify-between border-b border-border/80 bg-[#0f1422] px-3 py-1.5 text-xs text-muted-foreground min-w-0 gap-2">
                          <span className="flex items-center gap-1.5 font-mono text-[11px] text-cyan-400 min-w-0 truncate">
                            <Terminal className="h-3.5 w-3.5 shrink-0" />
                            <span className="truncate">Kaggle Python 實戰代碼 · 步驟 {step.stepNumber} ({step.codeLanguage || 'python'})</span>
                          </span>
                          <div className="flex items-center gap-1.5 shrink-0">
                            <button
                              type="button"
                              onClick={() => setIsCodeWrapped((prev) => !prev)}
                              className={cn(
                                "flex items-center gap-1 text-[11px] cursor-pointer px-2 py-0.5 rounded transition-colors",
                                isCodeWrapped
                                  ? "bg-amber-500/20 text-amber-400 border border-amber-500/40"
                                  : "text-slate-300 hover:text-white bg-slate-800/60 hover:bg-slate-800"
                              )}
                              title={isCodeWrapped ? "切換為單行橫向滾動模式" : "切換為自動換行模式 (適合窄屏閱讀)"}
                            >
                              <WrapText className="h-3 w-3" />
                              <span>{isCodeWrapped ? '取消換行' : '自動換行'}</span>
                            </button>
                            <button
                              type="button"
                              onClick={() => handleCopyCode(step.codeSnippet, idx)}
                              className="flex items-center gap-1 text-[11px] text-slate-300 hover:text-white cursor-pointer px-2 py-0.5 rounded bg-slate-800/60 hover:bg-slate-800 transition-colors"
                            >
                              {copiedStepIdx === idx ? (
                                <>
                                  <Check className="h-3 w-3 text-emerald-400" />
                                  <span className="text-emerald-400 font-semibold">已複製</span>
                                </>
                              ) : (
                                <>
                                  <Copy className="h-3 w-3" />
                                  <span>複製代碼</span>
                                </>
                              )}
                            </button>
                          </div>
                        </div>
                        <div className="p-3.5 overflow-x-auto bg-[#090d16]/95 max-h-[380px] min-w-0 max-w-full">
                          <pre className={cn(
                            "font-mono text-xs text-slate-100 leading-relaxed m-0",
                            isCodeWrapped ? "whitespace-pre-wrap break-all" : "whitespace-pre"
                          )}>
                            <code>{step.codeSnippet}</code>
                          </pre>
                        </div>
                      </div>
                    )}

                    {/* Engineering Takeaways */}
                    <div className="rounded-lg border border-border bg-muted/40 p-3 space-y-1.5 min-w-0">
                      <h5 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
                        <Sparkles className="h-3.5 w-3.5 text-amber-500 dark:text-amber-400 shrink-0" />
                        <span>工程要點與 Kaggle 實踐建議</span>
                      </h5>
                      <ul className="space-y-1 text-xs text-foreground/90 list-disc pl-4 min-w-0 break-words">
                        {step.takeaways.map((item, i) => (
                          <li key={i}>{item}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                ))}

                {/* END-TO-END CONSOLIDATED PYTHON SCRIPT CARD */}
                <div
                  id="full-pipeline-anchor"
                  className="scroll-mt-4 rounded-2xl border-2 border-amber-500/40 bg-gradient-to-b from-amber-500/5 to-card p-4 sm:p-5 space-y-4 shadow-lg min-w-0 max-w-full overflow-hidden"
                >
                  <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/80 pb-3 min-w-0">
                    <div className="space-y-1 min-w-0 flex-1">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="flex h-7 w-7 items-center justify-center rounded-xl bg-gradient-to-tr from-amber-500 to-orange-500 text-slate-950 font-bold shadow-md shadow-amber-500/20 shrink-0">
                          <Terminal className="h-4 w-4" />
                        </span>
                        <h4 className="text-sm sm:text-base font-bold text-foreground truncate">
                          ⚡ 端到端一體化完整可執行腳本 (Full End-to-End Pipeline)
                        </h4>
                      </div>
                      <p className="text-xs text-muted-foreground sm:pl-9 break-words">
                        已將步驟 1 至 4 串聯為自包含、可直接貼入 Kaggle / Colab 單元格直接運行的完整腳本。
                      </p>
                    </div>

                    <button
                      type="button"
                      onClick={handleCopyAllPipelineCode}
                      className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-amber-500 to-amber-600 text-slate-950 font-bold px-3.5 py-2 text-xs shadow-md shadow-amber-500/25 hover:opacity-90 transition-all cursor-pointer shrink-0"
                    >
                      {copiedAllCode ? (
                        <>
                          <Check className="h-4 w-4 text-slate-950" />
                          <span>已複製端到端完整腳本！</span>
                        </>
                      ) : (
                        <>
                          <Copy className="h-4 w-4" />
                          <span>一鍵複製端到端完整腳本</span>
                        </>
                      )}
                    </button>
                  </div>

                  {/* Complete Script Box */}
                  <div className="rounded-xl border border-border bg-[#090d16] overflow-hidden shadow-inner min-w-0 max-w-full">
                    <div className="flex items-center justify-between border-b border-border/80 bg-[#0f1422] px-3.5 py-2 text-xs text-muted-foreground min-w-0 gap-2">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="h-2.5 w-2.5 rounded-full bg-rose-500/80 shrink-0"></span>
                        <span className="h-2.5 w-2.5 rounded-full bg-amber-500/80 shrink-0"></span>
                        <span className="h-2.5 w-2.5 rounded-full bg-emerald-500/80 shrink-0"></span>
                        <span className="font-mono text-[11px] text-cyan-400 font-semibold ml-1 truncate">
                          {tutorial.kaggleNotebook}.py (完整流水線)
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5 shrink-0">
                        <button
                          type="button"
                          onClick={() => setIsCodeWrapped((prev) => !prev)}
                          className={cn(
                            "flex items-center gap-1 text-[11px] cursor-pointer px-2 py-0.5 rounded transition-colors",
                            isCodeWrapped
                              ? "bg-amber-500/20 text-amber-400 border border-amber-500/40"
                              : "text-slate-300 hover:text-white bg-slate-800/60 hover:bg-slate-800"
                          )}
                          title={isCodeWrapped ? "切換為單行橫向滾動模式" : "切換為自動換行模式 (適合窄屏閱讀)"}
                        >
                          <WrapText className="h-3 w-3" />
                          <span>{isCodeWrapped ? '取消換行' : '自動換行'}</span>
                        </button>
                        <button
                          type="button"
                          onClick={handleCopyAllPipelineCode}
                          className="flex items-center gap-1 text-[11px] text-slate-300 hover:text-white cursor-pointer px-2 py-0.5 rounded bg-slate-800/60 hover:bg-slate-800 transition-colors"
                        >
                          {copiedAllCode ? (
                            <>
                              <Check className="h-3 w-3 text-emerald-400" />
                              <span className="text-emerald-400 font-semibold">已複製</span>
                            </>
                          ) : (
                            <>
                              <Copy className="h-3 w-3" />
                              <span>複製腳本</span>
                            </>
                          )}
                        </button>
                      </div>
                    </div>
                    <div className="p-4 overflow-x-auto bg-[#090d16]/95 max-h-[480px] min-w-0 max-w-full">
                      <pre className={cn(
                        "font-mono text-xs text-slate-100 leading-relaxed m-0",
                        isCodeWrapped ? "whitespace-pre-wrap break-all" : "whitespace-pre"
                      )}>
                        <code>{getFullPipelineScript()}</code>
                      </pre>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* MODE 2: SINGLE STEP FOCUS (Step by step navigation) */}
            {viewMode === 'single' && (
              <div className="space-y-4 min-w-0 max-w-full">
                {/* Step Selection Pills */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 min-w-0">
                  {tutorial.steps.map((step, idx) => (
                    <button
                      key={step.stepNumber}
                      type="button"
                      onClick={() => setActiveStep(idx)}
                      className={cn(
                        'flex items-center gap-2 p-2.5 rounded-xl border text-left transition-all cursor-pointer min-w-0',
                        activeStep === idx
                          ? 'border-amber-500/60 bg-amber-500/12 text-amber-500 dark:text-amber-300 shadow-sm'
                          : 'border-border bg-muted/60 text-muted-foreground hover:text-foreground hover:bg-muted'
                      )}
                    >
                      <div
                        className={cn(
                          'flex h-6 w-6 shrink-0 items-center justify-center rounded-lg font-mono text-xs font-bold',
                          activeStep === idx
                            ? 'bg-amber-500 text-slate-950'
                            : 'bg-muted text-muted-foreground'
                        )}
                      >
                        {step.stepNumber}
                      </div>
                      <div className="truncate min-w-0">
                        <span className="block text-[11px] font-semibold truncate leading-tight">
                          {step.badge}
                        </span>
                        <span className="block text-[9.5px] opacity-70 truncate">
                          步驟 {step.stepNumber}
                        </span>
                      </div>
                    </button>
                  ))}
                </div>

                {/* Active Step Content */}
                <div className="rounded-xl border border-border bg-card/60 p-4 space-y-3.5 min-w-0 max-w-full overflow-hidden">
                  <div className="flex items-center justify-between min-w-0 gap-2">
                    <h4 className="text-sm font-bold text-foreground flex items-center gap-2 min-w-0">
                      <span className="flex h-5 w-5 items-center justify-center rounded-full bg-amber-500/20 text-amber-500 dark:text-amber-400 font-mono text-xs font-bold shrink-0">
                        {currentStep.stepNumber}
                      </span>
                      <span className="truncate">{currentStep.title}</span>
                    </h4>
                    <div className="flex items-center gap-2 shrink-0">
                      <Badge variant="outline" className="text-[10px] font-mono">
                        {currentStep.badge}
                      </Badge>
                      <button
                        type="button"
                        onClick={() => handleCopyCode(currentStep.codeSnippet, activeStep)}
                        className="flex items-center gap-1.5 text-xs text-amber-500 dark:text-amber-300 hover:text-amber-400 bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/30 px-2.5 py-1 rounded-lg transition-all cursor-pointer font-medium shrink-0"
                      >
                        {copiedStepIdx === activeStep ? (
                          <>
                            <Check className="h-3.5 w-3.5 text-emerald-400" />
                            <span className="text-emerald-400 font-semibold">已複製步驟 {currentStep.stepNumber}</span>
                          </>
                        ) : (
                          <>
                            <Copy className="h-3.5 w-3.5" />
                            <span>複製此步驟代碼</span>
                          </>
                        )}
                      </button>
                    </div>
                  </div>

                  {/* Objective Banner */}
                  <div className="rounded-lg border border-amber-500/25 bg-amber-500/10 px-3 py-2 text-xs text-amber-600 dark:text-amber-200/90 flex items-start gap-2 min-w-0">
                    <Target className="h-4 w-4 text-amber-500 dark:text-amber-400 shrink-0 mt-0.5" />
                    <div className="min-w-0 break-words">
                      <strong className="text-amber-600 dark:text-amber-300 font-bold">實戰目標：</strong>{' '}
                      {currentStep.objective}
                    </div>
                  </div>

                  {/* Code Snippet Box */}
                  {currentStep.codeSnippet && (
                    <div className="rounded-xl border border-border bg-[#090d16] overflow-hidden shadow-inner min-w-0 max-w-full">
                      <div className="flex items-center justify-between border-b border-border/80 bg-[#0f1422] px-3 py-1.5 text-xs text-muted-foreground min-w-0 gap-2">
                        <span className="flex items-center gap-1.5 font-mono text-[11px] text-cyan-400 min-w-0 truncate">
                          <Terminal className="h-3.5 w-3.5 shrink-0" />
                          <span className="truncate">Kaggle Python 實戰代碼 ({currentStep.codeLanguage || 'python'})</span>
                        </span>
                        <div className="flex items-center gap-1.5 shrink-0">
                          <button
                            type="button"
                            onClick={() => setIsCodeWrapped((prev) => !prev)}
                            className={cn(
                              "flex items-center gap-1 text-[11px] cursor-pointer px-2 py-0.5 rounded transition-colors",
                              isCodeWrapped
                                ? "bg-amber-500/20 text-amber-400 border border-amber-500/40"
                                : "text-slate-300 hover:text-white bg-slate-800/60 hover:bg-slate-800"
                            )}
                            title={isCodeWrapped ? "切換為單行橫向滾動模式" : "切換為自動換行模式 (適合窄屏閱讀)"}
                          >
                            <WrapText className="h-3 w-3" />
                            <span>{isCodeWrapped ? '取消換行' : '自動換行'}</span>
                          </button>
                          <button
                            type="button"
                            onClick={() => handleCopyCode(currentStep.codeSnippet, activeStep)}
                            className="flex items-center gap-1 text-[11px] text-slate-300 hover:text-white cursor-pointer px-2 py-0.5 rounded bg-slate-800/60 hover:bg-slate-800 transition-colors"
                          >
                            {copiedStepIdx === activeStep ? (
                              <>
                                <Check className="h-3 w-3 text-emerald-400" />
                                <span className="text-emerald-400 font-semibold">已複製步驟代碼</span>
                              </>
                            ) : (
                              <>
                                <Copy className="h-3 w-3" />
                                <span>複製代碼</span>
                              </>
                            )}
                          </button>
                        </div>
                      </div>
                      <div className="p-3.5 overflow-x-auto bg-[#090d16]/95 max-h-[380px] min-w-0 max-w-full">
                        <pre className={cn(
                          "font-mono text-xs text-slate-100 leading-relaxed m-0",
                          isCodeWrapped ? "whitespace-pre-wrap break-all" : "whitespace-pre"
                        )}>
                          <code>{currentStep.codeSnippet}</code>
                        </pre>
                      </div>
                    </div>
                  )}

                  {/* Engineering Takeaways */}
                  <div className="rounded-lg border border-border bg-muted/40 p-3 space-y-1.5 min-w-0">
                    <h5 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
                      <Sparkles className="h-3.5 w-3.5 text-amber-500 dark:text-amber-400 shrink-0" />
                      <span>工程要點與 Kaggle 實踐建議</span>
                    </h5>
                    <ul className="space-y-1 text-xs text-foreground/90 list-disc pl-4 min-w-0 break-words">
                      {currentStep.takeaways.map((item, i) => (
                        <li key={i}>{item}</li>
                      ))}
                    </ul>
                  </div>
                </div>

                {/* Bottom Navigation Controls for Single Mode */}
                <div className="flex flex-wrap items-center justify-between gap-3 pt-2 text-xs text-muted-foreground border-t border-border min-w-0">
                  <div className="flex items-center gap-1.5 font-mono text-[11px] min-w-0">
                    <FileCode className="h-3.5 w-3.5 text-amber-500 dark:text-amber-400 shrink-0" />
                    <span className="truncate">Showcase：{tutorial.kaggleNotebook}</span>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    {activeStep > 0 && (
                      <button
                        type="button"
                        onClick={() => setActiveStep((prev) => prev - 1)}
                        className="px-3 py-1 rounded-lg border border-border hover:bg-muted text-foreground text-xs cursor-pointer transition-colors shrink-0"
                      >
                        上一則步驟
                      </button>
                    )}
                    {activeStep < tutorial.steps.length - 1 ? (
                      <button
                        type="button"
                        onClick={() => setActiveStep((prev) => prev + 1)}
                        className="px-3 py-1 rounded-lg bg-amber-500 text-slate-950 font-bold text-xs hover:bg-amber-400 transition-colors cursor-pointer flex items-center gap-1 shrink-0"
                      >
                        <span>下一步驟</span>
                        <ChevronRight className="h-3 w-3" />
                      </button>
                    ) : isUnlocked ? (
                      <button
                        type="button"
                        onClick={() => setActiveTab('star')}
                        className="px-3 py-1 rounded-lg bg-emerald-500 text-slate-950 font-bold text-xs hover:bg-emerald-400 transition-colors cursor-pointer flex items-center gap-1 shrink-0 shadow-xs"
                      >
                        <CheckCircle2 className="h-3.5 w-3.5" />
                        <span>查看 STAR 答辯稿</span>
                      </button>
                    ) : (
                      <button
                        type="button"
                        onClick={onClose}
                        className="px-3 py-1 rounded-lg bg-primary text-primary-foreground font-semibold text-xs hover:opacity-90 transition-opacity cursor-pointer flex items-center gap-1 shrink-0 shadow-xs"
                      >
                        <CheckCircle2 className="h-3.5 w-3.5" />
                        <span>完成導讀</span>
                      </button>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* MODE 3: FULL SCRIPT ONLY */}
            {viewMode === 'script' && (
              <div className="space-y-4 min-w-0 max-w-full">
                <div className="rounded-2xl border-2 border-amber-500/40 bg-gradient-to-b from-amber-500/5 to-card p-4 sm:p-5 space-y-4 shadow-lg min-w-0 max-w-full overflow-hidden">
                  <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/80 pb-3 min-w-0">
                    <div className="space-y-1 min-w-0 flex-1">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="flex h-7 w-7 items-center justify-center rounded-xl bg-gradient-to-tr from-amber-500 to-orange-500 text-slate-950 font-bold shadow-md shadow-amber-500/20 shrink-0">
                          <Terminal className="h-4 w-4" />
                        </span>
                        <h4 className="text-sm sm:text-base font-bold text-foreground truncate">
                          ⚡ 端到端一體化完整可執行腳本 (Full End-to-End Pipeline)
                        </h4>
                      </div>
                      <p className="text-xs text-muted-foreground sm:pl-9 break-words">
                        已將步驟 1 至 4 串聯為自包含、可直接貼入 Kaggle / Colab 單元格直接運行的完整腳本。
                      </p>
                    </div>

                    <button
                      type="button"
                      onClick={handleCopyAllPipelineCode}
                      className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-amber-500 to-amber-600 text-slate-950 font-bold px-3.5 py-2 text-xs shadow-md shadow-amber-500/25 hover:opacity-90 transition-all cursor-pointer shrink-0"
                    >
                      {copiedAllCode ? (
                        <>
                          <Check className="h-4 w-4 text-slate-950" />
                          <span>已複製端到端完整腳本！</span>
                        </>
                      ) : (
                        <>
                          <Copy className="h-4 w-4" />
                          <span>一鍵複製端到端完整腳本</span>
                        </>
                      )}
                    </button>
                  </div>

                  <div className="rounded-xl border border-border bg-[#090d16] overflow-hidden shadow-inner min-w-0 max-w-full">
                    <div className="flex items-center justify-between border-b border-border/80 bg-[#0f1422] px-3.5 py-2 text-xs text-muted-foreground min-w-0 gap-2">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="h-2.5 w-2.5 rounded-full bg-rose-500/80 shrink-0"></span>
                        <span className="h-2.5 w-2.5 rounded-full bg-amber-500/80 shrink-0"></span>
                        <span className="h-2.5 w-2.5 rounded-full bg-emerald-500/80 shrink-0"></span>
                        <span className="font-mono text-[11px] text-cyan-400 font-semibold ml-1 truncate">
                          {tutorial.kaggleNotebook}.py
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5 shrink-0">
                        <button
                          type="button"
                          onClick={() => setIsCodeWrapped((prev) => !prev)}
                          className={cn(
                            "flex items-center gap-1 text-[11px] cursor-pointer px-2 py-0.5 rounded transition-colors",
                            isCodeWrapped
                              ? "bg-amber-500/20 text-amber-400 border border-amber-500/40"
                              : "text-slate-300 hover:text-white bg-slate-800/60 hover:bg-slate-800"
                          )}
                          title={isCodeWrapped ? "切換為單行橫向滾動模式" : "切換為自動換行模式 (適合窄屏閱讀)"}
                        >
                          <WrapText className="h-3 w-3" />
                          <span>{isCodeWrapped ? '取消換行' : '自動換行'}</span>
                        </button>
                        <button
                          type="button"
                          onClick={handleCopyAllPipelineCode}
                          className="flex items-center gap-1 text-[11px] text-slate-300 hover:text-white cursor-pointer px-2 py-0.5 rounded bg-slate-800/60 hover:bg-slate-800 transition-colors"
                        >
                          {copiedAllCode ? (
                            <>
                              <Check className="h-3 w-3 text-emerald-400" />
                              <span className="text-emerald-400 font-semibold">已複製</span>
                            </>
                          ) : (
                            <>
                              <Copy className="h-3 w-3" />
                              <span>複製腳本</span>
                            </>
                          )}
                        </button>
                      </div>
                    </div>
                    <div className="p-4 overflow-x-auto bg-[#090d16]/95 max-h-[560px] min-w-0 max-w-full">
                      <pre className={cn(
                        "font-mono text-xs text-slate-100 leading-relaxed m-0",
                        isCodeWrapped ? "whitespace-pre-wrap break-all" : "whitespace-pre"
                      )}>
                        <code>{getFullPipelineScript()}</code>
                      </pre>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Bottom Step Navigation / Actions Footer */}
            <div className="flex flex-wrap items-center justify-between gap-3 pt-3 text-xs text-muted-foreground border-t border-border min-w-0">
              <div className="flex items-center gap-1.5 font-mono text-[11px] min-w-0">
                <FileCode className="h-3.5 w-3.5 text-amber-500 dark:text-amber-400 shrink-0" />
                <span className="truncate">Showcase：{tutorial.kaggleNotebook}</span>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <button
                  type="button"
                  onClick={handleCopyAllPipelineCode}
                  className="px-3 py-1.5 rounded-lg border border-amber-500/40 bg-amber-500/10 hover:bg-amber-500/20 text-amber-500 dark:text-amber-300 font-semibold text-xs transition-colors cursor-pointer flex items-center gap-1.5 shrink-0"
                >
                  {copiedAllCode ? (
                    <>
                      <Check className="h-3.5 w-3.5 text-emerald-500" />
                      <span className="text-emerald-500 font-bold">已複製 4 步腳本</span>
                    </>
                  ) : (
                    <>
                      <Copy className="h-3.5 w-3.5" />
                      <span>複製完整 4 步驟腳本</span>
                    </>
                  )}
                </button>
                {isUnlocked ? (
                  <button
                    type="button"
                    onClick={() => setActiveTab('star')}
                    className="px-3 py-1.5 rounded-lg bg-emerald-500 text-slate-950 font-bold text-xs hover:bg-emerald-400 transition-colors cursor-pointer flex items-center gap-1.5 shadow-xs shrink-0"
                  >
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    <span>查看 STAR 答辯稿</span>
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={onClose}
                    className="px-3 py-1.5 rounded-lg bg-primary text-primary-foreground font-semibold text-xs hover:opacity-90 transition-opacity cursor-pointer flex items-center gap-1.5 shadow-xs shrink-0"
                  >
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    <span>完成導讀</span>
                  </button>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Tab 2: STAR Defense Workbench */}
        {isUnlocked && activeTab === 'star' && (
          <div className="space-y-4 pt-3 min-w-0 max-w-full">
            <div className="flex flex-wrap items-center justify-between gap-2 min-w-0">
              <div className="flex items-center gap-2 min-w-0">
                <Target className="h-4 w-4 text-amber-500 dark:text-amber-400 shrink-0" />
                <h4 className="text-sm font-bold text-foreground truncate">
                  STAR 系統架構決策分析 (Situation · Task · Action · Result)
                </h4>
              </div>
              <button
                type="button"
                onClick={handleCopyStarPlaybook}
                className="flex items-center gap-1 rounded-lg border border-amber-500/30 bg-amber-500/10 px-2.5 py-1 text-xs font-semibold text-amber-500 dark:text-amber-300 hover:bg-amber-500/20 cursor-pointer shrink-0"
              >
                {copiedStar ? (
                  <>
                    <Check className="h-3 w-3 text-emerald-500" />
                    <span className="text-emerald-500 font-bold">已複製完整答辯稿</span>
                  </>
                ) : (
                  <>
                    <Copy className="h-3 w-3" />
                    <span>複製完整 STAR 答辯稿</span>
                  </>
                )}
              </button>
            </div>

            {/* STAR 4-Cards Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 min-w-0 max-w-full">
              {/* Situation */}
              <div className="rounded-xl border border-rose-500/30 bg-rose-500/8 p-3.5 space-y-2 min-w-0 overflow-hidden">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-rose-500/20 text-rose-500 dark:text-rose-400 font-bold font-mono text-xs shrink-0">
                    S
                  </span>
                  <div className="min-w-0">
                    <h5 className="text-xs font-bold text-rose-500 dark:text-rose-300 uppercase tracking-wide truncate">
                      Situation · 業務背景與技術挑戰
                    </h5>
                    <span className="text-[10px] text-muted-foreground truncate block">痛點與未解決的問題</span>
                  </div>
                </div>
                <p className="text-xs text-foreground/90 leading-relaxed sm:pl-8 break-words">
                  {tutorial.starPlaybook.situation}
                </p>
              </div>

              {/* Task */}
              <div className="rounded-xl border border-amber-500/30 bg-amber-500/8 p-3.5 space-y-2 min-w-0 overflow-hidden">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-amber-500/20 text-amber-500 dark:text-amber-400 font-bold font-mono text-xs shrink-0">
                    T
                  </span>
                  <div className="min-w-0">
                    <h5 className="text-xs font-bold text-amber-500 dark:text-amber-300 uppercase tracking-wide truncate">
                      Task · 目標任務與約束條件
                    </h5>
                    <span className="text-[10px] text-muted-foreground truncate block">資源限制與成功判準</span>
                  </div>
                </div>
                <p className="text-xs text-foreground/90 leading-relaxed sm:pl-8 break-words">
                  {tutorial.starPlaybook.task}
                </p>
              </div>

              {/* Action */}
              <div className="rounded-xl border border-cyan-500/30 bg-cyan-500/8 p-3.5 space-y-2 min-w-0 overflow-hidden">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-cyan-500/20 text-cyan-500 dark:text-cyan-400 font-bold font-mono text-xs shrink-0">
                    A
                  </span>
                  <div className="min-w-0">
                    <h5 className="text-xs font-bold text-cyan-500 dark:text-cyan-300 uppercase tracking-wide truncate">
                      Action · 核心工程解法與架構設計
                    </h5>
                    <span className="text-[10px] text-muted-foreground truncate block">具體演算法與系統防禦</span>
                  </div>
                </div>
                <p className="text-xs text-foreground/90 leading-relaxed sm:pl-8 break-words">
                  {tutorial.starPlaybook.action}
                </p>
              </div>

              {/* Result */}
              <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/8 p-3.5 space-y-2 min-w-0 overflow-hidden">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-emerald-500/20 text-emerald-500 dark:text-emerald-400 font-bold font-mono text-xs shrink-0">
                    R
                  </span>
                  <div className="min-w-0">
                    <h5 className="text-xs font-bold text-emerald-500 dark:text-emerald-300 uppercase tracking-wide truncate">
                      Result · 量化產出與實證影響力
                    </h5>
                    <span className="text-[10px] text-muted-foreground truncate block">指標提升、延遲縮減與驗證</span>
                  </div>
                </div>
                <p className="text-xs text-foreground/90 leading-relaxed sm:pl-8 break-words">
                  {tutorial.starPlaybook.result}
                </p>
              </div>
            </div>

            {/* Defense Tips Card */}
            <div className="rounded-xl border border-border bg-muted/40 p-4 space-y-2 min-w-0">
              <h5 className="text-xs font-bold text-foreground flex items-center gap-1.5">
                <ShieldCheck className="h-4 w-4 text-primary shrink-0" />
                <span>系統架構防守攻略 (Defense Playbook)</span>
              </h5>
              <p className="text-xs text-muted-foreground leading-relaxed break-words">
                在進行架構評審與同行審查時，牢牢守住 <strong>硬體邊界（{tutorial.hardwareRequirements}）</strong> 與{' '}
                <strong>量化評估基線</strong>。強調在資源約束下的架構選型取捨 (Trade-offs)，並主動關注代碼的防作弊 (Reward Hacking) 機制。
              </p>
            </div>
          </div>
        )}

        {/* Tab 3: Frontier Lab Interview Q&A */}
        {isUnlocked && activeTab === 'interview' && (
          <div className="space-y-3.5 pt-3 min-w-0 max-w-full">
            <div className="flex items-center justify-between min-w-0 gap-2">
              <div className="flex items-center gap-2 min-w-0">
                <HelpCircle className="h-4 w-4 text-purple-400 shrink-0" />
                <h4 className="text-sm font-bold text-foreground truncate">
                  前沿實驗室核心架構思辨題 (Frontier Lab Architecture Deep Dive)
                </h4>
              </div>
              <Badge variant="violet" className="text-[10px] shrink-0">
                Tier-1 Architecture Q&A
              </Badge>
            </div>

            <div className="space-y-3 min-w-0 max-w-full">
              {tutorial.interviewQA && tutorial.interviewQA.length > 0 ? (
                tutorial.interviewQA.map((qa, i) => (
                  <div
                    key={i}
                    className="rounded-xl border border-purple-500/30 bg-purple-500/8 p-4 space-y-2.5 transition-all min-w-0 max-w-full overflow-hidden"
                  >
                    <div className="flex items-start gap-2.5 min-w-0">
                      <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-purple-500/25 text-purple-500 dark:text-purple-300 font-mono text-xs font-bold mt-0.5">
                        Q{i + 1}
                      </span>
                      <h5 className="text-xs sm:text-sm font-bold text-purple-600 dark:text-purple-200 leading-snug break-words">
                        {qa.question}
                      </h5>
                    </div>

                    <div className="sm:ml-7 rounded-lg border border-purple-500/20 bg-card/80 p-3.5 min-w-0">
                      <div className="text-[11px] font-semibold text-purple-500 dark:text-purple-400 uppercase tracking-wider mb-1 truncate">
                        Frontier Lab 專家級架構設計解析：
                      </div>
                      <p className="text-xs text-foreground/90 leading-relaxed whitespace-pre-line break-words">
                        {qa.answer}
                      </p>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-xs text-muted-foreground p-4 text-center">
                  暫無專屬架構題庫，請參考架構決策工作台。
                </div>
              )}
            </div>
          </div>
        )}

        {/* Tab 4: Resume Bullets Generator */}
        {isUnlocked && activeTab === 'resume' && (
          <div className="space-y-4 pt-3 min-w-0 max-w-full">
            <div className="flex items-center justify-between min-w-0 gap-2">
              <div className="flex items-center gap-2 min-w-0">
                <Award className="h-4 w-4 text-primary shrink-0" />
                <h4 className="text-sm font-bold text-foreground truncate">
                  工程實踐亮點與成果摘要 (Portfolio & Technical Highlights)
                </h4>
              </div>
              <Badge variant="secondary" className="text-[10px] shrink-0">
                Production Aligned
              </Badge>
            </div>

            <p className="text-xs text-muted-foreground break-words">
              直接複製符合頂級 AI 系統架構標準的 STAR 英文/中文工程成果亮點，可自由替換在訓練產出的實際指標數值。
            </p>

            {/* Bullet Variations */}
            <div className="space-y-3 min-w-0 max-w-full">
              {bulletVariants.map((item, idx) => (
                <div
                  key={idx}
                  className="rounded-xl border border-border bg-card/60 p-3.5 space-y-2 min-w-0 max-w-full overflow-hidden"
                >
                  <div className="flex items-center justify-between min-w-0 gap-2">
                    <span className="text-xs font-bold text-primary flex items-center gap-1.5 min-w-0 truncate">
                      <Sparkles className="h-3.5 w-3.5 text-amber-500 dark:text-amber-400 shrink-0" />
                      <span className="truncate">{item.label}</span>
                    </span>
                    <button
                      type="button"
                      onClick={() => handleCopyBullet(item.bullet, idx)}
                      className="flex items-center gap-1 text-[11px] font-medium text-primary hover:text-primary/80 bg-primary/10 hover:bg-primary/20 border border-primary/30 px-2.5 py-0.5 rounded-lg transition-colors cursor-pointer shrink-0"
                    >
                      {copiedBulletIdx === idx ? (
                        <>
                          <Check className="h-3 w-3 text-emerald-500" />
                          <span className="text-emerald-500 font-bold">已複製此亮點</span>
                        </>
                      ) : (
                        <>
                          <Copy className="h-3 w-3" />
                          <span>複製</span>
                        </>
                      )}
                    </button>
                  </div>

                  <div className="rounded-lg border border-border bg-muted/50 p-3 font-mono text-xs text-foreground leading-relaxed select-all break-words">
                    • {item.bullet}
                  </div>
                </div>
              ))}
            </div>

            {/* Resume Tips Box */}
            <div className="rounded-xl border border-amber-500/25 bg-amber-500/10 p-3 text-xs text-amber-600 dark:text-amber-200/90 space-y-1 min-w-0">
              <strong className="text-amber-600 dark:text-amber-300 font-bold block">
                💡 工程亮點提煉黃金原則 (Google X-Y-Z Formula)：
              </strong>
              <span className="break-words">
                "Accomplished [X] as measured by [Y], by doing [Z]" — 建議將上述亮點放置在開源專案或技術設計文檔中，並附上您的 Notebook 開源連結。
              </span>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};

