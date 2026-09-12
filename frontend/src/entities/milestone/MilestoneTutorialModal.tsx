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

export const MilestoneTutorialModal: React.FC<MilestoneTutorialModalProps> = ({
  part,
  isOpen,
  onClose,
}) => {
  const { isUnlocked } = usePrivacyStore();
  const [activeTab, setActiveTab] = useState<TabKey>('pipeline');
  const [activeStep, setActiveStep] = useState<number>(0);
  const [copiedCode, setCopiedCode] = useState<boolean>(false);
  const [copiedAllCode, setCopiedAllCode] = useState<boolean>(false);
  const [copiedBulletIdx, setCopiedBulletIdx] = useState<number | null>(null);
  const [copiedStar, setCopiedStar] = useState<boolean>(false);

  useEffect(() => {
    if (!isUnlocked && activeTab !== 'pipeline') {
      setActiveTab('pipeline');
    }
  }, [isUnlocked, activeTab]);

  if (!part || !isOpen) return null;

  const tutorial: MilestoneTutorial = getMilestoneTutorial(part);
  const currentStep = tutorial.steps[activeStep] || tutorial.steps[0];

  const handleCopyCode = (code?: string) => {
    if (!code) return;
    navigator.clipboard.writeText(code);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2000);
  };

  const handleCopyAllPipelineCode = () => {
    const fullScript = [
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

    navigator.clipboard.writeText(fullScript);
    setCopiedAllCode(true);
    setTimeout(() => setCopiedAllCode(false), 2200);
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
      <DialogContent className="max-w-4xl max-h-[92vh] overflow-y-auto p-5 sm:p-6 bg-card/95 backdrop-blur-md border border-border shadow-2xl rounded-2xl text-foreground">
        {/* Modal Header */}
        <DialogHeader className="pb-4 border-b border-border">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="flex items-start gap-3">
              <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-gradient-to-tr from-amber-500 via-orange-500 to-amber-600 text-white shadow-lg shadow-amber-500/25">
                <Trophy className="h-5 w-5" />
              </div>
              <div className="space-y-1">
                <div className="flex flex-wrap items-center gap-2">
                  <DialogTitle className="text-base sm:text-lg font-bold text-foreground font-title leading-snug">
                    {tutorial.title}
                  </DialogTitle>
                  <Badge variant="amber" className="text-[10px] px-1.5 py-0.5">
                    {tutorial.badge}
                  </Badge>
                </div>
                <DialogDescription className="text-xs text-muted-foreground flex flex-wrap items-center gap-2">
                  {isUnlocked && (
                    <>
                      <span className="inline-flex items-center gap-1 text-primary font-medium">
                        <Briefcase className="h-3.5 w-3.5" />
                        <span>架構定位：{tutorial.targetRole}</span>
                      </span>
                      <span className="text-muted-foreground/60">•</span>
                    </>
                  )}
                  <span className="inline-flex items-center gap-1 text-amber-500 dark:text-amber-300/90">
                    <Cpu className="h-3 w-3" />
                    <span>{tutorial.hardwareRequirements}</span>
                  </span>
                  <span className="text-muted-foreground/60">•</span>
                  <span className="inline-flex items-center gap-1 text-emerald-500 dark:text-emerald-300/90">
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
                className="flex items-center gap-1.5 rounded-lg border border-amber-500/40 bg-amber-500/10 px-2.5 py-1 text-xs font-semibold text-amber-500 dark:text-amber-300 hover:bg-amber-500/20 transition-all cursor-pointer shadow-sm"
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
                    <span>一鍵複製全部代碼</span>
                  </>
                )}
              </button>

              {isUnlocked && (
                <button
                  type="button"
                  onClick={() => handleCopyBullet(tutorial.portfolioBullet, -1)}
                  className="flex items-center gap-1.5 rounded-lg border border-primary/30 bg-primary/10 px-2.5 py-1 text-xs font-medium text-primary hover:bg-primary/20 transition-all cursor-pointer"
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

          <p className="mt-2.5 text-xs text-muted-foreground leading-relaxed">
            {tutorial.overview}
          </p>
        </DialogHeader>

        {/* 4-Tab Navigation Header (Filtered when locked) */}
        <div className="flex items-center gap-2 overflow-x-auto pb-1 pt-2 border-b border-border/60 scrollbar-none">
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
            <span>4 步驟實戰代碼</span>
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

        {/* Tab 1: 4-Step Pipeline Code */}
        {activeTab === 'pipeline' && (
          <div className="space-y-4 pt-3">
            {/* Step Selection Pills */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {tutorial.steps.map((step, idx) => (
                <button
                  key={step.stepNumber}
                  type="button"
                  onClick={() => setActiveStep(idx)}
                  className={cn(
                    'flex items-center gap-2 p-2.5 rounded-xl border text-left transition-all cursor-pointer',
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
                  <div className="truncate">
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
            <div className="rounded-xl border border-border bg-card/60 p-4 space-y-3.5">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-bold text-foreground flex items-center gap-2">
                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-amber-500/20 text-amber-500 dark:text-amber-400 font-mono text-xs font-bold">
                    {currentStep.stepNumber}
                  </span>
                  {currentStep.title}
                </h4>
                <Badge variant="outline" className="text-[10px] font-mono">
                  {currentStep.badge}
                </Badge>
              </div>

              {/* Objective Banner */}
              <div className="rounded-lg border border-amber-500/25 bg-amber-500/10 px-3 py-2 text-xs text-amber-600 dark:text-amber-200/90 flex items-start gap-2">
                <Target className="h-4 w-4 text-amber-500 dark:text-amber-400 shrink-0 mt-0.5" />
                <div>
                  <strong className="text-amber-600 dark:text-amber-300 font-bold">實戰目標：</strong>{' '}
                  {currentStep.objective}
                </div>
              </div>

              {/* Code Snippet Box */}
              {currentStep.codeSnippet && (
                <div className="rounded-xl border border-border bg-[#090d16] overflow-hidden shadow-inner">
                  <div className="flex items-center justify-between border-b border-border/80 bg-[#0f1422] px-3 py-1.5 text-xs text-muted-foreground">
                    <span className="flex items-center gap-1.5 font-mono text-[11px] text-cyan-400">
                      <Terminal className="h-3.5 w-3.5" />
                      Kaggle Python 實戰代碼 ({currentStep.codeLanguage || 'python'})
                    </span>
                    <button
                      type="button"
                      onClick={() => handleCopyCode(currentStep.codeSnippet)}
                      className="flex items-center gap-1 text-[11px] text-slate-300 hover:text-white cursor-pointer px-2 py-0.5 rounded bg-slate-800/60 hover:bg-slate-800 transition-colors"
                    >
                      {copiedCode ? (
                        <>
                          <Check className="h-3 w-3 text-emerald-400" />
                          <span className="text-emerald-400 font-semibold">已複製步驟代碼</span>
                        </>
                      ) : (
                        <>
                          <Copy className="h-3 w-3" />
                          <span>複製此步驟代碼</span>
                        </>
                      )}
                    </button>
                  </div>
                  <div className="p-3.5 overflow-x-auto bg-[#090d16]/95 max-h-[380px]">
                    <pre className="font-mono text-xs text-slate-100 leading-relaxed m-0 whitespace-pre">
                      <code>{currentStep.codeSnippet}</code>
                    </pre>
                  </div>
                </div>
              )}

              {/* Engineering Takeaways */}
              <div className="rounded-lg border border-border bg-muted/40 p-3 space-y-1.5">
                <h5 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
                  <Sparkles className="h-3.5 w-3.5 text-amber-500 dark:text-amber-400" />
                  <span>工程要點與 Kaggle 實踐建議</span>
                </h5>
                <ul className="space-y-1 text-xs text-foreground/90 list-disc pl-4">
                  {currentStep.takeaways.map((item, i) => (
                    <li key={i}>{item}</li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Bottom Step Navigation Controls */}
            <div className="flex flex-wrap items-center justify-between gap-3 pt-2 text-xs text-muted-foreground border-t border-border">
              <div className="flex items-center gap-1.5 font-mono text-[11px]">
                <FileCode className="h-3.5 w-3.5 text-amber-500 dark:text-amber-400" />
                <span>Showcase：{tutorial.kaggleNotebook}</span>
              </div>
              <div className="flex items-center gap-2">
                {activeStep > 0 && (
                  <button
                    type="button"
                    onClick={() => setActiveStep((prev) => prev - 1)}
                    className="px-3 py-1 rounded-lg border border-border hover:bg-muted text-foreground text-xs cursor-pointer transition-colors"
                  >
                    上一則步驟
                  </button>
                )}
                {activeStep < tutorial.steps.length - 1 ? (
                  <button
                    type="button"
                    onClick={() => setActiveStep((prev) => prev + 1)}
                    className="px-3 py-1 rounded-lg bg-amber-500 text-slate-950 font-bold text-xs hover:bg-amber-400 transition-colors cursor-pointer flex items-center gap-1"
                  >
                    <span>下一步驟</span>
                    <ChevronRight className="h-3 w-3" />
                  </button>
                ) : isUnlocked ? (
                  <button
                    type="button"
                    onClick={() => setActiveTab('star')}
                    className="px-3 py-1 rounded-lg bg-emerald-500 text-slate-950 font-bold text-xs hover:bg-emerald-400 transition-colors cursor-pointer flex items-center gap-1"
                  >
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    <span>查看 STAR 答辯稿</span>
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={onClose}
                    className="px-3 py-1 rounded-lg bg-primary text-primary-foreground font-semibold text-xs hover:opacity-90 transition-opacity cursor-pointer flex items-center gap-1"
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
          <div className="space-y-4 pt-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Target className="h-4 w-4 text-amber-500 dark:text-amber-400" />
                <h4 className="text-sm font-bold text-foreground">
                  STAR 系統架構決策分析 (Situation · Task · Action · Result)
                </h4>
              </div>
              <button
                type="button"
                onClick={handleCopyStarPlaybook}
                className="flex items-center gap-1 rounded-lg border border-amber-500/30 bg-amber-500/10 px-2.5 py-1 text-xs font-semibold text-amber-500 dark:text-amber-300 hover:bg-amber-500/20 cursor-pointer"
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
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {/* Situation */}
              <div className="rounded-xl border border-rose-500/30 bg-rose-500/8 p-3.5 space-y-2">
                <div className="flex items-center gap-2">
                  <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-rose-500/20 text-rose-500 dark:text-rose-400 font-bold font-mono text-xs">
                    S
                  </span>
                  <div>
                    <h5 className="text-xs font-bold text-rose-500 dark:text-rose-300 uppercase tracking-wide">
                      Situation · 業務背景與技術挑戰
                    </h5>
                    <span className="text-[10px] text-muted-foreground">痛點與未解決的問題</span>
                  </div>
                </div>
                <p className="text-xs text-foreground/90 leading-relaxed pl-8">
                  {tutorial.starPlaybook.situation}
                </p>
              </div>

              {/* Task */}
              <div className="rounded-xl border border-amber-500/30 bg-amber-500/8 p-3.5 space-y-2">
                <div className="flex items-center gap-2">
                  <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-amber-500/20 text-amber-500 dark:text-amber-400 font-bold font-mono text-xs">
                    T
                  </span>
                  <div>
                    <h5 className="text-xs font-bold text-amber-500 dark:text-amber-300 uppercase tracking-wide">
                      Task · 目標任務與約束條件
                    </h5>
                    <span className="text-[10px] text-muted-foreground">資源限制與成功判準</span>
                  </div>
                </div>
                <p className="text-xs text-foreground/90 leading-relaxed pl-8">
                  {tutorial.starPlaybook.task}
                </p>
              </div>

              {/* Action */}
              <div className="rounded-xl border border-cyan-500/30 bg-cyan-500/8 p-3.5 space-y-2">
                <div className="flex items-center gap-2">
                  <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-cyan-500/20 text-cyan-500 dark:text-cyan-400 font-bold font-mono text-xs">
                    A
                  </span>
                  <div>
                    <h5 className="text-xs font-bold text-cyan-500 dark:text-cyan-300 uppercase tracking-wide">
                      Action · 核心工程解法與架構設計
                    </h5>
                    <span className="text-[10px] text-muted-foreground">具體演算法與系統防禦</span>
                  </div>
                </div>
                <p className="text-xs text-foreground/90 leading-relaxed pl-8">
                  {tutorial.starPlaybook.action}
                </p>
              </div>

              {/* Result */}
              <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/8 p-3.5 space-y-2">
                <div className="flex items-center gap-2">
                  <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-emerald-500/20 text-emerald-500 dark:text-emerald-400 font-bold font-mono text-xs">
                    R
                  </span>
                  <div>
                    <h5 className="text-xs font-bold text-emerald-500 dark:text-emerald-300 uppercase tracking-wide">
                      Result · 量化產出與實證影響力
                    </h5>
                    <span className="text-[10px] text-muted-foreground">指標提升、延遲縮減與驗證</span>
                  </div>
                </div>
                <p className="text-xs text-foreground/90 leading-relaxed pl-8">
                  {tutorial.starPlaybook.result}
                </p>
              </div>
            </div>

            {/* Defense Tips Card */}
            <div className="rounded-xl border border-border bg-muted/40 p-4 space-y-2">
              <h5 className="text-xs font-bold text-foreground flex items-center gap-1.5">
                <ShieldCheck className="h-4 w-4 text-primary" />
                <span>系統架構防守攻略 (Defense Playbook)</span>
              </h5>
              <p className="text-xs text-muted-foreground leading-relaxed">
                在進行架構評審與同行審查時，牢牢守住 <strong>硬體邊界（{tutorial.hardwareRequirements}）</strong> 與{' '}
                <strong>量化評估基線</strong>。強調在資源約束下的架構選型取捨 (Trade-offs)，並主動關注代碼的防作弊 (Reward Hacking) 機制。
              </p>
            </div>
          </div>
        )}

        {/* Tab 3: Frontier Lab Interview Q&A */}
        {isUnlocked && activeTab === 'interview' && (
          <div className="space-y-3.5 pt-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <HelpCircle className="h-4 w-4 text-purple-400" />
                <h4 className="text-sm font-bold text-foreground">
                  前沿實驗室核心架構思辨題 (Frontier Lab Architecture Deep Dive)
                </h4>
              </div>
              <Badge variant="violet" className="text-[10px]">
                Tier-1 Architecture Q&A
              </Badge>
            </div>

            <div className="space-y-3">
              {tutorial.interviewQA && tutorial.interviewQA.length > 0 ? (
                tutorial.interviewQA.map((qa, i) => (
                  <div
                    key={i}
                    className="rounded-xl border border-purple-500/30 bg-purple-500/8 p-4 space-y-2.5 transition-all"
                  >
                    <div className="flex items-start gap-2.5">
                      <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-purple-500/25 text-purple-500 dark:text-purple-300 font-mono text-xs font-bold mt-0.5">
                        Q{i + 1}
                      </span>
                      <h5 className="text-xs sm:text-sm font-bold text-purple-600 dark:text-purple-200 leading-snug">
                        {qa.question}
                      </h5>
                    </div>

                    <div className="ml-7 rounded-lg border border-purple-500/20 bg-card/80 p-3.5">
                      <div className="text-[11px] font-semibold text-purple-500 dark:text-purple-400 uppercase tracking-wider mb-1">
                        Frontier Lab 專家級架構設計解析：
                      </div>
                      <p className="text-xs text-foreground/90 leading-relaxed whitespace-pre-line">
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
          <div className="space-y-4 pt-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Award className="h-4 w-4 text-primary" />
                <h4 className="text-sm font-bold text-foreground">
                  工程實踐亮點與成果摘要 (Portfolio & Technical Highlights)
                </h4>
              </div>
              <Badge variant="secondary" className="text-[10px]">
                Production Aligned
              </Badge>
            </div>

            <p className="text-xs text-muted-foreground">
              直接複製符合頂級 AI 系統架構標準的 STAR 英文/中文工程成果亮點，可自由替換在訓練產出的實際指標數值。
            </p>

            {/* Bullet Variations */}
            <div className="space-y-3">
              {bulletVariants.map((item, idx) => (
                <div
                  key={idx}
                  className="rounded-xl border border-border bg-card/60 p-3.5 space-y-2"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-primary flex items-center gap-1.5">
                      <Sparkles className="h-3.5 w-3.5 text-amber-500 dark:text-amber-400" />
                      {item.label}
                    </span>
                    <button
                      type="button"
                      onClick={() => handleCopyBullet(item.bullet, idx)}
                      className="flex items-center gap-1 text-[11px] font-medium text-primary hover:text-primary/80 bg-primary/10 hover:bg-primary/20 border border-primary/30 px-2.5 py-0.5 rounded-lg transition-colors cursor-pointer"
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

                  <div className="rounded-lg border border-border bg-muted/50 p-3 font-mono text-xs text-foreground leading-relaxed select-all">
                    • {item.bullet}
                  </div>
                </div>
              ))}
            </div>

            {/* Resume Tips Box */}
            <div className="rounded-xl border border-amber-500/25 bg-amber-500/10 p-3 text-xs text-amber-600 dark:text-amber-200/90 space-y-1">
              <strong className="text-amber-600 dark:text-amber-300 font-bold block">
                💡 工程亮點提煉黃金原則 (Google X-Y-Z Formula)：
              </strong>
              <span>
                "Accomplished [X] as measured by [Y], by doing [Z]" — 建議將上述亮點放置在開源專案或技術設計文檔中，並附上您的 Notebook 開源連結。
              </span>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};

