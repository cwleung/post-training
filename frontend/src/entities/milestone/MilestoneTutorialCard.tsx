// frontend/src/entities/milestone/MilestoneTutorialCard.tsx
import React, { useState } from 'react';
import {
  Trophy,
  Copy,
  Check,
  ChevronDown,
  Briefcase,
  Sparkles,
  Terminal,
  Maximize2,
  Cpu,
  Clock,
  Target,
  Award,
} from 'lucide-react';
import { Badge } from '@/shared/ui/Badge';
import { cn } from '@/shared/lib/utils';
import { getMilestoneTutorial, type MilestoneTutorial } from './milestoneTutorials';
import type { ManifestPart } from '@/shared/types';

interface MilestoneTutorialCardProps {
  part: ManifestPart;
  onOpenModal?: () => void;
}

export const MilestoneTutorialCard: React.FC<MilestoneTutorialCardProps> = ({
  part,
  onOpenModal,
}) => {
  const [isExpanded, setIsExpanded] = useState<boolean>(false);
  const [activeStepIdx, setActiveStepIdx] = useState<number>(0);
  const [copiedCode, setCopiedCode] = useState<boolean>(false);
  const [copiedBullet, setCopiedBullet] = useState<boolean>(false);

  const tutorial: MilestoneTutorial = getMilestoneTutorial(part);
  const currentStep = tutorial.steps[activeStepIdx] || tutorial.steps[0];

  const handleCopyCode = (code?: string) => {
    if (!code) return;
    navigator.clipboard.writeText(code);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2000);
  };

  const handleCopyBullet = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(tutorial.portfolioBullet);
    setCopiedBullet(true);
    setTimeout(() => setCopiedBullet(false), 2000);
  };

  return (
    <div className="rounded-2xl border border-amber-500/35 bg-gradient-to-br from-amber-500/10 via-card to-card p-4 sm:p-5 text-xs text-foreground shadow-lg shadow-amber-950/10 space-y-3.5">
      {/* Header Row */}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-start gap-2.5">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-amber-500/20 border border-amber-500/40 text-amber-500 dark:text-amber-400 shadow-sm mt-0.5">
            <Trophy className="h-4 w-4" />
          </div>
          <div className="space-y-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-bold text-amber-500 dark:text-amber-300 text-sm sm:text-base tracking-tight font-title">
                {tutorial.title}
              </span>
              <Badge variant="amber" className="text-[9px] px-1.5 py-0">
                Kaggle 實戰
              </Badge>
            </div>

            {/* Hardware & Runtime Badges */}
            <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
              <span className="inline-flex items-center gap-1 text-primary font-medium">
                <Briefcase className="h-3 w-3" />
                <span>{tutorial.targetRole}</span>
              </span>
              <span className="text-muted-foreground/60">•</span>
              <span className="inline-flex items-center gap-1 text-amber-500 dark:text-amber-300/80">
                <Cpu className="h-3 w-3" />
                <span>{tutorial.hardwareRequirements}</span>
              </span>
              <span className="text-muted-foreground/60">•</span>
              <span className="inline-flex items-center gap-1 text-emerald-500 dark:text-emerald-300/80">
                <Clock className="h-3 w-3" />
                <span>{tutorial.expectedRuntime}</span>
              </span>
            </div>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2 ml-auto shrink-0">
          <button
            type="button"
            onClick={handleCopyBullet}
            className="hidden sm:inline-flex items-center gap-1 rounded-lg border border-primary/30 bg-primary/10 px-2 py-1 text-[11px] font-medium text-primary hover:bg-primary/20 transition-all cursor-pointer"
            title="複製履歷亮點"
          >
            {copiedBullet ? (
              <>
                <Check className="h-3 w-3 text-emerald-500" />
                <span className="text-emerald-500 font-bold">已複製履歷</span>
              </>
            ) : (
              <>
                <Award className="h-3 w-3" />
                <span>複製履歷亮點</span>
              </>
            )}
          </button>

          {/* Fullscreen modal trigger */}
          {onOpenModal && (
            <button
              type="button"
              onClick={onOpenModal}
              className="flex items-center gap-1 rounded-lg border border-amber-500/40 bg-amber-500/20 px-2.5 py-1 text-xs font-semibold text-amber-500 dark:text-amber-200 hover:bg-amber-500/30 transition-all cursor-pointer shadow-sm"
              title="以全螢幕彈窗開啟 Kaggle 實戰教程與 STAR 答辯工作台"
            >
              <Maximize2 className="h-3 w-3" />
              <span>全景答辯工作台</span>
            </button>
          )}

          {/* Inline expand/collapse toggle */}
          <button
            type="button"
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex items-center gap-1 rounded-lg border border-border bg-secondary text-secondary-foreground px-2.5 py-1 text-xs hover:bg-secondary/80 hover:text-foreground transition-all cursor-pointer"
          >
            <span>{isExpanded ? '收起步驟' : '展開步驟'}</span>
            <ChevronDown
              className={cn(
                'h-3.5 w-3.5 transition-transform duration-200',
                isExpanded && 'rotate-180'
              )}
            />
          </button>
        </div>
      </div>

      {/* STAR Quick Highlights Bar */}
      <div className="rounded-xl border border-border bg-background/70 p-2.5 text-[11px] grid grid-cols-1 sm:grid-cols-2 gap-2">
        <div className="flex items-start gap-1.5">
          <span className="font-bold text-rose-500 dark:text-rose-400 shrink-0 font-mono">S:</span>
          <span className="text-foreground/90 line-clamp-1">{tutorial.starPlaybook.situation}</span>
        </div>
        <div className="flex items-start gap-1.5">
          <span className="font-bold text-emerald-500 dark:text-emerald-400 shrink-0 font-mono">R:</span>
          <span className="text-foreground/90 line-clamp-1">{tutorial.starPlaybook.result}</span>
        </div>
      </div>

      {/* Expanded Step-by-Step Practice Guide */}
      {isExpanded && (
        <div className="pt-2 border-t border-border space-y-3">
          {/* Step selector pills */}
          <div className="flex items-center gap-1.5 overflow-x-auto pb-1">
            {tutorial.steps.map((step, idx) => (
              <button
                key={step.stepNumber}
                type="button"
                onClick={() => setActiveStepIdx(idx)}
                className={cn(
                  'flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-[11px] font-medium transition-all shrink-0 cursor-pointer',
                  activeStepIdx === idx
                    ? 'bg-amber-500 text-slate-950 font-bold shadow'
                    : 'bg-muted text-muted-foreground hover:text-foreground hover:bg-muted/80'
                )}
              >
                <span>步驟 {step.stepNumber}</span>
                <span className="opacity-80">· {step.badge}</span>
              </button>
            ))}
          </div>

          {/* Step Detail Card */}
          <div className="rounded-xl border border-border bg-card/90 p-3.5 space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-foreground text-xs flex items-center gap-1.5">
                <Sparkles className="h-3.5 w-3.5 text-amber-500 dark:text-amber-400" />
                {currentStep.title}
              </span>
              <span className="text-[10px] text-muted-foreground font-mono">
                {currentStep.badge}
              </span>
            </div>

            <div className="rounded-lg border border-amber-500/25 bg-amber-500/10 px-2.5 py-1.5 text-xs text-amber-600 dark:text-amber-200/90 flex items-start gap-1.5">
              <Target className="h-3.5 w-3.5 text-amber-500 dark:text-amber-400 shrink-0 mt-0.5" />
              <span>{currentStep.objective}</span>
            </div>

            {/* Code Snippet if present */}
            {currentStep.codeSnippet && (
              <div className="rounded-lg border border-border bg-[#090d16] overflow-hidden">
                <div className="flex items-center justify-between border-b border-border/80 bg-[#0f1422] px-2.5 py-1 text-[11px] text-muted-foreground">
                  <span className="flex items-center gap-1 font-mono text-cyan-400">
                    <Terminal className="h-3 w-3" />
                    Kaggle Python
                  </span>
                  <button
                    type="button"
                    onClick={() => handleCopyCode(currentStep.codeSnippet)}
                    className="flex items-center gap-1 text-slate-300 hover:text-white cursor-pointer"
                  >
                    {copiedCode ? (
                      <>
                        <Check className="h-3 w-3 text-emerald-400" />
                        <span className="text-emerald-400">已複製</span>
                      </>
                    ) : (
                      <>
                        <Copy className="h-3 w-3" />
                        <span>複製代碼</span>
                      </>
                    )}
                  </button>
                </div>
                <div className="p-2.5 overflow-x-auto max-h-[300px]">
                  <pre className="font-mono text-[11.5px] text-slate-100 leading-relaxed m-0 whitespace-pre">
                    <code>{currentStep.codeSnippet}</code>
                  </pre>
                </div>
              </div>
            )}

            {/* Takeaways */}
            <div className="space-y-1">
              <span className="text-[10.5px] font-semibold text-muted-foreground uppercase tracking-wider">
                關鍵工程要點：
              </span>
              <ul className="text-[11px] text-foreground/90 list-disc pl-4 space-y-0.5">
                {currentStep.takeaways.map((t, i) => (
                  <li key={i}>{t}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

