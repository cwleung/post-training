import React, { useState, useMemo, useRef, useEffect } from 'react';
import {
  Bot,
  Brain,
  Rocket,
  ChevronRight,
  Sun,
  Moon,
  Search,
  X,
  PanelLeftClose,
  ChevronsUpDown,
  FlaskConical,
  Trophy,
} from 'lucide-react';
import { SITES, getManifest } from '@/entities/manifest';
import { useChapterStore } from '@/entities/chapter/chapterStore';
import { cn } from '@/shared/lib/utils';
import { Badge } from '@/shared/ui/Badge';
import { usePrivacyStore } from '@/entities/chapter/privacyStore';
import type { ChapterSummary } from '@/shared/types';

interface SidebarNavigationProps {
  className?: string;
}

export const SidebarNavigation: React.FC<SidebarNavigationProps> = ({ className = '' }) => {
  const {
    activeSite,
    setActiveSite,
    currentChapterId,
    setCurrentChapterId,
    sidebarOpen,
    toggleSidebar,
    setSidebarOpen,
    theme,
    setTheme,
    openMilestoneTutorial,
  } = useChapterStore();

  const closeSidebarIfMobile = () => {
    if (typeof window !== 'undefined' && window.innerWidth < 768) {
      setSidebarOpen(false);
    }
  };

  const { isUnlocked, openUnlockModal } = usePrivacyStore();

  const [collapsedParts, setCollapsedParts] = useState<Record<string, boolean>>({});
  const [searchQuery, setSearchQuery] = useState('');
  const [labsOnly, setLabsOnly] = useState(false);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const clickCountRef = useRef(0);
  const clickTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const manifest = getManifest(activeSite);

  const handleSecretClick = () => {
    clickCountRef.current += 1;
    if (clickTimerRef.current) clearTimeout(clickTimerRef.current);

    if (clickCountRef.current >= 5) {
      clickCountRef.current = 0;
      openUnlockModal();
      return;
    }

    clickTimerRef.current = setTimeout(() => {
      clickCountRef.current = 0;
    }, 2000);
  };

  // Keyboard shortcut: '/' focuses search bar, 'Escape' clears it, 'Ctrl+Shift+U' unlocks career console
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Secret developer shortcut: Ctrl+Shift+U or Cmd+Shift+U
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && (e.key === 'U' || e.key === 'u')) {
        e.preventDefault();
        openUnlockModal();
        return;
      }

      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        if (e.key === 'Escape') {
          setSearchQuery('');
          (e.target as HTMLElement).blur();
        }
        return;
      }

      if (e.key === '/' || (e.key === 'k' && (e.metaKey || e.ctrlKey))) {
        e.preventDefault();
        searchInputRef.current?.focus();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [openUnlockModal]);

  const togglePart = (partId: string) => {
    setCollapsedParts((prev) => ({ ...prev, [partId]: !prev[partId] }));
  };

  const toggleAllParts = () => {
    const allCollapsed = manifest.parts.every((p) => collapsedParts[p.id]);
    const nextState: Record<string, boolean> = {};
    manifest.parts.forEach((p) => {
      nextState[p.id] = !allCollapsed;
    });
    setCollapsedParts(nextState);
  };

  const toggleTheme = () => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
  };

  // Track-specific active styling
  const trackTheme = {
    deepagents: {
      activePill: 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30 shadow-cyan-500/10',
      activeItem: 'bg-cyan-500/10 text-cyan-300 border-l-2 border-cyan-400 font-semibold',
    },
    rl: {
      activePill: 'bg-purple-500/15 text-purple-300 border-purple-500/30 shadow-purple-500/10',
      activeItem: 'bg-purple-500/10 text-purple-300 border-l-2 border-purple-400 font-semibold',
    },
    rlvr: {
      activePill: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30 shadow-emerald-500/10',
      activeItem: 'bg-emerald-500/10 text-emerald-300 border-l-2 border-emerald-400 font-semibold',
    },
  }[activeSite];

  // Filtered chapters according to search query, labs-only toggle, and privacy unlock
  const filteredParts = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();

    return manifest.parts
      .map((part) => {
        const matchingChapters = part.chapters.filter((ch) => {
          if (ch.isCareerGated && !isUnlocked) return false;
          if (labsOnly && !ch.hasVisualizer && ch.tag !== 'LAB') return false;

          if (!q) return true;
          const matchNum = ch.num.toLowerCase().includes(q);
          const matchTitle = ch.title.toLowerCase().includes(q);
          const matchTag = ch.tag?.toLowerCase().includes(q);
          const matchSummary = ch.summary?.toLowerCase().includes(q);
          const matchId = ch.id.toLowerCase().includes(q);
          return matchNum || matchTitle || matchTag || matchSummary || matchId;
        });

        return {
          ...part,
          chapters: matchingChapters,
        };
      })
      .filter((part) => part.chapters.length > 0);
  }, [manifest, searchQuery, labsOnly, isUnlocked]);

  const totalFilteredChapters = useMemo(() => {
    return filteredParts.reduce((acc, part) => acc + part.chapters.length, 0);
  }, [filteredParts]);

  const allChaptersCount = useMemo(() => {
    return manifest.parts.reduce(
      (acc, part) => acc + part.chapters.filter((ch) => !ch.isCareerGated || isUnlocked).length,
      0
    );
  }, [manifest, isUnlocked]);

  return (
    <aside
      className={cn(
        'fixed inset-y-0 left-0 z-50 flex h-full w-[285px] sm:w-[300px] max-w-[85vw] flex-col border-r border-sidebar-border bg-sidebar text-sidebar-foreground backdrop-blur-md transition-all duration-300 select-none shadow-2xl md:shadow-none md:relative md:z-30 md:max-w-none',
        sidebarOpen
          ? 'translate-x-0 md:w-[300px] md:min-w-[300px] md:max-w-[300px]'
          : '-translate-x-full pointer-events-none md:pointer-events-auto md:w-0 md:min-w-0 md:max-w-0 md:overflow-hidden md:border-r-0 md:translate-x-0',
        className
      )}
    >
      {/* 1. Sleek Brand Header & Action Controls */}
      <div className="border-b border-sidebar-border p-3.5 space-y-3 bg-card/40">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <button
              type="button"
              onClick={handleSecretClick}
              className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-tr from-cyan-500 via-blue-600 to-indigo-600 text-white font-bold text-xs shadow-md shadow-cyan-500/25 ring-1 ring-white/20 hover:opacity-95 active:scale-95 transition-all cursor-default"
              title="ψ DeepAgents Core"
            >
              ψ
            </button>
            <div className="leading-tight">
              <div className="flex items-center gap-1.5">
                <span className="font-title text-sm font-bold text-foreground tracking-tight">
                  DeepAgents
                </span>
                <button
                  type="button"
                  onClick={handleSecretClick}
                  className="rounded bg-primary/15 px-1 py-0.2 text-[9px] font-mono font-medium text-primary hover:bg-primary/25 cursor-default transition-colors"
                  title="Version 4.0"
                >
                  v4
                </button>
                {isUnlocked && (
                  <button
                    type="button"
                    onClick={openUnlockModal}
                    className="flex items-center gap-0.5 text-[9px] text-emerald-400 hover:text-emerald-300 font-mono px-1 py-0.2 rounded bg-emerald-500/10 border border-emerald-500/20"
                    title="Developer Authorization Active (Click to manage)"
                  >
                    🔓
                  </button>
                )}
              </div>
              <span className="block text-[10px] text-muted-foreground font-mono">
                {allChaptersCount} 交互章節 · 18 實驗室
              </span>
            </div>
          </div>

          <div className="flex items-center gap-1">
            {/* Theme Toggle Button */}
            <button
              type="button"
              onClick={toggleTheme}
              className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-all cursor-pointer"
              title={theme === 'dark' ? '切換為淺色模式' : '切換為深色模式'}
            >
              {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>

            {/* Collapse Sidebar Button */}
            <button
              type="button"
              onClick={toggleSidebar}
              className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-all cursor-pointer"
              title="收合側邊欄"
            >
              <PanelLeftClose className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* 2. Floating Segmented 3-Way Track Switcher */}
        <div className="grid grid-cols-3 gap-1 rounded-xl border border-border bg-background/60 p-1 shadow-inner">
          <button
            type="button"
            onClick={() => setActiveSite('deepagents')}
            className={cn(
              'flex flex-col items-center justify-center py-1.5 px-1 rounded-lg text-xs font-medium transition-all cursor-pointer border border-transparent',
              activeSite === 'deepagents'
                ? cn(trackTheme.activePill, 'shadow-sm')
                : 'text-muted-foreground hover:text-foreground hover:bg-muted/60'
            )}
            title="DeepAgents 代理架構與評估體系"
          >
            <span className="flex items-center gap-1 text-[11px] font-semibold">
              <Bot className="h-3 w-3" /> Agents
            </span>
            <span className="text-[9px] font-mono opacity-70">
              {SITES.deepagents.totalChapters} 講
            </span>
          </button>

          <button
            type="button"
            onClick={() => setActiveSite('rl')}
            className={cn(
              'flex flex-col items-center justify-center py-1.5 px-1 rounded-lg text-xs font-medium transition-all cursor-pointer border border-transparent',
              activeSite === 'rl'
                ? cn(trackTheme.activePill, 'shadow-sm')
                : 'text-muted-foreground hover:text-foreground hover:bg-muted/60'
            )}
            title="強化學習與後訓練對齊實驗室"
          >
            <span className="flex items-center gap-1 text-[11px] font-semibold">
              <Brain className="h-3 w-3" /> RL
            </span>
            <span className="text-[9px] font-mono opacity-70">
              {SITES.rl.totalChapters} 講
            </span>
          </button>

          <button
            type="button"
            onClick={() => setActiveSite('rlvr')}
            className={cn(
              'flex flex-col items-center justify-center py-1.5 px-1 rounded-lg text-xs font-medium transition-all cursor-pointer border border-transparent',
              activeSite === 'rlvr'
                ? cn(trackTheme.activePill, 'shadow-sm')
                : 'text-muted-foreground hover:text-foreground hover:bg-muted/60'
            )}
            title="Post-Training MLE 實戰"
          >
            <span className="flex items-center gap-1 text-[11px] font-semibold">
              <Rocket className="h-3 w-3" /> Post-Train
            </span>
            <span className="text-[9px] font-mono opacity-70">
              {SITES.rlvr.totalChapters} 講
            </span>
          </button>
        </div>

        {/* 3. Search Bar & Controls */}
        <div className="space-y-1.5">
          <div className="relative flex items-center">
            <Search className="absolute left-2.5 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
            <input
              ref={searchInputRef}
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="搜尋章節 (快捷鍵 /)..."
              className="w-full rounded-lg border border-border bg-card/80 py-1.5 pl-8 pr-7 text-xs text-foreground placeholder:text-muted-foreground focus:border-primary/60 focus:outline-none focus:ring-1 focus:ring-primary/40 transition-all"
            />
            {searchQuery ? (
              <button
                type="button"
                onClick={() => setSearchQuery('')}
                className="absolute right-2 rounded p-0.5 text-muted-foreground hover:text-foreground cursor-pointer"
              >
                <X className="h-3 w-3" />
              </button>
            ) : (
              <kbd className="absolute right-2 rounded border border-border bg-muted/70 px-1 py-0.2 text-[9px] font-mono text-muted-foreground">
                /
              </kbd>
            )}
          </div>

          {/* Quick Filters / Controls row */}
          <div className="flex items-center justify-between text-[10.5px] pt-0.5">
            <button
              type="button"
              onClick={() => setLabsOnly(!labsOnly)}
              className={cn(
                'flex items-center gap-1 rounded-md px-2 py-0.5 font-medium transition-colors cursor-pointer',
                labsOnly
                  ? 'bg-purple-500/20 text-purple-400 font-semibold'
                  : 'text-muted-foreground hover:text-foreground'
              )}
            >
              <FlaskConical className="h-2.5 w-2.5" /> 僅顯示實驗室
            </button>

            <button
              type="button"
              onClick={toggleAllParts}
              className="flex items-center gap-1 text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
              title="展開 / 折疊全部單元"
            >
              <ChevronsUpDown className="h-3 w-3" />
              <span>切換展開</span>
            </button>
          </div>
        </div>
      </div>

      {/* 4. Scrollable Course Outline */}
      <div className="flex-1 overflow-y-auto px-2.5 py-3 space-y-3.5 scrollbar-thin">
        {searchQuery && (
          <div className="flex items-center justify-between px-2 text-[11px] text-muted-foreground">
            <span>找到 {totalFilteredChapters} 個相關章節</span>
            <button
              type="button"
              onClick={() => setSearchQuery('')}
              className="text-primary hover:underline cursor-pointer"
            >
              清除搜尋
            </button>
          </div>
        )}

        {filteredParts.length === 0 ? (
          <div className="py-8 text-center text-xs text-muted-foreground space-y-2">
            <p>沒有符合條件的章節</p>
            <button
              type="button"
              onClick={() => {
                setSearchQuery('');
                setLabsOnly(false);
              }}
              className="text-primary hover:underline cursor-pointer text-xs"
            >
              重設篩選條件
            </button>
          </div>
        ) : (
          filteredParts.map((part) => {
            const isCollapsed = !!collapsedParts[part.id] && !searchQuery;

            return (
              <div key={part.id} className="space-y-1">
                {/* Part Section Header */}
                <button
                  type="button"
                  onClick={() => togglePart(part.id)}
                  className="flex w-full items-center justify-between rounded-lg px-2 py-1.5 text-xs font-semibold text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors group cursor-pointer"
                >
                  <div className="flex items-center gap-1.5 truncate">
                    <ChevronRight
                      className={cn(
                        'h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform duration-200',
                        !isCollapsed && 'rotate-90 text-foreground'
                      )}
                    />
                    <span className="truncate text-[11.5px] font-medium tracking-tight">
                      {part.label}
                    </span>
                  </div>

                  <span className="shrink-0 font-mono text-[10px] text-muted-foreground group-hover:text-foreground px-1">
                    {part.chapters.length}
                  </span>
                </button>

                {/* Chapters List */}
                {!isCollapsed && (
                  <div className="space-y-0.5 pl-1.5">
                    {part.milestone && (
                      <button
                        type="button"
                        onClick={() => {
                          openMilestoneTutorial(part.id);
                          closeSidebarIfMobile();
                        }}
                        className="mb-1.5 mt-0.5 flex w-full items-center justify-between gap-1.5 rounded-lg border border-amber-500/30 bg-amber-500/10 px-2.5 py-1.5 text-left text-amber-300 hover:bg-amber-500/20 hover:border-amber-500/60 transition-all cursor-pointer group shadow-xs"
                        title="開啟 Kaggle 實戰里程碑步驟教程 (Step-by-Step)"
                      >
                        <div className="flex items-center gap-2 truncate">
                          <Trophy className="h-3.5 w-3.5 shrink-0 text-amber-400 group-hover:scale-110 transition-transform" />
                          <div className="truncate leading-tight">
                            <span className="block text-[11px] font-bold text-amber-200 group-hover:text-amber-100">
                              Kaggle 實戰教程
                            </span>
                            <span className="block text-[9.5px] text-amber-300/80 truncate">
                              {part.milestone}
                            </span>
                          </div>
                        </div>
                        <ChevronRight className="h-3 w-3 shrink-0 text-amber-400/80 group-hover:translate-x-0.5 transition-transform" />
                      </button>
                    )}

                    {part.chapters.map((ch: ChapterSummary) => {
                      const isSelected = currentChapterId === ch.id;

                      return (
                        <div
                          key={ch.id}
                          className={cn(
                            'group flex w-full items-center justify-between rounded-lg px-2.5 py-1.5 text-left text-xs transition-all cursor-pointer',
                            isSelected
                              ? cn(trackTheme.activeItem, 'shadow-xs')
                              : 'text-muted-foreground hover:bg-muted/60 hover:text-foreground'
                          )}
                          onClick={() => {
                            setCurrentChapterId(ch.id);
                            closeSidebarIfMobile();
                          }}
                          role="button"
                          tabIndex={0}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' || e.key === ' ') {
                              e.preventDefault();
                              setCurrentChapterId(ch.id);
                              closeSidebarIfMobile();
                            }
                          }}
                        >
                          <div className="flex items-center gap-2 truncate pr-1">
                            <span
                              className={cn(
                                'font-mono text-[11px] shrink-0 w-5 text-right',
                                isSelected
                                  ? 'text-primary font-bold'
                                  : 'text-muted-foreground group-hover:text-foreground'
                              )}
                            >
                              {ch.num}
                            </span>
                            <span
                              className={cn(
                                'truncate text-[11.5px]',
                                isSelected && 'font-medium text-foreground'
                              )}
                              title={ch.title}
                            >
                              {ch.title.split('(')[0].trim()}
                            </span>
                          </div>

                          <div className="flex items-center gap-1.5 shrink-0 ml-1">
                            {/* Tag Badge */}
                            {ch.tag && (
                              <Badge
                                variant={ch.tag === 'LAB' ? 'violet' : 'outline'}
                                className="text-[9px] px-1 py-0 border-border text-muted-foreground font-mono"
                              >
                                {ch.tag}
                              </Badge>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* 5. Sleek Bottom Status Bar */}
      <div className="border-t border-sidebar-border bg-card/40 p-2.5 flex items-center justify-between text-xs text-muted-foreground">
        <div className="flex items-center gap-1.5">
          <span className="flex h-2 w-2 rounded-full bg-primary" />
          <span className="font-medium text-foreground/90 text-[11px]">
            {manifest.shortName}
          </span>
        </div>
        <div className="flex items-center gap-1 text-[10px] font-mono text-muted-foreground">
          <span>{allChaptersCount} 講</span>
          <span>·</span>
          <span>按 <kbd className="text-foreground font-bold">/</kbd> 搜尋</span>
        </div>
      </div>
    </aside>
  );
};
