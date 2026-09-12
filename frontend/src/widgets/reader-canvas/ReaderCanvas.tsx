import React, { useEffect, useState, useRef, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import Prism from 'prismjs';
import 'prismjs/components/prism-python';
import 'prismjs/components/prism-bash';
import 'prismjs/components/prism-json';
import 'prismjs/components/prism-typescript';
import 'prismjs/components/prism-javascript';
import {
  Clock,
  FlaskConical,
  CheckCircle2,
  Circle,
  ArrowLeft,
  ArrowRight,
  Sparkles,
  Copy,
  Check,
  Info,
  Flame,
  AlertTriangle,
  ShieldAlert,
  ExternalLink,
  WrapText,
  Trophy,
  List,
  Network,
  Terminal,
  Code2,
} from 'lucide-react';
import type { ChapterData } from '@/shared/types';
import { getManifest, getAllChapters } from '@/entities/manifest';
import { useChapterStore } from '@/entities/chapter/chapterStore';
import { loadChapterData } from '@/entities/chapter/chapterLoader';
import { MermaidRenderer } from '@/shared/lib/MermaidRenderer';
import { Button } from '@/shared/ui/Button';
import { Badge } from '@/shared/ui/Badge';
import { cn } from '@/shared/lib/utils';

interface ReaderCanvasProps {
  onOpenLab?: (labId: string) => void;
}

// Helper to extract plain text from React nodes (for GitHub-style alert detection)
const extractNodeText = (node: any): string => {
  if (!node) return '';
  if (typeof node === 'string') return node;
  if (typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(extractNodeText).join('');
  if (node.props?.children) return extractNodeText(node.props.children);
  return '';
};

// Alert callout component supporting GitHub [!IMPORTANT], [!NOTE], [!TIP], [!WARNING], [!CAUTION]
const AlertBlockquote: React.FC<{ children?: React.ReactNode }> = ({ children }) => {
  const rawText = extractNodeText(children).trim();
  const alertMatch = rawText.match(/^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]/i);

  if (alertMatch) {
    const alertType = alertMatch[1].toUpperCase();

    const configs: Record<string, { border: string; bg: string; text: string; label: string; icon: any }> = {
      IMPORTANT: {
        border: 'border-l-violet-500 border-border/60',
        bg: 'bg-violet-500/[0.04]',
        text: 'text-violet-400 dark:text-violet-300',
        label: 'IMPORTANT · 核心考點與工業界陷阱',
        icon: Flame,
      },
      NOTE: {
        border: 'border-l-sky-500 border-border/60',
        bg: 'bg-sky-500/[0.04]',
        text: 'text-sky-400 dark:text-sky-300',
        label: 'NOTE · 重要說明',
        icon: Info,
      },
      TIP: {
        border: 'border-l-emerald-500 border-border/60',
        bg: 'bg-emerald-500/[0.04]',
        text: 'text-emerald-400 dark:text-emerald-300',
        label: 'TIP · 實戰技巧',
        icon: Sparkles,
      },
      WARNING: {
        border: 'border-l-amber-500 border-border/60',
        bg: 'bg-amber-500/[0.04]',
        text: 'text-amber-400 dark:text-amber-300',
        label: 'WARNING · 警告注意',
        icon: AlertTriangle,
      },
      CAUTION: {
        border: 'border-l-rose-500 border-border/60',
        bg: 'bg-rose-500/[0.04]',
        text: 'text-rose-400 dark:text-rose-300',
        label: 'CAUTION · 高危陷阱',
        icon: ShieldAlert,
      },
    };

    const config = configs[alertType] || configs.NOTE;
    const IconComp = config.icon;

    // Strips the [!TYPE] marker from children
    const stripMarker = (node: any): any => {
      if (!node) return node;
      if (typeof node === 'string') {
        return node.replace(/^\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*/i, '');
      }
      if (Array.isArray(node)) {
        let stripped = false;
        return node.map((child) => {
          if (!stripped) {
            const text = extractNodeText(child);
            if (/^\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]/i.test(text)) {
              stripped = true;
              return stripMarker(child);
            }
          }
          return child;
        });
      }
      if (React.isValidElement(node) && (node.props as any)?.children) {
        return React.cloneElement(node, {
          ...(node.props as any),
          children: stripMarker((node.props as any).children),
        });
      }
      return node;
    };

    const cleanedChildren = stripMarker(children);

    return (
      <div className={`my-3.5 rounded-r-lg rounded-l-xs border border-l-4 ${config.border} ${config.bg} p-3 sm:p-3.5 text-foreground/90 shadow-xs`}>
        <div className={`flex items-center gap-2 font-mono font-semibold text-[10.5px] uppercase tracking-wider mb-1 ${config.text}`}>
          <IconComp className="h-3.5 w-3.5 shrink-0" />
          <span>{config.label}</span>
        </div>
        <div className="space-y-1 text-[12.5px] sm:text-[13px] leading-relaxed text-foreground/85">
          {cleanedChildren}
        </div>
      </div>
    );
  }

  // Standard blockquote
  return (
    <blockquote className="my-3.5 border-l-3 border-primary/60 bg-muted/20 rounded-r-lg py-2 px-3.5 text-foreground/85 italic text-[12.5px] sm:text-[13px] leading-relaxed font-normal">
      {children}
    </blockquote>
  );
};

// Tri-Modal CodeBlock component: Architectural Blueprint / Terminal Telemetry / Source Code
const CodeBlock: React.FC<{ lang: string; codeString: string }> = ({ lang, codeString }) => {
  const [copied, setCopied] = useState(false);
  const [isWrapped, setIsWrapped] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(codeString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // 1. Detect if this block is an ASCII Architecture Map or Systems Topology Blueprint
  const isAsciiDiagram = useMemo(() => {
    if (lang === 'ascii' || lang === 'diagram' || lang === 'blueprint') return true;
    const isExplicitOutput =
      codeString.trim().startsWith('[Execution Output') ||
      codeString.trim().startsWith('[Output') ||
      codeString.trim().startsWith('[Telemetry Log');
    if (isExplicitOutput) return false;

    // Detect box-drawing characters or topology indicators
    const boxChars = ['┌', '─', '│', '└', '├', '╔', '═', '║', '╚', '╠', '▼', '▲', '◀', '▶'];
    return boxChars.some((char) => codeString.includes(char));
  }, [lang, codeString]);

  // 2. Detect if this block is an Execution Output / Telemetry Log
  const isOutputBlock = useMemo(() => {
    if (isAsciiDiagram) return false;
    return (
      lang === 'output' ||
      lang === 'terminal' ||
      lang === 'console' ||
      lang === 'log' ||
      codeString.trim().startsWith('[Execution Output') ||
      codeString.trim().startsWith('[Output') ||
      codeString.trim().startsWith('[Telemetry Log')
    );
  }, [isAsciiDiagram, lang, codeString]);

  // Prism syntax highlighting for source code
  const highlightedHtml = useMemo(() => {
    if (isOutputBlock || isAsciiDiagram) return null;
    const normalizedLang = lang ? lang.toLowerCase().trim() : 'python';
    const grammar = Prism.languages[normalizedLang] || Prism.languages.python;
    if (grammar) {
      try {
        return Prism.highlight(codeString, grammar, normalizedLang);
      } catch (e) {
        return null;
      }
    }
    return null;
  }, [codeString, lang, isOutputBlock, isAsciiDiagram]);

  // Case 1: Architectural Topology Blueprint Canvas
  if (isAsciiDiagram) {
    return (
      <div className="group relative my-4 overflow-hidden rounded-lg ascii-blueprint-container transition-all">
        {/* Top Blueprint Header Bar */}
        <div className="flex items-center justify-between border-b border-cyan-500/20 bg-muted/40 px-3 py-1.5 text-xs">
          <div className="flex items-center gap-2">
            <Network className="h-3.5 w-3.5 text-cyan-400 shrink-0" />
            <span className="font-mono text-[10px] sm:text-[10.5px] font-semibold uppercase tracking-wider text-cyan-400 dark:text-cyan-300">
              SYSTEM TOPOLOGY & ARCHITECTURAL BLUEPRINT
            </span>
          </div>

          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => setIsWrapped(!isWrapped)}
              className="rounded p-1 text-muted-foreground hover:text-foreground hover:bg-muted/60 transition-colors cursor-pointer"
              title={isWrapped ? '切換為不換行 (橫向滾動)' : '切換為自動換行'}
            >
              <WrapText className="h-3.5 w-3.5" />
            </button>
            <button
              type="button"
              onClick={handleCopy}
              className="rounded p-1 text-muted-foreground hover:text-foreground hover:bg-muted/60 transition-colors cursor-pointer"
              title="複製架構拓撲圖"
            >
              {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
            </button>
          </div>
        </div>

        {/* Monospace Blueprint Canvas */}
        <div className="overflow-x-auto p-3 sm:p-4 text-[11px] sm:text-[11.5px] ascii-blueprint-canvas">
          <pre className={cn("m-0 p-0 font-mono ascii-blueprint-text bg-transparent border-none", isWrapped ? "whitespace-pre-wrap break-all" : "whitespace-pre")}>
            <code>{codeString}</code>
          </pre>
        </div>
      </div>
    );
  }

  // Case 2: Specialized Sleek Terminal / Execution Output Block (Interactive Notebook / Jupyter style)
  if (isOutputBlock) {
    let displayTitle = '';
    let cleanedOutput = codeString;

    const titleMatch = codeString.match(/^\s*\[(?:Execution Output|Output|Telemetry Log)(?:\s*\/\s*([^\]]+))?\]\s*\n?/i);
    if (titleMatch) {
      displayTitle = titleMatch[1]?.trim() ? titleMatch[1].trim() : '';
      cleanedOutput = codeString.slice(titleMatch[0].length);
    }

    return (
      <div className="group relative -mt-2 mb-4 overflow-hidden rounded-lg terminal-telemetry-container transition-all">
        {/* Terminal Header Bar with Micro-dots & Title */}
        <div className="flex items-center justify-between border-b border-emerald-500/20 bg-black/40 px-3 py-1.5 text-xs">
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1 select-none">
              <span className="h-2 w-2 rounded-full bg-red-500/80 inline-block" />
              <span className="h-2 w-2 rounded-full bg-amber-500/80 inline-block" />
              <span className="h-2 w-2 rounded-full bg-emerald-500/80 inline-block" />
            </div>
            <Terminal className="h-3 w-3 text-emerald-400 ml-1 shrink-0" />
            <span className="font-mono text-[10px] font-semibold uppercase tracking-wider text-emerald-400">
              RUNTIME TELEMETRY {displayTitle ? `· ${displayTitle}` : ''}
            </span>
          </div>

          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => setIsWrapped(!isWrapped)}
              className="rounded p-1 text-slate-400 hover:text-emerald-300 hover:bg-emerald-950/40 transition-colors cursor-pointer"
              title={isWrapped ? '切換為不換行 (橫向滾動)' : '切換為自動換行'}
            >
              <WrapText className="h-3.5 w-3.5" />
            </button>
            <button
              type="button"
              onClick={handleCopy}
              className="rounded p-1 text-slate-400 hover:text-emerald-300 hover:bg-emerald-950/40 transition-colors cursor-pointer"
              title="複製執行輸出"
            >
              {copied ? (
                <Check className="h-3.5 w-3.5 text-emerald-400" />
              ) : (
                <Copy className="h-3.5 w-3.5" />
              )}
            </button>
          </div>
        </div>

        {/* Output Gutter & Content (Interactive Notebook Out: flow) */}
        <div className="flex items-start gap-2.5 p-2.5 sm:p-3 font-mono leading-relaxed">
          <div className="shrink-0 select-none text-emerald-500/80 font-mono text-[10px] pt-0.5 font-semibold">
            Out:
          </div>
          <div className="flex-1 min-w-0 overflow-x-auto">
            <pre className={cn("m-0 p-0 font-mono text-[11px] sm:text-[11.5px] text-emerald-300 leading-relaxed font-normal bg-transparent border-none", isWrapped ? "whitespace-pre-wrap break-all" : "whitespace-pre")}>
              <code>{cleanedOutput}</code>
            </pre>
          </div>
        </div>
      </div>
    );
  }

  // Case 3: Regular Source Code Block (Python, TS, Bash, etc. - Interactive Notebook In: flow)
  return (
    <div className="group relative my-3.5 overflow-hidden rounded-lg code-surface-container transition-all">
      {/* Code Header Toolbar */}
      <div className="flex items-center justify-between border-b border-border/50 bg-muted/30 px-3 py-1.5 text-xs">
        <div className="flex items-center gap-2">
          <Code2 className="h-3.5 w-3.5 text-sky-400 shrink-0" />
          <span className="font-mono text-[10.5px] font-semibold uppercase tracking-wider text-muted-foreground">
            {lang || 'python'}
          </span>
        </div>

        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => setIsWrapped(!isWrapped)}
            className="rounded p-1 text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors cursor-pointer"
            title={isWrapped ? '切換為不換行 (橫向滾動)' : '切換為自動換行'}
          >
            <WrapText className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            onClick={handleCopy}
            className="rounded p-1 text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors cursor-pointer"
            title="複製程式碼"
          >
            {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
          </button>
        </div>
      </div>

      {/* Code Area with In: Gutter */}
      <div className="flex items-start gap-2.5 p-3 sm:p-3.5">
        <div className="shrink-0 select-none text-sky-400/80 font-mono text-[10px] pt-0.5 font-semibold">
          In:
        </div>
        <div className="flex-1 min-w-0 overflow-x-auto">
          <pre className={cn("m-0 p-0 font-mono text-[11.5px] sm:text-[12px] leading-relaxed bg-transparent border-none", isWrapped ? "whitespace-pre-wrap break-all" : "whitespace-pre")}>
            {highlightedHtml ? (
              <code
                className="font-mono text-foreground"
                dangerouslySetInnerHTML={{ __html: highlightedHtml }}
              />
            ) : (
              <code className="font-mono text-foreground">{codeString}</code>
            )}
          </pre>
        </div>
      </div>
    </div>
  );
};

// MarkdownLink component to intercept relative .md tutorial links for seamless SPA navigation
const MarkdownLink: React.FC<{
  href?: string;
  children?: React.ReactNode;
  onNavigateChapter: (chapterId: string) => void;
  allChapters: Array<{ id: string; num: string; file?: string }>;
}> = ({ href = '', children, onNavigateChapter, allChapters }) => {
  const normalizedHref = href
    .replace(/^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?\//, '');

  const isMdLink = normalizedHref.endsWith('.md') || normalizedHref.includes('.md#');

  if (isMdLink) {
    const cleanHref = normalizedHref.split('#')[0].replace(/^\.\//, '').replace(/\.md$/, '');

    const matched = allChapters.find((ch) => {
      if (ch.file === cleanHref || ch.id === cleanHref) return true;
      const numMatch = cleanHref.match(/^(\d+)/);
      if (numMatch) {
        const num = numMatch[1].padStart(2, '0');
        return ch.num === num || ch.id.endsWith(num);
      }
      return false;
    });

    if (matched) {
      return (
        <a
          href={`#${matched.id}`}
          onClick={(e) => {
            e.preventDefault();
            onNavigateChapter(matched.id);
          }}
          className="inline-flex items-center gap-1 font-semibold text-cyan-400 hover:text-cyan-300 underline underline-offset-4 decoration-cyan-500/50 hover:decoration-cyan-300 transition-colors cursor-pointer"
        >
          <span>{children}</span>
        </a>
      );
    }
  }

  if (normalizedHref.startsWith('#')) {
    const rawTarget = normalizedHref.replace(/^#/, '');
    const cleanTarget = rawTarget.replace(/^guide\//, '');
    const matchedChapter = allChapters.find((ch) => ch.id === cleanTarget || ch.id === rawTarget);

    if (matchedChapter) {
      return (
        <a
          href={`#${matchedChapter.id}`}
          onClick={(e) => {
            e.preventDefault();
            onNavigateChapter(matchedChapter.id);
          }}
          className="inline-flex items-center gap-1 font-semibold text-cyan-400 hover:text-cyan-300 underline underline-offset-4 decoration-cyan-500/50 hover:decoration-cyan-300 transition-colors cursor-pointer"
        >
          <span>{children}</span>
        </a>
      );
    }

    return (
      <a
        href={normalizedHref}
        onClick={(e) => {
          const el = document.getElementById(rawTarget);
          if (el) {
            e.preventDefault();
            el.scrollIntoView({ behavior: 'smooth' });
          }
        }}
        className="inline-flex items-center gap-1 font-medium text-cyan-400 hover:text-cyan-300 underline underline-offset-4 decoration-cyan-500/50 hover:decoration-cyan-300 transition-colors cursor-pointer"
      >
        <span>{children}</span>
      </a>
    );
  }

  const isExternal = href.startsWith('http://') || href.startsWith('https://');
  return (
    <a
      href={href}
      target={isExternal ? '_blank' : undefined}
      rel={isExternal ? 'noopener noreferrer' : undefined}
      className="inline-flex items-center gap-1 font-medium text-cyan-400 hover:text-cyan-300 underline underline-offset-4 decoration-cyan-500/50 hover:decoration-cyan-300 transition-colors"
    >
      <span>{children}</span>
      {isExternal && <ExternalLink className="inline h-3 w-3 ml-0.5 opacity-70" />}
    </a>
  );
};

export const ReaderCanvas: React.FC<ReaderCanvasProps> = ({ onOpenLab }) => {
  const {
    activeSite,
    setActiveSite,
    currentChapterId,
    setCurrentChapterId,
    isChapterDone,
    toggleChapterDone,
    openMilestoneTutorial,
    resetToHome,
  } = useChapterStore();

  const [chapter, setChapter] = useState<ChapterData | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeHeadingId, setActiveHeadingId] = useState<string>('');
  const articleRef = useRef<HTMLElement>(null);

  const manifest = getManifest(activeSite);
  const isDone = isChapterDone(currentChapterId);

  // Extract In-Page Table of Contents (H2 and H3 headings)
  const tocHeadings = useMemo(() => {
    if (!chapter?.markdownContent) return [];
    const lines = chapter.markdownContent.split('\n');
    const items: Array<{ id: string; text: string; level: number }> = [];

    for (const line of lines) {
      const match = line.match(/^(#{2,3})\s+(.+)$/);
      if (match) {
        const level = match[1].length;
        const rawText = match[2].trim().replace(/\*\*([^*]+)\*\*/g, '$1').replace(/`([^`]+)`/g, '$1');
        const cleanText = rawText.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');
        const id = cleanText.toLowerCase().replace(/[^\w\u4e00-\u9fa5\s-]/g, '').replace(/\s+/g, '-');
        items.push({ id, text: cleanText, level });
      }
    }
    return items;
  }, [chapter?.markdownContent]);

  // Scrollspy: track active heading in reading body
  useEffect(() => {
    if (!chapter?.markdownContent) return;

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActiveHeadingId(entry.target.id);
            break;
          }
        }
      },
      { rootMargin: '-60px 0px -70% 0px', threshold: 0.1 }
    );

    const headingEls = document.querySelectorAll('article h2[id], article h3[id]');
    headingEls.forEach((el) => observer.observe(el));

    return () => {
      headingEls.forEach((el) => observer.unobserve(el));
    };
  }, [chapter?.markdownContent]);

  // Handle deep-link buttons within article content
  useEffect(() => {
    const articleEl = articleRef.current;
    if (!articleEl) return;

    const handleDeepLinkClick = (e: MouseEvent) => {
      const btn = (e.target as HTMLElement).closest('.deep-link-btn');
      if (!btn) return;
      e.preventDefault();

      const target = btn.getAttribute('data-target');
      if (target === 'viz' && chapter?.hasVisualizer) {
        onOpenLab?.(chapter.hasVisualizer);
      }
    };

    articleEl.addEventListener('click', handleDeepLinkClick);
    return () => articleEl.removeEventListener('click', handleDeepLinkClick);
  }, [chapter, onOpenLab]);

  // Load chapter data on change
  useEffect(() => {
    let isMounted = true;
    setLoading(true);

    loadChapterData(activeSite, currentChapterId).then((data) => {
      if (isMounted) {
        setChapter(data);
        setLoading(false);
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    });

    return () => {
      isMounted = false;
    };
  }, [activeSite, currentChapterId]);

  // Determine current Part, and Next / Previous chapters
  const allChapters = getAllChapters(activeSite);
  const currentIndex = allChapters.findIndex((c) => c.id === currentChapterId);
  const prevChapter = currentIndex > 0 ? allChapters[currentIndex - 1] : null;
  const nextChapter = currentIndex < allChapters.length - 1 ? allChapters[currentIndex + 1] : null;

  const currentPart = manifest.parts.find((p) =>
    p.chapters.some((c) => c.id === currentChapterId)
  ) || manifest.parts[0];

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center p-12">
        <div className="flex flex-col items-center gap-3">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-cyan-400 border-t-transparent" />
          <span className="text-xs text-slate-400 font-mono">載入章節資料中...</span>
        </div>
      </div>
    );
  }

  if (!chapter) {
    return (
      <div className="flex-1 flex items-center justify-center p-12 text-center text-slate-400">
        章節載入失敗，請重試
      </div>
    );
  }

  return (
    <main className="h-full w-full overflow-y-auto overflow-x-hidden bg-background px-3.5 pt-14 pb-8 sm:px-8 sm:py-8 md:px-10 lg:px-12 text-foreground transition-colors duration-200 selection:bg-cyan-500/20 selection:text-cyan-200">
      <div className="mx-auto max-w-7xl xl:flex xl:gap-12 xl:justify-center">
        {/* Main Reading Column */}
        <div className="w-full max-w-3xl lg:max-w-4xl min-w-0 space-y-7">
          {/* 1. Sleek Breadcrumbs (Sphinx RTD Style) */}
          <nav className="flex flex-wrap items-center gap-1.5 sm:gap-2 text-[11px] sm:text-xs text-muted-foreground/80 font-medium">
            <button
              type="button"
              onClick={resetToHome}
              className="hover:text-primary transition-colors cursor-pointer"
              title="返回首頁 (Home)"
            >
              Docs
            </button>
            <span>/</span>
            <button
              type="button"
              onClick={() => setActiveSite(manifest.id)}
              className="text-foreground/80 hover:text-primary transition-colors cursor-pointer truncate max-w-[100px] sm:max-w-none"
              title={`切換至 ${manifest.name}`}
            >
              {manifest.shortName}
            </button>
            <span>/</span>
            <span className="text-primary truncate max-w-[130px] sm:max-w-none">{currentPart.label.split('·')[0].trim()}</span>
            <span className="ml-auto font-mono text-[10.5px] sm:text-[11px] text-muted-foreground">
              {chapter.readTime || '15 min'}
            </span>
          </nav>

          {/* 2. Publication Header: Title, Summary & Sleek Badge Strip */}
          <div className="space-y-4 border-b border-border/60 pb-6">
            <div className="inline-flex items-center gap-2 rounded-full border border-primary/30 bg-primary/10 px-3 py-0.5 text-xs font-semibold text-primary">
              <span>{currentPart.label}</span>
            </div>

            <h1 className="font-title text-lg sm:text-2xl md:text-[1.65rem] font-bold tracking-tight text-foreground leading-snug break-words">
              {chapter.num} · {chapter.title}
            </h1>

            {chapter.summary && (
              <p className="text-[13px] sm:text-[14px] text-muted-foreground leading-relaxed font-normal">
                {chapter.summary}
              </p>
            )}

            {/* Sleek Resource & Action Strip (ReadTheDocs Badges Style) */}
            <div className="flex flex-wrap items-center gap-2 pt-1 text-xs">
              {/* Kaggle Milestone Playbook Pill */}
              {currentPart.milestone && (
                <button
                  type="button"
                  onClick={() => openMilestoneTutorial(currentPart.id)}
                  className="inline-flex items-center gap-1.5 rounded-full border border-amber-500/35 bg-amber-500/10 px-2.5 sm:px-3 py-1 text-[11px] sm:text-xs font-semibold text-amber-300 hover:bg-amber-500/20 hover:border-amber-500/60 transition-all cursor-pointer shadow-xs"
                  title="開啟 Kaggle 實戰里程碑步驟教程 (STAR 答辯 & 履歷亮點)"
                >
                  <Trophy className="h-3.5 w-3.5 text-amber-400 shrink-0" />
                  <span className="truncate">Kaggle 實戰：{currentPart.milestone.split('(')[0].trim()}</span>
                </button>
              )}

              {/* Simulation Lab Pill */}
              {chapter.hasVisualizer && (
                <button
                  type="button"
                  onClick={() => onOpenLab?.(chapter.hasVisualizer!)}
                  className="inline-flex items-center gap-1.5 rounded-full border border-purple-500/35 bg-purple-500/10 px-2.5 sm:px-3 py-1 text-[11px] sm:text-xs font-semibold text-purple-300 hover:bg-purple-500/20 hover:border-purple-500/60 transition-all cursor-pointer shadow-xs"
                  title={`開啟仿真實驗室: ${chapter.hasVisualizer}`}
                >
                  <FlaskConical className="h-3.5 w-3.5 text-purple-400 shrink-0" />
                  <span>仿真實驗室: {chapter.hasVisualizer}</span>
                </button>
              )}

              {/* Read Time Pill */}
              <span className="inline-flex items-center gap-1 rounded-full border border-border/60 bg-muted/40 px-2.5 py-1 font-mono text-[10.5px] sm:text-[11px] text-muted-foreground">
                <Clock className="h-3 w-3 text-primary" />
                <span>{chapter.readTime || '15 min'}</span>
              </span>

              {/* Competency Badges */}
              {chapter.competencies && chapter.competencies.map((comp) => (
                <Badge key={comp} variant="secondary" className="text-[10.5px] sm:text-[11px] py-0.5">
                  {comp}
                </Badge>
              ))}

              {/* Mark Completed Button */}
              <div className="w-full sm:w-auto sm:ml-auto pt-1 sm:pt-0">
                <Button
                  variant={isDone ? 'secondary' : 'default'}
                  size="sm"
                  onClick={() => toggleChapterDone(chapter.id)}
                  className="w-full sm:w-auto cursor-pointer text-xs h-7 px-3 justify-center"
                >
                  {isDone ? (
                    <>
                      <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                      <span>已完成學習</span>
                    </>
                  ) : (
                    <>
                      <Circle className="h-3.5 w-3.5" />
                      <span>標記為已完成</span>
                    </>
                  )}
                </Button>
              </div>
            </div>
          </div>

          {/* Mobile In-Page Table of Contents Dropdown (< xl) */}
          {tocHeadings.length > 0 && (
            <details className="xl:hidden my-2 rounded-xl border border-border/70 bg-card/50 p-2.5 text-xs group">
              <summary className="flex items-center justify-between font-mono font-semibold text-muted-foreground cursor-pointer select-none">
                <span className="flex items-center gap-1.5 text-foreground/90 text-[11.5px]">
                  <List className="h-3.5 w-3.5 text-primary" />
                  <span>本頁章節目錄 ({tocHeadings.length})</span>
                </span>
                <span className="text-[10px] text-primary">展開導航</span>
              </summary>
              <nav className="mt-2.5 space-y-1 pt-2 border-t border-border/50 max-h-56 overflow-y-auto">
                {tocHeadings.map((heading) => (
                  <a
                    key={heading.id}
                    href={`#${heading.id}`}
                    onClick={(e) => {
                      e.preventDefault();
                      const el = document.getElementById(heading.id);
                      if (el) el.scrollIntoView({ behavior: 'smooth' });
                    }}
                    className={cn(
                      "block py-1 transition-colors leading-snug cursor-pointer",
                      heading.level === 3 ? "pl-3 text-[10.5px] text-muted-foreground hover:text-foreground" : "text-[11.5px] font-medium text-foreground/85 hover:text-primary",
                      activeHeadingId === heading.id && "text-primary font-semibold"
                    )}
                  >
                    {heading.text}
                  </a>
                ))}
              </nav>
            </details>
          )}

          {/* 3. Main Chapter Body (Markdown / HTML) */}
          <article ref={articleRef} className="prose prose-slate dark:prose-invert max-w-none prose-p:leading-[1.68] prose-p:text-[13.5px] sm:prose-p:text-[14px] prose-headings:font-title prose-headings:tracking-tight prose-headings:scroll-mt-20 prose-pre:p-0 prose-pre:my-0 prose-pre:bg-transparent prose-table:my-0">
            {chapter.markdownContent ? (
              <ReactMarkdown
                remarkPlugins={[remarkMath, remarkGfm]}
                rehypePlugins={[[rehypeKatex, { output: 'html' }]]}
                components={{
                  h1({ children, ...props }) {
                    const text = extractNodeText(children).trim();
                    if (
                      text.toLowerCase().includes(chapter.title.toLowerCase()) ||
                      new RegExp(`^chapter\\s*${chapter.num}\\b`, 'i').test(text)
                    ) {
                      return null;
                    }
                    return (
                      <h1 className="font-title text-lg sm:text-xl md:text-[1.45rem] font-bold tracking-tight text-foreground mt-8 mb-4 pb-2 border-b border-border/60" {...props}>
                        {children}
                      </h1>
                    );
                  },
                  h2({ children, ...props }) {
                    const text = extractNodeText(children).trim();
                    const id = text.toLowerCase().replace(/[^\w\u4e00-\u9fa5\s-]/g, '').replace(/\s+/g, '-');
                    return (
                      <h2 id={id} className="font-title text-base sm:text-lg md:text-[1.2rem] font-semibold text-foreground mt-8 mb-3 pb-1.5 border-b border-border/50 flex items-center justify-between group scroll-mt-20" {...props}>
                        <span>{children}</span>
                        <a href={`#${id}`} className="opacity-0 group-hover:opacity-100 transition-opacity text-primary/50 hover:text-primary font-mono text-sm select-none pr-1 cursor-pointer" title="Permalink to this headline">
                          #
                        </a>
                      </h2>
                    );
                  },
                  h3({ children, ...props }) {
                    const text = extractNodeText(children).trim();
                    const id = text.toLowerCase().replace(/[^\w\u4e00-\u9fa5\s-]/g, '').replace(/\s+/g, '-');
                    return (
                      <h3 id={id} className="font-title text-[13.5px] sm:text-[14.5px] font-semibold text-foreground/95 mt-6 mb-2 flex items-center justify-between group scroll-mt-20" {...props}>
                        <span>{children}</span>
                        <a href={`#${id}`} className="opacity-0 group-hover:opacity-100 transition-opacity text-primary/40 hover:text-primary font-mono text-[11px] select-none pr-1 cursor-pointer" title="Permalink to this sub-headline">
                          ##
                        </a>
                      </h3>
                    );
                  },
                  h4({ children, ...props }) {
                    return (
                      <h4 className="font-title text-[12.5px] sm:text-[13px] font-semibold text-foreground/90 mt-4.5 mb-1.5" {...props}>
                        {children}
                      </h4>
                    );
                  },
                  p({ children, ...props }) {
                    return (
                      <p className="my-3.5 leading-[1.68] text-foreground/88 text-[13.5px] sm:text-[14px] font-normal tracking-normal" {...props}>
                        {children}
                      </p>
                    );
                  },
                  ul({ children, ...props }) {
                    return (
                      <ul className="my-3.5 pl-5 list-disc space-y-1 text-foreground/88 text-[13px] sm:text-[13.5px] leading-[1.65] marker:text-primary/50" {...props}>
                        {children}
                      </ul>
                    );
                  },
                  ol({ children, ...props }) {
                    return (
                      <ol className="my-3.5 pl-5 list-decimal space-y-1 text-foreground/88 text-[13px] sm:text-[13.5px] leading-[1.65] marker:text-primary/70 font-medium" {...props}>
                        {children}
                      </ol>
                    );
                  },
                  li({ children, ...props }) {
                    return (
                      <li className="leading-[1.65] pl-0.5 font-normal text-foreground/88" {...props}>
                        {children}
                      </li>
                    );
                  },
                  hr({ ...props }) {
                    return <hr className="my-8 border-border/50" {...props} />;
                  },
                  table({ children, ...props }) {
                    return (
                      <div className="my-4 overflow-x-auto rounded-lg border border-border/70 bg-card/40 shadow-xs max-w-full">
                        <table className="w-full border-collapse text-left text-xs min-w-[480px]" {...props}>
                          {children}
                        </table>
                      </div>
                    );
                  },
                  thead({ children, ...props }) {
                    return (
                      <thead className="bg-muted/70 text-[11px] font-mono font-semibold uppercase tracking-wider text-muted-foreground border-b border-border/70" {...props}>
                        {children}
                      </thead>
                    );
                  },
                  tbody({ children, ...props }) {
                    return <tbody className="divide-y divide-border/40 text-foreground/85" {...props}>{children}</tbody>;
                  },
                  tr({ children, ...props }) {
                    return (
                      <tr className="hover:bg-muted/30 transition-colors" {...props}>
                        {children}
                      </tr>
                    );
                  },
                  th({ children, ...props }) {
                    return (
                      <th className="px-3.5 py-2 font-semibold text-foreground font-mono text-[11px]" {...props}>
                        {children}
                      </th>
                    );
                  },
                  td({ children, ...props }) {
                    return (
                      <td className="px-3.5 py-2 text-[12px] sm:text-[12.5px] leading-relaxed text-foreground/85 font-normal" {...props}>
                        {children}
                      </td>
                    );
                  },
                  blockquote({ children }) {
                    return <AlertBlockquote>{children}</AlertBlockquote>;
                  },
                  a({ href, children }) {
                    return (
                      <MarkdownLink
                        href={href}
                        onNavigateChapter={setCurrentChapterId}
                        allChapters={allChapters}
                      >
                        {children}
                      </MarkdownLink>
                    );
                  },
                  pre({ children }) {
                    return <>{children}</>;
                  },
                  code({ className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || '');
                    const lang = match ? match[1] : '';
                    const codeString = String(children || '').replace(/\n$/, '');

                    if (lang === 'mermaid') {
                      return <MermaidRenderer chart={codeString} />;
                    }

                    // Inline code
                    if (!className && !codeString.includes('\n')) {
                      return (
                        <code
                          className="rounded bg-muted/80 border border-border/50 px-1.5 py-0.5 text-cyan-400 dark:text-cyan-300 font-mono text-[11.5px] sm:text-[12px] font-normal"
                          {...props}
                        >
                          {children}
                        </code>
                      );
                    }

                    // Block code
                    return (
                      <CodeBlock lang={lang} codeString={codeString} />
                    );
                  },
                }}
              >
                {chapter.markdownContent}
              </ReactMarkdown>
            ) : chapter.html ? (
              <div
                className="space-y-4"
                dangerouslySetInnerHTML={{ __html: chapter.html }}
              />
            ) : (
              <p className="text-muted-foreground">本章節正在載入詳細論述...</p>
            )}
          </article>

          {/* 4. Footer Navigation Buttons */}
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 border-t border-border pt-6">
            {prevChapter ? (
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setCurrentChapterId(prevChapter.id);
                  window.scrollTo({ top: 0, behavior: 'smooth' });
                }}
                className="flex items-center justify-center sm:justify-start gap-2 text-xs cursor-pointer py-2 sm:py-1.5"
              >
                <ArrowLeft className="h-3.5 w-3.5 shrink-0" />
                <span className="truncate">上一章：{prevChapter.num} {prevChapter.title.split('(')[0].trim()}</span>
              </Button>
            ) : (
              <div className="hidden sm:block" />
            )}

            {nextChapter ? (
              <Button
                variant="default"
                size="sm"
                onClick={() => {
                  setCurrentChapterId(nextChapter.id);
                  window.scrollTo({ top: 0, behavior: 'smooth' });
                }}
                className="flex items-center justify-center sm:justify-end gap-2 text-xs cursor-pointer py-2 sm:py-1.5"
              >
                <span className="truncate">下一章：{nextChapter.num} {nextChapter.title.split('(')[0].trim()}</span>
                <ArrowRight className="h-3.5 w-3.5 shrink-0" />
              </Button>
            ) : (
              <div className="hidden sm:block" />
            )}
          </div>
        </div>

        {/* Right Sticky Rail: On this page (TOC) - ReadTheDocs In-Page Outline */}
        {tocHeadings.length > 0 && (
          <aside className="hidden xl:block w-60 shrink-0">
            <div className="sticky top-8 max-h-[calc(100vh-5rem)] overflow-y-auto pl-4 border-l border-border/50 text-xs space-y-3.5 scrollbar-thin">
              <div className="flex items-center gap-1.5 font-mono text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                <List className="h-3.5 w-3.5 text-primary" />
                <span>On this page · 本頁導航</span>
              </div>
              <nav className="space-y-1">
                {tocHeadings.map((heading) => (
                  <a
                    key={heading.id}
                    href={`#${heading.id}`}
                    onClick={(e) => {
                      e.preventDefault();
                      const el = document.getElementById(heading.id);
                      if (el) el.scrollIntoView({ behavior: 'smooth' });
                    }}
                    className={cn(
                      "block py-0.5 transition-colors leading-snug cursor-pointer",
                      heading.level === 3 ? "pl-3 text-[10.5px] text-muted-foreground hover:text-foreground" : "text-[11.5px] font-medium text-foreground/80 hover:text-primary",
                      activeHeadingId === heading.id && "text-primary font-semibold border-l-2 border-primary -ml-[17px] pl-[15px]"
                    )}
                  >
                    {heading.text}
                  </a>
                ))}
              </nav>
            </div>
          </aside>
        )}
      </div>
    </main>
  );
};
