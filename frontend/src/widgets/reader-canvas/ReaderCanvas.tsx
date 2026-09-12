import React, { useEffect, useState, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
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
} from 'lucide-react';
import type { ChapterData } from '@/shared/types';
import { getManifest, getAllChapters } from '@/entities/manifest';
import { useChapterStore } from '@/entities/chapter/chapterStore';
import { loadChapterData } from '@/entities/chapter/chapterLoader';
import { MermaidRenderer } from '@/shared/lib/MermaidRenderer';
import { Button } from '@/shared/ui/Button';
import { Badge } from '@/shared/ui/Badge';
import { cn } from '@/shared/lib/utils';
import { MilestoneTutorialCard } from '@/entities/milestone';

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
        border: 'border-violet-500/40',
        bg: 'bg-violet-500/10',
        text: 'text-violet-500 dark:text-violet-300',
        label: 'IMPORTANT · 核心考點與工業界陷阱',
        icon: Flame,
      },
      NOTE: {
        border: 'border-cyan-500/40',
        bg: 'bg-cyan-500/10',
        text: 'text-cyan-600 dark:text-cyan-300',
        label: 'NOTE · 重要說明',
        icon: Info,
      },
      TIP: {
        border: 'border-emerald-500/40',
        bg: 'bg-emerald-500/10',
        text: 'text-emerald-600 dark:text-emerald-300',
        label: 'TIP · 實戰技巧',
        icon: Sparkles,
      },
      WARNING: {
        border: 'border-amber-500/40',
        bg: 'bg-amber-500/10',
        text: 'text-amber-600 dark:text-amber-300',
        label: 'WARNING · 警告注意',
        icon: AlertTriangle,
      },
      CAUTION: {
        border: 'border-rose-500/40',
        bg: 'bg-rose-500/10',
        text: 'text-rose-600 dark:text-rose-300',
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
      <div className={`my-6 rounded-xl border ${config.border} ${config.bg} p-4 sm:p-5 text-foreground/90 shadow-xl backdrop-blur`}>
        <div className={`flex items-center gap-2 font-title font-bold text-xs uppercase tracking-wider mb-3 ${config.text}`}>
          <IconComp className="h-4 w-4" />
          <span>{config.label}</span>
        </div>
        <div className="space-y-2 text-sm leading-relaxed">
          {cleanedChildren}
        </div>
      </div>
    );
  }

  // Standard blockquote
  return (
    <blockquote className="my-5 border-l-4 border-primary bg-muted/40 rounded-r-xl py-3 px-4 text-foreground/90 italic text-sm md:text-base">
      {children}
    </blockquote>
  );
};

// CodeBlock component with copy button and clean monospace formatting
const CodeBlock: React.FC<{ lang: string; codeString: string }> = ({ lang, codeString }) => {
  const [copied, setCopied] = useState(false);
  const [isWrapped, setIsWrapped] = useState(true);

  const handleCopy = () => {
    navigator.clipboard.writeText(codeString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="my-6 overflow-hidden rounded-xl border border-border bg-[#090d16] shadow-2xl">
      <div className="flex items-center justify-between border-b border-border/80 bg-[#0f1422] px-4 py-2 text-xs">
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full bg-rose-500/80" />
          <span className="h-2.5 w-2.5 rounded-full bg-amber-500/80" />
          <span className="h-2.5 w-2.5 rounded-full bg-emerald-500/80" />
          <span className="ml-2 font-mono text-[11px] font-semibold uppercase tracking-wider text-cyan-400">
            {lang || 'snippet'}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setIsWrapped(!isWrapped)}
            className="flex items-center gap-1 rounded border border-slate-700/60 bg-slate-800/80 px-2 py-0.5 text-[11px] text-slate-300 hover:text-white transition-colors cursor-pointer"
            title={isWrapped ? '切換為不換行 (橫向滾動)' : '切換為自動換行 (Code Wrap)'}
          >
            <WrapText className="h-3 w-3" />
            <span>{isWrapped ? '換行中' : '不換行'}</span>
          </button>
          <button
            type="button"
            onClick={handleCopy}
            className="flex items-center gap-1.5 rounded-md border border-slate-700/70 bg-slate-800/80 px-2.5 py-1 text-[11px] text-slate-300 hover:bg-slate-700 hover:text-white transition-colors cursor-pointer"
          >
            {copied ? (
              <>
                <Check className="h-3 w-3 text-emerald-400" />
                <span className="text-emerald-400">已複製</span>
              </>
            ) : (
              <>
                <Copy className="h-3 w-3" />
                <span>複製</span>
              </>
            )}
          </button>
        </div>
      </div>
      <div className="overflow-x-auto p-4 bg-[#090d16]/95">
        <pre className={cn("m-0 p-0 font-mono text-xs sm:text-[13px] text-slate-100 leading-relaxed font-normal bg-transparent border-none", isWrapped ? "whitespace-pre-wrap break-all" : "whitespace-pre")}>
          <code>{codeString}</code>
        </pre>
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
  // Normalize localhost URLs to relative paths
  const normalizedHref = href
    .replace(/^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?\//, '');

  const isMdLink = normalizedHref.endsWith('.md') || normalizedHref.includes('.md#');

  if (isMdLink) {
    const cleanHref = normalizedHref.split('#')[0].replace(/^\.\//, '').replace(/\.md$/, '');

    // Match by file, id, or numeric chapter prefix (e.g., '06_agentic_rlvr' -> prefix 06)
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

  // Handle in-page anchor links or chapter navigation smoothly
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
    currentChapterId,
    setCurrentChapterId,
    isChapterDone,
    toggleChapterDone,
    openMilestoneTutorial,
  } = useChapterStore();

  const [chapter, setChapter] = useState<ChapterData | null>(null);
  const [loading, setLoading] = useState(true);
  const articleRef = useRef<HTMLElement>(null);

  const manifest = getManifest(activeSite);
  const isDone = isChapterDone(currentChapterId);

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
    <main className="h-full w-full overflow-y-auto overflow-x-hidden bg-background p-4 md:p-8 text-foreground transition-colors duration-200">
      <div className="mx-auto max-w-4xl space-y-6">
        {/* 1. Breadcrumbs */}
        <nav className="flex items-center gap-2 text-xs text-muted-foreground font-medium">
          <span>Docs</span>
          <span>/</span>
          <span className="text-foreground/80">{manifest.shortName}</span>
          <span>/</span>
          <span className="text-primary">{currentPart.label.split('·')[0].trim()}</span>
          <span className="ml-auto font-mono text-[11px] text-muted-foreground">
            {chapter.readTime || '15 min'}
          </span>
        </nav>

        {/* 2. Chapter Title & Header Meta */}
        <div className="space-y-3 border-b border-border pb-5">
          <div className="inline-flex items-center gap-2 rounded-full border border-primary/30 bg-primary/10 px-3 py-1 text-xs font-semibold text-primary">
            <span>{currentPart.label}</span>
          </div>

          <h1 className="font-title text-2xl md:text-3xl font-bold tracking-tight text-foreground">
            {chapter.num} · {chapter.title}
          </h1>

          {chapter.summary && (
            <p className="text-sm text-muted-foreground leading-relaxed">
              {chapter.summary}
            </p>
          )}

          {/* Milestone Kaggle Practice Tutorial Card */}
          {currentPart.milestone && (
            <MilestoneTutorialCard
              part={currentPart}
              onOpenModal={() => openMilestoneTutorial(currentPart.id)}
            />
          )}

          {/* Competency tags */}
          {chapter.competencies && chapter.competencies.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5 pt-1">
              {chapter.competencies.map((comp) => (
                <Badge key={comp} variant="secondary" className="text-[11px]">
                  {comp}
                </Badge>
              ))}
            </div>
          )}

          {/* Action Toolbar */}
          <div className="flex flex-wrap items-center justify-between gap-3 pt-2 text-xs text-muted-foreground">
            <div className="flex items-center gap-4 font-mono text-[11.5px]">
              <span className="flex items-center gap-1">
                <Clock className="h-3.5 w-3.5 text-primary" />
                {chapter.readTime || '15 min'}
              </span>
              {chapter.hasVisualizer && (
                <button
                  type="button"
                  onClick={() => onOpenLab?.(chapter.hasVisualizer!)}
                  className="flex items-center gap-1 text-purple-500 dark:text-purple-400 hover:underline font-semibold cursor-pointer underline-offset-4"
                >
                  <FlaskConical className="h-3.5 w-3.5" />
                  <span>仿真實驗室: {chapter.hasVisualizer}</span>
                </button>
              )}
            </div>

            <Button
              variant={isDone ? 'secondary' : 'default'}
              size="sm"
              onClick={() => toggleChapterDone(chapter.id)}
              className="cursor-pointer"
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

        {/* 3. Main Chapter Body (Markdown / HTML) */}
        <article ref={articleRef} className="prose max-w-none">
          {chapter.markdownContent ? (
            <ReactMarkdown
              remarkPlugins={[remarkMath, remarkGfm]}
              rehypePlugins={[[rehypeKatex, { output: 'html' }]]}
              components={{
                h1({ children, ...props }) {
                  const text = extractNodeText(children).trim();
                  // Suppress redundant top-level h1 if it duplicates the chapter header card
                  if (
                    text.toLowerCase().includes(chapter.title.toLowerCase()) ||
                    new RegExp(`^chapter\\s*${chapter.num}\\b`, 'i').test(text)
                  ) {
                    return null;
                  }
                  return (
                    <h1 className="font-title text-2xl md:text-3xl font-bold tracking-tight text-foreground mt-8 mb-4" {...props}>
                      {children}
                    </h1>
                  );
                },
                h2({ children, ...props }) {
                  const text = extractNodeText(children).trim();
                  const id = text.toLowerCase().replace(/[^\w\u4e00-\u9fa5\s-]/g, '').replace(/\s+/g, '-');
                  return (
                    <h2 id={id} className="font-title text-xl md:text-2xl font-bold text-foreground mt-10 mb-4 pb-2 border-b border-border flex items-center gap-2 group" {...props}>
                      <span className="text-primary opacity-60 group-hover:opacity-100">#</span>
                      <span>{children}</span>
                    </h2>
                  );
                },
                h3({ children, ...props }) {
                  const text = extractNodeText(children).trim();
                  const id = text.toLowerCase().replace(/[^\w\u4e00-\u9fa5\s-]/g, '').replace(/\s+/g, '-');
                  return (
                    <h3 id={id} className="font-title text-lg md:text-xl font-semibold text-primary mt-7 mb-3" {...props}>
                      {children}
                    </h3>
                  );
                },
                h4({ children, ...props }) {
                  return (
                    <h4 className="font-title text-base font-semibold text-foreground mt-5 mb-2" {...props}>
                      {children}
                    </h4>
                  );
                },
                p({ children, ...props }) {
                  return (
                    <p className="my-3.5 leading-relaxed text-foreground/90 text-[15px]" {...props}>
                      {children}
                    </p>
                  );
                },
                ul({ children, ...props }) {
                  return (
                    <ul className="my-4 pl-6 list-disc space-y-2 text-foreground/90 marker:text-primary" {...props}>
                      {children}
                    </ul>
                  );
                },
                ol({ children, ...props }) {
                  return (
                    <ol className="my-4 pl-6 list-decimal space-y-2 text-foreground/90 marker:text-primary font-medium" {...props}>
                      {children}
                    </ol>
                  );
                },
                li({ children, ...props }) {
                  return (
                    <li className="leading-relaxed text-foreground/90 pl-1" {...props}>
                      {children}
                    </li>
                  );
                },
                hr({ ...props }) {
                  return <hr className="my-8 border-border" {...props} />;
                },
                table({ children, ...props }) {
                  return (
                    <div className="my-6 overflow-x-auto rounded-xl border border-border bg-card shadow-xl">
                      <table className="w-full border-collapse text-left text-sm" {...props}>
                        {children}
                      </table>
                    </div>
                  );
                },
                thead({ children, ...props }) {
                  return (
                    <thead className="bg-muted/80 text-xs uppercase tracking-wider text-foreground font-semibold border-b border-border" {...props}>
                      {children}
                    </thead>
                  );
                },
                tbody({ children, ...props }) {
                  return <tbody className="divide-y divide-border" {...props}>{children}</tbody>;
                },
                tr({ children, ...props }) {
                  return (
                    <tr className="hover:bg-muted/30 transition-colors group" {...props}>
                      {children}
                    </tr>
                  );
                },
                th({ children, ...props }) {
                  return (
                    <th className="px-4 py-3 font-semibold text-foreground font-mono text-xs" {...props}>
                      {children}
                    </th>
                  );
                },
                td({ children, ...props }) {
                  return (
                    <td className="px-4 py-3 text-foreground/90 text-[13.5px] leading-relaxed" {...props}>
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

                  // Inline code (no className and single-line)
                  if (!className && !codeString.includes('\n')) {
                    return (
                      <code
                        className="rounded-md bg-muted border border-border px-1.5 py-0.5 text-primary font-mono text-[12.5px] font-normal"
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


        {/* 5. Footer Navigation Buttons */}
        <div className="flex items-center justify-between border-t border-border pt-6">
          {prevChapter ? (
            <Button
              variant="outline"
              onClick={() => setCurrentChapterId(prevChapter.id)}
              className="flex items-center gap-2"
            >
              <ArrowLeft className="h-4 w-4" />
              <span>上一章：{prevChapter.num} {prevChapter.title.split('(')[0].trim()}</span>
            </Button>
          ) : (
            <div />
          )}

          {nextChapter ? (
            <Button
              variant="default"
              onClick={() => setCurrentChapterId(nextChapter.id)}
              className="flex items-center gap-2"
            >
              <span>下一章：{nextChapter.num} {nextChapter.title.split('(')[0].trim()}</span>
              <ArrowRight className="h-4 w-4" />
            </Button>
          ) : (
            <div />
          )}
        </div>
      </div>
    </main>
  );
};
