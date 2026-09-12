import React, { useEffect, useRef, useState, useId } from 'react';
import mermaid from 'mermaid';
import { ZoomIn, ZoomOut, RotateCcw, AlertCircle } from 'lucide-react';
import { useChapterStore } from '../../entities/chapter/chapterStore';

interface MermaidRendererProps {
  chart: string;
  className?: string;
}

function sanitizeMermaid(code: string): string {
  let counter = 0;
  let cleaned = code.replace(/\r\n/g, '\n');

  // 1. Sanitize unquoted subgraphs with spaces, non-ascii, colons, or special chars
  cleaned = cleaned.replace(/^([ \t]*subgraph)\s+([^\n\[]+)$/gm, (match, prefix, title) => {
    const trimmed = title.trim();
    if (trimmed.startsWith('"') || /^[a-zA-Z0-9_]+$/.test(trimmed)) {
      return match;
    }
    counter++;
    const safeTitle = trimmed.replace(/"/g, "'");
    return `${prefix} sg_auto_${counter} ["${safeTitle}"]`;
  });

  // 2. Sanitize edge labels between pipe delimiters: |...| (parentheses, braces, brackets break parser)
  cleaned = cleaned.replace(/(-->|<-->|-\.->|==>)\|([^|\n]+)\|/g, (_match, arrow, label) => {
    const safeLabel = label
      .replace(/\(/g, '（')
      .replace(/\)/g, '）')
      .replace(/\{/g, '｛')
      .replace(/\}/g, '｝')
      .replace(/\[/g, '〔')
      .replace(/\]/g, '〕');
    return `${arrow}|${safeLabel}|`;
  });

  return cleaned;
}

export const MermaidRenderer: React.FC<MermaidRendererProps> = ({ chart, className = '' }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [svg, setSvg] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [scale, setScale] = useState<number>(1);
  const id = useId().replace(/:/g, '_');
  const theme = useChapterStore((s) => s.theme);

  useEffect(() => {
    let isMounted = true;

    async function renderChart() {
      if (!chart.trim()) return;
      const uniqueId = `mermaid_${id}_${Date.now()}`;
      try {
        setError(null);
        mermaid.initialize({
          startOnLoad: false,
          theme: theme === 'light' ? 'default' : 'dark',
          themeVariables:
            theme === 'light'
              ? {
                  darkMode: false,
                  background: '#ffffff',
                  primaryColor: '#f1f5f9',
                  primaryTextColor: '#0f172a',
                  primaryBorderColor: '#0284c7',
                  lineColor: '#64748b',
                  secondaryColor: '#f8fafc',
                  tertiaryColor: '#f1f5f9',
                }
              : {
                  darkMode: true,
                  background: '#0f1422',
                  primaryColor: '#1e293b',
                  primaryTextColor: '#f8fafc',
                  primaryBorderColor: '#38bdf8',
                  lineColor: '#64748b',
                  secondaryColor: '#131929',
                  tertiaryColor: '#090d16',
                },
          securityLevel: 'loose',
          fontFamily: 'Inter, system-ui, sans-serif',
        });

        const cleanChart = sanitizeMermaid(chart.trim());
        const { svg: renderedSvg } = await mermaid.render(uniqueId, cleanChart);
        if (isMounted) {
          setSvg(renderedSvg);
        }
      } catch (err: unknown) {
        // Clean up any stray error elements mermaid adds to document.body
        const errorEl = document.getElementById(`d${uniqueId}`);
        if (errorEl) errorEl.remove();

        if (isMounted) {
          console.warn('Mermaid rendering error:', err);
          setError(err instanceof Error ? err.message : String(err));
        }
      }
    }

    renderChart();

    return () => {
      isMounted = false;
    };
  }, [chart, id, theme]);

  const handleZoomIn = () => setScale((s) => Math.min(s + 0.2, 2.5));
  const handleZoomOut = () => setScale((s) => Math.max(s - 0.2, 0.5));
  const handleReset = () => setScale(1);

  if (error) {
    return (
      <div className="my-4 rounded-xl border border-rose-500/30 bg-rose-950/20 p-4 text-xs text-rose-300">
        <div className="flex items-center gap-2 font-semibold text-rose-400 mb-1">
          <AlertCircle className="h-4 w-4" /> Mermaid Diagram Render Error
        </div>
        <pre className="font-mono text-[11px] overflow-x-auto p-2 bg-rose-950/40 rounded border border-rose-900/50">
          {error}
        </pre>
      </div>
    );
  }

  return (
    <div className={`group relative my-6 overflow-hidden rounded-xl border border-border bg-card/80 p-4 shadow-sm backdrop-blur ${className}`}>
      {/* Zoom / Pan Controls Toolbar */}
      <div className="absolute right-3 top-3 z-10 flex items-center gap-1 rounded-lg border border-border/80 bg-background/90 p-1 opacity-0 backdrop-blur transition-opacity group-hover:opacity-100 shadow-sm">
        <button
          type="button"
          onClick={handleZoomIn}
          className="rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
          title="Zoom In"
        >
          <ZoomIn className="h-3.5 w-3.5" />
        </button>
        <button
          type="button"
          onClick={handleZoomOut}
          className="rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
          title="Zoom Out"
        >
          <ZoomOut className="h-3.5 w-3.5" />
        </button>
        <button
          type="button"
          onClick={handleReset}
          className="rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
          title="Reset Zoom"
        >
          <RotateCcw className="h-3.5 w-3.5" />
        </button>
      </div>

      <div
        ref={containerRef}
        className="flex items-center justify-center overflow-x-auto py-2 transition-transform duration-150"
        style={{ transform: `scale(${scale})`, transformOrigin: 'top center' }}
        dangerouslySetInnerHTML={{ __html: svg }}
      />
    </div>
  );
};
