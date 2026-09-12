import React, { useState } from 'react';
import { Code, Copy, Check, Info, WrapText } from 'lucide-react';
import type { CodeLineItem } from '@/shared/types';
import { cn } from '@/shared/lib/utils';
import { Badge } from '@/shared/ui/Badge';

interface CodeInspectorProps {
  file?: string;
  lines?: CodeLineItem[];
  activeLine?: number | null;
  className?: string;
}

export const CodeInspector: React.FC<CodeInspectorProps> = ({ file, lines, activeLine, className = '' }) => {
  const [selectedLine, setSelectedLine] = useState<number | null>(null);
  const [isWrapped, setIsWrapped] = useState<boolean>(true);
  const [copied, setCopied] = useState<boolean>(false);

  React.useEffect(() => {
    if (activeLine) {
      setSelectedLine(activeLine);
    }
  }, [activeLine]);

  if (!lines || lines.length === 0) return null;

  const handleCopy = () => {
    const fullCode = lines.map((l) => l.text).join('\n');
    navigator.clipboard.writeText(fullCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleLineClick = (lineNum: number) => {
    setSelectedLine((prev) => (prev === lineNum ? null : lineNum));
  };

  // Determine file language display
  const fileName = file ? file.split('/').pop() || file : 'source_implementation.py';
  const fileExt = fileName.split('.').pop() || 'py';
  const languageLabel = fileExt === 'py' ? 'Python' : fileExt === 'js' || fileExt === 'ts' ? 'TypeScript' : fileExt.toUpperCase();

  return (
    <div className={cn('my-6 overflow-hidden rounded-xl border border-slate-800 bg-slate-900/90 shadow-2xl backdrop-blur', className)}>
      {/* ReadTheDocs-style Header Toolbar */}
      <div className="flex flex-wrap items-center justify-between border-b border-slate-800 bg-slate-950/80 px-4 py-2.5 gap-2">
        <div className="flex items-center gap-2">
          <Code className="h-4 w-4 text-cyan-400" />
          <span className="font-mono text-xs font-semibold text-slate-200">
            {file || 'Source Implementation'}
          </span>
          <Badge variant="secondary" className="text-[10px] py-0 px-1.5 font-mono">
            {languageLabel}
          </Badge>
          <span className="text-[11px] text-slate-500 font-mono">
            ({lines.length} lines)
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Wrap / No-wrap Toggle */}
          <button
            type="button"
            onClick={() => setIsWrapped(!isWrapped)}
            className={cn(
              'inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs transition-colors cursor-pointer',
              isWrapped
                ? 'border-cyan-500/40 bg-cyan-950/30 text-cyan-300'
                : 'border-slate-800 bg-slate-900 text-slate-400 hover:text-slate-200'
            )}
            title={isWrapped ? '切換為不換行 (橫向滾動)' : '切換為自動換行 (Code Wrap)'}
          >
            <WrapText className="h-3.5 w-3.5" />
            <span className="font-sans text-[11px]">{isWrapped ? '換行中' : '不換行'}</span>
          </button>

          {/* Copy Button */}
          <button
            type="button"
            onClick={handleCopy}
            className="inline-flex items-center gap-1.5 rounded-md border border-slate-700/60 bg-slate-800/80 px-2.5 py-1 text-xs text-slate-300 hover:bg-slate-700 hover:text-white transition-colors cursor-pointer"
          >
            {copied ? (
              <>
                <Check className="h-3.5 w-3.5 text-emerald-400" />
                <span className="text-emerald-400 font-sans text-[11px]">已複製</span>
              </>
            ) : (
              <>
                <Copy className="h-3.5 w-3.5" />
                <span className="font-sans text-[11px]">複製原始碼</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* ReadTheDocs Simple Code Wrap Body */}
      <div className="overflow-x-auto max-h-[600px] overflow-y-auto bg-slate-950/60 py-3 text-xs sm:text-[13px] font-mono leading-relaxed">
        <pre className="m-0 p-0 font-mono">
          <code>
            {lines.map((item) => {
              const isSelected = selectedLine === item.line;
              const hasAnnotation = Boolean(item.note);

              return (
                <React.Fragment key={item.line}>
                  <div
                    onClick={() => handleLineClick(item.line)}
                    className={cn(
                      'group flex items-start transition-colors px-3 py-0.5 select-text cursor-pointer',
                      isSelected
                        ? 'bg-cyan-500/15 text-cyan-200 font-medium'
                        : 'hover:bg-slate-850/60 text-slate-300'
                    )}
                  >
                    {/* Line number gutter (select-none) */}
                    <span className="w-10 sm:w-12 pr-3 sm:pr-4 text-right select-none text-slate-600 group-hover:text-slate-400 shrink-0 font-mono text-[11px] leading-relaxed">
                      {item.line}
                    </span>

                    {/* Code line content with simple wrap */}
                    <span
                      className={cn(
                        'flex-1 font-mono',
                        isWrapped ? 'whitespace-pre-wrap break-all' : 'whitespace-pre'
                      )}
                    >
                      {item.text || ' '}
                    </span>

                    {/* Subtle annotation indicator badge */}
                    {hasAnnotation && (
                      <span
                        className={cn(
                          'ml-2 px-1.5 py-0.5 rounded text-[10px] shrink-0 select-none transition-all font-sans',
                          isSelected
                            ? 'bg-cyan-500/30 text-cyan-200 border border-cyan-400/40'
                            : 'opacity-40 group-hover:opacity-100 text-slate-400 bg-slate-800/80 border border-slate-700/60'
                        )}
                        title="點擊展開/收合架構註解"
                      >
                        {isSelected ? '收合' : '註解'}
                      </span>
                    )}
                  </div>

                  {/* Inline Expandable Annotation Callout */}
                  {isSelected && hasAnnotation && (
                    <div className="my-2 mx-3 sm:mx-12 rounded-lg border border-cyan-500/30 bg-cyan-950/40 p-3 text-xs text-slate-200 shadow-lg backdrop-blur">
                      <div className="flex items-center gap-2 mb-1.5 font-semibold text-cyan-300 text-[12px] font-sans">
                        <Info className="h-3.5 w-3.5 text-cyan-400 shrink-0" />
                        <span>
                          {item.title && !/^Line\s*\d+$/i.test(item.title.trim())
                            ? item.title
                            : `第 ${item.line} 行核心邏輯架構解析`}
                        </span>
                        <Badge variant="secondary" className="text-[10px] py-0 px-1.5 ml-auto font-mono">
                          Line {item.line}
                        </Badge>
                      </div>
                      <p className="text-slate-300 text-[12px] leading-relaxed pl-5 font-sans">
                        {item.note}
                      </p>
                    </div>
                  )}
                </React.Fragment>
              );
            })}
          </code>
        </pre>
      </div>

      {/* Footer bar */}
      <div className="flex items-center justify-between border-t border-slate-800/70 bg-slate-950/60 px-4 py-2 text-[11px] text-slate-500">
        <span>點選任一行或「註解」標籤展開詳細解析</span>
        <span className="font-mono">{fileName}</span>
      </div>
    </div>
  );
};

