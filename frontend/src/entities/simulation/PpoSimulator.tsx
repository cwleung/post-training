import React, { useState, useMemo } from 'react';
import { Card } from '@/shared/ui/Card';
import { Slider } from '@/shared/ui/Slider';
import { Badge } from '@/shared/ui/Badge';
import { RefreshCw, Zap, TrendingUp, AlertTriangle } from 'lucide-react';

export const PpoSimulator: React.FC = () => {
  const [epsilon, setEpsilon] = useState(0.2);
  const [advantage, setAdvantage] = useState(1.5);
  const [currentRatio, setCurrentRatio] = useState(1.15);

  const { points, unclippedVal, clippedVal, minObjective, isClipped } = useMemo(() => {
    const pts: { r: number; unclipped: number; clipped: number; objective: number }[] = [];
    for (let r = 0.5; r <= 1.5; r += 0.02) {
      const unclipped = r * advantage;
      const clippedR = Math.max(1 - epsilon, Math.min(1 + epsilon, r));
      const clipped = clippedR * advantage;
      const objective = Math.min(unclipped, clipped);
      pts.push({
        r: Number(r.toFixed(2)),
        unclipped: Number(unclipped.toFixed(3)),
        clipped: Number(clipped.toFixed(3)),
        objective: Number(objective.toFixed(3)),
      });
    }

    const unclipped = currentRatio * advantage;
    const clippedR = Math.max(1 - epsilon, Math.min(1 + epsilon, currentRatio));
    const clipped = clippedR * advantage;
    const obj = Math.min(unclipped, clipped);
    const clippedActive = Math.abs(obj - unclipped) > 1e-4;

    return {
      points: pts,
      unclippedVal: unclipped,
      clippedVal: clipped,
      minObjective: obj,
      isClipped: clippedActive,
    };
  }, [epsilon, advantage, currentRatio]);

  // SVG Chart scaling
  const minX = 0.5;
  const maxX = 1.5;
  const minY = Math.min(-2, ...points.map((p) => Math.min(p.unclipped, p.objective)));
  const maxY = Math.max(2, ...points.map((p) => Math.max(p.unclipped, p.objective)));

  const toSvgX = (r: number) => ((r - minX) / (maxX - minX)) * 500 + 40;
  const toSvgY = (val: number) => 240 - ((val - minY) / (maxY - minY || 1)) * 200;

  const unclippedPath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${toSvgX(p.r)} ${toSvgY(p.unclipped)}`).join(' ');
  const objectivePath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${toSvgX(p.r)} ${toSvgY(p.objective)}`).join(' ');

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls */}
        <Card className="p-5 space-y-5 bg-card/60 backdrop-blur-sm border-border">
          <div className="flex items-center justify-between pb-3 border-b border-border">
            <h4 className="font-semibold text-sm flex items-center gap-2">
              <Zap className="w-4 h-4 text-primary" />
              策略參數調校
            </h4>
            <Badge variant="outline" className="text-xs">PPO-Clip</Badge>
          </div>

          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-xs mb-1.5 font-medium">
                <span className="text-muted-foreground">截斷臨界半徑 ε (Clip Range)</span>
                <span className="font-mono text-primary">{epsilon.toFixed(2)}</span>
              </div>
              <Slider
                min={0.05}
                max={0.4}
                step={0.01}
                value={[epsilon]}
                onValueChange={([v]) => setEpsilon(v)}
              />
              <span className="text-[11px] text-muted-foreground">有效比率範圍: [{(1 - epsilon).toFixed(2)}, {(1 + epsilon).toFixed(2)}]</span>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1.5 font-medium">
                <span className="text-muted-foreground">廣義優勢估計 Â (GAE Advantage)</span>
                <span className={`font-mono ${advantage >= 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                  {advantage >= 0 ? `+${advantage.toFixed(2)}` : advantage.toFixed(2)}
                </span>
              </div>
              <Slider
                min={-2}
                max={2}
                step={0.1}
                value={[advantage]}
                onValueChange={([v]) => setAdvantage(v)}
              />
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1.5 font-medium">
                <span className="text-muted-foreground">當前重要性採樣比率 r(θ)</span>
                <span className="font-mono text-cyan-400">{currentRatio.toFixed(2)}</span>
              </div>
              <Slider
                min={0.5}
                max={1.5}
                step={0.02}
                value={[currentRatio]}
                onValueChange={([v]) => setCurrentRatio(v)}
              />
            </div>
          </div>

          {/* Quick presets */}
          <div className="pt-2 border-t border-border flex flex-wrap gap-2">
            <button
              onClick={() => { setEpsilon(0.2); setAdvantage(1.2); setCurrentRatio(1.35); }}
              className="px-2.5 py-1 text-xs rounded-md bg-secondary hover:bg-secondary/80 transition-colors"
            >
              正優勢超界截斷
            </button>
            <button
              onClick={() => { setEpsilon(0.2); setAdvantage(-1.0); setCurrentRatio(0.7); }}
              className="px-2.5 py-1 text-xs rounded-md bg-secondary hover:bg-secondary/80 transition-colors"
            >
              負優勢下界觸發
            </button>
            <button
              onClick={() => { setEpsilon(0.2); setAdvantage(1.0); setCurrentRatio(1.0); }}
              className="px-2.5 py-1 text-xs rounded-md bg-secondary hover:bg-secondary/80 transition-colors flex items-center gap-1"
            >
              <RefreshCw className="w-3 h-3" /> 重置
            </button>
          </div>
        </Card>

        {/* Visual Canvas & Curve */}
        <Card className="lg:col-span-2 p-5 bg-card/60 backdrop-blur-sm border-border flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-emerald-400" />
                <span className="text-sm font-semibold">目標函數響應曲線 L^(CLIP)(θ)</span>
              </div>
              <div className="flex items-center gap-3 text-xs">
                <div className="flex items-center gap-1">
                  <span className="w-3 h-0.5 bg-muted-foreground inline-block dashed" />
                  <span className="text-muted-foreground">未截斷 r·Â</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="w-3 h-1 bg-primary inline-block rounded-full" />
                  <span className="font-semibold text-primary">PPO 悲觀目標</span>
                </div>
              </div>
            </div>

            {/* SVG Plot */}
            <div className="w-full h-[260px] bg-muted/20 rounded-lg p-2 relative overflow-hidden border border-border/50">
              <svg viewBox="0 0 580 270" className="w-full h-full">
                {/* Zero line */}
                <line
                  x1={toSvgX(minX)}
                  y1={toSvgY(0)}
                  x2={toSvgX(maxX)}
                  y2={toSvgY(0)}
                  stroke="currentColor"
                  strokeOpacity="0.15"
                  strokeDasharray="4 4"
                />

                {/* Ratio = 1 line */}
                <line
                  x1={toSvgX(1.0)}
                  y1={20}
                  x2={toSvgX(1.0)}
                  y2={250}
                  stroke="currentColor"
                  strokeOpacity="0.2"
                />
                <text x={toSvgX(1.0)} y={265} fill="currentColor" fillOpacity="0.5" fontSize="10" textAnchor="middle">
                  r=1.0 (策略無偏移)
                </text>

                {/* Epsilon boundary band */}
                <rect
                  x={toSvgX(1 - epsilon)}
                  y={20}
                  width={toSvgX(1 + epsilon) - toSvgX(1 - epsilon)}
                  height={230}
                  fill="currentColor"
                  fillOpacity="0.04"
                />
                <text x={toSvgX(1 - epsilon)} y={30} fill="currentColor" fillOpacity="0.4" fontSize="9" textAnchor="middle">
                  1-ε
                </text>
                <text x={toSvgX(1 + epsilon)} y={30} fill="currentColor" fillOpacity="0.4" fontSize="9" textAnchor="middle">
                  1+ε
                </text>

                {/* Curves */}
                <path d={unclippedPath} fill="none" stroke="currentColor" strokeOpacity="0.3" strokeWidth="2" strokeDasharray="4 4" />
                <path d={objectivePath} fill="none" stroke="var(--primary, #3b82f6)" strokeWidth="3" strokeLinecap="round" />

                {/* Current operating point */}
                <circle
                  cx={toSvgX(currentRatio)}
                  cy={toSvgY(minObjective)}
                  r="6"
                  fill={isClipped ? '#f59e0b' : '#10b981'}
                  stroke="#ffffff"
                  strokeWidth="2"
                />
              </svg>
            </div>
          </div>

          {/* Metrics summary */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4 pt-3 border-t border-border">
            <div className="p-2.5 rounded-md bg-muted/30">
              <div className="text-[11px] text-muted-foreground">未約束價值 r·Â</div>
              <div className="font-mono text-sm font-semibold">{unclippedVal.toFixed(3)}</div>
            </div>
            <div className="p-2.5 rounded-md bg-muted/30">
              <div className="text-[11px] text-muted-foreground">截斷值 clip(r)·Â</div>
              <div className="font-mono text-sm font-semibold">{clippedVal.toFixed(3)}</div>
            </div>
            <div className="p-2.5 rounded-md bg-primary/10 border border-primary/20">
              <div className="text-[11px] text-primary">有效梯度貢獻</div>
              <div className="font-mono text-sm font-bold text-primary">{minObjective.toFixed(3)}</div>
            </div>
            <div className={`p-2.5 rounded-md border ${isClipped ? 'bg-amber-500/10 border-amber-500/30 text-amber-500' : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-500'}`}>
              <div className="text-[11px] flex items-center gap-1">
                {isClipped ? <AlertTriangle className="w-3 h-3" /> : <Zap className="w-3 h-3" />}
                截斷保護狀態
              </div>
              <div className="text-xs font-semibold mt-0.5">
                {isClipped ? '防禦觸發 (飽和無梯度)' : '安全區 (正常傳播)'}
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};
