import React, { useState, useEffect, useId } from 'react';
import { Card } from '@/shared/ui/Card';
import { Slider } from '@/shared/ui/Slider';
import { Badge } from '@/shared/ui/Badge';
import { Button } from '@/shared/ui/Button';
import { LabSpec } from './labCatalog';
import { Play, RotateCcw, Activity } from 'lucide-react';
import katex from 'katex';

interface GenericInteractiveLabProps {
  lab: LabSpec;
}

export const GenericInteractiveLab: React.FC<GenericInteractiveLabProps> = ({ lab }) => {
  const [paramA, setParamA] = useState<number>(0.5);
  const [paramB, setParamB] = useState<number>(1.0);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [step, setStep] = useState<number>(0);
  const [history, setHistory] = useState<number[]>([0.2, 0.35, 0.45, 0.6, 0.72, 0.78, 0.85]);

  const mathContainerId = useId();

  // KaTeX rendering for the lab equation
  useEffect(() => {
    const el = document.getElementById(mathContainerId);
    if (el && lab.equation) {
      try {
        katex.render(lab.equation, el, {
          throwOnError: false,
          displayMode: true,
        });
      } catch (e) {
        console.error('KaTeX error:', e);
      }
    }
  }, [lab.equation, mathContainerId]);

  // Simulation tick
  useEffect(() => {
    let interval: any;
    if (isRunning) {
      interval = setInterval(() => {
        setStep((s) => s + 1);
        setHistory((prev) => {
          const last = prev[prev.length - 1] ?? 0.5;
          const noise = (Math.random() - 0.48) * 0.08;
          const target = (paramA * 0.6 + paramB * 0.4);
          const next = Math.max(0, Math.min(1, last + (target - last) * 0.15 + noise));
          const nextArr = [...prev.slice(-24), Number(next.toFixed(3))];
          return nextArr;
        });
      }, 400);
    }
    return () => clearInterval(interval);
  }, [isRunning, paramA, paramB]);

  const currentMetric = history[history.length - 1] ?? 0.5;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls */}
        <Card className="p-5 space-y-5 bg-card/60 backdrop-blur-sm border-border">
          <div className="flex items-center justify-between pb-3 border-b border-border">
            <h4 className="font-semibold text-sm flex items-center gap-2">
              <span className="text-lg">{lab.icon}</span>
              實驗室控制台
            </h4>
            <Badge variant="outline" className="text-xs capitalize">{lab.category}</Badge>
          </div>

          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-xs mb-1.5 font-medium">
                <span className="text-muted-foreground">探索與學習率權重 (α)</span>
                <span className="font-mono text-primary">{paramA.toFixed(2)}</span>
              </div>
              <Slider
                min={0.01}
                max={1.0}
                step={0.05}
                value={[paramA]}
                onValueChange={([v]) => setParamA(v)}
              />
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1.5 font-medium">
                <span className="text-muted-foreground">正則化與溫度係數 (β)</span>
                <span className="font-mono text-primary">{paramB.toFixed(2)}</span>
              </div>
              <Slider
                min={0.1}
                max={2.0}
                step={0.1}
                value={[paramB]}
                onValueChange={([v]) => setParamB(v)}
              />
            </div>

            <div className="pt-2 flex gap-2">
              <Button
                variant={isRunning ? 'destructive' : 'default'}
                className="flex-1 flex items-center justify-center gap-2"
                onClick={() => setIsRunning(!isRunning)}
              >
                {isRunning ? (
                  <>暫停模擬</>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-current" /> 啟動實時迭代
                  </>
                )}
              </Button>
              <Button
                variant="outline"
                size="icon"
                onClick={() => {
                  setIsRunning(false);
                  setStep(0);
                  setHistory([0.2, 0.35, 0.45, 0.6, 0.72]);
                }}
              >
                <RotateCcw className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </Card>

        {/* Visual Panel */}
        <Card className="lg:col-span-2 p-5 bg-card/60 backdrop-blur-sm border-border flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-emerald-400" />
                <span className="text-sm font-semibold">理論收斂曲線與動態響應</span>
              </div>
              <span className="text-xs font-mono text-muted-foreground">
                Step: {step} | 實時指標: {(currentMetric * 100).toFixed(1)}%
              </span>
            </div>

            {/* Live Chart */}
            <div className="w-full h-[200px] bg-muted/20 rounded-lg p-2 relative overflow-hidden border border-border/50">
              <svg viewBox="0 0 500 200" className="w-full h-full">
                <line x1="0" y1="180" x2="500" y2="180" stroke="currentColor" strokeOpacity="0.1" />
                <line x1="0" y1="100" x2="500" y2="100" stroke="currentColor" strokeOpacity="0.1" strokeDasharray="3 3" />
                <line x1="0" y1="20" x2="500" y2="20" stroke="currentColor" strokeOpacity="0.1" />

                {/* Path */}
                <path
                  d={history
                    .map((val, idx) => {
                      const x = (idx / (history.length - 1 || 1)) * 480 + 10;
                      const y = 180 - val * 160;
                      return `${idx === 0 ? 'M' : 'L'} ${x} ${y}`;
                    })
                    .join(' ')}
                  fill="none"
                  stroke="var(--primary, #3b82f6)"
                  strokeWidth="3"
                  strokeLinecap="round"
                />

                {/* Points */}
                {history.map((val, idx) => {
                  const x = (idx / (history.length - 1 || 1)) * 480 + 10;
                  const y = 180 - val * 160;
                  return (
                    <circle
                      key={idx}
                      cx={x}
                      cy={y}
                      r="3.5"
                      className="fill-primary stroke-background"
                      strokeWidth="1.5"
                    />
                  );
                })}
              </svg>
            </div>
          </div>

          {/* Mathematical formulation block */}
          <div className="mt-4 p-3 bg-muted/30 rounded-lg border border-border">
            <div className="text-xs font-semibold text-muted-foreground mb-1">對應理論數學公式</div>
            <div id={mathContainerId} className="text-sm overflow-x-auto py-1 text-center" />
          </div>
        </Card>
      </div>
    </div>
  );
};
