import React, { useState } from 'react';
import { Card } from '@/shared/ui/Card';
import { Button } from '@/shared/ui/Button';
import { Badge } from '@/shared/ui/Badge';
import { Slider } from '@/shared/ui/Slider';
import { RefreshCw, CheckCircle, XCircle, Sparkles, Layers } from 'lucide-react';

interface CompletionSample {
  id: number;
  reasoningLen: number;
  hasXmlTags: boolean;
  isCorrect: boolean;
}

export const GrpoSimulator: React.FC = () => {
  const [groupSize, setGroupSize] = useState<number>(4);
  const [accWeight, setAccWeight] = useState<number>(1.0);
  const [fmtWeight, setFmtWeight] = useState<number>(0.5);
  const [lenPenalty, setLenPenalty] = useState<number>(0.001);

  // Simulated completions in the group
  const [samples, setSamples] = useState<CompletionSample[]>([
    { id: 1, reasoningLen: 450, hasXmlTags: true, isCorrect: true },
    { id: 2, reasoningLen: 180, hasXmlTags: false, isCorrect: true },
    { id: 3, reasoningLen: 620, hasXmlTags: true, isCorrect: false },
    { id: 4, reasoningLen: 80, hasXmlTags: false, isCorrect: false },
    { id: 5, reasoningLen: 510, hasXmlTags: true, isCorrect: true },
    { id: 6, reasoningLen: 310, hasXmlTags: true, isCorrect: false },
    { id: 7, reasoningLen: 750, hasXmlTags: true, isCorrect: true },
    { id: 8, reasoningLen: 220, hasXmlTags: false, isCorrect: true },
  ]);

  const activeSamples = samples.slice(0, groupSize);

  // Compute rewards for active samples
  const evaluated = activeSamples.map((s) => {
    const rAcc = s.isCorrect ? accWeight : 0;
    const rFmt = s.hasXmlTags ? fmtWeight : 0;
    const rPenalty = s.reasoningLen * lenPenalty;
    const totalReward = rAcc + rFmt - rPenalty;
    return {
      ...s,
      rAcc,
      rFmt,
      rPenalty,
      totalReward,
    };
  });

  // Calculate Group Mean & Std
  const totalR = evaluated.reduce((sum, item) => sum + item.totalReward, 0);
  const meanR = totalR / groupSize;
  const variance = evaluated.reduce((sum, item) => sum + Math.pow(item.totalReward - meanR, 2), 0) / groupSize;
  const stdR = Math.sqrt(variance) + 1e-4;

  // Normalized Advantage
  const withAdvantage = evaluated.map((item) => ({
    ...item,
    advantage: (item.totalReward - meanR) / stdR,
  }));

  const resample = () => {
    setSamples(
      Array.from({ length: 8 }, (_, idx) => ({
        id: idx + 1,
        reasoningLen: Math.floor(Math.random() * 700) + 80,
        hasXmlTags: Math.random() > 0.35,
        isCorrect: Math.random() > 0.45,
      }))
    );
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls */}
        <Card className="p-5 space-y-5 bg-card/60 backdrop-blur-sm border-border">
          <div className="flex items-center justify-between pb-3 border-b border-border">
            <h4 className="font-semibold text-sm flex items-center gap-2">
              <Layers className="w-4 h-4 text-primary" />
              群組採樣超參數
            </h4>
            <Badge variant="outline" className="text-xs">DeepSeek-R1</Badge>
          </div>

          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-xs mb-1.5 font-medium">
                <span className="text-muted-foreground">群組候選數 G (Group Size)</span>
                <span className="font-mono text-primary">{groupSize} 組輸出</span>
              </div>
              <Slider
                min={2}
                max={8}
                step={1}
                value={[groupSize]}
                onValueChange={([v]) => setGroupSize(v)}
              />
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1.5 font-medium">
                <span className="text-muted-foreground">準確率獎勵權重 w_acc</span>
                <span className="font-mono text-emerald-400">+{accWeight.toFixed(2)}</span>
              </div>
              <Slider
                min={0.5}
                max={2.0}
                step={0.1}
                value={[accWeight]}
                onValueChange={([v]) => setAccWeight(v)}
              />
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1.5 font-medium">
                <span className="text-muted-foreground">格式合規獎勵權重 w_fmt</span>
                <span className="font-mono text-cyan-400">+{fmtWeight.toFixed(2)}</span>
              </div>
              <Slider
                min={0.0}
                max={1.0}
                step={0.05}
                value={[fmtWeight]}
                onValueChange={([v]) => setFmtWeight(v)}
              />
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1.5 font-medium">
                <span className="text-muted-foreground">思考長度懲罰係數 λ_len</span>
                <span className="font-mono text-amber-400">{lenPenalty.toFixed(4)}</span>
              </div>
              <Slider
                min={0.0}
                max={0.003}
                step={0.0002}
                value={[lenPenalty]}
                onValueChange={([v]) => setLenPenalty(v)}
              />
            </div>
          </div>

          <Button onClick={resample} variant="secondary" className="w-full flex items-center justify-center gap-2">
            <RefreshCw className="w-3.5 h-3.5" />
            重新生成隨機採樣群組
          </Button>
        </Card>

        {/* Group Advantage Visualizer */}
        <Card className="lg:col-span-2 p-5 bg-card/60 backdrop-blur-sm border-border flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-primary" />
                <span className="text-sm font-semibold">群組相對優勢 Â_i 分佈 (零和歸一化)</span>
              </div>
              <div className="text-xs text-muted-foreground font-mono">
                μ = {meanR.toFixed(3)} | σ = {stdR.toFixed(3)}
              </div>
            </div>

            {/* Table / Cards of Rollouts */}
            <div className="space-y-2.5">
              {withAdvantage.map((item) => (
                <div
                  key={item.id}
                  className="p-3 rounded-lg border border-border/70 bg-background/50 flex flex-col sm:flex-row sm:items-center justify-between gap-3"
                >
                  <div className="flex items-center gap-3">
                    <span className="w-6 h-6 rounded-full bg-muted flex items-center justify-center text-xs font-mono font-bold">
                      #{item.id}
                    </span>
                    <div className="space-y-0.5">
                      <div className="flex items-center gap-2 text-xs font-medium">
                        <span className="flex items-center gap-1">
                          {item.isCorrect ? (
                            <CheckCircle className="w-3.5 h-3.5 text-emerald-500" />
                          ) : (
                            <XCircle className="w-3.5 h-3.5 text-rose-500" />
                          )}
                          {item.isCorrect ? '答案正確' : '答案錯誤'}
                        </span>
                        <span className="text-muted-foreground">·</span>
                        <span className={item.hasXmlTags ? 'text-primary' : 'text-amber-500'}>
                          {item.hasXmlTags ? '<think>標籤合規' : '缺少XML標籤'}
                        </span>
                        <span className="text-muted-foreground">·</span>
                        <span className="text-muted-foreground">{item.reasoningLen} tokens</span>
                      </div>
                      <div className="text-[11px] text-muted-foreground font-mono">
                        R = {item.rAcc.toFixed(2)} (acc) + {item.rFmt.toFixed(2)} (fmt) - {item.rPenalty.toFixed(2)} (len) = {item.totalReward.toFixed(3)}
                      </div>
                    </div>
                  </div>

                  {/* Advantage Pill */}
                  <div className="flex items-center gap-3 self-end sm:self-center">
                    <div className="w-28 h-2 bg-muted rounded-full overflow-hidden relative">
                      <div
                        className={`h-full absolute ${item.advantage >= 0 ? 'bg-emerald-500 left-1/2' : 'bg-rose-500 right-1/2'}`}
                        style={{
                          width: `${Math.min(50, Math.abs(item.advantage) * 25)}%`,
                        }}
                      />
                    </div>
                    <span
                      className={`font-mono font-bold text-xs px-2.5 py-1 rounded-md border min-w-[70px] text-center ${
                        item.advantage >= 0
                          ? 'bg-emerald-500/10 text-emerald-500 border-emerald-500/30'
                          : 'bg-rose-500/10 text-rose-500 border-rose-500/30'
                      }`}
                    >
                      {item.advantage >= 0 ? `+${item.advantage.toFixed(2)}` : item.advantage.toFixed(2)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-border flex items-center justify-between text-xs text-muted-foreground">
            <span>
              💡 核心洞察：GRPO 徹底免除了 Critic 價值網絡，優勢和恆為 0，對標籤合規且答案正確的採樣施加正梯度推力。
            </span>
          </div>
        </Card>
      </div>
    </div>
  );
};
