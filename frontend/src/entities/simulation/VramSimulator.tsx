import React, { useState } from 'react';
import { Card } from '@/shared/ui/Card';
import { Slider } from '@/shared/ui/Slider';
import { Badge } from '@/shared/ui/Badge';
import { HardDrive, Server, AlertOctagon, CheckCircle2 } from 'lucide-react';

interface ModelPreset {
  name: string;
  paramsB: number;
  layers: number;
  hiddenDim: number;
  numHeads: number;
  numKVHeads: number;
}

const MODEL_PRESETS: Record<string, ModelPreset> = {
  'qwen-7b': { name: 'Qwen2.5-7B', paramsB: 7.6, layers: 28, hiddenDim: 3584, numHeads: 28, numKVHeads: 4 },
  'qwen-14b': { name: 'Qwen2.5-14B', paramsB: 14.7, layers: 48, hiddenDim: 5120, numHeads: 40, numKVHeads: 8 },
  'deepseek-r1-distill-32b': { name: 'DeepSeek-R1-Distill-32B', paramsB: 32.5, layers: 64, hiddenDim: 5120, numHeads: 40, numKVHeads: 8 },
  'llama-70b': { name: 'Llama-3.3-70B', paramsB: 70.6, layers: 80, hiddenDim: 8192, numHeads: 64, numKVHeads: 8 },
};

export const VramSimulator: React.FC = () => {
  const [modelKey, setModelKey] = useState<string>('qwen-7b');
  const [quantization, setQuantization] = useState<'4bit' | '8bit' | '16bit'>('16bit');
  const [loraRank, setLoraRank] = useState<number>(32);
  const [batchSize, setBatchSize] = useState<number>(2);
  const [seqLen, setSeqLen] = useState<number>(4096);
  const [use8BitOptimizer, setUse8BitOptimizer] = useState<boolean>(true);

  const model = MODEL_PRESETS[modelKey];

  // Bytes per parameter for base weights
  const bytesPerParam = quantization === '4bit' ? 0.5 : quantization === '8bit' ? 1.0 : 2.0;
  const baseWeightGB = model.paramsB * bytesPerParam;

  // LoRA Parameter estimation: 2 * r * hiddenDim * 4 * layers
  const loraParams = 2 * loraRank * model.hiddenDim * 4 * model.layers;
  const loraWeightGB = (loraParams * 2) / (1024 * 1024 * 1024); // fp16

  // Optimizer states: For LoRA params, AdamW stores 2 states (m and v)
  // Standard AdamW = 8 bytes per trainable param; 8-bit AdamW = 2 bytes per param
  const optBytesPerParam = use8BitOptimizer ? 2.0 : 8.0;
  const optMemoryGB = (loraParams * optBytesPerParam) / (1024 * 1024 * 1024);

  // KV Cache: 2 * layers * (numKVHeads * (hiddenDim / numHeads)) * 2 bytes * seqLen * batchSize
  const kvDim = model.numKVHeads * (model.hiddenDim / model.numHeads);
  const kvCacheGB = (2 * model.layers * kvDim * 2 * seqLen * batchSize) / (1024 * 1024 * 1024);

  // Activation memory estimate
  const activationGB = (batchSize * seqLen * model.hiddenDim * model.layers * 2 * 0.15) / (1024 * 1024 * 1024);

  // CUDA runtime + workspace overhead
  const cudaOverheadGB = 1.2;

  const totalVramGB = baseWeightGB + loraWeightGB + optMemoryGB + kvCacheGB + activationGB + cudaOverheadGB;

  const gpuBudgets = [
    { name: 'NVIDIA T4 (Colab)', vram: 15.0 },
    { name: 'RTX 3090 / 4090', vram: 24.0 },
    { name: 'A100 / H100 SXM', vram: 80.0 },
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls */}
        <Card className="p-5 space-y-5 bg-card/60 backdrop-blur-sm border-border">
          <div className="flex items-center justify-between pb-3 border-b border-border">
            <h4 className="font-semibold text-sm flex items-center gap-2">
              <Server className="w-4 h-4 text-primary" />
              訓練架構與顯存分配
            </h4>
            <Badge variant="outline" className="text-xs">LoRA VRAM</Badge>
          </div>

          <div className="space-y-4">
            <div>
              <label className="text-xs font-medium text-muted-foreground block mb-1.5">
                基礎模型選擇 (Base Model)
              </label>
              <select
                value={modelKey}
                onChange={(e) => setModelKey(e.target.value)}
                className="w-full text-xs font-medium px-3 py-2 rounded-md bg-secondary border border-border text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
              >
                {Object.entries(MODEL_PRESETS).map(([k, m]) => (
                  <option key={k} value={k}>
                    {m.name} ({m.paramsB}B)
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="text-xs font-medium text-muted-foreground block mb-1.5">
                底模量化精度 (Quantization)
              </label>
              <div className="grid grid-cols-3 gap-2">
                {(['4bit', '8bit', '16bit'] as const).map((q) => (
                  <button
                    key={q}
                    onClick={() => setQuantization(q)}
                    className={`py-1.5 text-xs font-medium rounded-md border transition-colors ${
                      quantization === q
                        ? 'bg-primary text-primary-foreground border-primary'
                        : 'bg-secondary text-secondary-foreground border-border hover:bg-secondary/80'
                    }`}
                  >
                    {q === '4bit' ? 'QLoRA 4-bit' : q === '8bit' ? 'NF4/8-bit' : 'BF16 / FP16'}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1.5 font-medium">
                <span className="text-muted-foreground">LoRA Rank (階數 r)</span>
                <span className="font-mono text-primary">r={loraRank}</span>
              </div>
              <Slider
                min={8}
                max={128}
                step={8}
                value={[loraRank]}
                onValueChange={([v]) => setLoraRank(v)}
              />
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1.5 font-medium">
                <span className="text-muted-foreground">上下文序列長度 (Max Seq Len)</span>
                <span className="font-mono text-primary">{seqLen.toLocaleString()} tokens</span>
              </div>
              <Slider
                min={1024}
                max={16384}
                step={1024}
                value={[seqLen]}
                onValueChange={([v]) => setSeqLen(v)}
              />
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1.5 font-medium">
                <span className="text-muted-foreground">批次大小 (Batch Size B)</span>
                <span className="font-mono text-primary">{batchSize}</span>
              </div>
              <Slider
                min={1}
                max={8}
                step={1}
                value={[batchSize]}
                onValueChange={([v]) => setBatchSize(v)}
              />
            </div>

            <div className="flex items-center justify-between pt-2 border-t border-border">
              <span className="text-xs text-muted-foreground">啟用 8-Bit AdamW 優化器</span>
              <input
                type="checkbox"
                checked={use8BitOptimizer}
                onChange={(e) => setUse8BitOptimizer(e.target.checked)}
                className="accent-primary h-4 w-4 rounded cursor-pointer"
              />
            </div>
          </div>
        </Card>

        {/* Breakdown & GPU Viability */}
        <Card className="lg:col-span-2 p-5 bg-card/60 backdrop-blur-sm border-border flex flex-col justify-between">
          <div>
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1.5 sm:gap-2 mb-4">
              <div className="flex items-center gap-2">
                <HardDrive className="w-4 h-4 text-primary" />
                <span className="text-sm font-semibold">顯存組成預算分析 (VRAM Breakdown)</span>
              </div>
              <div className="font-mono text-sm sm:text-base font-bold text-primary">
                總預計需求: {totalVramGB.toFixed(2)} GB
              </div>
            </div>

            {/* Stacked Memory Bar */}
            <div className="h-6 w-full bg-muted/40 rounded-lg overflow-hidden flex border border-border/70 mb-4">
              <div
                style={{ width: `${(baseWeightGB / totalVramGB) * 100}%` }}
                className="bg-blue-500 h-full flex items-center justify-center text-[10px] text-white font-mono"
                title={`底模權重: ${baseWeightGB.toFixed(2)} GB`}
              >
                底模
              </div>
              <div
                style={{ width: `${(loraWeightGB / totalVramGB) * 100}%` }}
                className="bg-purple-500 h-full flex items-center justify-center text-[10px] text-white font-mono"
                title={`LoRA 權重: ${loraWeightGB.toFixed(3)} GB`}
              >
                LoRA
              </div>
              <div
                style={{ width: `${(optMemoryGB / totalVramGB) * 100}%` }}
                className="bg-amber-500 h-full flex items-center justify-center text-[10px] text-white font-mono"
                title={`優化器狀態: ${optMemoryGB.toFixed(2)} GB`}
              >
                Opt
              </div>
              <div
                style={{ width: `${(kvCacheGB / totalVramGB) * 100}%` }}
                className="bg-emerald-500 h-full flex items-center justify-center text-[10px] text-white font-mono"
                title={`KV Cache: ${kvCacheGB.toFixed(2)} GB`}
              >
                KV
              </div>
              <div
                style={{ width: `${((activationGB + cudaOverheadGB) / totalVramGB) * 100}%` }}
                className="bg-rose-500 h-full flex items-center justify-center text-[10px] text-white font-mono"
                title={`激活值 & CUDA: ${(activationGB + cudaOverheadGB).toFixed(2)} GB`}
              >
                Act
              </div>
            </div>

            {/* Itemized Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
              <div className="p-2.5 rounded-md bg-muted/30 border-l-2 border-blue-500">
                <div className="text-[11px] text-muted-foreground">底模權重</div>
                <div className="font-mono text-sm font-semibold">{baseWeightGB.toFixed(2)} GB</div>
              </div>
              <div className="p-2.5 rounded-md bg-muted/30 border-l-2 border-purple-500">
                <div className="text-[11px] text-muted-foreground">LoRA 參數 (r={loraRank})</div>
                <div className="font-mono text-sm font-semibold">{(loraWeightGB * 1024).toFixed(1)} MB</div>
              </div>
              <div className="p-2.5 rounded-md bg-muted/30 border-l-2 border-amber-500">
                <div className="text-[11px] text-muted-foreground">AdamW 優化器狀態</div>
                <div className="font-mono text-sm font-semibold">{(optMemoryGB * 1024).toFixed(1)} MB</div>
              </div>
              <div className="p-2.5 rounded-md bg-muted/30 border-l-2 border-emerald-500">
                <div className="text-[11px] text-muted-foreground">KV Cache + 激活</div>
                <div className="font-mono text-sm font-semibold">{(kvCacheGB + activationGB).toFixed(2)} GB</div>
              </div>
            </div>

            {/* GPU Compatibility Matrix */}
            <h5 className="text-xs font-semibold text-muted-foreground mb-2">主流硬體相容度與 OOM 評估</h5>
            <div className="space-y-2">
              {gpuBudgets.map((gpu) => {
                const canFit = totalVramGB <= gpu.vram;
                const ratio = Math.min(100, (totalVramGB / gpu.vram) * 100);
                return (
                  <div
                    key={gpu.name}
                    className={`p-3 rounded-lg border flex items-center justify-between text-xs ${
                      canFit
                        ? 'bg-emerald-500/5 border-emerald-500/20'
                        : 'bg-rose-500/5 border-rose-500/20'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      {canFit ? (
                        <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                      ) : (
                        <AlertOctagon className="w-4 h-4 text-rose-500" />
                      )}
                      <span className="font-medium">{gpu.name}</span>
                      <span className="text-muted-foreground font-mono">({gpu.vram} GB)</span>
                    </div>

                    <div className="flex items-center gap-3">
                      <div className="w-24 h-2 bg-muted rounded-full overflow-hidden">
                        <div
                          className={`h-full ${canFit ? 'bg-emerald-500' : 'bg-rose-500'}`}
                          style={{ width: `${ratio}%` }}
                        />
                      </div>
                      <span className={`font-mono font-semibold ${canFit ? 'text-emerald-500' : 'text-rose-500'}`}>
                        {canFit ? `剩餘 ${(gpu.vram - totalVramGB).toFixed(1)} GB` : 'OOM 超限'}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-border text-xs text-muted-foreground">
            💡 建議：若在 24GB 顯卡 (RTX 3090/4090) 微調 14B 模型，推薦開啟 QLoRA 4-bit + 8-bit AdamW 以防 OOM。
          </div>
        </Card>
      </div>
    </div>
  );
};
