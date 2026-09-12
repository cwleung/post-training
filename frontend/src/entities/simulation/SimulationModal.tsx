import React from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/shared/ui/Dialog';
import { Badge } from '@/shared/ui/Badge';
import { LAB_CATALOG } from './labCatalog';
import { CartPoleSimulator } from './CartPoleSimulator';
import { PpoSimulator } from './PpoSimulator';
import { GrpoSimulator } from './GrpoSimulator';
import { VramSimulator } from './VramSimulator';
import { GenericInteractiveLab } from './GenericInteractiveLab';

interface SimulationModalProps {
  labId: string | null;
  isOpen: boolean;
  onClose: () => void;
  onSelectChapter?: (chapterId: string) => void;
}

export const SimulationModal: React.FC<SimulationModalProps> = ({
  labId,
  isOpen,
  onClose,
  onSelectChapter,
}) => {
  if (!labId || !isOpen) return null;

  const lab = LAB_CATALOG[labId] || {
    id: labId,
    title: `實驗室: ${labId}`,
    category: 'control',
    icon: '🧪',
    chapterId: 'ch01',
    equation: '',
    description: '動態參數調節與實時響應工作台。',
  };

  const renderSimulator = () => {
    switch (labId) {
      case 'cartpole':
        return <CartPoleSimulator />;
      case 'ppo':
        return <PpoSimulator />;
      case 'grpo':
        return <GrpoSimulator />;
      case 'rlvr_lora_vram':
        return <VramSimulator />;
      default:
        return <GenericInteractiveLab lab={lab} />;
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-4xl max-h-[92vh] overflow-y-auto p-6 bg-background/95 backdrop-blur-md border border-border shadow-2xl">
        <DialogHeader className="pb-4 border-b border-border">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <span className="text-2xl p-2 rounded-xl bg-secondary/80 border border-border">
                {lab.icon}
              </span>
              <div>
                <div className="flex items-center gap-2">
                  <DialogTitle className="text-lg font-bold">{lab.title}</DialogTitle>
                  <Badge variant="outline" className="text-xs capitalize font-mono">
                    {lab.category}
                  </Badge>
                </div>
                <DialogDescription className="text-xs text-muted-foreground mt-0.5">
                  {lab.description}
                </DialogDescription>
              </div>
            </div>

            {lab.chapterId && onSelectChapter && (
              <button
                onClick={() => {
                  onSelectChapter(lab.chapterId);
                  onClose();
                }}
                className="text-xs font-mono text-primary hover:underline flex items-center gap-1 shrink-0 px-3 py-1.5 rounded-md bg-primary/10 border border-primary/20"
              >
                前往對應章節 ({lab.chapterId}) →
              </button>
            )}
          </div>
        </DialogHeader>

        <div className="py-4">{renderSimulator()}</div>
      </DialogContent>
    </Dialog>
  );
};
