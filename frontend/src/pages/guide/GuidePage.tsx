import React, { useState } from 'react';
import { PanelLeft } from 'lucide-react';
import { SidebarNavigation } from '@/widgets/sidebar-nav/SidebarNavigation';
import { ReaderCanvas } from '@/widgets/reader-canvas/ReaderCanvas';
import { SimulationModal } from '@/entities/simulation/SimulationModal';
import { MilestoneTutorialModal } from '@/entities/milestone/MilestoneTutorialModal';
import { useChapterStore } from '@/entities/chapter/chapterStore';
import { getManifest } from '@/entities/manifest';

export const GuidePage: React.FC = () => {
  const {
    setChapter,
    sidebarOpen,
    toggleSidebar,
    activeSite,
    activeMilestonePartId,
    closeMilestoneTutorial,
  } = useChapterStore();
  const [activeLabId, setActiveLabId] = useState<string | null>(null);

  const handleOpenLab = (labId: string) => {
    setActiveLabId(labId);
  };

  const manifest = getManifest(activeSite);
  const activeMilestonePart =
    manifest.parts.find((p) => p.id === activeMilestonePartId) || null;

  return (
    <div className="flex w-full h-screen overflow-hidden bg-background">
      {/* Sleek Left Sidebar */}
      <SidebarNavigation />

      {/* Center Dynamic Reading Canvas */}
      <div className="relative flex-1 h-full overflow-hidden min-w-0 bg-background">
        {!sidebarOpen && (
          <button
            type="button"
            onClick={toggleSidebar}
            className="fixed top-4 left-4 z-40 flex items-center gap-1.5 rounded-xl border border-border bg-card/90 px-3 py-1.5 text-xs font-semibold text-foreground shadow-xl backdrop-blur-md hover:border-primary/50 hover:text-primary transition-all cursor-pointer group"
            title="展開章節導航 (Sidebar)"
          >
            <PanelLeft className="h-4 w-4 text-primary group-hover:scale-110 transition-transform" />
            <span className="tracking-tight">目錄導航</span>
          </button>
        )}
        <ReaderCanvas onOpenLab={handleOpenLab} />
      </div>

      {/* Global Simulation Modal */}
      <SimulationModal
        isOpen={!!activeLabId}
        labId={activeLabId}
        onClose={() => setActiveLabId(null)}
        onSelectChapter={(chId) => setChapter(chId)}
      />

      {/* Global Milestone Kaggle Tutorial Modal */}
      <MilestoneTutorialModal
        isOpen={!!activeMilestonePartId}
        part={activeMilestonePart}
        onClose={closeMilestoneTutorial}
      />
    </div>
  );
};
