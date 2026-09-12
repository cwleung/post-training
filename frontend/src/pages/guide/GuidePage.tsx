import React, { useState, useEffect, useRef } from 'react';
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
    setSidebarOpen,
    activeSite,
    activeMilestonePartId,
    closeMilestoneTutorial,
    resetToHome,
  } = useChapterStore();
  const [activeLabId, setActiveLabId] = useState<string | null>(null);

  const userDesktopPref = useRef<boolean>(
    typeof window !== 'undefined' ? window.innerWidth >= 768 : true
  );

  // Synchronize desktop preference when user toggles sidebar on desktop
  useEffect(() => {
    if (typeof window !== 'undefined' && window.innerWidth >= 768) {
      userDesktopPref.current = sidebarOpen;
    }
  }, [sidebarOpen]);

  // Responsive sidebar: Automatically collapse sidebar drawer when reducing window width to mobile (< 768px),
  // and restore desktop split pane when expanding to desktop (>= 768px).
  useEffect(() => {
    if (typeof window === 'undefined') return;

    let prevWidth = window.innerWidth;

    const handleResize = () => {
      const currentWidth = window.innerWidth;
      const isMobile = currentWidth < 768;
      const wasMobile = prevWidth < 768;

      if (isMobile && !wasMobile) {
        // Just crossed from desktop to mobile: hide sidebar drawer
        setSidebarOpen(false);
      } else if (!isMobile && wasMobile) {
        // Just crossed from mobile to desktop: restore desktop sidebar preference
        setSidebarOpen(userDesktopPref.current);
      }
      prevWidth = currentWidth;
    };

    // Initial check on mount: ensure sidebar is closed on mobile viewport
    if (window.innerWidth < 768) {
      setSidebarOpen(false);
    }

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [setSidebarOpen]);

  const handleOpenLab = (labId: string) => {
    setActiveLabId(labId);
  };

  const manifest = getManifest(activeSite);
  const activeMilestonePart =
    manifest.parts.find((p) => p.id === activeMilestonePartId) || null;

  return (
    <div className="relative flex w-full h-screen overflow-hidden bg-background">
      {/* Mobile Drawer Dimmed Backdrop */}
      {sidebarOpen && (
        <div
          onClick={() => setSidebarOpen(false)}
          className="fixed inset-0 z-40 bg-black/65 backdrop-blur-xs md:hidden transition-opacity duration-200"
          aria-hidden="true"
        />
      )}

      {/* Sleek Left Sidebar (Drawer on mobile, split pane on desktop) */}
      <SidebarNavigation />

      {/* Center Dynamic Reading Canvas */}
      <div className="relative flex-1 h-full overflow-hidden min-w-0 bg-background">
        {!sidebarOpen && (
          <div className="fixed top-3.5 left-3.5 sm:top-4 sm:left-4 z-40 flex items-center gap-1.5">
            <button
              type="button"
              onClick={resetToHome}
              className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-tr from-cyan-500 via-blue-600 to-indigo-600 text-white font-bold text-xs shadow-xl ring-1 ring-white/20 hover:opacity-95 hover:shadow-cyan-500/40 hover:scale-105 active:scale-95 transition-all cursor-pointer"
              title="返回首頁 (Home)"
            >
              ψ
            </button>
            <button
              type="button"
              onClick={toggleSidebar}
              className="flex items-center gap-1.5 rounded-xl border border-border bg-card/90 px-2.5 py-1.5 sm:px-3 text-xs font-semibold text-foreground shadow-xl backdrop-blur-md hover:border-primary/50 hover:text-primary transition-all cursor-pointer group"
              title="展開章節導航 (Sidebar)"
            >
              <PanelLeft className="h-4 w-4 text-primary group-hover:scale-110 transition-transform" />
              <span className="tracking-tight text-[11.5px] sm:text-xs">目錄導航</span>
            </button>
          </div>
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
