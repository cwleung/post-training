import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { SiteId } from '@/shared/types';
import { getDefaultChapterId, findChapter } from '@/entities/manifest';

interface ChapterState {
  activeSite: SiteId;
  currentChapterId: string;
  doneChapters: string[];
  theme: 'dark' | 'light';
  sidebarOpen: boolean;
  selectedCodeLine: number | null;
  activeMilestonePartId: string | null;

  setActiveSite: (site: SiteId) => void;
  setCurrentChapterId: (id: string) => void;
  setChapter: (id: string) => void;
  toggleChapterDone: (id: string) => void;
  isChapterDone: (id: string) => boolean;
  setTheme: (theme: 'dark' | 'light') => void;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  setSelectedCodeLine: (line: number | null) => void;
  openMilestoneTutorial: (partId: string) => void;
  closeMilestoneTutorial: () => void;
  resetToHome: () => void;
}

export const useChapterStore = create<ChapterState>()(
  persist(
    (set, get) => ({
      activeSite: 'deepagents',
      currentChapterId: 'da01',
      doneChapters: [],
      theme: 'dark',
      sidebarOpen: true,
      selectedCodeLine: null,
      activeMilestonePartId: null,

      setActiveSite: (site) => {
        const nextDefaultCh = getDefaultChapterId(site);
        set({ activeSite: site, currentChapterId: nextDefaultCh });
      },

      setCurrentChapterId: (id) => {
        const found = findChapter(id);
        set((state) => ({
          currentChapterId: id,
          activeSite: found ? found.siteId : state.activeSite,
          selectedCodeLine: null,
        }));
      },

      setChapter: (id) => {
        get().setCurrentChapterId(id);
      },

      toggleChapterDone: (id) => {
        const { doneChapters } = get();
        const exists = doneChapters.includes(id);
        const next = exists ? doneChapters.filter((c) => c !== id) : [...doneChapters, id];
        set({ doneChapters: next });
      },

      isChapterDone: (id) => get().doneChapters.includes(id),

      setTheme: (theme) => {
        document.documentElement.setAttribute('data-theme', theme);
        set({ theme });
      },

      toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),

      setSidebarOpen: (open) => set({ sidebarOpen: open }),

      setSelectedCodeLine: (line) => set({ selectedCodeLine: line }),

      openMilestoneTutorial: (partId) => set({ activeMilestonePartId: partId }),

      closeMilestoneTutorial: () => set({ activeMilestonePartId: null }),

      resetToHome: () => {
        const defaultSite = 'deepagents';
        const defaultCh = getDefaultChapterId(defaultSite);
        set({
          activeSite: defaultSite,
          currentChapterId: defaultCh,
          selectedCodeLine: null,
          activeMilestonePartId: null,
        });
        if (typeof window !== 'undefined') {
          window.location.hash = `#${defaultCh}`;
          window.scrollTo({ top: 0, behavior: 'smooth' });
          const mainElement = document.querySelector('main');
          if (mainElement) {
            mainElement.scrollTo({ top: 0, behavior: 'smooth' });
          }
        }
      },
    }),
    {
      name: 'deepagents-storage-v4',
      partialize: (state) => ({
        doneChapters: state.doneChapters,
        theme: state.theme,
        activeSite: state.activeSite,
        currentChapterId: state.currentChapterId,
      }),
    }
  )
);
