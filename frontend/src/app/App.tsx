import React, { useEffect } from 'react';
import { useChapterStore } from '@/entities/chapter/chapterStore';
import { findChapter } from '@/entities/manifest';
import { GuidePage } from '@/pages/guide/GuidePage';
import { CareerUnlockModal } from '@/widgets/privacy/CareerUnlockModal';

export const App: React.FC = () => {
  const { theme, currentChapterId, setCurrentChapterId } = useChapterStore();

  // Sync theme with document root attribute
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  // Initial URL hash parsing on mount
  useEffect(() => {
    const rawHash = window.location.hash.replace('#', '').trim();
    if (rawHash) {
      const match = findChapter(rawHash);
      if (match) {
        setCurrentChapterId(rawHash);
        return;
      }
    }
    // If empty or invalid hash, sync URL to current valid chapter
    window.history.replaceState(null, '', `#${currentChapterId}`);
  }, []);

  // Sync currentChapterId to URL hash without triggering hashchange event
  useEffect(() => {
    const currentHash = window.location.hash.replace('#', '').trim();
    if (currentHash !== currentChapterId) {
      window.history.replaceState(null, '', `#${currentChapterId}`);
    }
  }, [currentChapterId]);

  // Handle browser Back/Forward navigation
  useEffect(() => {
    const handlePopState = () => {
      const hash = window.location.hash.replace('#', '').trim();
      if (hash && hash !== currentChapterId) {
        const match = findChapter(hash);
        if (match) {
          setCurrentChapterId(hash);
        }
      }
    };

    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, [currentChapterId, setCurrentChapterId]);

  return (
    <div className="min-h-screen w-full bg-background text-foreground selection:bg-primary/20 selection:text-primary">
      <GuidePage />
      <CareerUnlockModal />
    </div>
  );
};
