import type { SiteId, SiteManifest, ChapterSummary, ManifestPart } from '@/shared/types';
import { DEEPAGENTS_MANIFEST } from './deepagentsManifest';
import { RL_MANIFEST } from './rlManifest';
import { RLVR_MANIFEST } from './rlvrManifest';

export { DEEPAGENTS_MANIFEST, RL_MANIFEST, RLVR_MANIFEST };

export const SITES: Record<SiteId, SiteManifest> = {
  deepagents: DEEPAGENTS_MANIFEST,
  rl: RL_MANIFEST,
  rlvr: RLVR_MANIFEST,
};

export const DEFAULT_SITE: SiteId = 'deepagents';

export function getManifest(siteId: SiteId): SiteManifest {
  return SITES[siteId] || SITES[DEFAULT_SITE];
}

export function getAllChapters(siteId?: SiteId): ChapterSummary[] {
  if (siteId) {
    const manifest = getManifest(siteId);
    return manifest.parts.flatMap((p) => p.chapters);
  }
  return Object.values(SITES).flatMap((manifest) =>
    manifest.parts.flatMap((p) => p.chapters)
  );
}

export function findChapter(chapterId: string): {
  siteId: SiteId;
  chapter: ChapterSummary;
  part: ManifestPart;
} | null {
  for (const [sId, manifest] of Object.entries(SITES)) {
    for (const part of manifest.parts) {
      const chapter = part.chapters.find((c) => c.id === chapterId);
      if (chapter) {
        return { siteId: sId as SiteId, chapter, part };
      }
    }
  }
  return null;
}

export function getDefaultChapterId(siteId: SiteId): string {
  const manifest = getManifest(siteId);
  return manifest.parts[0]?.chapters[0]?.id || 'da01';
}
