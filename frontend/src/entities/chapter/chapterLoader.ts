import type { ChapterData, SiteId } from '@/shared/types';

const chapterModules: Record<string, () => Promise<{ default: ChapterData }>> = {
  ...import.meta.glob<{ default: ChapterData }>('./data/deepagents/*.js'),
  ...import.meta.glob<{ default: ChapterData }>('./data/rl/*.js'),
  ...import.meta.glob<{ default: ChapterData }>('./data/rlvr/*.js'),
};

// Static raw markdown modules bundled across all three curriculum tracks
const deepagentsMarkdown: Record<string, () => Promise<string>> = import.meta.glob<string>(
  '../../../../tutorials/*.md',
  { query: '?raw', import: 'default' }
);
const rlMarkdown: Record<string, () => Promise<string>> = import.meta.glob<string>(
  '../../../../rl/tutorials/*.md',
  { query: '?raw', import: 'default' }
);
const rlvrMarkdown: Record<string, () => Promise<string>> = import.meta.glob<string>(
  '../../../../rlvr/tutorials/*.md',
  { query: '?raw', import: 'default' }
);

function findMarkdownLoader(
  modules: Record<string, () => Promise<string>>,
  targetFile?: string,
  chapterId?: string
): (() => Promise<string>) | null {
  const keys = Object.keys(modules);
  if (targetFile) {
    const cleanTarget = targetFile.replace(/\.md$/, '');
    const found = keys.find(k => k.endsWith(`/${cleanTarget}.md`) || k.includes(`/${cleanTarget}.`));
    if (found) return modules[found];
  }
  if (chapterId) {
    const num = chapterId.replace(/\D/g, '');
    if (num) {
      const paddedNum = num.padStart(2, '0');
      const found = keys.find(k => {
        const base = k.split('/').pop() || '';
        return (
          base.startsWith(`${paddedNum}-`) ||
          base.startsWith(`${paddedNum}_`) ||
          base.startsWith(`${num}_`) ||
          base.startsWith(`${num}-`)
        );
      });
      if (found) return modules[found];
    }
  }
  return null;
}

export async function loadChapterData(siteId: SiteId, chapterId: string): Promise<ChapterData | null> {
  const key = `./data/${siteId}/${chapterId}.js`;
  const loader = chapterModules[key];

  if (!loader) {
    console.warn(`Chapter module not found for key: ${key}`);
    return null;
  }

  try {
    const mod = await loader();
    const data = { ...mod.default };
    const targetFile = data.file || data.id;

    // 1. Try resolving markdown from static bundled assets (100% client-side / GitHub Pages)
    if (siteId === 'deepagents') {
      const mdLoader = findMarkdownLoader(deepagentsMarkdown, targetFile, data.id);
      if (mdLoader) {
        try {
          data.markdownContent = await mdLoader();
        } catch (e) {
          console.warn('Error loading static deepagents markdown', e);
        }
      }
    } else if (siteId === 'rlvr') {
      const mdLoader = findMarkdownLoader(rlvrMarkdown, targetFile, data.id);
      if (mdLoader) {
        try {
          data.markdownContent = await mdLoader();
        } catch (e) {
          console.warn('Error loading static rlvr markdown', e);
        }
      }
    } else if (siteId === 'rl') {
      const mdLoader = findMarkdownLoader(rlMarkdown, targetFile, data.id);
      if (mdLoader) {
        try {
          data.markdownContent = await mdLoader();
        } catch (e) {
          console.warn('Error loading static rl markdown', e);
        }
      }
    }

    // 2. If not loaded statically, fall back to backend API fetch (for local dev with FastAPI)
    if (!data.markdownContent) {
      if (siteId === 'deepagents') {
        try {
          const res = await fetch(`/api/deepagents/tutorials/${targetFile}`);
          if (res.ok) {
            const json = await res.json();
            if (json && json.content) {
              data.markdownContent = json.content;
            }
          }
        } catch (err) {
          console.warn('Could not fetch deepagents tutorial from API, using inline content', err);
        }
      } else if (siteId === 'rlvr') {
        try {
          let res = await fetch(`/api/rlvr/chapters/${targetFile}`);
          if (!res.ok && targetFile !== data.id) {
            res = await fetch(`/api/rlvr/chapters/${data.id}`);
          }
          if (res.ok) {
            const json = await res.json();
            if (json && json.content) {
              data.markdownContent = json.content;
            }
          }
        } catch (err) {
          console.warn('Could not fetch rlvr chapter from API, using inline content', err);
        }
      } else if (siteId === 'rl') {
        try {
          let res = await fetch(`/api/rl/chapters/${targetFile}`);
          if (!res.ok && targetFile !== data.id) {
            res = await fetch(`/api/rl/chapters/${data.id}`);
          }
          if (res.ok) {
            const json = await res.json();
            if (json && json.content) {
              data.markdownContent = json.content;
            }
          }
        } catch (err) {
          console.warn('Could not fetch rl chapter from API, using inline content', err);
        }
      }
    }

    return data;
  } catch (err) {
    console.error(`Error loading chapter ${chapterId}:`, err);
    return null;
  }
}
