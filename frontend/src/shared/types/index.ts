export type SiteId = 'deepagents' | 'rl' | 'rlvr';

export interface ChapterTocItem {
  level: number;
  text: string;
  anchor: string;
}

export interface CodeLineItem {
  line: number;
  text: string;
  title?: string;
  note?: string;
}

export interface ChapterSummary {
  id: string;
  num: string;
  title: string;
  file?: string;
  icon?: string;
  tag?: string;
  readTime?: string;
  summary?: string;
  competencies?: string[];
  hasVisualizer?: string;
  codeFile?: string;
  isCareerGated?: boolean;
}

export interface ChapterData extends ChapterSummary {
  file: string;
  html?: string;
  toc?: ChapterTocItem[];
  codeLines?: CodeLineItem[];
  markdownContent?: string;
}

export interface ManifestPart {
  id: string;
  label: string;
  icon?: string;
  description?: string;
  milestone?: string;
  jobTarget?: string;
  chapters: ChapterSummary[];
}

export interface SiteManifest {
  id: SiteId;
  name: string;
  shortName: string;
  description: string;
  icon: string;
  totalChapters: number;
  parts: ManifestPart[];
}
