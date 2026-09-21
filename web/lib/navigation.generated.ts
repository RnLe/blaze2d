/* Generated from content metadata. Run pnpm generate:content to update. */
import type { DocumentInfo } from './documents';

export interface NavigationItem {
  route: string;
  title: string;
  group: 'Use Blaze' | 'Research' | 'Theory' | 'Project';
  order: number;
  /** Set when the article is about a PDF, so navigation can advertise it. */
  document?: DocumentInfo;
}

export const navigation: NavigationItem[] = [
  {
    "route": "/blaze",
    "title": "Technical Report",
    "group": "Research",
    "order": 0
  },
  {
    "route": "/installation",
    "title": "Installation",
    "group": "Use Blaze",
    "order": 0
  },
  {
    "route": "/introduction",
    "title": "Introduction",
    "group": "Theory",
    "order": 0
  },
  {
    "route": "/roadmap",
    "title": "Optimization & Roadmap",
    "group": "Project",
    "order": 0
  },
  {
    "route": "/workbench?view=examples",
    "title": "Examples",
    "group": "Use Blaze",
    "order": 1
  },
  {
    "route": "/thesis",
    "title": "Master’s Thesis",
    "group": "Research",
    "order": 1
  },
  {
    "route": "/paper",
    "title": "Manuscript",
    "group": "Research",
    "order": 2,
    "document": {
      "href": "/paper/blaze2d.pdf",
      "pages": 11,
      "bytes": 970481
    }
  },
  {
    "route": "/workbench-guide",
    "title": "Workbench guide",
    "group": "Use Blaze",
    "order": 2
  },
  {
    "route": "/configuration",
    "title": "API & TOML",
    "group": "Use Blaze",
    "order": 3
  }
];
