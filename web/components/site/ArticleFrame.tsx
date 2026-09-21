import type { ReactNode } from 'react';
import { TableOfContents } from './TableOfContents';
export function ArticleFrame({ layout, toc, children }: { layout: string; toc: { text: string; id: string; depth: number }[]; children: ReactNode }) {
  const showToc = layout !== 'home' && toc.length > 1;
  return <div className={`article-layout article-layout-${layout}${showToc ? ' has-toc' : ''}`}>
    <article className={`article article-${layout}`} data-pagefind-body>
      {children}
    </article>
    {showToc && <TableOfContents headings={toc} />}
  </div>;
}
