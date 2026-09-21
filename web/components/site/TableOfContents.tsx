'use client';
import { useEffect, useState } from 'react';

/**
 * Slack below the anchor position, in rem, before a heading counts as read.
 *
 * A heading jumped to from this list parks exactly at the anchor offset, so the
 * line has to sit a little below it or the entry just clicked would not select
 * itself.
 */
const READING_SLACK = 0.5;

/**
 * "On this page", with the current section measured from the scroll position
 * rather than taken from intersection events.
 *
 * An IntersectionObserver only reports which headings are inside a band, which
 * leaves both ends of the page undefined: scrolled fully up, the first heading
 * can sit above the band and nothing is marked; scrolled fully down, the last
 * heading may never reach it, so the mark stops short. Measuring instead makes
 * the two ends exact by construction -- top pins the first entry, bottom pins
 * the last -- and in between the active entry is simply the last heading that
 * has crossed the reading line.
 *
 * That line is read back from `scroll-padding-top`, so it is the same offset the
 * browser scrolls an anchor to (see --scroll-offset in tokens.css) and the two
 * cannot drift apart.
 */
export function TableOfContents({ headings }: { headings: { text: string; id: string; depth: number }[] }) {
  const [active, setActive] = useState(headings[0]?.id ?? '');
  const rootDepth = headings.length ? Math.min(...headings.map(heading => heading.depth)) : 2;

  useEffect(() => {
    let frame = 0;
    const update = () => {
      frame = 0;
      const elements = headings.map(heading => document.getElementById(heading.id)).filter(element => element !== null);
      if (!elements.length) return;
      const furthest = document.documentElement.scrollHeight - window.innerHeight;
      if (furthest > 0 && window.scrollY >= furthest - 2) return setActive(elements[elements.length - 1].id);
      if (window.scrollY <= 2) return setActive(elements[0].id);
      const root = getComputedStyle(document.documentElement);
      const line = (parseFloat(root.scrollPaddingTop) || 0) + parseFloat(root.fontSize) * READING_SLACK;
      let current = elements[0];
      for (const element of elements) if (element.getBoundingClientRect().top <= line) current = element;
      setActive(current.id);
    };
    // Layout keeps shifting after hydration (fonts, charts, the PDF viewer), so
    // the page height is watched as well as the scroll position.
    const schedule = () => { if (!frame) frame = requestAnimationFrame(update); };
    const observer = new ResizeObserver(schedule);
    observer.observe(document.documentElement);
    window.addEventListener('scroll', schedule, { passive: true });
    window.addEventListener('resize', schedule);
    update();
    return () => {
      if (frame) cancelAnimationFrame(frame);
      observer.disconnect();
      window.removeEventListener('scroll', schedule);
      window.removeEventListener('resize', schedule);
    };
  }, [headings]);

  return <nav className="article-toc" aria-label="On this page"><span>On this page</span>{headings.map(heading =>
    <a href={`#${heading.id}`} key={heading.id} className={heading.depth > rootDepth ? 'toc-sub' : ''} aria-current={heading.id === active ? 'location' : undefined}>{heading.text}</a>)}</nav>;
}
