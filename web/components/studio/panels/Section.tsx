'use client';

import React, { useEffect, useRef } from 'react';
import { ChevronRight } from 'lucide-react';
import { useStudioStore } from '../../../lib/studio/store';

/** An accordion section keyed into the store's ui.accordionOpen map. */
export function Section({
  id,
  title,
  badge,
  children,
}: {
  id: string;
  title: string;
  badge?: { kind: 'error' | 'native'; text: string };
  children: React.ReactNode;
}) {
  const open = useStudioStore((s) => s.ui.accordionOpen[id] ?? false);
  const toggle = useStudioStore((s) => s.toggleAccordion);
  const headRef = useRef<HTMLButtonElement | null>(null);
  const wasOpen = useRef(open);

  // When a section opens (click or programmatically), keep it in view.
  useEffect(() => {
    if (open && !wasOpen.current) {
      requestAnimationFrame(() => {
        headRef.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      });
    }
    wasOpen.current = open;
  }, [open]);

  return (
    <div className="studio__section" data-section={id}>
      <button ref={headRef} type="button" className="studio__section-head" onClick={() => toggle(id)}>
        <span className={`studio__section-chevron${open ? ' studio__section-chevron--open' : ''}`}>
          <ChevronRight size={14} />
        </span>
        {title}
        {badge ? (
          <span className={`studio__section-badge studio__section-badge--${badge.kind}`}>
            {badge.text}
          </span>
        ) : null}
      </button>
      {open ? <div className="studio__section-body">{children}</div> : null}
    </div>
  );
}
