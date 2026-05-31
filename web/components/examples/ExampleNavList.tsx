'use client';

import Link from 'next/link';
import { useState } from 'react';
import { EXAMPLES, CATEGORIES } from '../../lib/examples/registry';

export interface ExampleNavListProps {
  /** Slug of the currently open example (highlighted). */
  activeSlug: string;
}

/**
 * ExampleNavList — a slim, file-explorer-styled sidebar listing every example
 * grouped by category. The active example is highlighted. Gives a fast way to
 * jump between examples without returning to the index page.
 */
export default function ExampleNavList({ activeSlug }: ExampleNavListProps) {
  const grouped = CATEGORIES.map((category) => ({
    category,
    items: EXAMPLES.filter((e) => e.category === category),
  })).filter((g) => g.items.length > 0);

  return (
    <div
      style={{
        border: '1px solid #1f2937',
        borderRadius: '10px',
        background: '#0a0a0a',
        overflow: 'hidden',
        fontFamily: 'var(--font-mono, ui-monospace, monospace)',
        fontSize: '0.8rem',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          padding: '5px 8px 5px 12px',
          fontSize: '0.66rem',
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          color: '#6b7280',
          borderBottom: '1px solid #1f2937',
        }}
      >
        <span>Examples</span>
      </div>
      <div style={{ padding: '6px 4px' }}>
        {grouped.map(({ category, items }) => (
          <div key={category} style={{ marginBottom: 6 }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                padding: '6px 10px 4px',
                color: '#ffffff',
                fontSize: '0.95rem',
                fontWeight: 600,
                letterSpacing: '0.02em',
                textTransform: 'uppercase',
                fontFamily:
                  "'OpenAI Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
              }}
            >
              <span>{category}</span>
            </div>
            {items.map((e) => (
              <NavRow key={e.slug} slug={e.slug} title={e.title} active={e.slug === activeSlug} />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

function NavRow({ slug, title, active }: { slug: string; title: string; active: boolean }) {
  const [hover, setHover] = useState(false);
  return (
    <Link
      href={`/examples/${slug}`}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        display: 'block',
        width: '100%',
        textDecoration: 'none',
        padding: '5px 10px 5px 26px',
        borderRadius: 6,
        color: active ? '#f3f4f6' : '#9ca3af',
        background: active ? '#1f2937' : hover ? '#111827' : 'transparent',
        fontSize: 'inherit',
        transition: 'background 0.12s',
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
      }}
      title={title}
    >
      {title}
    </Link>
  );
}
