'use client';

import { useMemo, useState } from 'react';
import { CATEGORIES, EXAMPLES, type Example } from '../../lib/examples/registry';
import ExampleCard from './ExampleCard';

function matches(example: Example, query: string): boolean {
  if (!query) return true;
  const q = query.toLowerCase();
  return (
    example.title.toLowerCase().includes(q) ||
    example.description.toLowerCase().includes(q) ||
    example.category.toLowerCase().includes(q)
  );
}

export default function ExampleLibrary() {
  const [query, setQuery] = useState('');
  const [activeCategory, setActiveCategory] = useState<string>('All');

  const filtered = useMemo(
    () =>
      EXAMPLES.filter(
        (e) =>
          matches(e, query) && (activeCategory === 'All' || e.category === activeCategory),
      ),
    [query, activeCategory],
  );

  const grouped = useMemo(() => {
    const map = new Map<string, Example[]>();
    for (const cat of CATEGORIES) {
      const items = filtered.filter((e) => e.category === cat);
      if (items.length > 0) map.set(cat, items);
    }
    return map;
  }, [filtered]);

  const categories = ['All', ...CATEGORIES];

  return (
    <div style={{ width: '100%' }}>
      {/* Search + category filter */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 12,
          alignItems: 'center',
          marginBottom: 24,
        }}
      >
        <div style={{ position: 'relative', flex: '1 1 280px', minWidth: 220 }}>
          <span
            style={{
              position: 'absolute',
              left: 14,
              top: '50%',
              transform: 'translateY(-50%)',
              color: '#6b7280',
              fontSize: '1.9rem',
              lineHeight: 1,
              pointerEvents: 'none',
            }}
          >
            ⌕
          </span>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search examples by title or description..."
            style={{
              width: '100%',
              boxSizing: 'border-box',
              background: '#0a0a0a',
              border: '1px solid #1f2937',
              borderRadius: 10,
              padding: '10px 14px 10px 46px',
              color: '#e5e7eb',
              fontSize: '0.9rem',
              outline: 'none',
            }}
          />
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {categories.map((cat) => {
            const active = cat === activeCategory;
            return (
              <button
                key={cat}
                onClick={() => setActiveCategory(cat)}
                style={{
                  background: active ? '#2563eb' : 'transparent',
                  color: active ? '#fff' : '#9ca3af',
                  border: `1px solid ${active ? '#2563eb' : '#1f2937'}`,
                  borderRadius: 999,
                  padding: '6px 14px',
                  fontSize: '0.8rem',
                  cursor: 'pointer',
                  whiteSpace: 'nowrap',
                }}
              >
                {cat}
              </button>
            );
          })}
        </div>
      </div>

      {filtered.length === 0 ? (
        <p style={{ color: '#6b7280', fontSize: '0.9rem' }}>
          No examples match “{query}”.
        </p>
      ) : (
        Array.from(grouped.entries()).map(([category, items], sectionIndex) => (
          <section
            key={category}
            style={{
              marginBottom: 36,
              ...(sectionIndex > 0
                ? { marginTop: 40, borderTop: '1px solid #ffffff', paddingTop: 32 }
                : {}),
            }}
          >
            <h2
              style={{
                fontSize: '1.1rem',
                fontWeight: 600,
                color: '#e5e7eb',
                margin: '0 0 14px 0',
                borderBottom: '1px solid #1f2937',
                paddingBottom: 8,
              }}
            >
              {category}
              <span style={{ color: '#4b5563', fontSize: '1.1rem', marginLeft: 10 }}>
                {items.length}
              </span>
            </h2>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))',
                gap: 24,
                alignItems: 'start',
              }}
            >
              {items.map((e) => (
                <ExampleCard key={e.slug} example={e} />
              ))}
            </div>
          </section>
        ))
      )}
    </div>
  );
}
