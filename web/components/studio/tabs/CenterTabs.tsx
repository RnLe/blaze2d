'use client';

import React from 'react';
import { useStudioStore, type CenterTab } from '../../../lib/studio/store';

const TABS: { id: CenterTab; label: string; shortcut: string }[] = [
  { id: 'geometry', label: 'Geometry', shortcut: '1' },
  { id: 'reciprocal', label: 'Reciprocal', shortcut: '2' },
  { id: 'plots', label: 'Plots', shortcut: '3' },
  { id: 'data', label: 'Data', shortcut: '4' },
];

export function CenterTabs() {
  const tab = useStudioStore((s) => s.ui.centerTab);
  const setTab = useStudioStore((s) => s.setCenterTab);
  const plotCount = useStudioStore((s) => s.plots.specs.length);
  const jobCount = useStudioStore((s) =>
    s.runs.reduce((sum, r) => sum + r.jobs.size, 0),
  );

  const badge = (id: CenterTab): number => {
    if (id === 'plots') return plotCount;
    if (id === 'data') return jobCount;
    return 0;
  };

  return (
    <div className="studio__tabs" role="tablist" aria-label="Workspace views">
      {TABS.map((t) => {
        const count = badge(t.id);
        return (
          <button
            key={t.id}
            type="button"
            role="tab"
            aria-selected={tab === t.id}
            className={`studio__tab${tab === t.id ? ' studio__tab--active' : ''}`}
            title={`${t.label} (${t.shortcut})`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
            {count > 0 ? <span className="studio__tab-badge">{count}</span> : null}
          </button>
        );
      })}
    </div>
  );
}
