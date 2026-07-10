'use client';

import React from 'react';
import { useStudioStore } from '../../../lib/studio/store';
import { timeAgo } from '../../../lib/studio/runData';

/**
 * Data tab: run/job browser with selection and export.
 * (Selection, add-to-plot, and exports land with the plot-builder milestone;
 * this stage lists the run history.)
 */
export function DataTab() {
  const runs = useStudioStore((s) => s.runs);

  if (runs.length === 0) {
    return (
      <div className="studio__tabbody">
        <div className="studio__placeholder studio__placeholder--tall">
          Completed runs appear here. The last 5 runs are kept in memory.
        </div>
      </div>
    );
  }

  return (
    <div className="studio__tabbody studio__tabbody--scroll subtle-scroll">
      {[...runs].reverse().map((run) => (
        <div key={run.id} className="studio__runrow">
          <span className="studio__runrow-label">{run.label}</span>
          <span className={`studio__status-chip studio__status-chip--${run.status === 'done' ? 'done' : run.status === 'error' ? 'error' : 'running'}`}>
            {run.status}
          </span>
          <span className="studio__runrow-meta">
            {run.jobs.size} job{run.jobs.size === 1 ? '' : 's'} · {timeAgo(run.startedAt)}
          </span>
        </div>
      ))}
    </div>
  );
}
