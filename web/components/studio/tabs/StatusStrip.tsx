'use client';

import React, { useEffect, useState } from 'react';
import { useStudioStore } from '../../../lib/studio/store';
import { estimateRun, formatBytes, formatDuration } from '../../../lib/studio/estimate';

function formatMs(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

/**
 * Persistent slim strip under the center tabs: run status + progress on the
 * left, config estimates in the middle (always visible, independent of the
 * TOML pane), elapsed/ETA on the right. Clicking jumps to the Plots tab.
 */
export function StatusStrip() {
  const live = useStudioStore((s) => s.live);
  const runs = useStudioStore((s) => s.runs);
  const summary = useStudioStore((s) => s.toml.summary);
  const setCenterTab = useStudioStore((s) => s.setCenterTab);

  // Local elapsed ticker while running (kept out of the store).
  const [now, setNow] = useState(() => Date.now());
  const running = live.status === 'running' || live.status === 'initializing';
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, [running]);

  const currentRun = live.runId ? runs.find((r) => r.id === live.runId) : null;
  const lastRun = runs.length > 0 ? runs[runs.length - 1] : null;

  const statusText = (() => {
    if (live.status === 'initializing') return 'Loading solver…';
    if (live.status === 'running') {
      const jobs = live.totalJobs > 1 ? `job ${live.jobIndex + 1}/${live.totalJobs} · ` : '';
      return `Solving ${jobs}${Math.round(live.progress * 100)}% · ${live.precision}`;
    }
    if (live.status === 'done' && currentRun) {
      const ms = currentRun.computeMs != null ? ` · ${formatMs(currentRun.computeMs)}` : '';
      const jobs = currentRun.jobs.size > 1 ? ` · ${currentRun.jobs.size} jobs` : '';
      return `Done${ms}${jobs}`;
    }
    if (live.status === 'error') return `Error: ${(live.error ?? 'solve failed').split('\n')[0]}`;
    if (live.status === 'aborted') {
      const kept = currentRun?.jobs.size ?? 0;
      return kept > 0 ? `Aborted · ${kept} job${kept === 1 ? '' : 's'} kept` : 'Aborted';
    }
    if (lastRun) return `${lastRun.label} · ${lastRun.status}`;
    return 'Idle';
  })();

  const dotClass = (() => {
    if (running) return ' studio__strip-dot--running';
    if (live.status === 'done') return ' studio__strip-dot--done';
    if (live.status === 'error') return ' studio__strip-dot--error';
    return '';
  })();

  const estimate = estimateRun(summary);
  const estimateText = estimate
    ? [
        `${estimate.jobs} job${estimate.jobs === 1 ? '' : 's'}`,
        `${estimate.kPointsTotal} k-pts`,
        `mem ${formatBytes(estimate.bytesPerSolve)}`,
        `est ${formatDuration(estimate.modeledSeconds)} (modeled)`,
      ].join(' · ')
    : null;

  const elapsedText = (() => {
    if (!running || !live.startedAt) return null;
    const elapsed = (now - live.startedAt) / 1000;
    let text = `${elapsed.toFixed(0)} s`;
    if (live.progress > 0.03) {
      const eta = (elapsed / live.progress) * (1 - live.progress);
      text += ` · ETA ${formatDuration(eta)}`;
    }
    return text;
  })();

  return (
    <button
      type="button"
      className="studio__strip"
      onClick={() => setCenterTab('plots')}
      title="Show plots"
    >
      <div
        className="studio__strip-progress"
        style={{ width: `${(running ? live.progress : live.status === 'done' ? 1 : 0) * 100}%` }}
      />
      <span className={`studio__strip-dot${dotClass}`} />
      <span className={`studio__strip-status${live.status === 'error' ? ' studio__strip-status--error' : ''}`}>
        {statusText}
      </span>
      <span className="studio__strip-mid">{estimateText}</span>
      <span className="studio__strip-right">{elapsedText}</span>
    </button>
  );
}
