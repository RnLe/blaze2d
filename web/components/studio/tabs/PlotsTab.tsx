'use client';

import React, { useLayoutEffect, useMemo, useRef, useState } from 'react';
import { Play } from 'lucide-react';
import { useStudioStore } from '../../../lib/studio/store';
import { LIVE_WINDOW } from '../../../lib/studio/runData';
import type { BandResult } from '../../../lib/examples/bandResult';
import BandPlot from '../../examples/BandPlot';

/**
 * Plots tab: the live/latest overlay (rolling window of the most recent
 * jobs) plus user-defined plot cards (added in the plot-builder milestone).
 */
export function PlotsTab({ onRun }: { onRun: () => void }) {
  const live = useStudioStore((s) => s.live);
  const runs = useStudioStore((s) => s.runs);
  const solverType = useStudioStore((s) => s.config.solver.type);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [plotW, setPlotW] = useState(640);

  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0].contentRect.width;
      setPlotW(Math.max(360, Math.floor(w - 48)));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // The run shown in the live card: the streaming one, else the latest.
  const shownRun = useMemo(() => {
    if (live.runId) {
      const r = runs.find((run) => run.id === live.runId);
      if (r) return r;
    }
    return runs.length > 0 ? runs[runs.length - 1] : null;
  }, [live.runId, runs]);

  const running = live.status === 'running' || live.status === 'initializing';

  // Rolling window: last N finished jobs + the streaming partial.
  const { results, totalFinished } = useMemo(() => {
    if (!shownRun) return { results: [] as BandResult[], totalFinished: 0 };
    const finished = [...shownRun.jobs.values()].sort((a, b) => a.jobIndex - b.jobIndex);
    const windowed = finished.slice(-LIVE_WINDOW).map((j) => j.result);
    if (live.liveResult && live.runId === shownRun.id) windowed.push(live.liveResult);
    return { results: windowed, totalFinished: finished.length };
  }, [shownRun, live.liveResult, live.runId]);

  const headerLabel = (() => {
    if (!shownRun) return 'Plots';
    const prefix = running && live.runId === shownRun.id ? 'Live · ' : '';
    const window =
      totalFinished > LIVE_WINDOW ? ` · showing last ${LIVE_WINDOW} of ${totalFinished}` : '';
    return `${prefix}${shownRun.label}${window}`;
  })();

  const isNative = solverType === 'operator_data';

  return (
    <div className="studio__tabbody studio__tabbody--scroll subtle-scroll" ref={wrapRef}>
      {results.length > 0 ? (
        <div className="studio__plotcard">
          <div className="studio__plotcard-head">
            <span className="studio__plotcard-title">{headerLabel}</span>
          </div>
          <BandPlot results={results} width={plotW} height={Math.max(300, plotW * 0.5)} />
        </div>
      ) : (
        <div className="studio__placeholder studio__placeholder--tall">
          {isNative ? (
            <span>
              Operator-data extraction runs on the native driver. Export the TOML and run it with
              the CLI or Python.
            </span>
          ) : running ? (
            <span>Waiting for the first k-point…</span>
          ) : (
            <>
              <span>Run this crystal to see its band structure here.</span>
              <button type="button" className="studio__btn studio__btn--run" onClick={onRun}>
                <Play size={13} /> Run
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}
