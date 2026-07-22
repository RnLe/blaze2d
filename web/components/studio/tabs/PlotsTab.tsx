'use client';

import React, { useMemo } from 'react';
import { Play, Plus } from 'lucide-react';
import { useStudioStore, type PlotSpec } from '../../../lib/studio/store';
import {
  LIVE_WINDOW,
  jobLabel,
  type ExportSeries,
} from '../../../lib/studio/runData';
import { PlotCard, useResolvedSpec } from '../plots/PlotCard';
import { seriesColor } from '../plots/palette';
import type { PlotSeries } from '../plots/StudioBandPlot';

function fileBaseOf(name: string, suffix: string): string {
  const base = name.trim().replace(/[^\w-]+/g, '_').toLowerCase() || 'crystal';
  return `${base}_${suffix}`;
}

/**
 * Plots tab: card 0 is the live/latest overlay (rolling window of the most
 * recent jobs, streaming job drawn thicker); below it, user plot cards built
 * from Data-tab selections, each individually exportable.
 */
export function PlotsTab({ onRun }: { onRun: () => void }) {
  const live = useStudioStore((s) => s.live);
  const runs = useStudioStore((s) => s.runs);
  const specs = useStudioStore((s) => s.plots.specs);
  const addPlot = useStudioStore((s) => s.addPlot);
  const solverType = useStudioStore((s) => s.config.solver.type);
  const projectName = useStudioStore((s) => s.project.name);

  const running = live.status === 'running' || live.status === 'initializing';

  // The run shown in the live card: the streaming one, else the latest.
  const shownRun = useMemo(() => {
    if (live.runId) {
      const r = runs.find((run) => run.id === live.runId);
      if (r) return r;
    }
    return runs.length > 0 ? runs[runs.length - 1] : null;
  }, [live.runId, runs]);

  // Rolling window: last N finished jobs + the streaming partial.
  const liveCard = useMemo(() => {
    if (!shownRun) return null;
    const finished = [...shownRun.jobs.values()].sort((a, b) => a.jobIndex - b.jobIndex);
    const windowed = finished.slice(-LIVE_WINDOW);
    const series: PlotSeries[] = windowed.map((j) => ({
      id: `${shownRun.id}:${j.jobIndex}`,
      // Stable color per job index so curves keep their color as the window slides.
      color: seriesColor(j.jobIndex),
      label: jobLabel(j, shownRun),
      result: j.result,
    }));
    if (live.liveResult && live.runId === shownRun.id) {
      series.push({
        id: 'live',
        color: seriesColor(live.jobIndex),
        label: `job ${live.jobIndex} (streaming)`,
        result: live.liveResult,
        live: true,
      });
    }
    const exportSeries: ExportSeries[] = windowed.map((j) => ({
      runLabel: shownRun.label,
      jobIndex: j.jobIndex,
      sweepValues: j.sweepValues,
      result: j.result,
    }));
    const extra =
      finished.length > LIVE_WINDOW
        ? `showing last ${LIVE_WINDOW} of ${finished.length}`
        : undefined;
    return { series, exportSeries, extra };
  }, [shownRun, live.liveResult, live.runId, live.jobIndex]);

  const isNative = solverType === 'operator_data';
  const empty = !liveCard || liveCard.series.length === 0;

  return (
    <div className="studio__tabbody studio__tabbody--scroll subtle-scroll">
      {empty ? (
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
              <GhostBands />
              <span>Run this crystal to see its band structure here.</span>
              <button type="button" className="studio__btn studio__btn--run" onClick={onRun}>
                <Play size={13} /> Run
              </button>
            </>
          )}
        </div>
      ) : (
        <PlotCard
          title={`${running ? 'Live · ' : ''}${shownRun?.label ?? ''}`}
          titleExtra={liveCard.extra}
          series={liveCard.series}
          exportSeries={liveCard.exportSeries}
          fileBase={fileBaseOf(projectName, `run${shownRun?.index ?? 0}`)}
        />
      )}

      {specs.map((spec) => (
        <SpecCard key={spec.id} spec={spec} projectName={projectName} />
      ))}

      {!empty || specs.length > 0 ? (
        <button type="button" className="studio__add-btn" onClick={() => addPlot()}>
          <Plus size={13} /> New plot
        </button>
      ) : null}
    </div>
  );
}

function SpecCard({ spec, projectName }: { spec: PlotSpec; projectName: string }) {
  const { series, exportSeries } = useResolvedSpec(spec);
  return (
    <PlotCard
      spec={spec}
      series={series}
      exportSeries={exportSeries}
      fileBase={fileBaseOf(projectName, spec.name.replace(/\s+/g, '_').toLowerCase())}
    />
  );
}

/** Decorative empty-state: faint band curves. */
function GhostBands() {
  return (
    <svg width="220" height="90" viewBox="0 0 220 90" aria-hidden="true" style={{ opacity: 0.14 }}>
      {[
        'M0,80 C40,78 70,72 110,70 C150,68 180,66 220,66',
        'M0,64 C45,60 75,42 110,40 C145,38 185,50 220,48',
        'M0,40 C40,44 80,26 110,24 C140,22 190,32 220,28',
        'M0,26 C35,20 80,16 110,18 C150,20 185,10 220,12',
      ].map((d, i) => (
        <path key={i} d={d} fill="none" stroke="#8ab4e8" strokeWidth={1.6} />
      ))}
    </svg>
  );
}
