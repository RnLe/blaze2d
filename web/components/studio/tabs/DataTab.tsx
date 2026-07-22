'use client';

import React, { useMemo, useRef, useState } from 'react';
import { ChevronDown, ChevronRight, Copy, Download, History, Plus } from 'lucide-react';
import { useStudioStore, type RunRecord, type SeriesRef } from '../../../lib/studio/store';
import {
  collectExportSeries,
  freqRange,
  jobLabel,
  parseSeriesKey,
  seriesKey,
  seriesToCsv,
  seriesToJsonPayload,
  timeAgo,
  type JobResult,
} from '../../../lib/studio/runData';
import { parseToml } from '../../../lib/studio/tomlParse';
import { copyText, downloadJson, downloadText } from '../../../lib/util/download';
import { seriesColor } from '../plots/palette';
import StudioBandPlot from '../plots/StudioBandPlot';

/**
 * Data tab: browse the run history, select jobs, send them to plots, and
 * export selections as CSV/JSON. Row click opens a detail dock with a mini
 * plot of that job.
 */
export function DataTab() {
  const runs = useStudioStore((s) => s.runs);
  const selection = useStudioStore((s) => s.selection);
  const toggleSelect = useStudioStore((s) => s.toggleSelect);
  const setRunSelected = useStudioStore((s) => s.setRunSelected);
  const clearSelection = useStudioStore((s) => s.clearSelection);
  const specs = useStudioStore((s) => s.plots.specs);
  const addPlot = useStudioStore((s) => s.addPlot);
  const updatePlot = useStudioStore((s) => s.updatePlot);
  const setCenterTab = useStudioStore((s) => s.setCenterTab);
  const setConfig = useStudioStore((s) => s.setConfig);
  const projectName = useStudioStore((s) => s.project.name);

  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const [detail, setDetail] = useState<SeriesRef | null>(null);
  const [addMenuOpen, setAddMenuOpen] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const flash = (msg: string) => {
    setToast(msg);
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 1800);
  };

  const selectedRefs = useMemo(() => {
    const refs: SeriesRef[] = [];
    for (const key of selection) {
      const ref = parseSeriesKey(key);
      if (ref) refs.push(ref);
    }
    // Stable order: run order, then job index.
    const runOrder = new Map(runs.map((r, i) => [r.id, i]));
    refs.sort(
      (a, b) =>
        (runOrder.get(a.runId) ?? 0) - (runOrder.get(b.runId) ?? 0) || a.jobIndex - b.jobIndex,
    );
    return refs;
  }, [selection, runs]);

  /** Fallback: everything in the latest run when nothing is selected. */
  const exportRefs = useMemo<SeriesRef[]>(() => {
    if (selectedRefs.length > 0) return selectedRefs;
    const latest = runs[runs.length - 1];
    if (!latest) return [];
    return [...latest.jobs.keys()]
      .sort((a, b) => a - b)
      .map((jobIndex) => ({ runId: latest.id, jobIndex }));
  }, [selectedRefs, runs]);

  const fileBase = projectName.trim().replace(/[^\w-]+/g, '_').toLowerCase() || 'crystal';

  const exportCsv = () => {
    const series = collectExportSeries(runs, exportRefs);
    if (series.length === 0) return;
    downloadText(seriesToCsv(series), `${fileBase}_data.csv`, 'text/csv');
  };

  const exportJson = () => {
    const series = collectExportSeries(runs, exportRefs);
    if (series.length === 0) return;
    downloadJson(seriesToJsonPayload(series, projectName), `${fileBase}_data.json`);
  };

  const addToPlot = (plotId: string | null) => {
    setAddMenuOpen(false);
    if (exportRefs.length === 0) return;
    if (plotId === null) {
      addPlot({ series: exportRefs });
    } else {
      const spec = specs.find((p) => p.id === plotId);
      if (!spec) return;
      const have = new Set(spec.series.map(seriesKey));
      const merged = [...spec.series, ...exportRefs.filter((r) => !have.has(seriesKey(r)))];
      updatePlot(plotId, { series: merged });
    }
    setCenterTab('plots');
  };

  const restoreConfig = (run: RunRecord) => {
    const parsed = parseToml(run.configToml);
    if (!parsed) {
      flash('Could not parse the stored config');
      return;
    }
    setConfig(parsed);
    flash(`Config of ${run.label} restored`);
  };

  if (runs.length === 0) {
    return (
      <div className="studio__tabbody">
        <div className="studio__placeholder studio__placeholder--tall">
          <History size={28} strokeWidth={1.4} style={{ opacity: 0.4 }} />
          Completed runs appear here. The last 5 runs stay available for plotting and export.
        </div>
      </div>
    );
  }

  const detailJob: { run: RunRecord; job: JobResult } | null = (() => {
    if (!detail) return null;
    const run = runs.find((r) => r.id === detail.runId);
    const job = run?.jobs.get(detail.jobIndex);
    return run && job ? { run, job } : null;
  })();

  return (
    <div className="studio__tabbody">
      {/* toolbar */}
      <div className="studio__datatoolbar">
        <span className="studio__datacount">
          {selectedRefs.length > 0
            ? `${selectedRefs.length} selected`
            : `latest run (${exportRefs.length} job${exportRefs.length === 1 ? '' : 's'})`}
        </span>
        {selectedRefs.length > 0 ? (
          <button type="button" className="studio__btn studio__btn--mini" onClick={clearSelection}>
            Clear
          </button>
        ) : null}
        <span className="studio__spacer" />
        <div className="studio__addmenu-wrap">
          <button
            type="button"
            className="studio__btn"
            disabled={exportRefs.length === 0}
            onClick={() => setAddMenuOpen((v) => !v)}
          >
            <Plus size={13} /> Add to plot
          </button>
          {addMenuOpen ? (
            <div className="studio__addmenu" onPointerLeave={() => setAddMenuOpen(false)}>
              <button type="button" onClick={() => addToPlot(null)}>
                New plot
              </button>
              {specs.map((p) => (
                <button key={p.id} type="button" onClick={() => addToPlot(p.id)}>
                  {p.name}
                </button>
              ))}
            </div>
          ) : null}
        </div>
        <button type="button" className="studio__btn" onClick={exportCsv} title="Export the selection as CSV">
          <Download size={13} /> CSV
        </button>
        <button type="button" className="studio__btn" onClick={exportJson} title="Export the selection as JSON">
          {'{ }'} JSON
        </button>
      </div>

      {/* run groups */}
      <div className="studio__datalist subtle-scroll">
        {[...runs].reverse().map((run) => {
          const jobs = [...run.jobs.values()].sort((a, b) => a.jobIndex - b.jobIndex);
          const selectedCount = jobs.filter((j) =>
            selection.has(seriesKey({ runId: run.id, jobIndex: j.jobIndex })),
          ).length;
          const allSelected = selectedCount === jobs.length && jobs.length > 0;
          const isCollapsed = collapsed[run.id] ?? false;

          return (
            <div key={run.id} className="studio__datarun">
              <div className="studio__datarun-head">
                <button
                  type="button"
                  className="studio__iconbtn"
                  aria-label={isCollapsed ? 'Expand run' : 'Collapse run'}
                  onClick={() => setCollapsed((c) => ({ ...c, [run.id]: !isCollapsed }))}
                >
                  {isCollapsed ? <ChevronRight size={14} /> : <ChevronDown size={14} />}
                </button>
                <input
                  type="checkbox"
                  checked={allSelected}
                  ref={(el) => {
                    if (el) el.indeterminate = selectedCount > 0 && !allSelected;
                  }}
                  aria-label={`Select all jobs of ${run.label}`}
                  onChange={(e) => setRunSelected(run.id, e.target.checked)}
                />
                <span className="studio__datarun-label">{run.label}</span>
                <span
                  className={`studio__status-chip studio__status-chip--${
                    run.status === 'done' ? 'done' : run.status === 'error' ? 'error' : 'running'
                  }`}
                >
                  {run.status}
                </span>
                <span className="studio__datarun-meta">
                  {timeAgo(run.startedAt)}
                  {run.computeMs != null ? ` · ${(run.computeMs / 1000).toFixed(2)} s` : ''}
                </span>
                <span className="studio__spacer" />
                <button
                  type="button"
                  className="studio__btn studio__btn--mini"
                  title="Load this run's exact configuration back into the editor"
                  onClick={() => restoreConfig(run)}
                >
                  Restore config
                </button>
                <button
                  type="button"
                  className="studio__iconbtn"
                  aria-label="Copy TOML"
                  title="Copy this run's TOML"
                  onClick={() => {
                    void copyText(run.configToml).then((ok) => flash(ok ? 'TOML copied' : 'Copy failed'));
                  }}
                >
                  <Copy size={13} />
                </button>
              </div>

              {!isCollapsed
                ? jobs.map((job) => {
                    const ref: SeriesRef = { runId: run.id, jobIndex: job.jobIndex };
                    const key = seriesKey(ref);
                    const [fLo, fHi] = freqRange(job.result);
                    const isDetail =
                      detail?.runId === run.id && detail.jobIndex === job.jobIndex;
                    return (
                      <div
                        key={key}
                        className={`studio__datajob${isDetail ? ' studio__datajob--active' : ''}`}
                        onClick={() => setDetail(isDetail ? null : ref)}
                      >
                        <input
                          type="checkbox"
                          checked={selection.has(key)}
                          aria-label={`Select job ${job.jobIndex}`}
                          onClick={(e) => e.stopPropagation()}
                          onChange={() => toggleSelect(ref)}
                        />
                        <span className="studio__datajob-idx">#{job.jobIndex}</span>
                        {job.sweepValues.length > 0 ? (
                          <span className="studio__chiprow">
                            {job.sweepValues.map(([k, v]) => (
                              <span key={k} className="studio__chip studio__chip--plain">
                                {k.replace(/^atom\d+\./, '')}={typeof v === 'number' ? +v.toFixed(4) : v}
                              </span>
                            ))}
                          </span>
                        ) : (
                          <span className="studio__datajob-meta">single job</span>
                        )}
                        <span className="studio__spacer" />
                        <span className="studio__datajob-meta">
                          {job.result.n_bands} bands · {job.result.n_kpoints} k · ω {fLo.toFixed(2)}–{fHi.toFixed(2)}
                        </span>
                      </div>
                    );
                  })
                : null}
            </div>
          );
        })}
      </div>

      {/* detail dock */}
      {detailJob ? (
        <div className="studio__datadock">
          <div className="studio__datadock-head">
            <span className="studio__plotcard-title">
              {detailJob.run.label} · {jobLabel(detailJob.job, detailJob.run)}
            </span>
            <span className="studio__spacer" />
            <button
              type="button"
              className="studio__btn studio__btn--mini"
              onClick={() => {
                const series = collectExportSeries(runs, [detail!]);
                downloadText(seriesToCsv(series), `${fileBase}_job${detailJob.job.jobIndex}.csv`, 'text/csv');
              }}
            >
              CSV
            </button>
            <button
              type="button"
              className="studio__btn studio__btn--mini"
              onClick={() => setDetail(null)}
            >
              Close
            </button>
          </div>
          <StudioBandPlot
            series={[
              {
                id: 'detail',
                label: jobLabel(detailJob.job, detailJob.run),
                color: seriesColor(detailJob.job.jobIndex),
                result: detailJob.job.result,
              },
            ]}
            width={560}
            height={200}
            interactive={false}
            showLegend={false}
          />
        </div>
      ) : null}

      {toast ? <div className="studio__toast">{toast}</div> : null}
    </div>
  );
}
