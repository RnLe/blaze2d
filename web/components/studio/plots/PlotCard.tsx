'use client';

import React, { useLayoutEffect, useMemo, useRef, useState } from 'react';
import { Trash2, X } from 'lucide-react';
import StudioBandPlot, { type PlotSeries } from './StudioBandPlot';
import { seriesColor } from './palette';
import { useStudioStore, type PlotSpec } from '../../../lib/studio/store';
import {
  collectExportSeries,
  jobLabel,
  seriesToCsv,
  type ExportSeries,
} from '../../../lib/studio/runData';
import { downloadText } from '../../../lib/util/download';
import { downloadRaster, downloadSvg, webpSupported } from '../../../lib/util/svgExport';

interface PlotCardProps {
  /** Static title (live card) or editable via spec. */
  title?: string;
  titleExtra?: string;
  series: PlotSeries[];
  /** When present, the card is an editable user plot. */
  spec?: PlotSpec;
  exportSeries: ExportSeries[];
  fileBase: string;
  height?: number;
}

export function PlotCard({
  title,
  titleExtra,
  series,
  spec,
  exportSeries,
  fileBase,
  height,
}: PlotCardProps) {
  const updatePlot = useStudioStore((s) => s.updatePlot);
  const removePlot = useStudioStore((s) => s.removePlot);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [plotW, setPlotW] = useState(640);
  const [busy, setBusy] = useState(false);
  const canWebp = useMemo(() => (typeof document !== 'undefined' ? webpSupported() : false), []);

  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0].contentRect.width;
      setPlotW(Math.max(360, Math.floor(w)));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const maxBands = Math.max(1, ...series.map((s) => s.result.n_bands));
  const bandRange = spec?.bandRange ?? null;

  const raster = async (mime: 'image/png' | 'image/webp', ext: string) => {
    const svg = svgRef.current;
    if (!svg || busy) return;
    setBusy(true);
    try {
      await downloadRaster(svg, `${fileBase}.${ext}`, { mime, scale: 2 });
    } catch {
      // encoding unsupported; button should already be hidden for webp
    } finally {
      setBusy(false);
    }
  };

  const exportCsv = () => {
    if (exportSeries.length === 0) return;
    downloadText(seriesToCsv(exportSeries, bandRange), `${fileBase}.csv`, 'text/csv');
  };

  return (
    <div className="studio__plotcard" ref={wrapRef}>
      <div className="studio__plotcard-head">
        {spec ? (
          <input
            className="studio__plotcard-name"
            value={spec.name}
            spellCheck={false}
            aria-label="Plot name"
            onChange={(e) => updatePlot(spec.id, { name: e.target.value })}
          />
        ) : (
          <span className="studio__plotcard-title">{title}</span>
        )}
        {titleExtra ? <span className="studio__plotcard-extra">{titleExtra}</span> : null}
        <span className="studio__spacer" />

        {spec ? (
          <>
            <label className="studio__plotcard-opt" title="Bands shown (1-based, inclusive)">
              bands
              <input
                className="studio__input studio__input--tiny"
                type="number"
                min={1}
                max={maxBands}
                value={bandRange?.[0] ?? 1}
                onChange={(e) => {
                  const lo = Math.max(1, Number(e.target.value) || 1);
                  updatePlot(spec.id, { bandRange: [lo, bandRange?.[1] ?? maxBands] });
                }}
              />
              –
              <input
                className="studio__input studio__input--tiny"
                type="number"
                min={1}
                max={maxBands}
                value={bandRange?.[1] ?? maxBands}
                onChange={(e) => {
                  const hi = Math.max(1, Number(e.target.value) || maxBands);
                  updatePlot(spec.id, { bandRange: [bandRange?.[0] ?? 1, hi] });
                }}
              />
            </label>
            <label className="studio__plotcard-opt">
              <input
                type="checkbox"
                checked={spec.showLegend}
                onChange={(e) => updatePlot(spec.id, { showLegend: e.target.checked })}
              />
              legend
            </label>
            <button
              type="button"
              className="studio__iconbtn"
              aria-label="Delete plot"
              title="Delete plot"
              onClick={() => removePlot(spec.id)}
            >
              <Trash2 size={13} />
            </button>
          </>
        ) : null}
      </div>

      {spec && spec.series.length > 0 ? (
        <div className="studio__chiprow">
          {series.map((s) => (
            <span key={s.id} className="studio__chip" style={{ borderColor: s.color }}>
              <span className="studio__chip-swatch" style={{ background: s.color }} />
              {s.label}
              <button
                type="button"
                className="studio__chip-x"
                aria-label={`Remove ${s.label}`}
                title="Remove series"
                onClick={() =>
                  updatePlot(spec.id, {
                    series: spec.series.filter(
                      (ref) => `${ref.runId}:${ref.jobIndex}` !== s.id,
                    ),
                  })
                }
              >
                <X size={10} />
              </button>
            </span>
          ))}
        </div>
      ) : null}

      {series.length > 0 ? (
        <StudioBandPlot
          series={series}
          width={plotW}
          height={height ?? Math.max(300, Math.min(480, Math.round(plotW * 0.52)))}
          bandRange={bandRange}
          showLegend={spec?.showLegend ?? true}
          showGrid={spec?.showGrid ?? true}
          svgRef={svgRef}
        />
      ) : (
        <div className="studio__placeholder">
          No series yet. Select jobs in the Data tab and use “Add to plot”.
        </div>
      )}

      {series.length > 0 ? (
        <div className="studio__plotcard-foot">
          <span className="studio__plotcard-hint">export</span>
          <button type="button" className="studio__btn studio__btn--mini" onClick={() => svgRef.current && downloadSvg(svgRef.current, `${fileBase}.svg`)}>
            SVG
          </button>
          <button type="button" className="studio__btn studio__btn--mini" disabled={busy} onClick={() => raster('image/png', 'png')}>
            PNG
          </button>
          {canWebp ? (
            <button type="button" className="studio__btn studio__btn--mini" disabled={busy} onClick={() => raster('image/webp', 'webp')}>
              WebP
            </button>
          ) : null}
          <button type="button" className="studio__btn studio__btn--mini" onClick={exportCsv}>
            CSV
          </button>
        </div>
      ) : null}
    </div>
  );
}

/** Resolve a plot spec into renderable series + export payload. */
export function useResolvedSpec(spec: PlotSpec): {
  series: PlotSeries[];
  exportSeries: ExportSeries[];
} {
  const runs = useStudioStore((s) => s.runs);
  return useMemo(() => {
    const exportSeries = collectExportSeries(runs, spec.series);
    const series: PlotSeries[] = exportSeries.map((e, i) => ({
      id: `${spec.series[i]?.runId}:${e.jobIndex}`,
      label: labelFor(e),
      color: seriesColor(i),
      result: e.result,
    }));
    return { series, exportSeries };
  }, [runs, spec.series]);
}

function labelFor(e: ExportSeries): string {
  if (e.sweepValues.length > 0) {
    return jobLabel({ jobIndex: e.jobIndex, sweepValues: e.sweepValues, result: e.result });
  }
  return `${e.runLabel.replace(/ · .*/, '')} · job ${e.jobIndex}`;
}
