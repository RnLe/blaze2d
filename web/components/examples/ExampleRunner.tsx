'use client';

import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { useMemo, useState, type ReactNode } from 'react';
import { getExample, type PreparedRun } from '../../lib/examples/registry';
import { useExampleRunner } from '../../lib/examples/useExampleRunner';
import type { BandResult } from '../../lib/examples/bandResult';
import BandPlot from './BandPlot';
import DataInspector from './DataInspector';
import FileExplorer, { type ExplorerFile } from './FileExplorer';
import ExampleNavList from './ExampleNavList';
import CodeWindow from './CodeWindow';
import OutputBox from './OutputBox';
import { downloadText, downloadZip } from '../../lib/util/download';

export interface ExampleRunnerProps {
  slug: string;
  /** Optional server-rendered primer (markdown + math). Falls back to the
   *  registry's plain-text primer when not provided. */
  primer?: ReactNode;
}

const STATUS_LABEL: Record<string, string> = {
  idle: 'Ready',
  initializing: 'Loading WASM…',
  running: 'Computing…',
  done: 'Done',
  error: 'Error',
  aborted: 'Aborted',
};

export default function ExampleRunner({ slug, primer }: ExampleRunnerProps) {
  const example = getExample(slug)!;
  const runner = useExampleRunner();
  const [prepared, setPrepared] = useState<PreparedRun | null>(null);
  const [prepareError, setPrepareError] = useState<string | null>(null);

  // Prepare statically (pure) so the TOML config can be shown before running.
  const staticPrepared = useMemo(() => {
    try {
      return example.prepare();
    } catch {
      return null;
    }
  }, [example]);

  // Files that make up this example.
  const files: ExplorerFile[] = useMemo(() => {
    const list: ExplorerFile[] = [{ name: example.pyFile, language: 'python' }];
    if (example.showToml && example.tomlFile) {
      list.push({ name: example.tomlFile, language: 'toml' });
    }
    return list;
  }, [example]);

  const [activeFile, setActiveFile] = useState(example.pyFile);
  const isPython = activeFile === example.pyFile;
  const tomlCode = staticPrepared?.toml ?? '';

  /** Download all files in the explorer (zip when >1, raw file when 1). */
  const handleDownloadAll = () => {
    const all: { name: string; content: string }[] = [
      { name: example.pyFile, content: example.code },
    ];
    if (example.showToml && example.tomlFile) {
      all.push({ name: example.tomlFile, content: tomlCode });
    }
    if (all.length === 1) {
      downloadText(all[0].content, all[0].name);
    } else {
      downloadZip(all, `${example.slug}.zip`);
    }
  };

  const handleRun = () => {
    try {
      const run = example.prepare();
      setPrepareError(null);
      setPrepared(run);
      runner.run({
        configToml: run.toml,
        mode: run.mode,
        meta: run.meta,
        totalJobs: run.totalJobs,
        kIndices: run.kIndices,
        bandIndices: run.bandIndices,
      });
    } catch (e) {
      setPrepareError(e instanceof Error ? e.message : String(e));
    }
  };

  const isBusy = runner.status === 'running' || runner.status === 'initializing';
  const activePrepared = prepared ?? staticPrepared;

  // Decide what to plot and what to inspect based on the display kind.
  const plotResults: BandResult[] = useMemo(() => {
    const kind = activePrepared?.displayKind;
    if (kind === 'list') {
      return runner.results.filter(Boolean);
    }
    if (kind === 'BandResult') {
      if (runner.results[0]) return [runner.results[0]];
      return runner.liveResult ? [runner.liveResult] : [];
    }
    // streamDict — for filtered mode, map the collected result dicts; for live
    // streaming, show the progressive live result.
    if (activePrepared?.mode === 'filtered') {
      return runner.rawStream
        .map((d) => maxwellDictToBandResult(d, activePrepared.meta))
        .filter((r): r is BandResult => r !== null);
    }
    return runner.liveResult ? [runner.liveResult] : [];
  }, [activePrepared, runner.results, runner.liveResult, runner.rawStream]);

  const inspectData = useMemo(() => {
    const kind = activePrepared?.displayKind;
    if (kind === 'list') {
      return runner.results.filter(Boolean);
    }
    if (kind === 'BandResult') {
      return runner.results[0] ?? runner.liveResult ?? null;
    }
    // streamDict: show the most recent streamed dict (or the collected list for filtered).
    if (activePrepared?.mode === 'filtered') {
      return runner.rawStream;
    }
    return runner.rawStream.length > 0
      ? runner.rawStream[runner.rawStream.length - 1]
      : null;
  }, [activePrepared, runner.results, runner.liveResult, runner.rawStream]);

  const resultVar = activePrepared?.resultVar ?? 'result';
  const resultLabel =
    activePrepared?.displayKind === 'list'
      ? `list[BandResult] (${runner.results.filter(Boolean).length})`
      : activePrepared?.resultLabel;

  // Live stdout for examples with print(...) statements.
  const outputText = useMemo(() => {
    if (!example.output) return null;
    try {
      return example.output({
        results: runner.results,
        live: runner.liveResult,
        rawStream: runner.rawStream,
        status: runner.status,
      });
    } catch {
      return null;
    }
  }, [example, runner.results, runner.liveResult, runner.rawStream, runner.status]);

  const runActions = (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <button onClick={handleRun} disabled={isBusy} style={runBtnStyle(isBusy)}>
        {isBusy ? 'Running…' : '▶ Run'}
      </button>
      <button onClick={runner.abort} disabled={!isBusy} style={abortBtnStyle(isBusy)}>
        ■
      </button>
    </div>
  );

  return (
    <div
      style={{
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        background: '#000',
        color: '#e5e7eb',
      }}
    >
      {/* Top bar: back arrow + title only.
          Using <div> (not <header>) because docs-chrome CSS hides every <header>
          inside `.blaze-layout`. */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '14px',
          padding: '12px 24px',
          borderBottom: '1px solid #1f2937',
          background: 'rgba(0,0,0,0.85)',
          backdropFilter: 'blur(8px)',
          flexShrink: 0,
          zIndex: 10,
        }}
      >
        <Link
          href="/examples#library"
          style={{
            color: '#9ca3af',
            textDecoration: 'none',
            fontSize: '0.85rem',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.4rem',
            border: '1px solid #1f2937',
            borderRadius: 8,
            padding: '6px 12px',
          }}
        >
          <ArrowLeft size={16} /> Back
        </Link>
        <h1 style={{ fontSize: '1.55rem', fontWeight: 600, margin: 0, flex: 1 }}>
          {example.title}
        </h1>
        <span
          style={{
            fontSize: '0.78rem',
            color: '#6b7280',
            border: '1px solid #1f2937',
            borderRadius: '999px',
            padding: '3px 12px',
          }}
        >
          {example.category}
        </span>
      </div>

      {/* Per-example primer: a short context paragraph that flags the core
          lesson, common pitfalls, or things to look at while the example runs. */}
      {(primer ?? example.primer) && (
        <div
          style={{
            padding: '14px 24px 16px',
            borderBottom: '1px solid #1f2937',
            background: '#050505',
            color: '#cbd5e1',
            fontSize: '0.98rem',
            lineHeight: 1.6,
            flexShrink: 0,
            maxWidth: 1640,
            width: '100%',
            margin: '0 auto',
            boxSizing: 'border-box',
          }}
        >
          {primer ?? example.primer}
        </div>
      )}

      <div
        style={{
          display: 'grid',
          gridTemplateColumns:
            'minmax(170px, 200px) minmax(0, 1.05fr) minmax(0, 1fr) minmax(180px, 220px)',
          gap: '20px',
          padding: '20px 24px',
          maxWidth: 1860,
          width: '100%',
          margin: '0 auto',
          boxSizing: 'border-box',
          alignItems: 'stretch',
          flex: 1,
          minHeight: 0,
          overflow: 'hidden',
        }}
      >
        {/* Col 1: file explorer */}
        <div className="subtle-scroll" style={{ minHeight: 0, overflowY: 'auto' }}>
          <FileExplorer
            files={files}
            activeFile={activeFile}
            onSelect={setActiveFile}
            folder={example.slug}
            onDownloadAll={handleDownloadAll}
          />
        </div>

        {/* Col 2: code + controls + output */}
        <section
          className="subtle-scroll"
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '14px',
            minWidth: 0,
            minHeight: 0,
            overflowY: 'auto',
          }}
        >
          {isPython ? (
            <CodeWindow
              code={example.code}
              language="python"
              filename={example.pyFile}
              actions={runActions}
              showLineNumbers
              maxHeight={420}
            />
          ) : (
            <CodeWindow
              code={tomlCode}
              language="toml"
              filename={example.tomlFile ?? 'config.toml'}
              actions={runActions}
              maxHeight={420}
            />
          )}

          {/* Progress (under the code window) */}
          {(isBusy || runner.progress > 0) && (
            <div>
              <div
                style={{
                  height: 6,
                  background: '#1f2937',
                  borderRadius: 999,
                  overflow: 'hidden',
                }}
              >
                <div
                  style={{
                    height: '100%',
                    width: `${runner.progress}%`,
                    background: 'linear-gradient(90deg, #2563eb, #60a5fa)',
                    transition: 'width 0.1s linear',
                  }}
                />
              </div>
              <div
                style={{
                  marginTop: 6,
                  fontSize: '0.74rem',
                  color: '#6b7280',
                  display: 'flex',
                  gap: 8,
                  flexWrap: 'wrap',
                }}
              >
                <span>{STATUS_LABEL[runner.status] ?? runner.status}</span>
                {runner.computeTime != null && runner.status === 'done' && (
                  <span>· {runner.computeTime.toFixed(0)} ms</span>
                )}
                {runner.totalJobs > 1 && (
                  <span>· job {runner.jobIndex + 1}/{runner.totalJobs}</span>
                )}
                <span>
                  · k {runner.currentKIndex + 1}/{runner.totalKPoints || '?'} ·{' '}
                  {runner.progress.toFixed(0)}%
                </span>
              </div>
            </div>
          )}

          {example.singleCore && (
            <div style={noteStyle}>
              ⚠ The browser runs a single-threaded WASM build. Thread settings are accepted by the
              API but ignored for in-browser compute (jobs run sequentially on one core).
            </div>
          )}

          {(runner.error || prepareError) && (
            <div style={errorStyle}>{prepareError ?? runner.error}</div>
          )}

          {/* Output box (live stdout) */}
          {example.output && (outputText || isBusy) && (
            <OutputBox text={outputText ?? ''} running={isBusy} />
          )}
        </section>

        {/* Col 3: results */}
        <section
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '14px',
            minWidth: 0,
            minHeight: 0,
            overflow: 'hidden',
          }}
        >
          <div style={{ width: '100%', flexShrink: 0 }}>
            <h3 style={sectionTitleStyle}>Band diagram</h3>
            <BandPlot
              results={plotResults}
              xMaxOverride={staticPrepared?.meta.k_label_distances?.at(-1)}
            />
          </div>

          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              minHeight: 0,
              flex: 1,
            }}
          >
            <h3 style={{ ...sectionTitleStyle, flexShrink: 0 }}>
              Result{resultLabel ? ` · ${resultLabel}` : ''}
            </h3>
            {inspectData != null ? (
              <div className="subtle-scroll" style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
                <DataInspector
                  data={inspectData}
                  rootName={resultVar}
                  rootType={resultLabel}
                  defaultOpen
                />
              </div>
            ) : (
              <div
                style={{
                  fontSize: '0.85rem',
                  color: '#6b7280',
                  border: '1px solid #1f2937',
                  borderRadius: '10px',
                  padding: '16px',
                }}
              >
                Run the example to inspect the result object.
              </div>
            )}
          </div>
        </section>

        {/* Col 4: quick navigation between examples */}
        <div className="subtle-scroll" style={{ minHeight: 0, overflowY: 'auto' }}>
          <ExampleNavList activeSlug={slug} />
        </div>
      </div>
    </div>
  );
}

function runBtnStyle(isBusy: boolean): React.CSSProperties {
  return {
    background: isBusy ? '#1f2937' : '#2563eb',
    color: '#fff',
    border: 'none',
    borderRadius: '7px',
    padding: '5px 14px',
    fontSize: '0.82rem',
    fontWeight: 600,
    cursor: isBusy ? 'default' : 'pointer',
  };
}

function abortBtnStyle(isBusy: boolean): React.CSSProperties {
  return {
    background: 'transparent',
    color: isBusy ? '#f87171' : '#4b5563',
    border: `1px solid ${isBusy ? '#7f1d1d' : '#1f2937'}`,
    borderRadius: '7px',
    padding: '5px 11px',
    fontSize: '0.82rem',
    cursor: isBusy ? 'pointer' : 'default',
  };
}

const sectionTitleStyle: React.CSSProperties = {
  fontSize: '0.72rem',
  fontWeight: 600,
  textTransform: 'uppercase',
  letterSpacing: '0.05em',
  color: '#9ca3af',
  margin: '0 0 8px 0',
};

const noteStyle: React.CSSProperties = {
  fontSize: '0.8rem',
  color: '#fcd34d',
  background: 'rgba(251, 191, 36, 0.08)',
  border: '1px solid rgba(251, 191, 36, 0.25)',
  borderRadius: '8px',
  padding: '10px 14px',
};

const errorStyle: React.CSSProperties = {
  fontSize: '0.82rem',
  color: '#fca5a5',
  background: 'rgba(248, 113, 113, 0.08)',
  border: '1px solid rgba(248, 113, 113, 0.25)',
  borderRadius: '8px',
  padding: '10px 14px',
};

/** Map a WASM Maxwell result dict (job-level/filtered) to a plottable BandResult. */
function maxwellDictToBandResult(
  d: Record<string, unknown>,
  meta: { lattice_type: string; k_labels: string[]; k_label_distances: number[] },
): BandResult | null {
  const bands = d.bands as number[][] | undefined;
  const distances = d.distances as number[] | undefined;
  const kPath = d.k_path as number[][] | undefined;
  if (!bands || bands.length === 0 || !distances) return null;
  const params = (d.params as Record<string, unknown>) ?? {};
  const atoms = (params.atoms as Array<Record<string, unknown>>) ?? [];
  const first = atoms[0] ?? {};
  return {
    freqs: bands,
    distances,
    k_points: kPath ?? [],
    lattice_type: (params.lattice_type as string) ?? meta.lattice_type,
    polarization: (params.polarization as string) ?? 'TM',
    epsilon_background: (params.eps_bg as number) ?? 0,
    epsilon_atoms: (first.eps_inside as number) ?? 0,
    radius_atom: (first.radius as number) ?? 0,
    resolution: (params.resolution as number) ?? 0,
    n_bands: bands[0]?.length ?? 0,
    n_kpoints: bands.length,
    k_labels: meta.k_labels,
    k_label_distances: meta.k_label_distances,
  };
}
