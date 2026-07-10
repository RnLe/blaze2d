/**
 * Workbench Studio state.
 *
 * `config` (the canonical StudioConfig) is the single source of truth; the
 * TOML text is derived from it via `tomlSerialize`. Undo/redo (zundo temporal)
 * tracks only the config slice. Run history, plot specs, project metadata,
 * and UI layout live in the same store so every panel shares one
 * subscription surface (the archboard store pattern).
 */

'use client';

import { create } from 'zustand';
import { temporal } from 'zundo';
import type { TemporalState } from 'zundo';
import { useStore } from 'zustand';

import {
  type StudioConfig,
  defaultConfig,
  cloneConfig,
} from './configModel';
import { serializeConfig } from './tomlSerialize';
import type { BandResult } from '../examples/bandResult';
import type { Diagnostic, ValidationSummary } from './utilWorker';
import {
  type JobResult,
  type PlotSpec,
  type RunRecord,
  type SeriesRef,
  evictRuns,
  makeRunRecord,
  pruneRefs,
  runLabel,
  seriesKey,
} from './runData';

export type { JobResult, PlotSpec, RunRecord, SeriesRef } from './runData';

export type RunStatus = 'idle' | 'initializing' | 'running' | 'done' | 'error' | 'aborted';

export type CenterTab = 'geometry' | 'reciprocal' | 'plots' | 'data';

/** Pointer to the streaming run (transient; RunRecords hold the history). */
export interface LiveState {
  runId: string | null;
  status: RunStatus;
  /** Overall progress 0..1 across all jobs. */
  progress: number;
  jobIndex: number;
  totalJobs: number;
  precision: string;
  /** Partial band result of the job currently streaming. */
  liveResult: BandResult | null;
  startedAt: number | null;
  error: string | null;
}

const idleLive = (): LiveState => ({
  runId: null,
  status: 'idle',
  progress: 0,
  jobIndex: 0,
  totalJobs: 0,
  precision: 'f64',
  liveResult: null,
  startedAt: null,
  error: null,
});

export interface StudioState {
  // --- canonical config (undo-tracked) ---
  config: StudioConfig;

  // --- derived TOML + validation ---
  toml: {
    text: string;
    /** true while the TOML editor pane has focus (TOML wins conflicts). */
    focused: boolean;
    diagnostics: Diagnostic[];
    summary: ValidationSummary | null;
    /** Set when the current TOML text failed to parse into a config. */
    invalid: boolean;
  };

  // --- run history + live pointer ---
  /** Completed + in-flight runs, oldest first. Capped (see runData). */
  runs: RunRecord[];
  live: LiveState;

  // --- plotting + selection ---
  plots: { specs: PlotSpec[] };
  /** Data-tab selection: seriesKey() strings. */
  selection: Set<string>;

  // --- project meta ---
  project: {
    name: string;
    dirty: boolean;
  };

  // --- UI layout ---
  ui: {
    /** [leftFrac, centerFrac, rightFrac] of the three columns. */
    paneFractions: [number, number, number];
    accordionOpen: Record<string, boolean>;
    centerTab: CenterTab;
    selectedAtom: number | null;
    tomlVisible: boolean;
    fullscreen: boolean;
    /** Config panel drawer (narrow layouts only). */
    leftDrawerOpen: boolean;
  };

  // --- actions ---
  /** Mutate the config in place (immer-free); re-derives the TOML text. */
  patchConfig: (mutator: (draft: StudioConfig) => void) => void;
  /** Replace the whole config (e.g. loading a project). */
  setConfig: (config: StudioConfig, opts?: { markClean?: boolean }) => void;
  /** Apply a config parsed from edited TOML WITHOUT re-deriving the text
   *  (the editor holds the authoritative text while focused). */
  applyParsedConfig: (config: StudioConfig) => void;
  setTomlText: (text: string) => void;
  setTomlFocused: (focused: boolean) => void;
  setDiagnostics: (diags: Diagnostic[], summary: ValidationSummary | null, invalid: boolean) => void;

  // run lifecycle
  beginRun: (meta: {
    configToml: string;
    precision: string;
    polarization: string;
    solverType: string;
  }) => string;
  setRunMeta: (runId: string, patch: { totalJobs?: number; precision?: string }) => void;
  appendJob: (runId: string, job: JobResult) => void;
  finishRun: (
    runId: string,
    patch: { status: 'done' | 'error' | 'aborted'; computeMs?: number; error?: string },
  ) => void;
  setLive: (patch: Partial<LiveState>) => void;

  // plots + selection
  addPlot: (init?: Partial<Omit<PlotSpec, 'id'>>) => string;
  updatePlot: (id: string, patch: Partial<Omit<PlotSpec, 'id'>>) => void;
  removePlot: (id: string) => void;
  toggleSelect: (ref: SeriesRef) => void;
  setRunSelected: (runId: string, on: boolean) => void;
  clearSelection: () => void;

  setProjectName: (name: string) => void;
  markClean: () => void;

  setPaneFractions: (f: [number, number, number]) => void;
  toggleAccordion: (key: string) => void;
  setAccordion: (key: string, open: boolean) => void;
  setCenterTab: (tab: CenterTab) => void;
  selectAtom: (i: number | null) => void;
  setTomlVisible: (v: boolean) => void;
  setFullscreen: (v: boolean) => void;
  setLeftDrawerOpen: (v: boolean) => void;
}

function initialConfig(): StudioConfig {
  return defaultConfig();
}

let plotCounter = 0;

export const useStudioStore = create<StudioState>()(
  temporal(
    (set, get) => ({
      config: initialConfig(),
      toml: {
        text: serializeConfig(initialConfig()),
        focused: false,
        diagnostics: [],
        summary: null,
        invalid: false,
      },
      runs: [],
      live: idleLive(),
      plots: { specs: [] },
      selection: new Set<string>(),
      project: {
        name: 'Untitled crystal',
        dirty: false,
      },
      ui: {
        paneFractions: [0.24, 0.5, 0.26],
        accordionOpen: {
          geometry: true,
          grid: true,
          solver: true,
          path: false,
          sweeps: false,
          eigensolver: false,
          dielectric: false,
          output: false,
          advanced: false,
        },
        centerTab: 'geometry',
        selectedAtom: 0,
        tomlVisible: true,
        fullscreen: false,
        leftDrawerOpen: false,
      },

      patchConfig: (mutator) => {
        const next = cloneConfig(get().config);
        mutator(next);
        set((s) => ({
          config: next,
          toml: { ...s.toml, text: serializeConfig(next) },
          project: { ...s.project, dirty: true },
        }));
      },

      setConfig: (config, opts) => {
        const clone = cloneConfig(config);
        set((s) => ({
          config: clone,
          toml: { ...s.toml, text: serializeConfig(clone), invalid: false },
          project: { ...s.project, dirty: opts?.markClean ? false : true },
        }));
      },

      applyParsedConfig: (config) => {
        const clone = cloneConfig(config);
        set((s) => ({ config: clone, project: { ...s.project, dirty: true } }));
      },

      setTomlText: (text) =>
        set((s) => ({ toml: { ...s.toml, text }, project: { ...s.project, dirty: true } })),

      setTomlFocused: (focused) => set((s) => ({ toml: { ...s.toml, focused } })),

      setDiagnostics: (diagnostics, summary, invalid) =>
        set((s) => ({ toml: { ...s.toml, diagnostics, summary, invalid } })),

      /* ------------------------------ run lifecycle ------------------------------ */

      beginRun: (meta) => {
        const record = makeRunRecord(meta);
        set((s) => {
          const runs = evictRuns([...s.runs, record], record.id);
          const pruned = pruneRefs(runs, s.plots.specs, s.selection);
          return {
            runs,
            plots: { specs: pruned.specs },
            selection: pruned.selection,
            live: {
              ...idleLive(),
              runId: record.id,
              status: 'initializing',
              precision: meta.precision,
              startedAt: record.startedAt,
            },
          };
        });
        return record.id;
      },

      setRunMeta: (runId, patch) =>
        set((s) => ({
          runs: s.runs.map((r) => {
            if (r.id !== runId) return r;
            const next = { ...r, ...patch };
            next.label = runLabel(next);
            return next;
          }),
        })),

      appendJob: (runId, job) =>
        set((s) => {
          let runs = s.runs.map((r) => {
            if (r.id !== runId) return r;
            const jobs = new Map(r.jobs);
            jobs.set(job.jobIndex, job);
            return { ...r, jobs };
          });
          runs = evictRuns(runs, s.live.runId);
          const pruned = pruneRefs(runs, s.plots.specs, s.selection);
          return { runs, plots: { specs: pruned.specs }, selection: pruned.selection };
        }),

      finishRun: (runId, patch) =>
        set((s) => ({
          runs: s.runs.map((r) =>
            r.id === runId
              ? {
                  ...r,
                  status: patch.status,
                  computeMs: patch.computeMs ?? r.computeMs,
                  error: patch.error ?? null,
                }
              : r,
          ),
        })),

      setLive: (patch) => set((s) => ({ live: { ...s.live, ...patch } })),

      /* ----------------------------- plots + selection ---------------------------- */

      addPlot: (init) => {
        plotCounter += 1;
        const id = `p_${Date.now().toString(36)}_${plotCounter}`;
        const spec: PlotSpec = {
          id,
          name: init?.name ?? `Plot ${plotCounter}`,
          series: init?.series ?? [],
          bandRange: init?.bandRange ?? null,
          showLegend: init?.showLegend ?? true,
          showGrid: init?.showGrid ?? true,
        };
        set((s) => ({ plots: { specs: [...s.plots.specs, spec] } }));
        return id;
      },

      updatePlot: (id, patch) =>
        set((s) => ({
          plots: {
            specs: s.plots.specs.map((p) => (p.id === id ? { ...p, ...patch } : p)),
          },
        })),

      removePlot: (id) =>
        set((s) => ({ plots: { specs: s.plots.specs.filter((p) => p.id !== id) } })),

      toggleSelect: (ref) =>
        set((s) => {
          const key = seriesKey(ref);
          const selection = new Set(s.selection);
          if (selection.has(key)) selection.delete(key);
          else selection.add(key);
          return { selection };
        }),

      setRunSelected: (runId, on) =>
        set((s) => {
          const run = s.runs.find((r) => r.id === runId);
          if (!run) return {};
          const selection = new Set(s.selection);
          for (const jobIndex of run.jobs.keys()) {
            const key = seriesKey({ runId, jobIndex });
            if (on) selection.add(key);
            else selection.delete(key);
          }
          return { selection };
        }),

      clearSelection: () => set(() => ({ selection: new Set<string>() })),

      /* ---------------------------------- misc ---------------------------------- */

      setProjectName: (name) =>
        set((s) => ({ project: { ...s.project, name, dirty: true } })),

      markClean: () => set((s) => ({ project: { ...s.project, dirty: false } })),

      setPaneFractions: (paneFractions) => set((s) => ({ ui: { ...s.ui, paneFractions } })),
      toggleAccordion: (key) =>
        set((s) => ({
          ui: { ...s.ui, accordionOpen: { ...s.ui.accordionOpen, [key]: !s.ui.accordionOpen[key] } },
        })),
      setAccordion: (key, open) =>
        set((s) => ({
          ui: { ...s.ui, accordionOpen: { ...s.ui.accordionOpen, [key]: open } },
        })),
      setCenterTab: (centerTab) => set((s) => ({ ui: { ...s.ui, centerTab } })),
      selectAtom: (selectedAtom) => set((s) => ({ ui: { ...s.ui, selectedAtom } })),
      setTomlVisible: (tomlVisible) => set((s) => ({ ui: { ...s.ui, tomlVisible } })),
      setFullscreen: (fullscreen) => set((s) => ({ ui: { ...s.ui, fullscreen } })),
      setLeftDrawerOpen: (leftDrawerOpen) => set((s) => ({ ui: { ...s.ui, leftDrawerOpen } })),
    }),
    {
      // Undo/redo tracks only the config slice.
      partialize: (state) => ({ config: state.config }),
      limit: 100,
      equality: (a, b) => JSON.stringify(a.config) === JSON.stringify(b.config),
    },
  ),
);

/**
 * Hook into zundo's temporal store (undo/redo/clear + past/future lengths).
 * When the config is restored via undo/redo, re-derive the TOML text.
 */
export function useTemporalStore<T>(
  selector: (state: TemporalState<{ config: StudioConfig }>) => T,
): T {
  return useStore(useStudioStore.temporal, selector);
}

/** After a temporal undo/redo, the config changes but toml.text does not.
 *  Call this to re-sync derived state. */
export function resyncDerived(): void {
  const s = useStudioStore.getState();
  useStudioStore.setState({ toml: { ...s.toml, text: serializeConfig(s.config) } });
}
