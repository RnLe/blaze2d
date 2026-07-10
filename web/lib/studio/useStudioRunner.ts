/**
 * Studio run + validation orchestration.
 *
 * Owns two workers built on the shared WASM loader:
 *  - solveWorker: runs the driver, streaming k-points; aborted by terminate.
 *  - utilWorker:  authoritative validation, alive during a running solve.
 *
 * Results accumulate per job via BandResultBuilder (keyed by job index). A job
 * is finalized into the run history the moment its LAST k-point streams in,
 * so the live overlay can show finished + streaming curves without waiting
 * for the whole sweep. Runs live in the store's capped history (see runData).
 */

'use client';

import { useCallback, useEffect, useRef } from 'react';

import { BandResultBuilder, type BandResult } from '../examples/bandResult';
import { PRESET_LABELS, resolvePreset } from './geometry';
import { useStudioStore } from './store';
import { computeLabelDistances } from './runData';
import type { SolveWorkerOutMessage } from './solveWorker';
import type { UtilWorkerOutMessage } from './utilWorker';

interface KPointData {
  job_index: number;
  k_index: number;
  total_k_points: number;
  k_point: [number, number];
  distance: number;
  omegas: number[];
  bands: number[];
  is_gamma: boolean;
  num_bands: number;
  progress: number;
  params: {
    eps_bg: number;
    resolution: number;
    polarization: string;
    lattice_type?: string;
    atoms?: { index: number; pos: [number, number]; radius: number; eps_inside: number }[];
    sweep_values?: Record<string, number | string>;
  };
}

interface JobAccumulator {
  builder: BandResultBuilder;
  labels: string[];
  distances: number[];
  sweepValues: [string, string | number][];
}

function basePath(): string {
  return process.env.NEXT_PUBLIC_BASE_PATH || '';
}

/** Short label for a custom k-path point. */
function pointLabel(i: number): string {
  return `k${i}`;
}

export function useStudioRunner() {
  const solveWorkerRef = useRef<Worker | null>(null);
  const utilWorkerRef = useRef<Worker | null>(null);
  const jobsRef = useRef<Map<number, JobAccumulator>>(new Map());
  const runIdRef = useRef<string | null>(null);
  /** Label plan for the current run, captured from the config at Run click. */
  const labelPlanRef = useRef<{ mode: 'preset' | 'points'; customCount: number }>({
    mode: 'preset',
    customCount: 0,
  });
  const validateIdRef = useRef(0);

  const setDiagnostics = useStudioStore((s) => s.setDiagnostics);
  const beginRun = useStudioStore((s) => s.beginRun);
  const setRunMeta = useStudioStore((s) => s.setRunMeta);
  const appendJob = useStudioStore((s) => s.appendJob);
  const finishRun = useStudioStore((s) => s.finishRun);
  const setLive = useStudioStore((s) => s.setLive);

  // --- util worker: spin up once, keep alive ---
  useEffect(() => {
    const worker = new Worker(new URL('./utilWorker.ts', import.meta.url), { type: 'module' });
    utilWorkerRef.current = worker;
    worker.postMessage({ type: 'init', basePath: basePath() });

    worker.onmessage = (ev: MessageEvent<UtilWorkerOutMessage>) => {
      const msg = ev.data;
      if (msg.type === 'validated') {
        // Drop stale responses.
        if (msg.id !== validateIdRef.current) return;
        setDiagnostics(msg.errors, msg.summary ?? null, !msg.ok);
      }
    };

    return () => {
      worker.terminate();
      utilWorkerRef.current = null;
    };
  }, [setDiagnostics]);

  const validate = useCallback((configToml: string) => {
    const worker = utilWorkerRef.current;
    if (!worker) return;
    validateIdRef.current += 1;
    worker.postMessage({ type: 'validate', id: validateIdRef.current, configToml });
  }, []);

  const accumulatorFor = useCallback((k: KPointData): JobAccumulator => {
    const existing = jobsRef.current.get(k.job_index);
    if (existing) return existing;

    const p = k.params;
    const sweepValues = p.sweep_values
      ? (Object.entries(p.sweep_values) as [string, string | number][])
      : [];

    const plan = labelPlanRef.current;
    let labels: string[];
    if (plan.mode === 'points' && plan.customCount >= 2) {
      labels = Array.from({ length: plan.customCount }, (_, i) => pointLabel(i));
    } else {
      const latticeType = p.lattice_type ?? 'square';
      const presetName = resolvePreset('auto', latticeType);
      labels = presetName ? (PRESET_LABELS[presetName] ?? []) : [];
    }

    const atom0 = p.atoms?.[0];
    const acc: JobAccumulator = {
      builder: new BandResultBuilder({
        lattice_type: p.lattice_type ?? 'square',
        polarization: p.polarization,
        epsilon_background: p.eps_bg,
        epsilon_atoms: atom0?.eps_inside ?? 1,
        radius_atom: atom0?.radius ?? 0,
        resolution: p.resolution,
        k_labels: labels,
        k_label_distances: [],
      }),
      labels,
      distances: [],
      sweepValues,
    };
    jobsRef.current.set(k.job_index, acc);
    return acc;
  }, []);

  /** builder.finish() plus the streamed-distance label positions. */
  const finalize = useCallback((acc: JobAccumulator): BandResult => {
    const base = acc.builder.finish();
    return {
      ...base,
      k_labels: acc.labels,
      k_label_distances: computeLabelDistances(base.distances, acc.labels.length),
    };
  }, []);

  const stopSolve = useCallback(() => {
    if (solveWorkerRef.current) {
      solveWorkerRef.current.terminate();
      solveWorkerRef.current = null;
    }
  }, []);

  /** Finalize any builders that never saw their last k-point (abort, done). */
  const flushPending = useCallback(() => {
    const runId = runIdRef.current;
    if (!runId) return;
    for (const [jobIndex, acc] of jobsRef.current) {
      if (acc.builder.count === 0) continue;
      appendJob(runId, { jobIndex, result: finalize(acc), sweepValues: acc.sweepValues });
    }
    jobsRef.current.clear();
  }, [appendJob, finalize]);

  const run = useCallback(
    (configToml: string) => {
      // Fresh worker every run (also our only way to abort a prior run).
      stopSolve();
      jobsRef.current = new Map();

      const cfg = useStudioStore.getState().config;
      labelPlanRef.current = {
        mode: cfg.path.mode,
        customCount: cfg.path.points.length,
      };
      const runId = beginRun({
        configToml,
        precision: cfg.solver.precision,
        polarization: cfg.solver.polarization.toUpperCase(),
        solverType: cfg.solver.type,
      });
      runIdRef.current = runId;

      const worker = new Worker(new URL('./solveWorker.ts', import.meta.url), { type: 'module' });
      solveWorkerRef.current = worker;
      worker.postMessage({ type: 'init', basePath: basePath() });

      worker.onmessage = (ev: MessageEvent<SolveWorkerOutMessage>) => {
        const msg = ev.data;
        switch (msg.type) {
          case 'ready':
            worker.postMessage({ type: 'run', configToml });
            setLive({ status: 'running' });
            break;
          case 'meta':
            setRunMeta(runId, { totalJobs: msg.totalJobs, precision: msg.precision });
            setLive({ totalJobs: msg.totalJobs, precision: msg.precision });
            break;
          case 'kpoint': {
            const k = msg.data as unknown as KPointData;
            const acc = accumulatorFor(k);
            acc.builder.addKPoint({ k_point: k.k_point, distance: k.distance, bands: k.bands });

            const jobDone = k.k_index >= k.total_k_points - 1;
            if (jobDone) {
              appendJob(runId, {
                jobIndex: k.job_index,
                result: finalize(acc),
                sweepValues: acc.sweepValues,
              });
              jobsRef.current.delete(k.job_index);
              setLive({ jobIndex: k.job_index, progress: k.progress, liveResult: null });
            } else {
              setLive({
                jobIndex: k.job_index,
                progress: k.progress,
                liveResult: finalize(acc),
              });
            }
            break;
          }
          case 'done': {
            flushPending();
            finishRun(runId, { status: 'done', computeMs: msg.totalTime });
            setLive({ status: 'done', progress: 1, liveResult: null });
            stopSolve();
            break;
          }
          case 'error':
            flushPending();
            finishRun(runId, { status: 'error', error: msg.message });
            setLive({ status: 'error', error: msg.message, liveResult: null });
            stopSolve();
            break;
        }
      };
    },
    [accumulatorFor, appendJob, beginRun, finalize, finishRun, flushPending, setLive, setRunMeta, stopSolve],
  );

  const abort = useCallback(() => {
    stopSolve();
    flushPending();
    const runId = runIdRef.current;
    if (runId) finishRun(runId, { status: 'aborted' });
    setLive({ status: 'aborted', liveResult: null });
  }, [finishRun, flushPending, setLive, stopSolve]);

  // Clean up on unmount.
  useEffect(() => () => stopSolve(), [stopSolve]);

  return { run, abort, validate };
}
