'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { BandResultBuilder, type BandResult } from './bandResult';
import type {
  ExampleWorkerInMessage,
  ExampleWorkerOutMessage,
} from './exampleWorker';

export type RunStatus =
  | 'idle'
  | 'initializing'
  | 'running'
  | 'done'
  | 'error'
  | 'aborted';

export interface RunMeta {
  lattice_type: string;
  k_labels: string[];
  k_label_distances: number[];
}

export interface RunRequest {
  configToml: string;
  mode: 'stream' | 'filtered';
  meta: RunMeta;
  totalJobs?: number;
  kIndices?: number[] | null;
  bandIndices?: number[] | null;
}

export interface UseExampleRunnerResult {
  status: RunStatus;
  isInitialized: boolean;
  progress: number;
  currentKIndex: number;
  totalKPoints: number;
  jobIndex: number;
  totalJobs: number;
  /** Snapshot of the current/last job as a band-sorted BandResult (for live plotting). */
  liveResult: BandResult | null;
  /** Finalized per-job results (one entry per sweep job). */
  results: BandResult[];
  /** Raw streaming dicts (k-point dicts in `stream` mode, result dicts in `filtered` mode). */
  rawStream: Record<string, unknown>[];
  stats: Record<string, unknown> | null;
  computeTime: number | null;
  error: string | null;
  run: (req: RunRequest) => void;
  abort: () => void;
  reset: () => void;
}

interface JobAccumulator {
  builder: BandResultBuilder;
  done: boolean;
}

function buildMetaFromParams(
  params: Record<string, unknown>,
  meta: RunMeta,
): ConstructorParameters<typeof BandResultBuilder>[0] {
  const atoms = (params.atoms as Array<Record<string, unknown>>) ?? [];
  const first = atoms[0] ?? {};
  return {
    lattice_type: (params.lattice_type as string) ?? meta.lattice_type,
    polarization: (params.polarization as string) ?? 'TM',
    epsilon_background: (params.eps_bg as number) ?? 0,
    epsilon_atoms: (first.eps_inside as number) ?? 0,
    radius_atom: (first.radius as number) ?? 0,
    resolution: (params.resolution as number) ?? 0,
    k_labels: meta.k_labels,
    k_label_distances: meta.k_label_distances,
  };
}

export function useExampleRunner(): UseExampleRunnerResult {
  const [status, setStatus] = useState<RunStatus>('idle');
  const [isInitialized, setIsInitialized] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentKIndex, setCurrentKIndex] = useState(0);
  const [totalKPoints, setTotalKPoints] = useState(0);
  const [jobIndex, setJobIndex] = useState(0);
  const [totalJobs, setTotalJobs] = useState(1);
  const [liveResult, setLiveResult] = useState<BandResult | null>(null);
  const [results, setResults] = useState<BandResult[]>([]);
  const [rawStream, setRawStream] = useState<Record<string, unknown>[]>([]);
  const [stats, setStats] = useState<Record<string, unknown> | null>(null);
  const [computeTime, setComputeTime] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const workerRef = useRef<Worker | null>(null);
  const metaRef = useRef<RunMeta | null>(null);
  const accumulatorsRef = useRef<Map<number, JobAccumulator>>(new Map());
  const maxJobIndexRef = useRef<number>(0);
  // Total job count for the active run, captured at run() time so the
  // progress denominator stays fixed for the entire run (otherwise it grows
  // as new jobs arrive and the bar jumps backwards).
  const totalJobsRef = useRef<number>(1);
  // Number of fully-completed filtered results (job-level) received so far.
  const completedFilteredRef = useRef<number>(0);

  const createWorker = useCallback(() => {
    const worker = new Worker(new URL('./exampleWorker.ts', import.meta.url), {
      type: 'module',
    });
    workerRef.current = worker;

    const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';
    worker.postMessage({ type: 'init', basePath } as ExampleWorkerInMessage);

    worker.onmessage = (event: MessageEvent<ExampleWorkerOutMessage>) => {
      const msg = event.data;
      switch (msg.type) {
        case 'ready':
          setIsInitialized(true);
          break;

        case 'kpoint': {
          const data = msg.data;
          const ji = (data.job_index as number) ?? 0;
          const kIndex = (data.k_index as number) ?? 0;
          const total = (data.total_k_points as number) ?? 0;
          const prog = (data.progress as number) ?? 0;
          const meta = metaRef.current!;

          setJobIndex(ji);
          setCurrentKIndex(kIndex);
          setTotalKPoints(total);
          maxJobIndexRef.current = Math.max(maxJobIndexRef.current, ji);

          let acc = accumulatorsRef.current.get(ji);
          if (!acc) {
            const jobMeta = buildMetaFromParams(
              (data.params as Record<string, unknown>) ?? {},
              meta,
            );
            acc = { builder: new BandResultBuilder(jobMeta), done: false };
            accumulatorsRef.current.set(ji, acc);
          }
          acc.builder.addKPoint({
            k_point: (data.k_point as [number, number]) ?? [0, 0],
            distance: (data.distance as number) ?? 0,
            bands: (data.bands as number[]) ?? [],
          });

          // Live snapshot of the active job for progressive plotting.
          setLiveResult(acc.builder.finish());

          setRawStream((prev) => [...prev, data]);

          // Overall progress across all jobs, using the known total so the
          // bar advances monotonically and matches the work being done.
          const totalJ = Math.max(totalJobsRef.current, ji + 1);
          const kFraction = total > 0 ? (kIndex + 1) / total : prog;
          setProgress(((ji + kFraction) / totalJ) * 100);

          // Finalize this job when its last k-point arrives.
          if (kIndex + 1 >= total && total > 0 && !acc.done) {
            acc.done = true;
            const finalised = acc.builder.finish();
            setResults((prev) => {
              const next = [...prev];
              next[ji] = finalised;
              return next;
            });
          }
          break;
        }

        case 'result': {
          // Filtered job-level result dict.
          setRawStream((prev) => [...prev, msg.data]);
          completedFilteredRef.current += 1;
          const totalJ = Math.max(totalJobsRef.current, completedFilteredRef.current);
          setProgress((completedFilteredRef.current / totalJ) * 100);
          break;
        }

        case 'done': {
          setStats(msg.stats);
          setComputeTime(msg.totalTime);
          setProgress(100);
          setStatus('done');
          break;
        }

        case 'error':
          setError(msg.message);
          setStatus('error');
          break;
      }
    };

    worker.onerror = (e) => {
      setError(`Worker error: ${e.message}`);
      setStatus('error');
    };

    return worker;
  }, []);

  useEffect(() => {
    createWorker();
    return () => {
      workerRef.current?.terminate();
      workerRef.current = null;
    };
  }, [createWorker]);

  const resetState = useCallback(() => {
    accumulatorsRef.current = new Map();
    maxJobIndexRef.current = 0;
    completedFilteredRef.current = 0;
    setProgress(0);
    setCurrentKIndex(0);
    setTotalKPoints(0);
    setJobIndex(0);
    setLiveResult(null);
    setResults([]);
    setRawStream([]);
    setStats(null);
    setComputeTime(null);
    setError(null);
  }, []);

  const run = useCallback(
    (req: RunRequest) => {
      if (!workerRef.current) return;
      resetState();
      metaRef.current = req.meta;
      totalJobsRef.current = Math.max(req.totalJobs ?? 1, 1);
      setTotalJobs(req.totalJobs ?? 1);
      setStatus(isInitialized ? 'running' : 'initializing');
      workerRef.current.postMessage({
        type: 'run',
        configToml: req.configToml,
        mode: req.mode,
        kIndices: req.kIndices ?? null,
        bandIndices: req.bandIndices ?? null,
      } as ExampleWorkerInMessage);
    },
    [isInitialized, resetState],
  );

  const abort = useCallback(() => {
    // A single-threaded WASM call cannot be interrupted; terminate the worker
    // and spin up a fresh one. Partial results accumulated so far are kept.
    workerRef.current?.terminate();
    setIsInitialized(false);
    setStatus('aborted');
    createWorker();
  }, [createWorker]);

  const reset = useCallback(() => {
    resetState();
    setStatus('idle');
  }, [resetState]);

  return {
    status,
    isInitialized,
    progress,
    currentKIndex,
    totalKPoints,
    jobIndex,
    totalJobs,
    liveResult,
    results,
    rawStream,
    stats,
    computeTime,
    error,
    run,
    abort,
    reset,
  };
}
