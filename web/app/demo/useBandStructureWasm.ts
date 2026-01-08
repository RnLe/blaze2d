'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import type { 
  WorkerInMessage, 
  WorkerOutMessage, 
  WorkerKPointMessage,
  WorkerDoneMessage,
} from './bandWorker';

// =============================================================================
// Types
// =============================================================================

export interface AtomParams {
  index: number;
  pos: [number, number];
  radius: number;
  eps_inside: number;
}

export interface JobParams {
  eps_bg: number;
  resolution: number;
  polarization: 'TE' | 'TM';
  lattice_type?: string;
  atoms: AtomParams[];
}

/** K-point streaming result (emitted after each k-point solve) */
export interface KPointResult {
  stream_type: 'k_point';
  job_index: number;
  k_index: number;
  total_k_points: number;
  k_point: [number, number];
  distance: number;
  omegas: number[];
  bands: number[];
  iterations: number;
  is_gamma: boolean;
  num_bands: number;
  progress: number;
  params: JobParams;
}

export interface DriverStats {
  total_jobs: number;
  completed: number;
  failed: number;
  total_time_ms: number;
  jobs_per_second?: number;
}

// =============================================================================
// Hook Result Interface
// =============================================================================

/** Progressive band data structure (for k-point streaming) */
export interface StreamingBandData {
  distances: number[];
  bands: number[][]; // bands[kIndex][bandIndex] = frequency
  numBands: number;
  numKPoints: number;
  polarization: 'TE' | 'TM';
}

export interface UseBandStructureWasmResult {
  /** TM band data (computed first) - grows progressively */
  tmData: StreamingBandData | null;
  
  /** TE band data (computed second) - grows progressively */
  teData: StreamingBandData | null;
  
  /** Whether computation is in progress */
  isComputing: boolean;
  
  /** Current phase: 'idle', 'tm', 'te', 'done' */
  phase: 'idle' | 'tm' | 'te' | 'done';
  
  /** Progress percentage (0-100) */
  progress: number;
  
  /** TM progress percentage (0-100) */
  tmProgress: number;
  
  /** TE progress percentage (0-100) */
  teProgress: number;
  
  /** Current k-point index */
  currentKIndex: number;
  
  /** Total k-points */
  totalKPoints: number;
  
  /** Error message if computation failed */
  error: string | null;
  
  /** Total computation time in ms */
  computeTime: number | null;
  
  /** TM computation time in ms */
  tmTime: number | null;
  
  /** TE computation time in ms */
  teTime: number | null;
  
  /** Start computation with the given TOML config (will compute both polarizations) */
  compute: (configToml: string) => void;
  
  /** Reset state and clear results */
  reset: () => void;
  
  /** Whether WASM module is initialized */
  isInitialized: boolean;
}

// =============================================================================
// Hook Implementation (Web Worker based)
// =============================================================================

export function useBandStructureWasm(): UseBandStructureWasmResult {
  const [tmData, setTmData] = useState<StreamingBandData | null>(null);
  const [teData, setTeData] = useState<StreamingBandData | null>(null);
  const [isComputing, setIsComputing] = useState(false);
  const [phase, setPhase] = useState<'idle' | 'tm' | 'te' | 'done'>('idle');
  const [progress, setProgress] = useState(0);
  const [tmProgress, setTmProgress] = useState(0);
  const [teProgress, setTeProgress] = useState(0);
  const [currentKIndex, setCurrentKIndex] = useState(0);
  const [totalKPoints, setTotalKPoints] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [computeTime, setComputeTime] = useState<number | null>(null);
  const [tmTime, setTmTime] = useState<number | null>(null);
  const [teTime, setTeTime] = useState<number | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  
  const workerRef = useRef<Worker | null>(null);
  const configRef = useRef<string>('');
  const startTimeRef = useRef<number>(0);
  const tmTimeRef = useRef<number>(0);
  
  // Refs for accumulating streaming data
  const tmDataRef = useRef<StreamingBandData>({
    distances: [],
    bands: [],
    numBands: 0,
    numKPoints: 0,
    polarization: 'TM',
  });
  const teDataRef = useRef<StreamingBandData>({
    distances: [],
    bands: [],
    numBands: 0,
    numKPoints: 0,
    polarization: 'TE',
  });
  
  // Initialize Web Worker on mount
  useEffect(() => {
    // Create worker from the bundled file
    const worker = new Worker(new URL('./bandWorker.ts', import.meta.url), { type: 'module' });
    workerRef.current = worker;
    
    // Send init message with base path for GitHub Pages compatibility
    const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';
    worker.postMessage({ type: 'init', basePath } as WorkerInMessage);
    
    // Handle messages from worker
    worker.onmessage = (event: MessageEvent<WorkerOutMessage>) => {
      const msg = event.data;
      
      switch (msg.type) {
        case 'ready':
          console.log('Worker ready');
          setIsInitialized(true);
          break;
          
        case 'kpoint': {
          const kMsg = msg as WorkerKPointMessage;
          setCurrentKIndex(kMsg.kIndex);
          setTotalKPoints(kMsg.totalKPoints);
          
          if (kMsg.polarization === 'TM') {
            // Accumulate TM data
            tmDataRef.current.distances.push(kMsg.distance);
            tmDataRef.current.bands.push([...kMsg.omegas]);
            tmDataRef.current.numBands = kMsg.numBands;
            tmDataRef.current.numKPoints = kMsg.kIndex + 1;
            setProgress(kMsg.progress * 50);
            setTmProgress(kMsg.progress * 100);
            // Update state with new copy to trigger re-render
            setTmData({ ...tmDataRef.current });
          } else {
            // Accumulate TE data
            teDataRef.current.distances.push(kMsg.distance);
            teDataRef.current.bands.push([...kMsg.omegas]);
            teDataRef.current.numBands = kMsg.numBands;
            teDataRef.current.numKPoints = kMsg.kIndex + 1;
            setProgress(50 + kMsg.progress * 50);
            setTeProgress(kMsg.progress * 100);
            // Update state with new copy to trigger re-render
            setTeData({ ...teDataRef.current });
          }
          break;
        }
          
        case 'done': {
          const dMsg = msg as WorkerDoneMessage;
          console.log(`${dMsg.polarization} completed in ${dMsg.totalTime.toFixed(0)}ms`);
          
          if (dMsg.polarization === 'TM') {
            tmTimeRef.current = dMsg.totalTime;
            setTmTime(dMsg.totalTime);
            setTmProgress(100);
            // Start TE computation
            setPhase('te');
            setCurrentKIndex(0);
            
            // Reset TE data ref
            teDataRef.current = {
              distances: [],
              bands: [],
              numBands: 0,
              numKPoints: 0,
              polarization: 'TE',
            };
            
            // Send TE compute request
            const teConfig = configRef.current.replace(
              /polarization\s*=\s*"(TE|TM)"/i, 
              'polarization = "TE"'
            );
            worker.postMessage({ type: 'compute', configToml: teConfig, polarization: 'TE' } as WorkerInMessage);
          } else {
            // TE done - all complete
            const endTime = performance.now();
            setComputeTime(endTime - startTimeRef.current);
            setTeTime(dMsg.totalTime);
            setTeProgress(100);
            setProgress(100);
            setPhase('done');
            setIsComputing(false);
          }
          break;
        }
          
        case 'error':
          console.error('Worker error:', msg.message);
          setError(msg.message);
          setPhase('idle');
          setIsComputing(false);
          break;
      }
    };
    
    worker.onerror = (e) => {
      console.error('Worker error event:', e);
      setError(`Worker error: ${e.message}`);
      setIsComputing(false);
    };
    
    // Cleanup
    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, []);
  
  const compute = useCallback((configToml: string) => {
    if (!workerRef.current || !isInitialized) {
      setError('Worker not ready');
      return;
    }
    
    // Reset state
    setIsComputing(true);
    setError(null);
    setProgress(0);
    setTmProgress(0);
    setTeProgress(0);
    setCurrentKIndex(0);
    setTotalKPoints(0);
    setTmData(null);
    setTeData(null);
    setPhase('tm');
    setComputeTime(null);
    setTmTime(null);
    setTeTime(null);
    
    // Reset data refs
    tmDataRef.current = {
      distances: [],
      bands: [],
      numBands: 0,
      numKPoints: 0,
      polarization: 'TM',
    };
    teDataRef.current = {
      distances: [],
      bands: [],
      numBands: 0,
      numKPoints: 0,
      polarization: 'TE',
    };
    
    // Store config for TE phase
    configRef.current = configToml;
    startTimeRef.current = performance.now();
    
    // Start TM computation
    const tmConfig = configToml.replace(/polarization\s*=\s*"(TE|TM)"/i, 'polarization = "TM"');
    console.log('Starting TM computation via worker');
    workerRef.current.postMessage({ type: 'compute', configToml: tmConfig, polarization: 'TM' } as WorkerInMessage);
  }, [isInitialized]);
  
  const reset = useCallback(() => {
    setTmData(null);
    setTeData(null);
    setIsComputing(false);
    setPhase('idle');
    setProgress(0);
    setTmProgress(0);
    setTeProgress(0);
    setCurrentKIndex(0);
    setTotalKPoints(0);
    setError(null);
    setComputeTime(null);
    setTmTime(null);
    setTeTime(null);
  }, []);
  
  return {
    tmData,
    teData,
    isComputing,
    phase,
    progress,
    tmProgress,
    teProgress,
    currentKIndex,
    totalKPoints,
    error,
    computeTime,
    tmTime,
    teTime,
    compute,
    reset,
    isInitialized,
  };
}
