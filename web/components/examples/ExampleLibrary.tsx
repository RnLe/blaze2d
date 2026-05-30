'use client';

import {
  createContext,
  isValidElement,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactElement,
  type ReactNode,
} from 'react';
import { Play, Search, Square } from 'lucide-react';
import ExampleBandPlot from './ExampleBandPlot';
import VariableInspector from './VariableInspector';
import styles from './Examples.module.css';
import type {
  DryRunResult,
  DriverStats,
  ExampleRunMode,
  ExampleWorkerInMessage,
  ExampleWorkerOutMessage,
  KPointResult,
  MaxwellResult,
  VariableEntry,
} from './types';

interface ExampleLibraryContextValue {
  getRun: (id: string) => CardRunState;
  runExample: (request: RunRequest) => void;
  abortExample: (id: string) => void;
  activeId: string | null;
}

interface RunRequest {
  id: string;
  toml: string;
  mode: ExampleRunMode;
  kIndices?: number[];
  bandIndices?: number[];
}

interface CardRunState {
  status: 'idle' | 'loading' | 'running' | 'done' | 'error' | 'aborted';
  progress: number;
  message: string;
  dryRun?: DryRunResult;
  kPoints: KPointResult[];
  results: MaxwellResult[];
  stats?: DriverStats;
  error?: string;
}

export interface ExampleCardProps extends RunRequest {
  category: string;
  title: string;
  description: string;
  tags?: string[];
  runtimeNote?: string;
  children: ReactNode;
}

const emptyRun: CardRunState = {
  status: 'idle',
  progress: 0,
  message: 'Ready to run in the browser.',
  kPoints: [],
  results: [],
};

const ExampleLibraryContext = createContext<ExampleLibraryContextValue | null>(null);

function useExampleLibrary() {
  const context = useContext(ExampleLibraryContext);
  if (!context) {
    throw new Error('ExampleCard must be used inside ExampleLibrary');
  }
  return context;
}

function cardText(props: ExampleCardProps) {
  return `${props.category} ${props.title} ${props.description} ${(props.tags ?? []).join(' ')}`.toLowerCase();
}

function statusText(run: CardRunState) {
  if (run.status === 'done' && run.stats) {
    return `Completed ${run.stats.completed}/${run.stats.total_jobs} jobs in ${run.stats.total_time_secs.toFixed(2)}s.`;
  }
  if (run.status === 'error') return run.error ?? 'The browser run failed.';
  return run.message;
}

function variablesFor(run: CardRunState): VariableEntry[] {
  const variables: VariableEntry[] = [];
  if (run.kPoints.length > 0) {
    variables.push({ name: 'stream', type: 'KPointResult[]', value: run.kPoints });
    variables.push({ name: 'k_point', type: 'KPointResult', value: run.kPoints[run.kPoints.length - 1] });
  }

  if (run.results.length === 1) {
    variables.push({ name: 'result', type: 'MaxwellResult', value: run.results[0] });
  } else if (run.results.length > 1) {
    variables.push({ name: 'results', type: 'MaxwellResult[]', value: run.results });
  }

  if (run.stats) {
    variables.push({ name: 'stats', type: 'DriverStats', value: run.stats });
  }

  if (run.dryRun) {
    variables.push({ name: 'browser_runtime', type: 'DryRunResult', value: run.dryRun });
  }

  return variables;
}

export function ExampleCard({
  id,
  title,
  description,
  category,
  tags = [],
  runtimeNote,
  children,
  toml,
  mode,
  kIndices,
  bandIndices,
}: ExampleCardProps) {
  const { getRun, runExample, abortExample, activeId } = useExampleLibrary();
  const run = getRun(id);
  const running = run.status === 'loading' || run.status === 'running';
  const blockedByOther = Boolean(activeId && activeId !== id);
  const variables = variablesFor(run);

  return (
    <article className={styles.card}>
      <header className={styles.cardHeader}>
        <div className={styles.category}>{category}</div>
        <h2 className={styles.title}>{title}</h2>
        <p className={styles.description}>{description}</p>
        <div className={styles.tagRow}>
          {tags.map((tag) => (
            <span className={styles.tag} key={tag}>
              {tag}
            </span>
          ))}
        </div>
        {runtimeNote && <div className={styles.note}>{runtimeNote}</div>}
      </header>

      <div className={styles.codeFrame}>{children}</div>

      <div className={styles.runPanel}>
        <div className={styles.actions}>
          <button
            className={styles.button}
            onClick={() => runExample({ id, toml, mode, kIndices, bandIndices })}
            disabled={running || blockedByOther}
          >
            <Play size={16} />
            {running ? 'Running' : 'Run in browser'}
          </button>
          <button
            className={`${styles.button} ${styles.buttonSecondary}`}
            onClick={() => abortExample(id)}
            disabled={!running}
          >
            <Square size={15} />
            Abort
          </button>
          <span className={styles.status}>{statusText(run)}</span>
        </div>
        <div className={styles.progressTrack}>
          <div className={styles.progressFill} style={{ width: `${Math.round(run.progress * 100)}%` }} />
        </div>
        <div className={styles.resultArea}>
          <ExampleBandPlot results={run.results} kPoints={run.kPoints} />
          <VariableInspector variables={variables} />
        </div>
      </div>
    </article>
  );
}

export function ExampleLibrary({ children }: { children: ReactNode }) {
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('All');
  const [runs, setRuns] = useState<Record<string, CardRunState>>({});
  const [activeId, setActiveId] = useState<string | null>(null);
  const workerRef = useRef<Worker | null>(null);

  const cards = useMemo(
    () =>
      (Array.isArray(children) ? children : [children]).filter(isValidElement) as ReactElement<ExampleCardProps>[],
    [children]
  );

  const categories = useMemo(() => {
    const names = Array.from(new Set(cards.map((card) => card.props.category)));
    return ['All', ...names];
  }, [cards]);

  const visibleCards = useMemo(() => {
    const q = query.trim().toLowerCase();
    return cards.filter((card) => {
      const inCategory = category === 'All' || card.props.category === category;
      const matches = !q || cardText(card.props).includes(q);
      return inCategory && matches;
    });
  }, [cards, category, query]);

  const patchRun = useCallback((id: string, patch: Partial<CardRunState>) => {
    setRuns((current) => ({
      ...current,
      [id]: {
        ...(current[id] ?? emptyRun),
        ...patch,
      },
    }));
  }, []);

  const terminateWorker = useCallback(() => {
    workerRef.current?.terminate();
    workerRef.current = null;
  }, []);

  const abortExample = useCallback(
    (id: string) => {
      if (activeId !== id) return;
      terminateWorker();
      patchRun(id, {
        status: 'aborted',
        progress: 0,
        message: 'Browser run aborted.',
      });
      setActiveId(null);
    },
    [activeId, patchRun, terminateWorker]
  );

  const runExample = useCallback(
    (request: RunRequest) => {
      if (activeId && activeId !== request.id) return;

      terminateWorker();
      setActiveId(request.id);
      patchRun(request.id, {
        status: 'loading',
        progress: 0,
        message: 'Loading WASM runtime...',
        dryRun: undefined,
        kPoints: [],
        results: [],
        stats: undefined,
        error: undefined,
      });

      const worker = new Worker(new URL('./examplesWorker.ts', import.meta.url), { type: 'module' });
      workerRef.current = worker;

      worker.onmessage = (event: MessageEvent<ExampleWorkerOutMessage>) => {
        const message = event.data;
        if (message.type === 'ready') {
          patchRun(request.id, {
            status: 'running',
            message: 'WASM ready. Starting calculation...',
          });
          worker.postMessage({ type: 'start', ...request } satisfies ExampleWorkerInMessage);
          return;
        }

        if (message.type === 'metadata') {
          patchRun(message.id, {
            dryRun: message.dryRun,
            status: 'running',
            message: `Running ${message.jobCount} job${message.jobCount === 1 ? '' : 's'} on a browser worker.`,
          });
          return;
        }

        if (message.type === 'kpoint') {
          setRuns((current) => {
            const previous = current[message.id] ?? emptyRun;
            return {
              ...current,
              [message.id]: {
                ...previous,
                status: 'running',
                progress: message.progress,
                message: `Computed k-point ${message.result.k_index + 1}/${message.result.total_k_points}.`,
                kPoints: [...previous.kPoints, message.result],
              },
            };
          });
          return;
        }

        if (message.type === 'result') {
          setRuns((current) => {
            const previous = current[message.id] ?? emptyRun;
            return {
              ...current,
              [message.id]: {
                ...previous,
                status: 'running',
                progress: message.progress,
                message: `Completed job ${message.result.job_index + 1}.`,
                results: [...previous.results, message.result],
              },
            };
          });
          return;
        }

        if (message.type === 'done') {
          patchRun(message.id, {
            status: 'done',
            progress: 1,
            message: 'Browser run complete.',
            results: message.results,
            stats: message.stats,
          });
          setActiveId(null);
          terminateWorker();
          return;
        }

        if (message.type === 'error') {
          const id = message.id ?? request.id;
          patchRun(id, {
            status: 'error',
            progress: 0,
            error: message.message,
            message: message.message,
          });
          setActiveId(null);
          terminateWorker();
        }
      };

      worker.onerror = (error) => {
        patchRun(request.id, {
          status: 'error',
          progress: 0,
          error: error.message,
          message: error.message,
        });
        setActiveId(null);
        terminateWorker();
      };

      const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';
      worker.postMessage({ type: 'init', basePath } satisfies ExampleWorkerInMessage);
    },
    [activeId, patchRun, terminateWorker]
  );

  useEffect(() => () => terminateWorker(), [terminateWorker]);

  const value = useMemo<ExampleLibraryContextValue>(
    () => ({
      getRun: (id) => runs[id] ?? emptyRun,
      runExample,
      abortExample,
      activeId,
    }),
    [abortExample, activeId, runExample, runs]
  );

  return (
    <ExampleLibraryContext.Provider value={value}>
      <div className={styles.library}>
        <div className={styles.controls}>
          <div className={styles.searchWrap}>
            <Search size={17} className={styles.searchIcon} />
            <input
              className={styles.search}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search examples"
            />
          </div>
          <div className={styles.tabs} role="tablist" aria-label="Example categories">
            {categories.map((name) => (
              <button
                key={name}
                className={`${styles.tab} ${category === name ? styles.tabActive : ''}`}
                onClick={() => setCategory(name)}
                type="button"
              >
                {name}
              </button>
            ))}
          </div>
        </div>
        <div className={styles.grid}>
          {visibleCards.length > 0 ? visibleCards : <div className={styles.empty}>No examples match the current search.</div>}
        </div>
      </div>
    </ExampleLibraryContext.Provider>
  );
}
