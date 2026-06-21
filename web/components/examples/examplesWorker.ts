import type {
  DriverStats,
  ExampleRunMode,
  ExampleWorkerInMessage,
  ExampleWorkerOutMessage,
  KPointResult,
  MaxwellResult,
} from './types';

const isWorker = typeof self !== 'undefined' && typeof Window === 'undefined';

type WasmModule = {
  default: (input?: string) => Promise<unknown>;
  initPanicHook?: () => void;
  WasmBulkDriver: new (config: string) => {
    jobCount: number;
    dryRun: () => unknown;
    runWithKPointStreaming: (callback: (result: unknown) => void) => DriverStats;
    runWithCallback: (callback: (result: unknown) => void) => DriverStats;
    runCollect: () => { results: unknown[]; stats: DriverStats };
    runCollectFiltered: (
      kIndices?: Uint32Array | null,
      bandIndices?: Uint32Array | null
    ) => { results: unknown[]; stats: DriverStats };
  };
};

let wasmModule: WasmModule | null = null;
let basePath = '';

function post(message: ExampleWorkerOutMessage) {
  self.postMessage(message);
}

function toPlain<T>(value: unknown): T {
  if (Array.isArray(value)) {
    return value.map((item) => toPlain(item)) as T;
  }

  if (value && typeof value === 'object') {
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(value as Record<string, unknown>)) {
      out[key] = toPlain((value as Record<string, unknown>)[key]);
    }
    return out as T;
  }

  return value as T;
}

async function initWasm() {
  if (wasmModule) return wasmModule;

  const wasmJsUrl = `${basePath}/wasm-blaze/blaze2d_backend_wasm.js`;
  const wasmBinaryUrl = `${basePath}/wasm-blaze/blaze2d_backend_wasm_bg.wasm`;
  const mod = (await import(/* webpackIgnore: true */ wasmJsUrl)) as WasmModule;
  await mod.default(wasmBinaryUrl);
  mod.initPanicHook?.();
  wasmModule = mod;
  return mod;
}

function makeResultFromKPoints(jobIndex: number, events: KPointResult[]): MaxwellResult | null {
  if (events.length === 0) return null;
  const first = events[0];
  return {
    result_type: 'maxwell',
    job_index: jobIndex,
    params: first.params,
    sweep_values: first.sweep_values ?? {},
    sweep_order: first.sweep_order ?? '',
    k_path: events.map((event) => event.k_point),
    distances: events.map((event) => event.distance),
    bands: events.map((event) => event.omegas),
    num_k_points: events.length,
    num_bands: first.num_bands,
  };
}

function normalizeStats(stats: DriverStats): DriverStats {
  return {
    ...stats,
    total_time_secs: stats.total_time_secs ?? stats.total_time_ms / 1000,
  };
}

function runKPoint(id: string, driver: InstanceType<WasmModule['WasmBulkDriver']>) {
  const byJob = new Map<number, KPointResult[]>();
  const stats = driver.runWithKPointStreaming((raw) => {
    const result = toPlain<KPointResult>(raw);
    const events = byJob.get(result.job_index) ?? [];
    events.push(result);
    byJob.set(result.job_index, events);
    post({ type: 'kpoint', id, result, progress: result.progress });
  });

  const results = Array.from(byJob.entries())
    .map(([jobIndex, events]) => makeResultFromKPoints(jobIndex, events))
    .filter((result): result is MaxwellResult => Boolean(result));

  post({ type: 'done', id, results, stats: normalizeStats(stats) });
}

function runJobStream(id: string, driver: InstanceType<WasmModule['WasmBulkDriver']>) {
  const results: MaxwellResult[] = [];
  const total = Math.max(driver.jobCount, 1);
  let completed = 0;

  const stats = driver.runWithCallback((raw) => {
    completed += 1;
    const result = toPlain<MaxwellResult>(raw);
    results.push(result);
    post({ type: 'result', id, result, progress: completed / total });
  });

  post({ type: 'done', id, results, stats: normalizeStats(stats) });
}

function runCollect(
  id: string,
  driver: InstanceType<WasmModule['WasmBulkDriver']>,
  mode: ExampleRunMode,
  kIndices?: number[],
  bandIndices?: number[]
) {
  const output =
    mode === 'collect-filtered'
      ? driver.runCollectFiltered(
          kIndices ? new Uint32Array(kIndices) : null,
          bandIndices ? new Uint32Array(bandIndices) : null
        )
      : driver.runCollect();

  post({
    type: 'done',
    id,
    results: toPlain<MaxwellResult[]>(output.results),
    stats: normalizeStats(output.stats),
  });
}

async function start(message: Extract<ExampleWorkerInMessage, { type: 'start' }>) {
  const mod = await initWasm();
  const driver = new mod.WasmBulkDriver(message.toml);

  post({
    type: 'metadata',
    id: message.id,
    dryRun: toPlain(driver.dryRun()),
    jobCount: driver.jobCount,
  });

  if (message.mode === 'kpoint') {
    runKPoint(message.id, driver);
  } else if (message.mode === 'job-stream') {
    runJobStream(message.id, driver);
  } else {
    runCollect(message.id, driver, message.mode, message.kIndices, message.bandIndices);
  }
}

if (isWorker) {
  self.onmessage = async (event: MessageEvent<ExampleWorkerInMessage>) => {
    try {
      if (event.data.type === 'init') {
        basePath = event.data.basePath || '';
        await initWasm();
        post({ type: 'ready' });
      } else {
        await start(event.data);
      }
    } catch (error) {
      post({
        type: 'error',
        id: event.data.type === 'start' ? event.data.id : undefined,
        message: error instanceof Error ? error.message : String(error),
      });
    }
  };
}
