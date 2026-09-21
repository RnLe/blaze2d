import type { BandPoint, Config, Diagnostic, PlannedJob, ValidationReport } from '@/lib/contract/generated';
import { resultBytes, type Failure, type Result, type RunEvent } from '@/lib/contract/records';
import { diagnostic, EXECUTION_BUDGET, type BuildInfo, type SolveRequest, type SolveResponse, type ValidationRequest, type ValidationResponse } from './protocol';
import { holdRun, saveHeader, saveRecord, type RunHeader } from './storage';

export type Applied = { source: string; report: ValidationReport; build: BuildInfo };
export interface BrowserRun { header: RunHeader; results: Result[]; errors: Failure[]; failure?: Diagnostic }
export interface ExecutionState {
  run: BrowserRun | null;
  job?: PlannedJob;
  progress?: Extract<RunEvent, { event: 'progress' }>;
  live: (BandPoint & { sampleIndex: number })[];
  storageError?: string;
}

export class Validator {
  private worker: Worker;
  private requests = new Map<string, { resolve: (reply: ValidationResponse) => void; reject: (error: unknown) => void }>();
  constructor(private basePath: string) {
    this.worker = new Worker(new URL('./validation.worker.ts', import.meta.url));
    this.worker.onmessage = ({ data }: MessageEvent<ValidationResponse>) => {
      const pending = this.requests.get(data.requestId);
      this.requests.delete(data.requestId);
      if (data.kind === 'error') pending?.reject(data.diagnostic); else pending?.resolve(data);
    };
    this.worker.onerror = event => this.reject(new Error(event.message || 'The validation worker could not load.'));
  }
  private reject(error: unknown) { for (const request of this.requests.values()) request.reject(error); this.requests.clear(); }
  request(input: Omit<Extract<ValidationRequest, { action: 'validate' }>, 'requestId' | 'basePath'>
    | Omit<Extract<ValidationRequest, { action: 'defaults' }>, 'requestId' | 'basePath'>
    | Omit<Extract<ValidationRequest, { action: 'edit' }>, 'requestId' | 'basePath'>
    | Omit<Extract<ValidationRequest, { action: 'task' }>, 'requestId' | 'basePath'>
    | Omit<Extract<ValidationRequest, { action: 'normalize' | 'preview' }>, 'requestId' | 'basePath'>): Promise<ValidationResponse> {
    const requestId = crypto.randomUUID();
    return new Promise((resolve, reject) => {
      this.requests.set(requestId, { resolve, reject });
      this.worker.postMessage({ ...input, requestId, basePath: this.basePath } satisfies ValidationRequest);
    });
  }
  async validate(source?: string): Promise<Applied> {
    const reply = await this.request(source === undefined ? { action: 'defaults' } : { action: 'validate', source });
    if (reply.kind !== 'validated') throw new Error('Unexpected validation response.');
    return reply;
  }
  async edit(applied: Applied, config: Config): Promise<Applied> {
    const reply = await this.request({ action: 'edit', source: applied.source, config });
    if (reply.kind !== 'validated') throw new Error('Unexpected edit response.');
    return reply;
  }
  dispose() { this.worker.terminate(); this.reject(new Error('Validation was cancelled.')); }
}

export class ExecutionController {
  state: ExecutionState = { run: null, live: [] };
  private worker?: Worker;
  private releaseLock?: () => void;
  private listeners = new Set<() => void>();
  private queue: Promise<void> = Promise.resolve();
  private cache = true;
  private frame?: number;
  constructor(private basePath: string) {}
  subscribe = (listener: () => void) => { this.listeners.add(listener); return () => { this.listeners.delete(listener); }; };
  snapshot = () => this.state;
  private publish(state: ExecutionState, defer = false) {
    this.state = state;
    const notify = () => { this.frame = undefined; this.listeners.forEach(listener => listener()); };
    if (defer) { if (this.frame === undefined) this.frame = requestAnimationFrame(notify); }
    else { if (this.frame !== undefined) cancelAnimationFrame(this.frame); notify(); }
  }
  start(applied: Applied, title = 'Calculation', errorPolicy: 'stop' | 'continue' = 'stop') {
    if (!applied.report.ok || !applied.report.config || !applied.report.summary) throw new Error('Apply a valid configuration before running.');
    if (applied.report.summary.estimated_peak_bytes > EXECUTION_BUDGET) throw new Error('This study exceeds the 512 MiB browser memory budget.');
    this.cancel();
    const id = crypto.randomUUID();
    const header: RunHeader = { id, schema: 'blaze2d/browser-run/1', title, source: applied.source,
      config: structuredClone(applied.report.config), build: { ...applied.build }, startedAt: new Date().toISOString(),
      status: 'running', bytes: 0, completed: 0, failed: 0,
      runtime: { threads: 1, error_policy: errorPolicy, memory_budget: EXECUTION_BUDGET } };
    this.cache = true;
    this.releaseLock = holdRun(id);
    this.publish({ run: { header, results: [], errors: [] }, live: [] });
    this.queue = saveHeader(header).catch(error => this.storageFailure(error, id));
    const worker = new Worker(new URL('./solve.worker.ts', import.meta.url));
    this.worker = worker;
    worker.onmessage = ({ data }: MessageEvent<SolveResponse>) => {
      if (data.runId !== this.state.run?.header.id || worker !== this.worker) return;
      this.queue = this.queue.then(() => this.handle(data, worker)).catch(error => this.fail(error, id));
    };
    worker.onerror = event => { if (worker === this.worker) this.fail(new Error(event.message || 'The solve worker stopped unexpectedly.'), id); };
    worker.postMessage({ action: 'start', runId: id, source: header.source, basePath: this.basePath,
      budget: EXECUTION_BUDGET, errorPolicy, build: applied.build } satisfies SolveRequest);
  }
  private storageFailure(error: unknown, id: string) {
    if (this.state.run?.header.id !== id) return;
    this.cache = false;
    this.publish({ ...this.state, storageError: `History unavailable: ${diagnostic(error).message} Current results remain available for export.` });
  }
  private async handle(message: SolveResponse, worker: Worker) {
    const run = this.state.run;
    if (!run || message.runId !== run.header.id || worker !== this.worker) return;
    if (message.kind === 'ready') return;
    if (message.kind === 'fatal') { this.publish({ ...this.state, run: { ...run, failure: message.diagnostic } }); return; }
    const event = message.event;
    if (event.event === 'job_start') this.publish({ ...this.state, job: event.job, progress: undefined, live: [] });
    if (event.event === 'progress') {
      const stride = Math.max(1, Math.ceil(event.total / 4096));
      const keep = event.sample_index % stride === 0 || event.completed === event.total
        || this.state.job?.resolved.k_label_indices.includes(event.sample_index);
      this.publish({ ...this.state, progress: event, live: event.band_point && keep
        ? [...this.state.live, { ...event.band_point, sampleIndex: event.sample_index }] : this.state.live }, true);
    }
    if (event.event === 'result' || event.event === 'job_failure') {
      const result = event.event === 'result' ? event.result : undefined;
      const error = event.event === 'job_failure' ? event.error : undefined;
      const data = result ?? error?.partial_result;
      const next = { ...run, header: { ...run.header, bytes: run.header.bytes + (data ? resultBytes(data) : 0),
        completed: run.header.completed + Number(!!result), failed: run.header.failed + Number(!!error) },
        results: result ? [...run.results, result] : run.results, errors: error ? [...run.errors, error] : run.errors };
      this.publish({ ...this.state, run: next, live: [] });
      if (this.cache) await saveRecord(next.header, result, error).catch(error => this.storageFailure(error, run.header.id));
      if (worker === this.worker) worker.postMessage({ action: 'ack', runId: run.header.id,
        jobIndex: result?.job_index ?? error!.job_index, retainedBytes: next.header.bytes } satisfies SolveRequest);
    }
    if (event.event === 'terminal') this.finish(event.status);
  }
  private fail(error: unknown, id: string) {
    if (this.state.run?.header.id !== id) return;
    this.publish({ ...this.state, run: { ...this.state.run, failure: diagnostic(error) } });
    this.finish('failed');
  }
  private finish(status: RunHeader['status']) {
    this.worker?.terminate(); this.worker = undefined;
    const run = this.state.run;
    if (!run || run.header.status !== 'running') return;
    const header = { ...run.header, status, endedAt: new Date().toISOString(), cacheComplete: this.cache };
    this.publish({ ...this.state, run: { ...run, header }, live: [] });
    const release = this.releaseLock; this.releaseLock = undefined;
    void this.queue.then(() => saveHeader(header)).catch(error => this.storageFailure(error, header.id)).finally(() => release?.());
  }
  cancel() { if (this.state.run?.header.status === 'running') this.finish('cancelled'); }
  select(run: BrowserRun) {
    if (this.state.run?.header.status === 'running') throw new Error('Cancel the active run before opening history.');
    this.publish({ run, live: [] });
  }
  dispose() { this.cancel(); this.listeners.clear(); if (this.frame !== undefined) cancelAnimationFrame(this.frame); }
}
