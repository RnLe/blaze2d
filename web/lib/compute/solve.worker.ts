/// <reference lib="webworker" />
import { loadWasm } from '../wasm/loadWasm';
import type { PlanSummary } from '../contract/generated';
import { resultBuffers, type Failure, type Result, type RunEvent } from '../contract/records';
import { diagnostic, type BuildInfo, type SolveRequest, type SolveResponse } from './protocol';

const scope = self as unknown as DedicatedWorkerGlobalScope;
let activeId: string | undefined;
let acknowledge: { jobIndex: number; resolve: (bytes: number) => void } | undefined;
scope.onmessage = ({ data }: MessageEvent<SolveRequest>) => {
  if (data.action === 'ack') {
    if (data.runId === activeId && data.jobIndex === acknowledge?.jobIndex) {
      acknowledge.resolve(data.retainedBytes);
      acknowledge = undefined;
    }
  } else if (!activeId) {
    activeId = data.runId;
    void run(data);
  }
};

async function run(request: Extract<SolveRequest, { action: 'start' }>) {
  const reply = (message: SolveResponse, transfer: Transferable[] = []) => scope.postMessage(message, transfer);
  const event = (event: RunEvent, transfer: Transferable[] = []) => reply({ runId: request.runId, kind: 'event', event }, transfer);
  let calculation: InstanceType<Awaited<ReturnType<typeof loadWasm>>['Calculation']> | undefined;
  let completed = 0, failed = 0, retained = 0;
  try {
    const wasm = await loadWasm(request.basePath);
    const build = wasm.buildInfo() as BuildInfo;
    if (build.source_revision !== request.build.source_revision || build.version !== request.build.version) {
      throw new Error('The solver changed since validation. Reload this page before running.');
    }
    calculation = new wasm.Calculation(request.source);
    const summary = calculation.summary as PlanSummary;
    if (summary.estimated_peak_bytes > request.budget) throw new Error('This study exceeds the 512 MiB browser memory budget. Reduce the grid, band window, or retained fields.');
    reply({ runId: request.runId, kind: 'ready', build, summary });
    event({ event: 'run_start', jobs: summary.jobs, solves: summary.solves });
    for (let index = 0; index < summary.jobs; index++) {
      if (retained + summary.estimated_peak_bytes > request.budget) throw new Error('The browser memory budget is full. Completed results remain available. Export them and run a smaller study, or use Python streaming.');
      let result: Result | undefined, failure: Failure | undefined;
      try {
        result = calculation.runJob(index, (value: RunEvent) => event(value)) as Result;
        if (result.schema !== build.result_schema) throw new Error('The solver returned an unsupported result schema.');
        resultBuffers(result);
        completed++;
      } catch (error) {
        const partial = error && typeof error === 'object' && 'partial_result' in error ? error.partial_result as Result : undefined;
        failure = { job_index: index, diagnostic: diagnostic(error), partial_result: partial };
        failed++;
      }
      const delivered = new Promise<number>(resolve => { acknowledge = { jobIndex: index, resolve }; });
      if (result) event({ event: 'result', result }, resultBuffers(result));
      if (failure) event({ event: 'job_failure', error: failure }, failure.partial_result ? resultBuffers(failure.partial_result) : []);
      retained = await delivered;
      // The acknowledgement yields to the worker event loop and bounds result delivery to one job.
      if (failure && request.errorPolicy === 'stop') break;
    }
    event({ event: 'terminal', completed, failed, status: failed ? request.errorPolicy === 'stop' ? 'failed' : 'completed_with_errors' : 'completed' });
  } catch (error) {
    reply({ runId: request.runId, kind: 'fatal', diagnostic: diagnostic(error) });
    event({ event: 'terminal', completed, failed: failed + 1, status: 'failed' });
  } finally { calculation?.free(); }
}
