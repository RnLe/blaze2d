/// <reference lib="webworker" />
import { loadWasm } from '@/lib/wasm/loadWasm';
import type { ValidationReport } from '@/lib/contract/generated';
import { diagnostic, EXECUTION_BUDGET, type BuildInfo, type Preview, type ValidationRequest, type ValidationResponse } from './protocol';

const scope = self as unknown as DedicatedWorkerGlobalScope;
scope.onmessage = async ({ data: request }: MessageEvent<ValidationRequest>) => {
  const reply = (response: ValidationResponse, transfer: Transferable[] = []) => scope.postMessage(response, transfer);
  try {
    const wasm = await loadWasm(request.basePath);
    let source = request.action === 'defaults' ? wasm.defaultToml() : request.source;
    if (request.action === 'edit') source = wasm.applyConfig(source, JSON.stringify(request.config));
    if (request.action === 'normalize') source = wasm.normalizeToml(source);
    if (request.action === 'task') source = wasm.selectTask(source, request.task);
    if (request.action === 'preview') {
      const calculation = new wasm.Calculation(source);
      try {
        if (calculation.summary.estimated_peak_bytes > EXECUTION_BUDGET) throw new Error('This model exceeds the browser memory budget. Reduce the grid or use Python.');
        const preview = calculation.geometryPreview() as Preview;
        reply({ requestId: request.requestId, kind: 'preview', preview }, [preview.epsilon.buffer]);
      } finally { calculation.free(); }
    } else {
      reply({ requestId: request.requestId, kind: 'validated', source,
        report: wasm.validateToml(source) as ValidationReport, build: wasm.buildInfo() as BuildInfo });
    }
  } catch (error) { reply({ requestId: request.requestId, kind: 'error', diagnostic: diagnostic(error) }); }
};
