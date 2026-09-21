import type { Config, Diagnostic, PlanSummary, ValidationReport } from '@/lib/contract/generated';
import type { RunEvent } from '@/lib/contract/records';

export const EXECUTION_BUDGET = 512 * 1024 * 1024;
export type BuildInfo = { version: string; source_revision: string; config_schema: string; result_schema: string };
export type Preview = { epsilon: Float64Array; resolution: [number, number]; lattice_vectors: [[number, number], [number, number]] };
export type ValidationRequest = { requestId: string; basePath: string } & (
  | { action: 'validate'; source: string }
  | { action: 'defaults' }
  | { action: 'edit'; source: string; config: Config }
  | { action: 'normalize'; source: string }
  | { action: 'task'; source: string; task: 'bands' | 'operators' }
  | { action: 'preview'; source: string }
);
export type ValidationResponse = { requestId: string } & (
  | { kind: 'validated'; source: string; report: ValidationReport; build: BuildInfo }
  | { kind: 'preview'; preview: Preview }
  | { kind: 'error'; diagnostic: Diagnostic }
);
export type SolveRequest =
  | { action: 'start'; runId: string; source: string; basePath: string; errorPolicy: 'stop' | 'continue'; budget: number; build: BuildInfo }
  | { action: 'ack'; runId: string; jobIndex: number; retainedBytes: number };
export type SolveResponse = { runId: string } & (
  | { kind: 'event'; event: RunEvent }
  | { kind: 'ready'; summary: PlanSummary; build: BuildInfo }
  | { kind: 'fatal'; diagnostic: Diagnostic }
);

export function diagnostic(error: unknown): Diagnostic {
  if (error && typeof error === 'object' && 'diagnostic' in error) return diagnostic(error.diagnostic);
  if (error && typeof error === 'object' && 'message' in error) {
    const value = error as Partial<Diagnostic>;
    return { code: typeof value.code === 'string' ? value.code : 'execution', path: value.path ?? '', message: String(value.message), span: value.span };
  }
  return { code: 'execution', path: '', message: String(error) };
}
