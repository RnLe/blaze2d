import type { Array as ArrayRecord, Event, JobFailure, ResultRecord, SampleRecord } from './generated';

export type NumericArray = Omit<ArrayRecord, 'data'> & { data: Float64Array };
export type Arrays = Record<string, NumericArray>;
export type Sample = Omit<SampleRecord, 'arrays'> & { arrays: Arrays };
export type Result = Omit<ResultRecord, 'arrays' | 'samples'> & { arrays: Arrays; samples?: Sample[] };
export type Failure = Omit<JobFailure, 'partial_result'> & { partial_result?: Result | null };
export type RunEvent = Exclude<Event, { event: 'result' | 'job_failure' }>
  | { event: 'result'; result: Result }
  | { event: 'job_failure'; error: Failure };

export function object(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' ? value as Record<string, unknown> : {};
}

export function resultBuffers(result: Result): ArrayBuffer[] {
  const buffers = new Set<ArrayBuffer>();
  for (const arrays of [result.arrays, ...(result.samples ?? []).map(sample => sample.arrays)]) {
    for (const [name, array] of Object.entries(arrays)) {
      const count = array.shape.reduce((a, b) => a * b, 1) * (array.dtype === 'complex128' ? 2 : 1);
      if (!['float64','complex128'].includes(array.dtype) || array.shape.some(n => !Number.isSafeInteger(n) || n < 0) || !(array.data instanceof Float64Array) || array.data.length !== count || array.order !== 'C'
        || array.dimensions.length !== array.shape.length || !Number.isSafeInteger(count)) {
        throw new Error(`Invalid array descriptor: ${name}`);
      }
      buffers.add(array.data.buffer as ArrayBuffer);
    }
  }
  return [...buffers];
}

export function resultBytes(result: Result): number {
  return resultBuffers(result).reduce((size, buffer) => size + buffer.byteLength, 0)
    + new TextEncoder().encode(JSON.stringify([result.metadata, result.samples?.map(s => s.metadata)])).byteLength;
}
