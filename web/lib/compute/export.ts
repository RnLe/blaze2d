import { Zip, ZipPassThrough } from 'fflate';
import type { BrowserRun } from './controller';
import type { NumericArray } from '@/lib/contract/records';

const encoder = new TextEncoder();
const yieldToBrowser = () => new Promise<void>(resolve => setTimeout(resolve, 0));

export function download(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
}

function study(run: BrowserRun) {
  return { schema: 'blaze2d/run/1', config: run.header.config, source_toml: run.header.source,
    results: run.results, errors: run.errors,
    statistics: { status: run.header.status, completed: run.header.completed, failed: run.header.failed,
      started_at: run.header.startedAt, ended_at: run.header.endedAt, runtime: run.header.runtime,
      build: run.header.build, diagnostic: run.failure } };
}

async function* jsonChunks(value: unknown): AsyncGenerator<string> {
  if (value instanceof Float64Array) {
    yield '[';
    for (let start = 0; start < value.length; start += 4096) {
      const chunk = value.subarray(start, start + 4096);
      if (chunk.some(number => !Number.isFinite(number))) throw new Error('Non-finite array data cannot be exported as lossless JSON.');
      yield (start ? ',' : '') + Array.from(chunk, number => Object.is(number, -0) ? '-0' : String(number)).join(',');
      await yieldToBrowser();
    }
    yield ']';
  } else if (Array.isArray(value)) {
    yield '[';
    for (let index = 0; index < value.length; index++) { if (index) yield ','; yield* jsonChunks(value[index]); }
    yield ']';
  } else if (value && typeof value === 'object') {
    yield '{'; let first = true;
    for (const [key, item] of Object.entries(value)) {
      if (item === undefined) continue;
      yield (first ? '' : ',') + JSON.stringify(key) + ':'; first = false; yield* jsonChunks(item);
    }
    yield '}';
  } else {
    if (typeof value === 'number' && !Number.isFinite(value)) throw new Error('Non-finite metadata cannot be exported.');
    yield Object.is(value, -0) ? '-0' : JSON.stringify(value) ?? 'null';
  }
}

function npyHeader(dtype: string, shape: number[]): Uint8Array {
  const tuple = shape.length === 1 ? `${shape[0]},` : shape.join(', ');
  const dictionary = `{'descr': '${dtype}', 'fortran_order': False, 'shape': (${tuple}), }`;
  const padding = (64 - (10 + dictionary.length + 1) % 64) % 64;
  const text = encoder.encode(dictionary + ' '.repeat(padding) + '\n');
  const header = new Uint8Array(10 + text.length);
  header.set([147, 78, 85, 77, 80, 89, 1, 0]);
  new DataView(header.buffer).setUint16(8, text.length, true); header.set(text, 10);
  return header;
}

export async function exportRun(run: BrowserRun, format: 'json' | 'ndjson' | 'npz'): Promise<Blob> {
  const data = study(run);
  if (format !== 'npz') {
    const chunks: Blob[] = [];
    for await (const chunk of jsonChunks(data)) chunks.push(new Blob([chunk]));
    return new Blob([...chunks, '\n'], { type: format === 'json' ? 'application/json' : 'application/x-ndjson' });
  }
  const chunks: Blob[] = [], arrays: NumericArray[] = [];
  let zipError: Error | undefined;
  const zip = new Zip((error, chunk) => { if (error) zipError = error; else chunks.push(new Blob([chunk as Uint8Array<ArrayBuffer>])); });
  const manifest = JSON.stringify(data, (_key, value) => {
    if (value?.data instanceof Float64Array) {
      const { data: _unused, ...descriptor } = value as NumericArray;
      void _unused;
      const buffer = `array_${String(arrays.length).padStart(6, '0')}`;
      arrays.push(value); return { ...descriptor, buffer };
    }
    return value;
  });
  const manifestBytes = encoder.encode(manifest), file = new ZipPassThrough('manifest.npy');
  zip.add(file); file.push(npyHeader('|u1', [manifestBytes.length])); file.push(manifestBytes, true);
  for (const [index, array] of arrays.entries()) {
    const file = new ZipPassThrough(`array_${String(index).padStart(6, '0')}.npy`);
    zip.add(file); file.push(npyHeader(array.dtype === 'complex128' ? '<c16' : '<f8', array.shape));
    for (let offset = 0; offset < array.data.length; offset += 262144) {
      const count = Math.min(262144, array.data.length - offset), chunk = new Uint8Array(count * 8), view = new DataView(chunk.buffer);
      for (let i = 0; i < count; i++) {
        const value = array.data[offset + i];
        if (!Number.isFinite(value)) throw new Error('Non-finite array data cannot be exported.');
        view.setFloat64(i * 8, value, true);
      }
      file.push(chunk); await yieldToBrowser();
    }
    file.push(new Uint8Array(), true);
  }
  zip.end(); if (zipError) throw zipError;
  return new Blob(chunks, { type: 'application/octet-stream' });
}
