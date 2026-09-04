import type { Config, RunStatus } from '../contract/generated';
import { resultBytes, type Failure, type Result } from '../contract/records';
import type { BuildInfo } from './protocol';

export interface RunHeader {
  id: string;
  schema: 'blaze2d/browser-run/1';
  title: string;
  startedAt: string;
  endedAt?: string;
  status: 'running' | 'interrupted' | RunStatus;
  source: string;
  config: Config;
  build: BuildInfo;
  bytes: number;
  cacheComplete?: boolean;
  completed: number;
  failed: number;
  runtime: { threads: 1; error_policy: 'stop' | 'continue'; memory_budget: number };
}
type StoredRecord = { key: [string, number]; runId: string; result?: Result; error?: Failure };
const DB_NAME = 'blaze2d-workbench-v1';
let connection: Promise<IDBDatabase> | undefined;
const activeRuns = new Set<string>();

function database(): Promise<IDBDatabase> {
  if (!connection) connection = new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore('runs', { keyPath: 'id' });
      request.result.createObjectStore('records', { keyPath: 'key' }).createIndex('runId', 'runId');
    };
    request.onsuccess = () => { request.result.onversionchange = () => request.result.close(); resolve(request.result); };
    request.onerror = () => reject(request.error);
    request.onblocked = () => reject(new Error('Run storage is blocked by another tab.'));
  }).catch(error => { connection = undefined; throw error; });
  return connection;
}
function value<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => { request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error); });
}
function finished(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => { transaction.oncomplete = () => resolve(); transaction.onabort = () => reject(transaction.error ?? new Error('Storage transaction aborted.')); transaction.onerror = () => reject(transaction.error); });
}
function remove(transaction: IDBTransaction, id: string) {
  transaction.objectStore('runs').delete(id);
  const cursor = transaction.objectStore('records').index('runId').openKeyCursor(IDBKeyRange.only(id));
  cursor.onsuccess = () => { if (cursor.result) { transaction.objectStore('records').delete(cursor.result.primaryKey); cursor.result.continue(); } };
}

export async function cacheLimit(): Promise<number> {
  const quota = await navigator.storage?.estimate?.();
  return Math.min(256 * 1024 * 1024, (quota?.quota ?? 2560 * 1024 * 1024) * 0.1);
}

function charge(header: RunHeader): number {
  return header.bytes + new TextEncoder().encode(JSON.stringify(header)).byteLength;
}
async function reserve(transaction: IDBTransaction, header: RunHeader, limit: number) {
  const all = await value<RunHeader[]>(transaction.objectStore('runs').getAll());
  let total = charge(header) + all.filter(run => run.id !== header.id).reduce((sum, run) => sum + charge(run), 0);
  for (const run of all.sort((a, b) => a.startedAt.localeCompare(b.startedAt))) {
    if (total <= limit) break;
    if (run.id === header.id || run.status === 'running') continue;
    remove(transaction, run.id); total -= charge(run);
  }
  if (total > limit) throw new Error('This run exceeds the history cache limit. Export current results before leaving.');
}
async function store(header: RunHeader, record?: StoredRecord) {
  const db = await database(), limit = await cacheLimit();
  const transaction = db.transaction(['runs', 'records'], 'readwrite'), done = finished(transaction);
  try {
    await reserve(transaction, header, limit);
    transaction.objectStore('runs').put(header);
    if (record) transaction.objectStore('records').put(record);
  } catch (error) {
    transaction.abort(); await done.catch(() => {}); throw error;
  }
  await done;
}
export async function saveHeader(header: RunHeader): Promise<void> { await store(header); }
export async function saveRecord(header: RunHeader, result?: Result, error?: Failure): Promise<void> {
  const index = result?.job_index ?? error!.job_index;
  await store(header, { key: [header.id, index], runId: header.id, result, error });
}

export async function listRuns(): Promise<RunHeader[]> {
  const db = await database();
  const rows = await value<RunHeader[]>(db.transaction('runs').objectStore('runs').getAll());
  const locks = await navigator.locks?.query?.();
  for (const row of rows) {
    if (row.status === 'running' && !activeRuns.has(row.id) && !locks?.held?.some(lock => lock.name === `blaze-run-${row.id}`)) {
      row.status = 'interrupted';
      await saveHeader(row);
    }
  }
  return rows.sort((a, b) => b.startedAt.localeCompare(a.startedAt));
}

export async function loadRun(id: string) {
  const db = await database();
  const transaction = db.transaction(['runs', 'records']);
  const [header, rows] = await Promise.all([
    value<RunHeader | undefined>(transaction.objectStore('runs').get(id)),
    value<StoredRecord[]>(transaction.objectStore('records').index('runId').getAll(IDBKeyRange.only(id))),
  ]);
  if (!header || header.schema !== 'blaze2d/browser-run/1') throw new Error('This run uses an unsupported schema. Its stored data has been preserved.');
  const results = rows.flatMap(row => row.result ? [row.result] : []);
  results.forEach(resultBytes);
  rows.forEach(row => { if (row.error?.partial_result) resultBytes(row.error.partial_result); });
  return { header, results, errors: rows.flatMap(row => row.error ? [row.error] : []) };
}

export async function deleteRun(id: string): Promise<void> {
  const db = await database();
  const transaction = db.transaction(['runs', 'records'], 'readwrite');
  remove(transaction, id);
  await finished(transaction);
}

export function holdRun(id: string): () => void {
  activeRuns.add(id);
  let release = () => {};
  const lifetime = new Promise<void>(resolve => { release = resolve; });
  void navigator.locks?.request(`blaze-run-${id}`, () => lifetime).catch(() => {});
  return () => { activeRuns.delete(id); release(); };
}
