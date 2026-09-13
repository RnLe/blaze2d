import { readFile, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const [directory, input, output] = process.argv.slice(2);
const wasm = await import(pathToFileURL(path.resolve(directory, 'blaze2d_backend_wasm.js')));
await wasm.default({ module_or_path: await readFile(path.join(directory, 'blaze2d_backend_wasm_bg.wasm')) });
const source = await readFile(input, 'utf8'), report = wasm.validateToml(source);
if (!report.ok) throw new Error(JSON.stringify(report.errors));
const calculation = new wasm.Calculation(source);
try {
  const results = Array.from({ length: calculation.summary.jobs }, (_, index) => calculation.runJob(index, () => {}));
  await writeFile(output, JSON.stringify({ schema: 'blaze2d/run/1', config: report.config, results },
    (_, value) => value instanceof Float64Array ? Array.from(value) : value));
} finally { calculation.free(); }
