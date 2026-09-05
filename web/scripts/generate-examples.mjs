import { readFile, writeFile, mkdir } from 'node:fs/promises';

const directory = new URL('../../examples/calculations/', import.meta.url);
const catalog = JSON.parse(await readFile(new URL('catalog.json', directory), 'utf8'));
const examples = await Promise.all(catalog.map(async entry => ({ ...entry,
  source: await readFile(new URL(`${entry.slug}.toml`, directory), 'utf8'),
  python: await readFile(new URL(`${entry.slug}.py`, directory), 'utf8'),
})));
await mkdir(new URL('../lib/examples/', import.meta.url), { recursive: true });
await writeFile(new URL('../lib/examples/catalog.generated.ts', import.meta.url),
  '/* Generated from examples/calculations. Run pnpm generate:examples to update. */\nexport const examples = ' + JSON.stringify(examples, null, 2) + ' as const;\n');
