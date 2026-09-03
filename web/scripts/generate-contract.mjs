import { execFileSync } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { compile } from 'json-schema-to-typescript';

const schema = JSON.parse(execFileSync('cargo', ['run', '--quiet', '-p', 'blaze2d-interface', '--example', 'contract'], {
  cwd: new URL('../..', import.meta.url), encoding: 'utf8', maxBuffer: 8 * 1024 * 1024,
}));
const directory = new URL('../lib/contract/', import.meta.url);
await mkdir(directory, { recursive: true });
await writeFile(new URL('schema.json', directory), JSON.stringify(schema, null, 2) + '\n');
await writeFile(new URL('generated.ts', directory), await compile(schema, 'BrowserContract', {
  bannerComment: '/* Generated from blaze2d-interface. Run pnpm generate:contract to update. */',
  unknownAny: true, additionalProperties: false, unreachableDefinitions: false,
}));
