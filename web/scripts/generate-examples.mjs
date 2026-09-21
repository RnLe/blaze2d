import { readFile, writeFile, mkdir } from 'node:fs/promises';

const directory = new URL('../../examples/calculations/', import.meta.url);
const catalog = JSON.parse(await readFile(new URL('catalog.json', directory), 'utf8'));
const wasm = await import(new URL('../public/wasm-blaze/blaze2d_backend_wasm.js', import.meta.url));
await wasm.default({ module_or_path: await readFile(new URL('../public/wasm-blaze/blaze2d_backend_wasm_bg.wasm', import.meta.url)) });
const images = new URL('../public/examples/', import.meta.url);
await mkdir(images, { recursive: true });
// A blue ramp: distinct enough to tell cards apart, close enough to stay one family.
// Mirrors the accent in app/styles/tokens.css.
const palette = ['#59b6ff', '#7ba6ff', '#8ed2ff', '#5f92e0', '#a6ccff', '#4aa6e8', '#86bdf5'];
const escape = value => value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('"', '&quot;');
const examples = [];
for (const [index, entry] of catalog.entries()) {
  const source = await readFile(new URL(`${entry.slug}.toml`, directory), 'utf8');
  const report = wasm.validateToml(source);
  if (!report.ok) throw new Error(`${entry.slug}: ${JSON.stringify(report.errors)}`);
  const { geometry } = report.config, [a, b] = report.resolved.lattice_vectors;
  const size = Math.max(Math.hypot(...a), Math.hypot(...b)), scale = 63 / size;
  const xy = (u, v) => [320 + (a[0] * u + b[0] * v) * scale, 155 - (a[1] * u + b[1] * v) * scale];
  const lines = [], objects = [], accent = palette[index % palette.length];
  for (let n = -5; n <= 5; n++) {
    for (const ends of [[xy(n,-5),xy(n,5)],[xy(-5,n),xy(5,n)]]) lines.push(`<path d="M${ends[0].join(',')}L${ends[1].join(',')}"/>`);
  }
  for (let iy = -6; iy <= 6; iy++) for (let ix = -6; ix <= 6; ix++) for (const object of geometry.objects) {
    const [x, y] = xy(ix + object.center[0], iy + object.center[1]);
    const hole = object.epsilon < geometry.background_epsilon;
    objects.push(`<circle cx="${x.toFixed(3)}" cy="${y.toFixed(3)}" r="${(object.radius * scale).toFixed(3)}" fill="${hole ? '#07080c' : 'url(#material)'}" stroke="${accent}" stroke-opacity="${hole ? '.55' : '.75'}" stroke-width=".7"/>`);
  }
  const cell = [[0,0],[1,0],[1,1],[0,1]].map(([u,v]) => xy(u,v).join(',')).join(' ');
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="640" height="310" viewBox="0 0 640 310" role="img" aria-label="${escape(entry.title)} periodic lattice">
<defs><radialGradient id="material"><stop stop-color="${accent}"/><stop offset="1" stop-color="${accent}" stop-opacity=".48"/></radialGradient></defs>
<rect width="640" height="310" fill="#060809"/><rect width="640" height="310" fill="${accent}" fill-opacity="${geometry.background_epsilon > 1 ? '.26' : '.06'}"/><g fill="none" stroke="${accent}" stroke-opacity=".12">${lines.join('')}</g>${objects.join('')}<polygon points="${cell}" fill="none" stroke="#e4f0ff" stroke-opacity=".75" stroke-width="1.1"/></svg>\n`;
  await writeFile(new URL(`${entry.slug}.svg`, images), svg);
  examples.push({ ...entry, image: `/examples/${entry.slug}.svg`, accent, source,
    python: await readFile(new URL(`${entry.slug}.py`, directory), 'utf8') });
}
await mkdir(new URL('../lib/examples/', import.meta.url), { recursive: true });
await writeFile(new URL('../lib/examples/catalog.generated.ts', import.meta.url),
  '/* Generated from examples/calculations. Run pnpm generate:examples to update. */\nexport const examples = ' + JSON.stringify(examples, null, 2) + ' as const;\n');
console.log(`Generated ${examples.length} examples and lattice images.`);
