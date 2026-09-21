import { watch } from 'node:fs';
import { spawn } from 'node:child_process';

const next = spawn('pnpm', ['exec', 'next', 'dev', ...process.argv.slice(2)], { stdio: 'inherit' });
let timer, generating = false, queued = false;
function generate() {
  if (generating) { queued = true; return; }
  generating = true;
  const child = spawn(process.execPath, ['scripts/generate-content.mjs'], { stdio: 'inherit' });
  child.on('exit', () => { generating = false; if (queued) { queued = false; generate(); } });
}
const watcher = watch('content', (_, filename) => {
  if (!filename || !/\.mdx?$/.test(filename)) return;
  clearTimeout(timer); timer = setTimeout(generate, 150);
});
function close(signal = 'SIGTERM') { watcher.close(); clearTimeout(timer); next.kill(signal); }
process.on('SIGINT', () => close('SIGINT'));
process.on('SIGTERM', () => close());
next.on('exit', code => { watcher.close(); clearTimeout(timer); process.exitCode = code ?? 0; });
