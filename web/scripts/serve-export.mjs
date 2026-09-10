import http from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve('out'), base = process.env.NEXT_BASE_PATH ?? '';
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.wasm': 'application/wasm', '.json': 'application/json', '.svg': 'image/svg+xml', '.woff2': 'font/woff2', '.webp': 'image/webp', '.pdf': 'application/pdf', '.png': 'image/png' };
http.createServer(async (request, response) => {
  try {
    let pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    if (base && pathname !== base && !pathname.startsWith(base + '/')) throw new Error('Missing base path');
    pathname = pathname.slice(base.length);
    let file = path.resolve(root, '.' + pathname);
    if (!file.startsWith(root + path.sep) && file !== root) throw new Error('Invalid path');
    if ((await stat(file)).isDirectory()) file = path.join(file, 'index.html');
    response.setHeader('Content-Type', types[path.extname(file)] ?? 'application/octet-stream');
    response.end(await readFile(file));
  } catch { response.statusCode = 404; response.end('Not found'); }
}).listen(3211, '127.0.0.1');
