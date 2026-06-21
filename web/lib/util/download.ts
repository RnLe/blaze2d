/**
 * Tiny browser helpers for copying text and downloading files.
 *
 * Includes a minimal STORE-mode (uncompressed) ZIP writer — small enough to
 * inline so we don't need a dependency for the file-explorer's
 * "download all" action. Modern operating systems and archive utilities
 * extract stored ZIPs without issue.
 */

export async function copyText(text: string): Promise<boolean> {
  try {
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // Fall through to legacy path.
  }
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand('copy');
    document.body.removeChild(ta);
    return ok;
  } catch {
    return false;
  }
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function downloadText(text: string, filename: string, mime = 'text/plain'): void {
  downloadBlob(new Blob([text], { type: `${mime};charset=utf-8` }), filename);
}

export function downloadJson(value: unknown, filename: string): void {
  const text = JSON.stringify(value, jsonReplacer, 2);
  downloadText(text, filename, 'application/json');
}

/** JSON replacer that turns NaN/Infinity into a safe string. */
function jsonReplacer(_k: string, v: unknown): unknown {
  if (typeof v === 'number' && !Number.isFinite(v)) return String(v);
  return v;
}

// ---------------------------------------------------------------------------
// Minimal STORE-mode ZIP writer (no compression)
// ---------------------------------------------------------------------------

interface ZipEntry {
  name: string;
  data: Uint8Array;
}

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(buf: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function dosTime(d: Date): { time: number; date: number } {
  const time = (d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1);
  const date = ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate();
  return { time, date };
}

/** Build a STORE-mode ZIP archive from a list of `{name, content}` files. */
export function buildZip(files: { name: string; content: string }[]): Blob {
  const enc = new TextEncoder();
  const entries: ZipEntry[] = files.map((f) => ({ name: f.name, data: enc.encode(f.content) }));
  const { time, date } = dosTime(new Date());

  const localChunks: Uint8Array[] = [];
  const centralChunks: Uint8Array[] = [];
  let offset = 0;

  for (const entry of entries) {
    const nameBytes = enc.encode(entry.name);
    const crc = crc32(entry.data);
    const size = entry.data.length;

    // Local file header (30 bytes + name)
    const local = new Uint8Array(30 + nameBytes.length);
    const ldv = new DataView(local.buffer);
    ldv.setUint32(0, 0x04034b50, true);    // sig
    ldv.setUint16(4, 20, true);             // version needed
    ldv.setUint16(6, 0, true);              // flags
    ldv.setUint16(8, 0, true);              // method = STORE
    ldv.setUint16(10, time, true);
    ldv.setUint16(12, date, true);
    ldv.setUint32(14, crc, true);
    ldv.setUint32(18, size, true);          // compressed size
    ldv.setUint32(22, size, true);          // uncompressed size
    ldv.setUint16(26, nameBytes.length, true);
    ldv.setUint16(28, 0, true);             // extra length
    local.set(nameBytes, 30);
    localChunks.push(local, entry.data);

    // Central directory header (46 bytes + name)
    const central = new Uint8Array(46 + nameBytes.length);
    const cdv = new DataView(central.buffer);
    cdv.setUint32(0, 0x02014b50, true);     // sig
    cdv.setUint16(4, 20, true);              // version made by
    cdv.setUint16(6, 20, true);              // version needed
    cdv.setUint16(8, 0, true);               // flags
    cdv.setUint16(10, 0, true);              // method
    cdv.setUint16(12, time, true);
    cdv.setUint16(14, date, true);
    cdv.setUint32(16, crc, true);
    cdv.setUint32(20, size, true);
    cdv.setUint32(24, size, true);
    cdv.setUint16(28, nameBytes.length, true);
    cdv.setUint16(30, 0, true);              // extra
    cdv.setUint16(32, 0, true);              // comment
    cdv.setUint16(34, 0, true);              // disk #
    cdv.setUint16(36, 0, true);              // internal attrs
    cdv.setUint32(38, 0, true);              // external attrs
    cdv.setUint32(42, offset, true);         // local header offset
    central.set(nameBytes, 46);
    centralChunks.push(central);

    offset += local.length + entry.data.length;
  }

  const centralSize = centralChunks.reduce((s, c) => s + c.length, 0);
  const centralOffset = offset;

  // End of central directory record (22 bytes)
  const end = new Uint8Array(22);
  const edv = new DataView(end.buffer);
  edv.setUint32(0, 0x06054b50, true);
  edv.setUint16(4, 0, true);
  edv.setUint16(6, 0, true);
  edv.setUint16(8, entries.length, true);
  edv.setUint16(10, entries.length, true);
  edv.setUint32(12, centralSize, true);
  edv.setUint32(16, centralOffset, true);
  edv.setUint16(20, 0, true);

  return new Blob([...localChunks, ...centralChunks, end] as BlobPart[], {
    type: 'application/zip',
  });
}

export function downloadZip(files: { name: string; content: string }[], zipName: string): void {
  downloadBlob(buildZip(files), zipName);
}
