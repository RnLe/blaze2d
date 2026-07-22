/**
 * SVG plot export: serialize to standalone .svg, or rasterize to PNG/WebP.
 *
 * Contract with the plot components: everything visual must be inline SVG
 * attributes/styles (no CSS classes inside the svg), and interactive-only
 * chrome sits under `<g data-export="omit">` so it can be stripped here.
 */

import { downloadBlob } from './download';

export interface RasterizeOptions {
  /** Device-pixel multiple of the viewBox size. */
  scale?: number;
  mime?: 'image/png' | 'image/webp';
  /** Encoder quality for lossy formats. */
  quality?: number;
  background?: string;
}

/** Serialize with xmlns + explicit dimensions, export-omit nodes stripped. */
export function svgToString(svg: SVGSVGElement): string {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.querySelectorAll('[data-export="omit"]').forEach((node) => node.remove());

  const viewBox = svg.viewBox.baseVal;
  const width = viewBox && viewBox.width > 0 ? viewBox.width : svg.clientWidth || 800;
  const height = viewBox && viewBox.height > 0 ? viewBox.height : svg.clientHeight || 400;
  clone.setAttribute('width', String(width));
  clone.setAttribute('height', String(height));
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  clone.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');

  const markup = new XMLSerializer().serializeToString(clone);
  return `<?xml version="1.0" encoding="UTF-8"?>\n${markup}`;
}

/** Can this browser ENCODE webp? (Safari renders webp but cannot encode it.) */
export function webpSupported(): boolean {
  try {
    const canvas = document.createElement('canvas');
    canvas.width = 2;
    canvas.height = 2;
    return canvas.toDataURL('image/webp').startsWith('data:image/webp');
  } catch {
    return false;
  }
}

export async function rasterizeSvg(
  svg: SVGSVGElement,
  opts: RasterizeOptions = {},
): Promise<Blob> {
  const { scale = 2, mime = 'image/png', quality = 0.95, background = '#0a0a0a' } = opts;

  const text = svgToString(svg);
  const viewBox = svg.viewBox.baseVal;
  const width = viewBox && viewBox.width > 0 ? viewBox.width : svg.clientWidth || 800;
  const height = viewBox && viewBox.height > 0 ? viewBox.height : svg.clientHeight || 400;

  const svgBlob = new Blob([text], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(svgBlob);
  try {
    const img = new Image();
    img.decoding = 'sync';
    img.src = url;
    await img.decode();

    const canvas = document.createElement('canvas');
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('canvas 2d context unavailable');
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, mime, quality),
    );
    if (!blob || blob.type !== mime) {
      throw new Error(`encoding to ${mime} is not supported in this browser`);
    }
    return blob;
  } finally {
    URL.revokeObjectURL(url);
  }
}

export function downloadSvg(svg: SVGSVGElement, filename: string): void {
  const blob = new Blob([svgToString(svg)], { type: 'image/svg+xml;charset=utf-8' });
  downloadBlob(blob, filename);
}

export async function downloadRaster(
  svg: SVGSVGElement,
  filename: string,
  opts?: RasterizeOptions,
): Promise<void> {
  const blob = await rasterizeSvg(svg, opts);
  downloadBlob(blob, filename);
}
