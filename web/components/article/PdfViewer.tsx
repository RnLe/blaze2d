'use client';

import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import type { EmbedPdfContainer, CommandButtonItem, PluginRegistry, ToolbarItem, UIPlugin } from '@embedpdf/react-pdf-viewer';
import { getAssetPath } from '@/lib/paths';

// EmbedPDF drives canvases and a WASM engine, so it must never run during
// static export. Load it on the client only.
const PDFViewer = dynamic(() => import('@embedpdf/react-pdf-viewer').then(m => m.PDFViewer), {
  ssr: false,
  loading: () => <ViewerPlaceholder />,
});

function ViewerPlaceholder() {
  return (
    <div className="pdf-viewer-placeholder" role="status">
      Loading viewer…
    </div>
  );
}

/**
 * The viewer's own dark theme is a blue-grey that does not belong to this
 * palette, so every surface it exposes is mapped onto the site's tokens. These
 * are literals because the viewer renders inside a shadow root and resolves its
 * colours in JavaScript, where `var()` would not reach.
 *
 * Mirrors app/styles/tokens.css; see lib/theme.ts for the same constraint.
 */
const VIEWER_DARK = {
  background: {
    app: '#050505',
    surface: '#0a0a0a',
    surfaceAlt: '#0c0c0c',
    elevated: '#141414',
    overlay: 'rgba(0, 0, 0, 0.72)',
    input: '#101010',
  },
  foreground: {
    primary: '#ededed',
    secondary: '#b5b5b5',
    muted: '#8f8f8f',
    disabled: '#5a5a5a',
    onAccent: '#04121e',
  },
  border: { default: '#292929', subtle: '#202020', strong: '#3b3b3b' },
  accent: {
    primary: '#59b6ff',
    primaryHover: '#8acfff',
    primaryActive: '#3890d1',
    primaryLight: '#071929',
    primaryForeground: '#04121e',
  },
  interactive: {
    hover: '#161616',
    active: '#1c1c1c',
    selected: '#071929',
    focus: '#59b6ff',
    focusRing: 'rgba(89, 182, 255, 0.35)',
  },
  scrollbar: { track: '#0a0a0a', thumb: '#2c2c2c', thumbHover: '#3b3b3b' },
  tooltip: { background: '#141414', foreground: '#ededed' },
};

/**
 * Fit-to-page and fit-to-width ship as real commands (`Ctrl+0` and `Ctrl+1`)
 * but the default toolbar only reaches them through the zoom dropdown. In a
 * column this narrow they are the first two controls a reader wants, so they
 * are promoted to buttons beside the zoom control.
 */
const FIT_BUTTONS: CommandButtonItem[] = [
  { type: 'command-button', id: 'zoom-fit-page-button', commandId: 'zoom:fit-page', variant: 'icon', categories: ['zoom', 'zoom-fit-page'] },
  { type: 'command-button', id: 'zoom-fit-width-button', commandId: 'zoom:fit-width', variant: 'icon', categories: ['zoom', 'zoom-fit-width'] },
];

/** Returns the items with the fit buttons after the zoom control, or null if it is not here. */
function withFitButtons(items: ToolbarItem[]): ToolbarItem[] | null {
  const at = items.findIndex(item => item.id === 'zoom-toolbar');
  if (at >= 0) return [...items.slice(0, at + 1), ...FIT_BUTTONS, ...items.slice(at + 1)];
  for (let index = 0; index < items.length; index++) {
    const item = items[index];
    if (item.type !== 'group') continue;
    const inner = withFitButtons(item.items);
    if (inner) return [...items.slice(0, index), { ...item, items: inner }, ...items.slice(index + 1)];
  }
  return null;
}

/**
 * The schema is read back and merged rather than declared, so the toolbar keeps
 * whatever else the installed viewer version ships; if a future version moves
 * or renames the zoom control, nothing is inserted and the toolbar is left
 * exactly as it came.
 */
function promoteFitButtons(registry: PluginRegistry) {
  const ui = registry.getPlugin<UIPlugin>('ui')?.provides();
  if (!ui) return;
  const toolbars = ui.getSchema().toolbars;
  const toolbar = toolbars['main-toolbar'];
  const items = toolbar && withFitButtons(toolbar.items);
  if (items) ui.mergeSchema({ toolbars: { ...toolbars, 'main-toolbar': { ...toolbar, items } } });
}

/** Keep the viewport size stable while pages and zoom controls initialize.
 * Auto scrollbars feed size changes back into EmbedPDF's ResizeObserver in WebKit.
 * Reserve both scrollbars inside the viewer's shadow root instead.
 */
function stabilizeViewport(container: EmbedPdfContainer) {
  const style = document.createElement('style');
  style.textContent = '.bg-bg-app[style*="overflow: auto"] { overflow: scroll !important; }';
  container.shadowRoot?.append(style);
}

interface PdfViewerProps {
  /** Public path to the PDF (e.g. '/paper/blaze2d.pdf'); the base path is added here. */
  src: string;
  height?: string;
}

export default function PdfViewer({ src, height }: PdfViewerProps) {
  // The engine runs in a blob: worker, which cannot resolve origin-less paths.
  // Every URL handed to the viewer must be fully qualified.
  const [origin, setOrigin] = useState<string | null>(null);
  useEffect(() => setOrigin(window.location.origin), []);

  return (
    <div className="pdf-viewer" style={height ? { height } : undefined}>
      {origin ? (
        <PDFViewer
          config={{
            src: origin + getAssetPath(src),
            // Self-hosted engine: keeps the viewer offline-capable and immune to
            // bundler and base-path asset resolution surprises.
            wasmUrl: origin + getAssetPath('/paper/pdfium.wasm'),
            theme: { preference: 'dark', dark: VIEWER_DARK },
            // A read-only document on a static site: nothing can be persisted,
            // so these tool categories would only clutter the toolbar. Disabling
            // a category hides its items and disables its commands and shortcuts.
            disabledCategories: ['annotation', 'redaction', 'mode', 'insert', 'form', 'panel-comment'],
          }}
          onInit={stabilizeViewport}
          onReady={promoteFitButtons}
          style={{ width: '100%', height: '100%' }}
        />
      ) : (
        <ViewerPlaceholder />
      )}
    </div>
  );
}
