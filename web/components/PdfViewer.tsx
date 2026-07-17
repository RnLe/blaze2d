'use client';

import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import { getAssetPath } from '../lib/paths';

// EmbedPDF drives canvases and a WASM engine, so it must never run during
// static export — load it on the client only.
const PDFViewer = dynamic(
  () => import('@embedpdf/react-pdf-viewer').then((m) => m.PDFViewer),
  { ssr: false, loading: () => <ViewerPlaceholder /> }
);

function ViewerPlaceholder() {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        opacity: 0.6,
      }}
    >
      Loading viewer…
    </div>
  );
}

interface PdfViewerProps {
  /** Public path to the PDF (e.g. '/paper/blaze2d.pdf'); base path is added here. */
  src: string;
  height?: string;
}

export default function PdfViewer({ src, height = '85vh' }: PdfViewerProps) {
  // The engine runs in a blob: worker, which cannot resolve origin-less
  // paths — every URL handed to the viewer must be fully qualified.
  const [origin, setOrigin] = useState<string | null>(null);
  useEffect(() => {
    setOrigin(window.location.origin);
  }, []);

  return (
    <div
      style={{
        width: '100%',
        height,
        borderRadius: '12px',
        overflow: 'hidden',
        border: '1px solid rgba(128, 128, 128, 0.25)',
      }}
    >
      {origin ? (
        <PDFViewer
          config={{
            src: origin + getAssetPath(src),
            // Self-hosted engine: keeps the viewer fully offline-capable and
            // immune to bundler/base-path asset resolution surprises.
            wasmUrl: origin + getAssetPath('/paper/pdfium.wasm'),
            theme: { preference: 'system' },
            // Read-only document: annotation/redaction/mode/comment tools
            // would only clutter the toolbar (nothing can be persisted on a
            // static site). Disabling a category hides its UI items AND
            // disables the underlying commands/shortcuts.
            disabledCategories: [
              'annotation',
              'redaction',
              'mode',
              'insert',
              'form',
              'panel-comment',
            ],
          }}
          onReady={(registry: unknown) => {
            // Hoist fit-width / fit-page / fullscreen out of the zoom
            // dropdown into the main toolbar. The UI schema cannot be set
            // statically (config.ui.schema replaces the default wholesale),
            // so mutate it at runtime instead.
            /* eslint-disable @typescript-eslint/no-explicit-any */
            const reg = registry as any;
            const ui = reg?.getPlugin?.('ui')?.provides?.();
            const bar = ui?.getSchema?.()?.toolbars?.['main-toolbar'];
            if (!ui || !bar) return;
            const stripped = new Set(['mode-tabs', 'mode-select-button', 'comment-button']);
            const extraButtons = [
              { type: 'command-button', id: 'fit-width-btn', commandId: 'zoom:fit-width', variant: 'icon', categories: ['zoom'] },
              { type: 'command-button', id: 'fit-page-btn', commandId: 'zoom:fit-page', variant: 'icon', categories: ['zoom'] },
              { type: 'command-button', id: 'fullscreen-btn', commandId: 'document:fullscreen', variant: 'icon', categories: ['document'] },
            ];
            const items = bar.items
              .filter((item: any) => !stripped.has(item.id))
              .map((item: any) => {
                if (item.id === 'center-group' && Array.isArray(item.items)) {
                  return { ...item, items: [...item.items, ...extraButtons.slice(0, 2)] };
                }
                if (item.id === 'right-group' && Array.isArray(item.items)) {
                  return {
                    ...item,
                    items: [...item.items.filter((c: any) => !stripped.has(c.id)), extraButtons[2]],
                  };
                }
                return item;
              });
            ui.mergeSchema({ toolbars: { 'main-toolbar': { items } } });
            /* eslint-enable @typescript-eslint/no-explicit-any */
          }}
          style={{ width: '100%', height: '100%' }}
        />
      ) : (
        <ViewerPlaceholder />
      )}
    </div>
  );
}
