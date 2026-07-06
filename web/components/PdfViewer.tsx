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
            // Read-only document: annotation/redaction tools would only
            // clutter the toolbar (nothing can be persisted on a static site).
            disabledCategories: ['annotation', 'redaction'],
          }}
          style={{ width: '100%', height: '100%' }}
        />
      ) : (
        <ViewerPlaceholder />
      )}
    </div>
  );
}
