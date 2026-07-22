'use client';

import dynamic from 'next/dynamic';

// The studio is a heavy, client-only app (WASM workers, canvas, live state).
// Load it with ssr:false so the static export does not try to prerender it.
const StudioApp = dynamic(() => import('../../components/studio/StudioApp'), {
  ssr: false,
  loading: () => (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#060606',
        color: '#7c7c7c',
        fontSize: 14,
      }}
    >
      Loading workbench…
    </div>
  ),
});

export default function WorkbenchPage() {
  return <StudioApp />;
}
