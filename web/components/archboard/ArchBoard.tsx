'use client';

import dynamic from 'next/dynamic';

// pixi touches the DOM and WebGL at init time; it must never run during
// static export. Everything pixi-related loads behind this boundary.
const ArchBoardInner = dynamic(() => import('./ArchBoardInner'), {
  ssr: false,
  loading: () => (
    <div
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#0a0a0a',
        border: '1px solid rgba(128, 128, 128, 0.25)',
        borderRadius: '12px',
        color: 'rgba(255,255,255,0.5)',
        fontSize: 14,
      }}
    >
      Loading the architecture board…
    </div>
  ),
});

export default function ArchBoard({ height = '72vh' }: { height?: string }) {
  return (
    <div style={{ width: '100%', height }}>
      <ArchBoardInner />
    </div>
  );
}
