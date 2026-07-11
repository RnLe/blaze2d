import type { ReactNode } from 'react';

/**
 * Self-centering breakout row for chart components. Escapes the 600px
 * `.narrow` column (same trick as the board wrapper on the architecture
 * index) and lays children out side by side, wrapping to one column when
 * the viewport is too small. A single child just gets centered.
 */
export default function ChartRow({ children, width = 1104 }: { children: ReactNode; width?: number }) {
  return (
    <div
      style={{
        width: `${width}px`,
        maxWidth: '92vw',
        marginLeft: '50%',
        transform: 'translateX(-50%)',
        marginTop: '1.75rem',
        marginBottom: '1.75rem',
      }}
    >
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '24px',
          justifyContent: 'center',
          alignItems: 'flex-start',
        }}
      >
        {children}
      </div>
    </div>
  );
}
