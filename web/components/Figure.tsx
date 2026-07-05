'use client';

import { getAssetPath } from '../lib/paths';

interface FigureProps {
  /** Path to the image relative to /public (e.g. '/figures/intro/crystal.svg'). */
  src: string;
  alt: string;
  /** Optional caption shown beneath the image. */
  caption?: string;
  /** Max width of the figure (CSS value). Defaults to filling its container. */
  maxWidth?: string;
}

/**
 * A captioned image that resolves the GitHub-Pages base path via getAssetPath.
 * Thesis plots are exported on transparent/white backgrounds, so the image sits
 * on a light panel that reads cleanly in both light and dark themes.
 */
export default function Figure({ src, alt, caption, maxWidth }: FigureProps) {
  return (
    <figure
      style={{
        margin: '0 auto',
        // Grows to share a FigureRow evenly; wraps below ~320px. Harmless when
        // the figure is used standalone (outside a flex container).
        flex: '1 1 320px',
        width: '100%',
        maxWidth: maxWidth ?? '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
      }}
    >
      <div
        style={{
          width: '100%',
          background: '#ffffff',
          border: '1px solid rgba(120, 140, 170, 0.18)',
          borderRadius: '12px',
          padding: '14px',
          boxShadow: '0 1px 4px rgba(0, 0, 0, 0.06)',
          display: 'flex',
          justifyContent: 'center',
        }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={getAssetPath(src)}
          alt={alt}
          style={{ width: '100%', height: 'auto', display: 'block' }}
          loading="lazy"
        />
      </div>
      {caption && (
        <figcaption
          style={{
            fontSize: '0.82rem',
            lineHeight: 1.5,
            color: 'var(--blaze-muted, #6b7280)',
            textAlign: 'center',
            marginTop: '0.6rem',
            maxWidth: '46ch',
          }}
        >
          {caption}
        </figcaption>
      )}
    </figure>
  );
}

/**
 * Lays out two or more <Figure> elements in a responsive row that wraps to a
 * single column on narrow screens.
 */
export function FigureRow({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: '20px',
        alignItems: 'flex-start',
        justifyContent: 'center',
        margin: '1.75rem 0',
      }}
    >
      {children}
    </div>
  );
}
