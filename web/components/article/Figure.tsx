import type { CSSProperties, ReactNode } from 'react';
import { getAssetPath } from '@/lib/paths';

interface FigureProps {
  /** Path to the image relative to /public (e.g. '/figures/intro/crystal.svg'). */
  src: string;
  alt: string;
  /** Optional caption shown beneath the image. */
  caption?: string;
  /** Max width of the figure (CSS value). Defaults to filling its container. */
  maxWidth?: string;
}

/** A captioned image that resolves the deployment base path. */
export default function Figure({ src, alt, caption, maxWidth }: FigureProps) {
  return (
    <figure className="figure" style={{ maxWidth: `min(100%, ${maxWidth ?? '100%'})` }}>
      <div className="figure-frame">
        <img src={getAssetPath(src)} alt={alt} loading="lazy" />
      </div>
      {caption && <figcaption>{caption}</figcaption>}
    </figure>
  );
}

/** Lays out two or more figures in a row that wraps to one column when narrow. */
export function FigureRow({ children }: { children: ReactNode }) {
  return <div className="figure-row">{children}</div>;
}

interface HeroImageProps {
  src: string;
  alt: string;
  /** Fraction of the image's 16:9 height to show (1 = full, 0.6 = 60% cropped). */
  cropHeight?: number;
}

/** Full-width image for an article header, optionally cropped about its centre. */
export function HeroImage({ src, alt, cropHeight }: HeroImageProps) {
  const cropped = cropHeight !== undefined;
  return (
    <div className="narrow-centered">
      <div
        className={`hero-image${cropped ? ' hero-image-cropped' : ''}`}
        style={cropped ? ({ '--hero-crop': cropHeight } as CSSProperties) : undefined}
      >
        <img src={getAssetPath(src)} alt={alt} loading="eager" fetchPriority="high" />
      </div>
    </div>
  );
}
