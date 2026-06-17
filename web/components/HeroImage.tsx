'use client';

import { getAssetPath } from '../lib/paths';

interface HeroImageProps {
  src: string;
  alt: string;
  /**
   * Fraction of the image's 16:9 height to show (1.0 = full, 0.6 = 60% cropped).
   * Defaults to 1.0 (no cropping).
   */
  cropHeight?: number;
}

/**
 * Hero image component for article headers.
 * Fits to the narrow-centered width with rounded borders.
 * Optionally crops the image vertically via cropHeight (default: full height).
 */
export default function HeroImage({ src, alt, cropHeight }: HeroImageProps) {
  return (
    <div
      className="narrow-centered"
      style={{ marginBottom: '2rem' }}
    >
      <div
        style={{
          ...(cropHeight !== undefined && {
            position: 'relative',
            paddingBottom: `${56.25 * cropHeight}%`,
            overflow: 'hidden',
          }),
          borderRadius: '12px',
        }}
      >
        <img
          src={getAssetPath(src)}
          alt={alt}
          style={{
            ...(cropHeight !== undefined
              ? {
                  position: 'absolute',
                  top: '50%',
                  left: '0',
                  width: '100%',
                  height: 'auto',
                  transform: 'translateY(-50%)',
                }
              : {
                  width: '100%',
                  height: 'auto',
                }),
            display: 'block',
          }}
          loading="eager"
          fetchPriority="high"
        />
      </div>
    </div>
  );
}
