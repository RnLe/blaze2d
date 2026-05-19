'use client';

import { getAssetPath } from '../lib/paths';

interface HeroImageProps {
  src: string;
  alt: string;
  /** Crop factor: 1.0 = full height, 0.8 = 80% height (cropped). Default: 0.8 */
  cropHeight?: number;
}

/**
 * Hero image component for article headers.
 * Fits to the narrow-centered width with rounded borders.
 * Crops the image vertically to reduce height while maintaining full width.
 */
export default function HeroImage({ src, alt, cropHeight = 0.8 }: HeroImageProps) {
  // Calculate padding-bottom for aspect ratio container
  // Original aspect ratio is typically 16:9 (56.25%), we reduce it by cropHeight
  const aspectRatio = 56.25 * cropHeight; // percentage
  
  return (
    <div 
      className="narrow-centered"
      style={{
        marginBottom: '2rem',
      }}
    >
      <div
        style={{
          position: 'relative',
          width: '100%',
          paddingBottom: `${aspectRatio}%`,
          borderRadius: '12px',
          overflow: 'hidden',
        }}
      >
        <img
          src={getAssetPath(src)}
          alt={alt}
          style={{
            position: 'absolute',
            top: '50%',
            left: '0',
            width: '100%',
            height: 'auto',
            transform: 'translateY(-42%)',
            display: 'block',
          }}
          // Prioritize loading - this is the hero image
          loading="eager"
          fetchPriority="high"
        />
      </div>
    </div>
  );
}
