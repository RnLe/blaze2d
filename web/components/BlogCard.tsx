'use client';

import Link from 'next/link';
import Image from 'next/image';
import { useRef } from 'react';
import { getAssetPath } from '../lib/paths';

export interface BlogCardProps {
  /** URL path to navigate to (e.g., "/blaze") */
  href: string;
  /** Title of the blog post */
  title: string;
  /** Short description/subtitle */
  description?: string;
  /**
   * Path to the banner image (relative to /public). Optional: when omitted, an
   * artistic, blurred title panel is shown in place of the image and sharpens
   * on hover.
   */
  image?: string;
  /** Date string to display */
  date?: string;
  /** Optional tags/categories */
  tags?: string[];
  /**
   * When false, the date/tags meta row above the title is hidden. Defaults to
   * true.
   */
  showMeta?: boolean;
  /**
   * When true, the image is shown at its natural aspect ratio (no cropping).
   * The card height adapts to the image instead of forcing 16:9.
   */
  naturalHeight?: boolean;
}

export default function BlogCard({
  href,
  title,
  description,
  image,
  date,
  tags,
  showMeta = true,
  naturalHeight = false,
}: BlogCardProps) {
  const imageRef = useRef<HTMLImageElement>(null);
  const titleArtRef = useRef<HTMLDivElement>(null);

  return (
    <Link 
      href={href} 
      className="blog-card-link"
      style={{ textDecoration: 'none', color: 'inherit', display: 'block', height: naturalHeight ? 'auto' : '100%' }}
      onMouseEnter={() => {
        if (imageRef.current) {
          imageRef.current.style.transform = 'scale(1.05)';
        }
        if (titleArtRef.current) {
          titleArtRef.current.style.filter = 'blur(0px)';
          titleArtRef.current.style.opacity = '1';
          titleArtRef.current.style.transform = 'scale(1.04)';
        }
      }}
      onMouseLeave={() => {
        if (imageRef.current) {
          imageRef.current.style.transform = 'scale(1)';
        }
        if (titleArtRef.current) {
          titleArtRef.current.style.filter = 'blur(7px)';
          titleArtRef.current.style.opacity = '0.78';
          titleArtRef.current.style.transform = 'scale(1)';
        }
      }}
    >
      <article
        style={{
          display: 'flex',
          flexDirection: 'column',
          borderRadius: '12px',
          backgroundColor: '#000000',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
          cursor: 'pointer',
          width: '100%',
          height: naturalHeight ? 'auto' : '100%',
          overflow: 'hidden',
        }}
      >
        {/* Image container (or artistic title panel when no image) */}
        <div
          style={{
            position: naturalHeight ? 'static' : 'relative',
            width: '100%',
            ...(naturalHeight ? {} : { aspectRatio: '2 / 1' }),
            overflow: 'hidden',
            backgroundColor: '#000000',
            borderRadius: '12px',
            isolation: 'isolate',
            willChange: 'transform',
          }}
        >
          {!image ? (
            // No image: an artistic, blurred title panel that sharpens on hover.
            <div
              style={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '18px 22px',
                background:
                  'radial-gradient(120% 120% at 30% 20%, #1d2740 0%, #0b1120 55%, #05070d 100%)',
                borderRadius: '12px',
              }}
            >
              <div
                ref={titleArtRef}
                style={{
                  filter: 'blur(7px)',
                  opacity: 0.78,
                  transform: 'scale(1)',
                  transition: 'filter 0.35s ease, opacity 0.35s ease, transform 0.35s ease',
                  fontSize: 'clamp(20px, 3.4vw, 30px)',
                  fontWeight: 700,
                  lineHeight: 1.18,
                  textAlign: 'center',
                  letterSpacing: '-0.01em',
                  background: 'linear-gradient(110deg, #93c5fd 0%, #c4b5fd 50%, #f0abfc 100%)',
                  WebkitBackgroundClip: 'text',
                  backgroundClip: 'text',
                  color: 'transparent',
                  userSelect: 'none',
                }}
              >
                {title}
              </div>
            </div>
          ) : naturalHeight ? (
            // Natural mode: image sets its own height, no cropping
            <Image
              ref={imageRef}
              src={getAssetPath(image)}
              loader={({ src }) => src}
              alt={title}
              width={0}
              height={0}
              sizes="100vw"
              style={{
                width: '100%',
                height: 'auto',
                display: 'block',
                transition: 'transform 0.3s ease',
              }}
            />
          ) : (
            // Default cover mode: fixed 2:1 frame, image scaled to fit so the
            // whole banner is visible (no left/right cropping). The black
            // backdrop makes any letterboxing invisible against the card.
            <Image
              ref={imageRef}
              src={getAssetPath(image)}
              loader={({ src }) => src}
              alt={title}
              fill
              style={{ objectFit: 'contain' }}
              sizes="(max-width: 768px) 100vw, 480px"
            />
          )}
        </div>

        {/* Content */}
        <div style={{ padding: '20px', flex: 1, display: 'flex', flexDirection: 'column' }}>
          {/* Date and tags row */}
          {showMeta && (date || tags) && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                marginBottom: '8px',
                flexWrap: 'wrap',
              }}
            >
              {date && (
                <span
                  style={{
                    fontSize: '12px',
                    color: '#666',
                    fontVariantNumeric: 'tabular-nums',
                  }}
                >
                  {date}
                </span>
              )}
              {tags && tags.length > 0 && (
                <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                  {tags.map((tag) => (
                    <span
                      key={tag}
                      style={{
                        fontSize: '10px',
                        padding: '2px 8px',
                        borderRadius: '12px',
                        backgroundColor: '#eef2ff',
                        color: '#435f9d',
                        fontWeight: 500,
                        textTransform: 'uppercase',
                        letterSpacing: '0.5px',
                      }}
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Title */}
          <h2
            style={{
              fontSize: '18px',
              fontWeight: 600,
              margin: 0,
              marginBottom: description ? '8px' : 0,
              color: '#1a1a1a',
              lineHeight: 1.3,
            }}
          >
            {title}
          </h2>

          {/* Description */}
          {description && (
            <p
              style={{
                fontSize: '14px',
                color: '#666',
                margin: 0,
                lineHeight: 1.5,
                flex: 1,
              }}
            >
              {description}
            </p>
          )}
        </div>
      </article>
    </Link>
  );
}

/** Grid container for multiple blog cards */
export function BlogCardGrid({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="blog-card-grid"
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
        gap: '24px',
        padding: '24px 0',
        width: '100%',
        maxWidth: '1500px',
        marginLeft: 'auto',
        marginRight: 'auto',
        alignItems: 'start',
      }}
    >
      {children}
    </div>
  );
}
