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
  /** Path to the banner image (relative to /public, e.g., "/banners/blaze_intro.png") */
  image: string;
  /** Date string to display */
  date?: string;
  /** Optional tags/categories */
  tags?: string[];
}

export default function BlogCard({
  href,
  title,
  description,
  image,
  date,
  tags,
}: BlogCardProps) {
  const imageRef = useRef<HTMLImageElement>(null);

  return (
    <Link 
      href={href} 
      className="blog-card-link"
      style={{ textDecoration: 'none', color: 'inherit', display: 'block' }}
      onMouseEnter={() => {
        if (imageRef.current) {
          imageRef.current.style.transform = 'scale(1.05)';
        }
      }}
      onMouseLeave={() => {
        if (imageRef.current) {
          imageRef.current.style.transform = 'scale(1)';
        }
      }}
    >
      <article
        style={{
          display: 'flex',
          flexDirection: 'column',
          borderRadius: '12px',
          backgroundColor: '#fff',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
          cursor: 'pointer',
          maxWidth: '400px',
          overflow: 'hidden',
        }}
      >
        {/* Image container */}
        <div
          style={{
            position: 'relative',
            width: '100%',
            aspectRatio: '16 / 9',
            overflow: 'hidden',
            backgroundColor: '#f0f0f0',
            borderRadius: '12px',
          }}
        >
          <Image
            ref={imageRef}
            src={getAssetPath(image)}
            loader={({ src }) => src} // Bypass automatic base path handling since we do it manually
            alt={title}
            fill
            style={{
              objectFit: 'cover',
              borderRadius: '12px',
            }}
            sizes="(max-width: 768px) 100vw, 400px"
          />
        </div>

        {/* Content */}
        <div style={{ padding: '20px' }}>
          {/* Date and tags row */}
          {(date || tags) && (
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
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
        gap: '24px',
        padding: '24px 0',
        justifyItems: 'center',
      }}
    >
      {children}
    </div>
  );
}
