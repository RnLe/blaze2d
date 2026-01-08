'use client';

import Link from 'next/link';
import { getAssetPath } from '../../lib/paths';

export default function PitchHeader() {
  return (
    <Link 
      href="/"
      style={{
        position: 'fixed',
        left: '2rem',
        top: '2rem', // Reduced from 50% to align top-left
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        textDecoration: 'none',
        zIndex: 100,
        cursor: 'pointer',
        // Minimal hover effect
        transition: 'opacity 0.2s ease',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.opacity = '0.8';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.opacity = '1';
      }}
    >
      <img src={getAssetPath('/icons/blaze_bw.svg')} alt="Blaze" style={{ height: '1.5rem', width: 'auto' }} />
      <span style={{ 
        color: '#ffffff', 
        fontWeight: 600, 
        fontSize: '1.5rem', 
        letterSpacing: '-0.02em' 
      }}>
        Blaze 2D
      </span>
    </Link>
  );
}
