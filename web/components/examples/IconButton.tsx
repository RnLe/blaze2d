'use client';

import { Check } from 'lucide-react';
import { useState, type ReactNode } from 'react';

export interface IconButtonProps {
  /** Icon (typically a lucide icon at 14-15px). */
  children: ReactNode;
  /** Click handler. */
  onClick: (e: React.MouseEvent) => void;
  /** Tooltip / accessibility label. */
  label: string;
  /** Optional flash text shown briefly after click (e.g. "Copied to clipboard"). */
  flashOnClick?: string;
}

// While the flash is active, the icon is swapped for a green check and a short
// label is shown next to it. After ~1.1s everything reverts.
const FLASH_MS = 1100;

export default function IconButton({ children, onClick, label, flashOnClick }: IconButtonProps) {
  const [hover, setHover] = useState(false);
  const [flashed, setFlashed] = useState(false);

  const showFlashLabel = flashed && !!flashOnClick;

  return (
    <button
      type="button"
      aria-label={label}
      title={showFlashLabel ? flashOnClick : label}
      onClick={(e) => {
        onClick(e);
        if (flashOnClick) {
          setFlashed(true);
          setTimeout(() => setFlashed(false), FLASH_MS);
        }
      }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        background: 'transparent',
        border: 'none',
        cursor: 'pointer',
        padding: '4px 6px',
        borderRadius: 5,
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        justifyContent: 'center',
        color: flashed ? '#34d399' : hover ? '#e5e7eb' : '#9ca3af',
        backgroundColor: hover && !flashed ? 'rgba(255,255,255,0.05)' : 'transparent',
        transition: 'color 0.12s, background-color 0.12s',
        lineHeight: 0,
        fontSize: '0.74rem',
        fontWeight: 500,
      }}
    >
      {flashed ? <Check size={14} strokeWidth={2.5} /> : children}
      {showFlashLabel && <span style={{ whiteSpace: 'nowrap' }}>{flashOnClick}</span>}
    </button>
  );
}
