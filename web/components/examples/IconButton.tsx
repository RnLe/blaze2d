'use client';
import { Check } from 'lucide-react';
import { useEffect, useRef, useState, type ReactNode } from 'react';

export interface IconButtonProps {
  /** Icon, typically a lucide icon at 14-15px. */
  children: ReactNode;
  onClick: () => void;
  /** Tooltip and accessible name. */
  label: string;
  /** Shown briefly beside a check mark after a click, e.g. "Copied to clipboard". */
  flashOnClick?: string;
}

const FLASH_MS = 1100;

/** Icon-only affordance used in the code window chrome. */
export default function IconButton({ children, onClick, label, flashOnClick }: IconButtonProps) {
  const [flashed, setFlashed] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);
  useEffect(() => () => clearTimeout(timer.current), []);

  return (
    <button
      type="button"
      className="icon-button"
      data-flashed={flashed}
      aria-label={label}
      title={flashed && flashOnClick ? flashOnClick : label}
      onClick={() => {
        onClick();
        if (!flashOnClick) return;
        setFlashed(true);
        clearTimeout(timer.current);
        timer.current = setTimeout(() => setFlashed(false), FLASH_MS);
      }}
    >
      {flashed ? <Check size={14} strokeWidth={2.5} /> : children}
      {flashed && flashOnClick && <span>{flashOnClick}</span>}
    </button>
  );
}
