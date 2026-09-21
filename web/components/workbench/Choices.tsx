'use client';
import type { ReactNode } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';

export function Choices<T extends string>({ label, value, options, onChange, compact = false, disabled = false }: {
  label: string; value: T; options: readonly { value: T; label: string; description?: string; icon?: ReactNode }[];
  onChange: (value: T) => void; compact?: boolean; disabled?: boolean;
}) {
  return <div className={`wb-choices${compact ? ' wb-choices-compact' : ''}`} role="group" aria-label={label}>
    {!compact && <span className="wb-control-label">{label}</span>}
    <div className="wb-choice-options">{options.map(option => <button type="button" key={option.value} aria-pressed={value === option.value}
      disabled={disabled} onClick={() => { if (option.value !== value) onChange(option.value); }} title={compact ? option.description : undefined}>
      {option.icon}<span>{option.label}{option.description && !compact && <small>{option.description}</small>}</span>
    </button>)}</div>
  </div>;
}

/** Large studies use a bounded index navigator instead of rendering thousands of choices. */
export function IndexControl({ label, value, count, onChange }: { label: string; value: number; count: number; onChange: (index: number) => void }) {
  return <div className="wb-index-control" role="group" aria-label={label}>
    <span>{label}</span><button aria-label={`Previous ${label.toLowerCase()}`} disabled={value <= 0} onClick={() => onChange(value - 1)}><ChevronLeft size={14} /></button>
    <input aria-label={label} type="number" min={0} max={Math.max(0, count - 1)} value={value}
      onChange={event => { const index = event.currentTarget.valueAsNumber; if (Number.isInteger(index)) onChange(Math.max(0, Math.min(count - 1, index))); }} />
    <button aria-label={`Next ${label.toLowerCase()}`} disabled={value >= count - 1} onClick={() => onChange(value + 1)}><ChevronRight size={14} /></button>
    <small>of {Math.max(0, count - 1)}</small>
  </div>;
}

export function LatticeIcon({ type }: { type: string }) {
  const skew = type === 'triangular' ? 7 : type === 'oblique' || type === 'custom' ? 5 : 0;
  const dx = type === 'rectangular' ? 17 : 13;
  return <svg width="48" height="35" viewBox="0 0 52 38" fill="none" aria-hidden="true">
    <path d={`M${10 + skew} 25h${dx}l${skew} -12H${10 + 2 * skew}Z`} stroke="currentColor" opacity=".55" />
    {[0, 1, 2].flatMap(row => [0, 1, 2].map(column => <circle key={`${row}:${column}`} cx={7 + column * dx + (2 - row) * skew} cy={6 + row * 12} r="2" fill="currentColor" />))}
  </svg>;
}
