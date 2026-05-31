'use client';

import { Copy } from 'lucide-react';
import IconButton from './IconButton';
import { copyText } from '../../lib/util/download';

export interface OutputBoxProps {
  /** The text to display (live stdout from the run). */
  text: string;
  /** Whether the run is still producing output. */
  running?: boolean;
}

/**
 * OutputBox — a terminal-like panel showing the live stdout an example's
 * `print(...)` statements would produce, filled from the actual WASM results.
 */
export default function OutputBox({ text, running = false }: OutputBoxProps) {
  return (
    <div
      style={{
        border: '1px solid #1f2937',
        borderRadius: '10px',
        overflow: 'hidden',
        background: '#0a0a0a',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '7px 12px',
          borderBottom: '1px solid #1f2937',
          fontSize: '0.66rem',
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          color: '#6b7280',
        }}
      >
        <span>Output</span>
        {running && (
          <span style={{ color: '#60a5fa', textTransform: 'none', letterSpacing: 0 }}>
            ▋
          </span>
        )}
        <div style={{ flex: 1 }} />
        <IconButton
          label="Copy output"
          flashOnClick="Copied to clipboard"
          onClick={() => copyText(text)}
        >
          <Copy size={13} />
        </IconButton>
      </div>
      <pre
        className="blaze-no-scrollbar"
        style={{
          margin: 0,
          padding: '12px 14px',
          maxHeight: 180,
          overflow: 'auto',
          fontFamily: 'var(--font-mono, ui-monospace, monospace)',
          fontSize: '0.78rem',
          lineHeight: 1.6,
          color: '#d1d5db',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}
      >
        {text || '\u00A0'}
      </pre>
    </div>
  );
}
