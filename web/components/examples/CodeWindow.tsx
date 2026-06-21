'use client';

import type { ReactNode } from 'react';
import { Copy, Download } from 'lucide-react';
import CodeBlock from './CodeBlock';
import IconButton from './IconButton';
import { copyText, downloadText } from '../../lib/util/download';

export interface CodeWindowProps {
  code: string;
  language?: 'python' | 'toml';
  /** File name shown at the top-right of the window chrome. */
  filename: string;
  /** Optional controls (e.g. Run / Abort) rendered at the top-right. */
  actions?: ReactNode;
  showLineNumbers?: boolean;
  /** Max height of the scrollable code area. */
  maxHeight?: number;
}

/**
 * CodeWindow — a reusable editor-window chrome (traffic-light dots, a filename
 * badge, and an optional actions slot) wrapping a syntax-highlighted CodeBlock.
 * Used for both the Python script and the TOML config of an example.
 */
export default function CodeWindow({
  code,
  language = 'python',
  filename,
  actions,
  showLineNumbers = false,
  maxHeight,
}: CodeWindowProps) {
  return (
    <div
      style={{
        border: '1px solid #1f2937',
        borderRadius: '10px',
        overflow: 'clip',
        background: '#0d1117',
        display: 'flex',
        flexDirection: 'column',
        minWidth: 0,
      }}
    >
      {/* Window chrome */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          padding: '8px 12px',
          borderBottom: '1px solid #1f2937',
          background: '#0b0e14',
        }}
      >
        <div style={{ display: 'flex', gap: 6 }}>
          <Dot color="#ff5f56" />
          <Dot color="#ffbd2e" />
          <Dot color="#27c93f" />
        </div>
        <span
          style={{
            fontFamily: 'var(--font-mono, ui-monospace, monospace)',
            fontSize: '0.74rem',
            color: '#9ca3af',
          }}
        >
          {filename}
        </span>
        <div style={{ flex: 1 }} />
        <IconButton label="Copy code" flashOnClick="Copied to clipboard" onClick={() => copyText(code)}>
          <Copy size={14} />
        </IconButton>
        <IconButton
          label={`Download ${filename}`}
          onClick={() => downloadText(code, filename)}
        >
          <Download size={14} />
        </IconButton>
        {actions}
      </div>

      {/* Code body */}
      <div
        style={{
          overflow: 'auto',
          maxHeight: maxHeight ?? undefined,
        }}
        className="subtle-scroll"
      >
        <CodeBlock code={code} language={language} variant="full" showLineNumbers={showLineNumbers} />
      </div>
    </div>
  );
}

function Dot({ color }: { color: string }) {
  return (
    <span
      style={{
        width: 11,
        height: 11,
        borderRadius: '50%',
        background: color,
        display: 'inline-block',
      }}
    />
  );
}
