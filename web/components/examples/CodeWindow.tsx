'use client';

import type { CSSProperties, ReactNode } from 'react';
import { Copy, Download } from 'lucide-react';
import CodeBlock, { type CodeLanguage } from './CodeBlock';
import IconButton from './IconButton';
import { copyText, downloadText } from '@/lib/util/download';

export interface CodeWindowProps {
  code: string;
  language?: CodeLanguage;
  /** File name shown in the window chrome, and the name used when downloading. */
  filename: string;
  /** Optional controls rendered after the copy and download buttons. */
  actions?: ReactNode;
  showLineNumbers?: boolean;
  /** Caps the height of the scrollable code area. */
  maxHeight?: number;
}

const DOT_COLORS = ['#ff5f56', '#ffbd2e', '#27c93f'];

/**
 * An editor-window frame (traffic lights, file name, copy and download) around a
 * syntax-highlighted listing. Used for the Python and TOML source of an example.
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
    <div className="code-window">
      <div className="code-window-bar">
        <div className="code-window-dots" aria-hidden="true">
          {DOT_COLORS.map(color => (
            <span key={color} style={{ color } as CSSProperties} />
          ))}
        </div>
        <span className="code-window-filename">{filename}</span>
        <IconButton label="Copy code" flashOnClick="Copied to clipboard" onClick={() => void copyText(code)}>
          <Copy size={14} />
        </IconButton>
        <IconButton label={`Download ${filename}`} onClick={() => downloadText(code, filename)}>
          <Download size={14} />
        </IconButton>
        {actions}
      </div>
      <div className="code-window-body" style={{ maxHeight }}>
        <CodeBlock code={code} language={language} showLineNumbers={showLineNumbers} />
      </div>
    </div>
  );
}
