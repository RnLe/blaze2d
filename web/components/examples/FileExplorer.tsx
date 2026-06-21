'use client';

import { useState } from 'react';
import { Download, FileArchive } from 'lucide-react';
import IconButton from './IconButton';

export interface ExplorerFile {
  name: string;
  language: 'python' | 'toml';
}

export interface FileExplorerProps {
  files: ExplorerFile[];
  activeFile: string;
  onSelect: (name: string) => void;
  /** Folder label shown at the top of the tree. */
  folder?: string;
  /** Download the whole folder (zip when >1 file, the single file otherwise). */
  onDownloadAll?: () => void;
}

/** Small VS-Code-like file icon. */
function FileIcon({ language }: { language: ExplorerFile['language'] }) {
  const color = language === 'python' ? '#3b82f6' : '#f59e0b';
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden style={{ flexShrink: 0 }}>
      <path
        d="M6 2h8l4 4v16H6z"
        fill="none"
        stroke={color}
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
      <path d="M14 2v4h4" fill="none" stroke={color} strokeWidth="1.6" strokeLinejoin="round" />
    </svg>
  );
}

/**
 * FileExplorer — a mock IDE file tree. Lists the files that make up an example
 * (a single `.py`, optionally a `.toml` config) and lets the user switch the
 * active file shown in the code window.
 */
export default function FileExplorer({
  files,
  activeFile,
  onSelect,
  folder = 'example',
  onDownloadAll,
}: FileExplorerProps) {
  const multi = files.length > 1;
  return (
    <div
      style={{
        border: '1px solid #1f2937',
        borderRadius: '10px',
        background: '#0a0a0a',
        overflow: 'hidden',
        fontFamily: 'var(--font-mono, ui-monospace, monospace)',
        fontSize: '0.8rem',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          padding: '5px 8px 5px 12px',
          fontSize: '0.66rem',
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          color: '#6b7280',
          borderBottom: '1px solid #1f2937',
        }}
      >
        <span>Explorer</span>
        <div style={{ flex: 1 }} />
        {onDownloadAll && (
          <IconButton
            label={multi ? `Download all files as .zip` : `Download ${files[0]?.name ?? 'file'}`}
            onClick={onDownloadAll}
          >
            {multi ? <FileArchive size={13} /> : <Download size={13} />}
          </IconButton>
        )}
      </div>
      <div style={{ padding: '6px 4px' }}>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '4px 10px',
            color: '#9ca3af',
          }}
        >
          <span style={{ fontSize: '0.7rem' }}>▾</span>
          <span>{folder}/</span>
        </div>
        {files.map((f) => {
          const active = f.name === activeFile;
          return (
            <FileRow
              key={f.name}
              file={f}
              active={active}
              onSelect={() => onSelect(f.name)}
            />
          );
        })}
      </div>
    </div>
  );
}

function FileRow({
  file,
  active,
  onSelect,
}: {
  file: ExplorerFile;
  active: boolean;
  onSelect: () => void;
}) {
  const [hover, setHover] = useState(false);
  return (
    <button
      onClick={onSelect}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        width: '100%',
        textAlign: 'left',
        border: 'none',
        cursor: 'pointer',
        padding: '5px 10px 5px 26px',
        borderRadius: 6,
        color: active ? '#f3f4f6' : '#9ca3af',
        background: active ? '#1f2937' : hover ? '#111827' : 'transparent',
        fontFamily: 'inherit',
        fontSize: 'inherit',
        transition: 'background 0.12s',
      }}
    >
      <FileIcon language={file.language} />
      <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
        {file.name}
      </span>
    </button>
  );
}
