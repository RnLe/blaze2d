'use client';

import { useState } from 'react';
import { Copy, Download } from 'lucide-react';
import IconButton from './IconButton';
import { copyText, downloadJson } from '../../lib/util/download';

/**
 * DataInspector — a typed, expandable tree view for inspecting computation
 * results in the browser. Designed to make WASM/Python-aligned result objects
 * (e.g. a `BandResult` or a streaming dict) feel like inspecting a typed
 * variable in a REPL: named fields with type badges, fully expandable
 * dictionaries, and a compact "first 3 … last 3" view for large arrays.
 */

const COLORS = {
  border: '#1f2937',
  bg: '#0a0a0a',
  rowHover: '#111827',
  name: '#93c5fd',
  type: '#6b7280',
  string: '#86efac',
  number: '#fbbf24',
  boolean: '#c084fc',
  null: '#6b7280',
  toggle: '#9ca3af',
  shape: '#5eead4',
};

interface NodeProps {
  name: string;
  value: unknown;
  depth: number;
  defaultOpen?: boolean;
}

function isNumberArray(v: unknown): v is number[] {
  return Array.isArray(v) && v.length > 0 && v.every((x) => typeof x === 'number');
}

function isMatrix(v: unknown): v is number[][] {
  return (
    Array.isArray(v) &&
    v.length > 0 &&
    v.every((row) => Array.isArray(row) && row.every((x) => typeof x === 'number'))
  );
}

function numberType(n: number): string {
  return Number.isInteger(n) ? 'int' : 'float';
}

function fmtNumber(n: number): string {
  if (Number.isInteger(n)) return String(n);
  if (Math.abs(n) !== 0 && (Math.abs(n) < 1e-3 || Math.abs(n) >= 1e6)) {
    return n.toExponential(4);
  }
  return n.toFixed(6).replace(/0+$/, '').replace(/\.$/, '.0');
}

function typeLabel(value: unknown): string {
  if (value === null || value === undefined) return 'None';
  if (typeof value === 'string') return 'str';
  if (typeof value === 'boolean') return 'bool';
  if (typeof value === 'number') return numberType(value);
  if (isMatrix(value)) {
    const rows = value.length;
    const cols = (value as number[][])[0].length;
    return `ndarray (${rows}, ${cols})`;
  }
  if (isNumberArray(value)) return `ndarray (${value.length},)`;
  if (Array.isArray(value)) return `list[${value.length}]`;
  return 'dict';
}

const indentStep = 16;

function Badge({ label }: { label: string }) {
  return (
    <span
      style={{
        color: COLORS.shape,
        fontSize: '0.72rem',
        fontFamily: 'var(--font-mono, monospace)',
        marginLeft: '0.5rem',
        opacity: 0.85,
      }}
    >
      {label}
    </span>
  );
}

function PrimitiveValue({ value }: { value: unknown }) {
  if (value === null || value === undefined) {
    return <span style={{ color: COLORS.null }}>None</span>;
  }
  if (typeof value === 'string') {
    return <span style={{ color: COLORS.string }}>&quot;{value}&quot;</span>;
  }
  if (typeof value === 'boolean') {
    return <span style={{ color: COLORS.boolean }}>{value ? 'True' : 'False'}</span>;
  }
  if (typeof value === 'number') {
    return <span style={{ color: COLORS.number }}>{fmtNumber(value)}</span>;
  }
  return <span>{String(value)}</span>;
}

/** Compact preview of a numeric array: first 3 … last 3. */
function ArrayPreview({ arr }: { arr: number[] }) {
  if (arr.length <= 8) {
    return (
      <span style={{ color: COLORS.number }}>
        [{arr.map(fmtNumber).join(', ')}]
      </span>
    );
  }
  const head = arr.slice(0, 3).map(fmtNumber).join(', ');
  const tail = arr.slice(-3).map(fmtNumber).join(', ');
  return (
    <span style={{ color: COLORS.number }}>
      [{head}, <span style={{ color: COLORS.type }}>… {arr.length - 6} more …</span>, {tail}]
    </span>
  );
}

function Row({
  depth,
  children,
}: {
  depth: number;
  children: React.ReactNode;
}) {
  const [hover, setHover] = useState(false);
  return (
    <div
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        display: 'flex',
        alignItems: 'flex-start',
        padding: '2px 8px',
        paddingLeft: `${8 + depth * indentStep}px`,
        background: hover ? COLORS.rowHover : 'transparent',
        borderRadius: '4px',
        fontFamily: 'var(--font-mono, monospace)',
        fontSize: '0.82rem',
        lineHeight: 1.6,
      }}
    >
      {children}
    </div>
  );
}

function Name({ name }: { name: string }) {
  return <span style={{ color: COLORS.name }}>{name}</span>;
}

function InspectorNode({ name, value, depth, defaultOpen = false }: NodeProps) {
  const [open, setOpen] = useState(defaultOpen);

  // Numeric array (ndarray 1D) — expandable to show all values.
  if (isNumberArray(value)) {
    const arr = value as number[];
    const expandable = arr.length > 8;
    return (
      <div>
        <Row depth={depth}>
          {expandable ? (
            <button
              onClick={() => setOpen((o) => !o)}
              style={toggleBtnStyle}
              aria-label={open ? 'Collapse' : 'Expand'}
            >
              {open ? '▾' : '▸'}
            </button>
          ) : (
            <span style={{ display: 'inline-block', width: '1.1em' }} />
          )}
          <Name name={name} />
          <span style={{ color: COLORS.type, margin: '0 0.4rem' }}>:</span>
          {open ? (
            <span style={{ color: COLORS.shape }}>ndarray</span>
          ) : (
            <ArrayPreview arr={arr} />
          )}
          <Badge label={typeLabel(value)} />
        </Row>
        {open &&
          arr.map((v, i) => (
            <Row depth={depth + 1} key={i}>
              <span style={{ display: 'inline-block', width: '1.1em' }} />
              <span style={{ color: COLORS.type }}>[{i}]</span>
              <span style={{ color: COLORS.type, margin: '0 0.4rem' }}>:</span>
              <PrimitiveValue value={v} />
            </Row>
          ))}
      </div>
    );
  }

  // 2D numeric array (ndarray) — expandable to rows, each row expandable.
  if (isMatrix(value)) {
    const mat = value as number[][];
    return (
      <div>
        <Row depth={depth}>
          <button onClick={() => setOpen((o) => !o)} style={toggleBtnStyle}>
            {open ? '▾' : '▸'}
          </button>
          <Name name={name} />
          <span style={{ color: COLORS.type, margin: '0 0.4rem' }}>:</span>
          <span style={{ color: COLORS.shape }}>ndarray</span>
          <Badge label={typeLabel(value)} />
        </Row>
        {open &&
          (mat.length > 12
            ? [
                ...mat.slice(0, 3).map((row, i) => [i, row] as const),
                ...mat.slice(-3).map((row, i) => [mat.length - 3 + i, row] as const),
              ]
            : mat.map((row, i) => [i, row] as const)
          ).map(([i, row], idx, list) => {
            const gap = idx > 0 && (list[idx][0] as number) - (list[idx - 1][0] as number) > 1;
            return (
              <div key={i}>
                {gap && (
                  <Row depth={depth + 1}>
                    <span style={{ color: COLORS.type, marginLeft: '1.1em' }}>
                      … {mat.length - 6} more rows …
                    </span>
                  </Row>
                )}
                <InspectorNode name={`[${i}]`} value={row} depth={depth + 1} />
              </div>
            );
          })}
      </div>
    );
  }

  // Generic list (non-numeric) — expandable.
  if (Array.isArray(value)) {
    return (
      <div>
        <Row depth={depth}>
          <button onClick={() => setOpen((o) => !o)} style={toggleBtnStyle}>
            {open ? '▾' : '▸'}
          </button>
          <Name name={name} />
          <span style={{ color: COLORS.type, margin: '0 0.4rem' }}>:</span>
          <Badge label={typeLabel(value)} />
        </Row>
        {open &&
          value.map((v, i) => (
            <InspectorNode key={i} name={`[${i}]`} value={v} depth={depth + 1} />
          ))}
      </div>
    );
  }

  // Object / dict — fully expandable.
  if (value !== null && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>);
    return (
      <div>
        <Row depth={depth}>
          <button onClick={() => setOpen((o) => !o)} style={toggleBtnStyle}>
            {open ? '▾' : '▸'}
          </button>
          <Name name={name} />
          <span style={{ color: COLORS.type, margin: '0 0.4rem' }}>:</span>
          <Badge label={`dict (${entries.length})`} />
        </Row>
        {open &&
          entries.map(([k, v]) => (
            <InspectorNode key={k} name={k} value={v} depth={depth + 1} />
          ))}
      </div>
    );
  }

  // Primitive.
  return (
    <Row depth={depth}>
      <span style={{ display: 'inline-block', width: '1.1em' }} />
      <Name name={name} />
      <span style={{ color: COLORS.type, margin: '0 0.4rem' }}>:</span>
      <PrimitiveValue value={value} />
      <Badge label={typeLabel(value)} />
    </Row>
  );
}

const toggleBtnStyle: React.CSSProperties = {
  background: 'none',
  border: 'none',
  color: COLORS.toggle,
  cursor: 'pointer',
  padding: 0,
  width: '1.1em',
  fontSize: '0.8rem',
  lineHeight: 'inherit',
  fontFamily: 'inherit',
};

export interface DataInspectorProps {
  /** The value to inspect (object, array, BandResult, streaming dict, …). */
  data: unknown;
  /** Display name for the root variable. */
  rootName?: string;
  /** A short type annotation shown next to the root (e.g. "BandResult"). */
  rootType?: string;
  /** Whether the root node starts expanded. */
  defaultOpen?: boolean;
}

export default function DataInspector({
  data,
  rootName = 'result',
  rootType,
  defaultOpen = true,
}: DataInspectorProps) {
  const [open, setOpen] = useState(defaultOpen);
  const isContainer = data !== null && typeof data === 'object';
  const entries = isContainer && !Array.isArray(data)
    ? Object.entries(data as Record<string, unknown>)
    : null;

  return (
    <div
      style={{
        border: `1px solid ${COLORS.border}`,
        borderRadius: '10px',
        background: COLORS.bg,
        overflow: 'hidden',
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
          borderBottom: `1px solid ${COLORS.border}`,
        }}
      >
        <span>Result</span>
        <div style={{ flex: 1 }} />
        <IconButton
          label="Copy result as JSON"
          flashOnClick="Copied to clipboard"
          onClick={() => copyText(serializeJson(data))}
        >
          <Copy size={13} />
        </IconButton>
        <IconButton
          label={`Download ${rootName}.json`}
          onClick={() => downloadJson(data, `${rootName}.json`)}
        >
          <Download size={13} />
        </IconButton>
      </div>
      <div style={{ padding: '8px 4px', overflowX: 'auto' }}>
        <Row depth={0}>
          {isContainer ? (
            <button onClick={() => setOpen((o) => !o)} style={toggleBtnStyle}>
              {open ? '▾' : '▸'}
            </button>
          ) : (
            <span style={{ display: 'inline-block', width: '1.1em' }} />
          )}
          <Name name={rootName} />
          <span style={{ color: COLORS.type, margin: '0 0.4rem' }}>:</span>
          <Badge label={rootType ?? typeLabel(data)} />
        </Row>
        {open &&
          (entries
            ? entries.map(([k, v]) => (
                <InspectorNode key={k} name={k} value={v} depth={1} />
              ))
            : Array.isArray(data)
              ? (data as unknown[]).map((v, i) => (
                  <InspectorNode key={i} name={`[${i}]`} value={v} depth={1} />
                ))
              : null)}
      </div>
    </div>
  );
}

function serializeJson(v: unknown): string {
  return JSON.stringify(
    v,
    (_k, val) => (typeof val === 'number' && !Number.isFinite(val) ? String(val) : val),
    2,
  );
}
