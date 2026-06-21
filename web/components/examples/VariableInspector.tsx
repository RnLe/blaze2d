'use client';

import { useState } from 'react';
import { ChevronRight } from 'lucide-react';
import styles from './Examples.module.css';
import type { VariableEntry } from './types';

interface VariableInspectorProps {
  variables: VariableEntry[];
}

function valueType(value: unknown): string {
  if (Array.isArray(value)) {
    const first = value.find((item) => item !== undefined && item !== null);
    return `${valueType(first)}[]`;
  }
  if (value === null) return 'null';
  return typeof value;
}

function primitiveSummary(value: unknown): string {
  if (typeof value === 'number') return Number.isInteger(value) ? `${value}` : value.toPrecision(6);
  if (typeof value === 'string') return `"${value}"`;
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (value === null) return 'null';
  if (value === undefined) return 'undefined';
  return String(value);
}

function compactArray(value: unknown[]): string {
  if (value.length === 0) return '[]';
  const preview = value.length > 6 ? [...value.slice(0, 3), '...', ...value.slice(-3)] : value;
  return `[${preview
    .map((item) => (item === '...' ? '...' : Array.isArray(item) || typeof item === 'object' ? valueType(item) : primitiveSummary(item)))
    .join(', ')}]`;
}

function summary(value: unknown): string {
  if (Array.isArray(value)) return `${compactArray(value)} (${value.length})`;
  if (value && typeof value === 'object') return `{${Object.keys(value).slice(0, 4).join(', ')}}`;
  return primitiveSummary(value);
}

function TreeNode({
  name,
  value,
  type,
  depth = 0,
  defaultOpen = false,
}: {
  name: string;
  value: unknown;
  type?: string;
  depth?: number;
  defaultOpen?: boolean;
}) {
  const expandable = Boolean(value && typeof value === 'object');
  const [open, setOpen] = useState(defaultOpen && depth < 2);
  const inferredType = type ?? valueType(value);

  if (!expandable) {
    return (
      <div className={styles.node}>
        <span className={styles.nodeLabel}>{name}</span>
        <span className={styles.nodeType}>: {inferredType}</span>
        <span className={styles.nodeValue}> = {summary(value)}</span>
      </div>
    );
  }

  const entries = Array.isArray(value)
    ? value.map((item, index) => [String(index), item] as const)
    : Object.entries(value as Record<string, unknown>);

  return (
    <div className={styles.node}>
      <button className={styles.nodeButton} onClick={() => setOpen(!open)}>
        <ChevronRight size={14} style={{ transform: open ? 'rotate(90deg)' : 'none' }} />
        <span className={styles.nodeLabel}>{name}</span>
        <span className={styles.nodeType}>: {inferredType}</span>
        <span className={styles.nodeValue}> = {summary(value)}</span>
      </button>
      {open && (
        <div className={styles.children}>
          {entries.map(([childName, childValue]) => (
            <TreeNode key={childName} name={childName} value={childValue} depth={depth + 1} defaultOpen={depth < 1} />
          ))}
        </div>
      )}
    </div>
  );
}

export default function VariableInspector({ variables }: VariableInspectorProps) {
  if (variables.length === 0) return null;

  return (
    <div className={styles.inspector}>
      <div className={styles.inspectorHeader}>
        <span>Returned variables</span>
        <span>{variables.length}</span>
      </div>
      <div className={styles.tree}>
        {variables.map((variable) => (
          <TreeNode
            key={variable.name}
            name={variable.name}
            type={variable.type}
            value={variable.value}
            defaultOpen
          />
        ))}
      </div>
    </div>
  );
}
