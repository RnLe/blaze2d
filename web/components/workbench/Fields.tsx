'use client';
import { createContext, useContext, useEffect, useId, useState, type ReactNode } from 'react';

export const FieldDrafts = createContext<(id: string, dirty: boolean) => void>(() => {});
function useDraft(dirty: boolean) {
  const id = useId(), report = useContext(FieldDrafts);
  useEffect(() => { report(id, dirty); return () => report(id, false); }, [id, dirty, report]);
}
export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return <label className="wb-field"><span>{label}</span>{children}{hint && <small>{hint}</small>}</label>;
}
export function ValueField({ label, value, onCommit, numeric = false, optional = false, hint, displayLabel, displayPrecision }: {
  label: string; value: string | number | undefined | null; onCommit: (value: string) => void; numeric?: boolean; optional?: boolean; hint?: string; displayLabel?: string; displayPrecision?: number;
}) {
  const id = useId();
  const initial = String(typeof value === 'number' && displayPrecision ? Number(value.toPrecision(displayPrecision)) : value ?? ''), [text, setText] = useState(initial), [error, setError] = useState('');
  useDraft(text !== initial);
  useEffect(() => { setText(initial); setError(''); }, [initial]);
  function commit() {
    if (text === initial) return;
    if (numeric && !(optional && text.trim() === '') && (text.trim() === '' || !Number.isFinite(Number(text)))) {
      setError('Enter a finite number.'); return;
    }
    const next = numeric && text.trim() !== '' ? String(Number(text)) : text;
    setText(next); setError(''); if (next !== initial) onCommit(next);
  }
  return <Field label={displayLabel ?? label} hint={hint}><input aria-label={label} aria-describedby={error ? id : undefined} value={text} inputMode={numeric ? 'decimal' : undefined} aria-invalid={!!error}
    onChange={event => setText(event.target.value)} onBlur={commit}
    onKeyDown={event => { if (event.key === 'Enter') event.currentTarget.blur(); if (event.key === 'Escape') { setText(initial); setError(''); } }} />
    {error && <small id={id} role="alert">{error}</small>}</Field>;
}
export function PairField({ label, value, onCommit, axes = ['x', 'y'], hint }: {
  label: string; value: number[]; onCommit: (value: number[]) => void; axes?: [string, string]; hint?: string;
}) {
  return <fieldset className="wb-pair"><legend>{label}</legend><div className="wb-fields">
    {axes.map((axis, index) => <ValueField key={axis} label={`${label} ${axis}`} displayLabel={axis} displayPrecision={13} numeric value={value[index]}
      onCommit={text => { const next = [...value]; next[index] = Number(text); onCommit(next); }} />)}
  </div>{hint && <p className="wb-muted">{hint}</p>}</fieldset>;
}
export function JsonField({ label, value, onCommit, hint }: { label: string; value: unknown; onCommit: (value: unknown) => void; hint?: string }) {
  const initial = JSON.stringify(value), [text, setText] = useState(initial), [error, setError] = useState('');
  const id = useId(); useDraft(text !== initial);
  useEffect(() => { setText(initial); setError(''); }, [initial]);
  return <Field label={label} hint={hint}><textarea aria-label={label} rows={2} value={text} aria-invalid={!!error} aria-describedby={error ? id : undefined}
    onChange={event => setText(event.target.value)} onKeyDown={event => { if (event.key === 'Escape') { setText(initial); setError(''); } }} onBlur={() => {
      if (text === initial) return;
      try { const value: unknown = JSON.parse(text); setText(JSON.stringify(value)); setError(''); onCommit(value); }
      catch { setError('Enter valid JSON. Press Escape to restore the applied value.'); }
    }} />{error && <small id={id} role="alert">{error}</small>}</Field>;
}
export function Section({ title, children }: { title: string; children: ReactNode }) {
  return <section className="wb-section"><h2>{title}</h2>{children}</section>;
}
