'use client';

import React, { useEffect, useRef } from 'react';
import { EditorState, type Extension } from '@codemirror/state';
import {
  EditorView,
  keymap,
  lineNumbers,
  highlightActiveLine,
  highlightActiveLineGutter,
} from '@codemirror/view';
import { defaultKeymap, history, historyKeymap } from '@codemirror/commands';
import {
  syntaxHighlighting,
  HighlightStyle,
  StreamLanguage,
} from '@codemirror/language';
import { toml as tomlMode } from '@codemirror/legacy-modes/mode/toml';
import { linter, lintGutter, type Diagnostic as CmDiagnostic } from '@codemirror/lint';
import { tags as t } from '@lezer/highlight';
import type { Diagnostic } from '../../../lib/studio/utilWorker';

// A compact dark highlight style matching the site's editor look.
const highlight = HighlightStyle.define([
  { tag: [t.heading, t.tagName], color: '#6fb0f0' }, // [tables]
  { tag: t.propertyName, color: '#d0d0d0' }, // keys
  { tag: t.string, color: '#8fce8f' },
  { tag: [t.number, t.bool], color: '#f0a163' },
  { tag: t.comment, color: '#6c6c6c', fontStyle: 'italic' },
  { tag: t.punctuation, color: '#9a9a9a' },
]);

const editorTheme = EditorView.theme(
  {
    '&': { backgroundColor: 'transparent', color: '#cfcfcf', height: '100%' },
    '.cm-scroller': {
      fontFamily: 'var(--font-mono, ui-monospace, monospace)',
      fontSize: '12px',
      lineHeight: '1.55',
    },
    '.cm-content': { padding: '10px 0' },
    '.cm-gutters': {
      backgroundColor: 'transparent',
      color: '#555',
      border: 'none',
    },
    '.cm-activeLine': { backgroundColor: 'rgba(255,255,255,0.03)' },
    '.cm-activeLineGutter': { backgroundColor: 'rgba(255,255,255,0.03)' },
    '&.cm-focused': { outline: 'none' },
    '.cm-lintRange-error': {
      textDecoration: 'underline wavy #e06666',
      textDecorationSkipInk: 'none',
    },
    '.cm-diagnostic-error': { borderLeftColor: '#e06666' },
  },
  { dark: true },
);

export interface CodeMirrorEditorProps {
  value: string;
  onChange: (text: string) => void;
  onFocusChange: (focused: boolean) => void;
  /** Structured diagnostics (byte spans) to render as lint underlines. */
  diagnostics: Diagnostic[];
}

export function CodeMirrorEditor({
  value,
  onChange,
  onFocusChange,
  diagnostics,
}: CodeMirrorEditorProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const viewRef = useRef<EditorView | null>(null);
  const diagsRef = useRef<Diagnostic[]>(diagnostics);
  diagsRef.current = diagnostics;

  // Build the view once.
  useEffect(() => {
    if (!hostRef.current) return;

    const tomlLinter = linter((view): CmDiagnostic[] => {
      const docLen = view.state.doc.length;
      return diagsRef.current.map((d) => {
        let from = 0;
        let to = docLen;
        if (d.span) {
          from = Math.min(d.span[0], docLen);
          to = Math.min(Math.max(d.span[1], from + 1), docLen);
        } else {
          // No span: underline the first line.
          const firstLine = view.state.doc.line(1);
          from = firstLine.from;
          to = firstLine.to;
        }
        return {
          from,
          to,
          severity: 'error',
          message: d.path ? `${d.path}: ${d.message}` : d.message,
        };
      });
    });

    const extensions: Extension[] = [
      lineNumbers(),
      highlightActiveLine(),
      highlightActiveLineGutter(),
      history(),
      keymap.of([...defaultKeymap, ...historyKeymap]),
      StreamLanguage.define(tomlMode),
      syntaxHighlighting(highlight),
      lintGutter(),
      tomlLinter,
      editorTheme,
      EditorView.updateListener.of((update) => {
        if (update.docChanged) {
          onChange(update.state.doc.toString());
        }
        if (update.focusChanged) {
          onFocusChange(update.view.hasFocus);
        }
      }),
    ];

    const view = new EditorView({
      state: EditorState.create({ doc: value, extensions }),
      parent: hostRef.current,
    });
    viewRef.current = view;

    return () => {
      view.destroy();
      viewRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Push external value changes into the editor (only when they differ, to
  // avoid clobbering the user's cursor on round-tripped edits).
  useEffect(() => {
    const view = viewRef.current;
    if (!view) return;
    const current = view.state.doc.toString();
    if (current !== value) {
      view.dispatch({
        changes: { from: 0, to: current.length, insert: value },
      });
    }
  }, [value]);

  // Re-run linting when diagnostics change.
  useEffect(() => {
    const view = viewRef.current;
    if (!view) return;
    // A no-op dispatch forces the linter source to re-run.
    view.dispatch({});
  }, [diagnostics]);

  return <div ref={hostRef} style={{ height: '100%', overflow: 'hidden' }} />;
}
