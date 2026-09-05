'use client';
import { useEffect, useRef } from 'react';
import { EditorState } from '@codemirror/state';
import { EditorView, keymap, lineNumbers } from '@codemirror/view';
import { defaultKeymap, history, historyKeymap } from '@codemirror/commands';
import { StreamLanguage, syntaxHighlighting, defaultHighlightStyle } from '@codemirror/language';
import { toml } from '@codemirror/legacy-modes/mode/toml';
import { lintGutter, setDiagnostics } from '@codemirror/lint';
import type { Diagnostic } from '../../lib/contract/generated';
import { editorSpan } from '../../lib/compute/editor';

export function TomlEditor({ value, diagnostics, onChange }: { value: string; diagnostics: Diagnostic[]; onChange: (text: string) => void }) {
  const host = useRef<HTMLDivElement>(null), view = useRef<EditorView | null>(null);
  const change = useRef(onChange), external = useRef(false); change.current = onChange;
  useEffect(() => {
    if (!host.current) return;
    const editor = new EditorView({ parent: host.current, state: EditorState.create({ doc: value, extensions: [
      lineNumbers(), history(), keymap.of([...defaultKeymap, ...historyKeymap]), StreamLanguage.define(toml),
      syntaxHighlighting(defaultHighlightStyle), lintGutter(), EditorView.lineWrapping,
      EditorView.contentAttributes.of({ 'aria-label': 'Calculation TOML', spellcheck: 'false' }),
      EditorView.theme({ '&': { background: '#0b1111', color: '#d7e3df', height: '100%' },
        '.cm-scroller': { overflow: 'auto', fontSize: '13px', fontFamily: 'ui-monospace, monospace', lineHeight: '1.65' },
        '.cm-gutters': { color: '#99aaa5', background: '#0b1111', border: 'none' }, '.cm-content': { padding: '12px 0' },
        '.cm-selectionBackground': { background: '#345951 !important' }, '&.cm-focused': { outline: '2px solid #84d4bd', outlineOffset: '-2px' },
      }, { dark: true }),
      EditorView.updateListener.of(update => { if (update.docChanged && !external.current) change.current(update.state.doc.toString()); }),
    ] }) });
    view.current = editor;
    return () => { editor.destroy(); view.current = null; };
  }, []); // The current callback is kept in a ref; text updates preserve the editor and selection.
  useEffect(() => {
    const editor = view.current;
    if (editor && value !== editor.state.doc.toString()) {
      external.current = true;
      editor.dispatch({ changes: { from: 0, to: editor.state.doc.length, insert: value } });
      external.current = false;
    }
  }, [value]);
  useEffect(() => {
    const editor = view.current;
    if (!editor) return;
    editor.dispatch(setDiagnostics(editor.state, diagnostics.map(diagnostic => {
      const [from, to] = diagnostic.span ? editorSpan(value, diagnostic.span) : [0, editor.state.doc.line(1).to];
      return { from, to, severity: 'error', message: diagnostic.path ? `${diagnostic.path}: ${diagnostic.message}` : diagnostic.message };
    })));
  }, [diagnostics, value]);
  return <div className="wb-editor" ref={host} />;
}
