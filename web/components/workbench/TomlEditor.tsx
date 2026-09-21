'use client';
import { useEffect, useRef } from 'react';
import { EditorState } from '@codemirror/state';
import { EditorView, keymap, lineNumbers } from '@codemirror/view';
import { defaultKeymap, history, historyKeymap } from '@codemirror/commands';
import { StreamLanguage, syntaxHighlighting, HighlightStyle } from '@codemirror/language';
import { tags } from '@lezer/highlight';
import { toml } from '@codemirror/legacy-modes/mode/toml';
import { lintGutter, setDiagnostics } from '@codemirror/lint';
import type { Diagnostic } from '@/lib/contract/generated';
import { editorSpan } from '@/lib/compute/editor';
import { syntax, theme } from '@/lib/theme';

export function TomlEditor({ value, diagnostics, onChange }: { value: string; diagnostics: Diagnostic[]; onChange: (text: string) => void }) {
  const host = useRef<HTMLDivElement>(null), view = useRef<EditorView | null>(null);
  const change = useRef(onChange), external = useRef(false); change.current = onChange;
  useEffect(() => {
    if (!host.current) return;
    const editor = new EditorView({ parent: host.current, state: EditorState.create({ doc: value, extensions: [
      lineNumbers(), history(), keymap.of([...defaultKeymap, ...historyKeymap]), StreamLanguage.define(toml),
      syntaxHighlighting(HighlightStyle.define([
        { tag: tags.comment, color: syntax.comment, fontStyle: 'italic' },
        { tag: tags.string, color: syntax.string },
        { tag: [tags.number, tags.bool, tags.atom], color: syntax.number },
        { tag: [tags.heading, tags.keyword, tags.tagName], color: syntax.keyword },
        { tag: [tags.propertyName, tags.attributeName], color: syntax.property },
        { tag: [tags.punctuation, tags.bracket], color: syntax.punctuation },
      ])), lintGutter(), EditorView.lineWrapping,
      EditorView.contentAttributes.of({ 'aria-label': 'Calculation TOML', role: 'textbox', 'aria-multiline': 'true', spellcheck: 'false' }),
      EditorView.theme({ '&': { background: theme.surface, color: theme.textPrimary, height: '100%' },
        '.cm-scroller': { overflow: 'auto', fontSize: '13px', fontFamily: 'var(--font-mono)', lineHeight: '1.65' },
        '.cm-gutters': { color: theme.textSubtle, background: theme.surface, border: 'none' }, '.cm-content': { padding: '12px 0' },
        '.cm-selectionBackground': { background: `${syntax.selection} !important` }, '&.cm-focused': { outline: `2px solid ${theme.accent}`, outlineOffset: '-2px' },
      }, { dark: true }),
      EditorView.updateListener.of(update => { if (update.docChanged && !external.current) change.current(update.state.doc.toString()); }),
    ] }) });
    editor.scrollDOM.tabIndex = 0;
    editor.scrollDOM.setAttribute('role', 'region');
    editor.scrollDOM.setAttribute('aria-label', 'TOML source viewport');
    view.current = editor;
    return () => { editor.destroy(); view.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  // The editor is created once; `value` seeds the initial document and later
  // changes arrive through this effect, which preserves cursor and selection.
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
