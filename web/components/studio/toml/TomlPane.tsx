'use client';

import React from 'react';
import { FileCode2, Copy, Download, RotateCcw } from 'lucide-react';
import { useStudioStore } from '../../../lib/studio/store';
import { serializeConfig } from '../../../lib/studio/tomlSerialize';
import { copyText, downloadText } from '../../../lib/util/download';
import { CodeMirrorEditor } from './CodeMirrorEditor';

/**
 * Editable TOML pane (CodeMirror 6) with a lint gutter fed by the authoritative
 * WASM diagnostics, plus a live validation + estimate footer.
 *
 * Two-way sync: user edits flow to the store via setTomlText (parsed back to a
 * config by the debounced sync in StudioApp). UI edits re-derive the text; the
 * editor only replaces its doc when the incoming text actually differs, so the
 * cursor is preserved on round-trips.
 */
export function TomlPane() {
  const text = useStudioStore((s) => s.toml.text);
  const summary = useStudioStore((s) => s.toml.summary);
  const diagnostics = useStudioStore((s) => s.toml.diagnostics);
  const invalid = useStudioStore((s) => s.toml.invalid);
  const focused = useStudioStore((s) => s.toml.focused);
  const projectName = useStudioStore((s) => s.project.name);

  const setTomlText = useStudioStore((s) => s.setTomlText);
  const setTomlFocused = useStudioStore((s) => s.setTomlFocused);

  const fileBase = projectName.trim().replace(/[^\w-]+/g, '_').toLowerCase() || 'crystal';

  const revert = () => {
    // Discard invalid text by re-deriving from the last-good config.
    const good = serializeConfig(useStudioStore.getState().config);
    setTomlText(good);
  };

  return (
    <div className="studio__toml">
      <div className="studio__panel-title" style={{ justifyContent: 'space-between' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <FileCode2 size={13} /> TOML
        </span>
        <span style={{ display: 'flex', gap: 4 }}>
          {invalid ? (
            <button
              type="button"
              className="studio__iconbtn"
              aria-label="Revert to last valid"
              title="Revert to last valid config"
              onClick={revert}
            >
              <RotateCcw size={14} />
            </button>
          ) : null}
          <button
            type="button"
            className="studio__iconbtn"
            aria-label="Copy TOML"
            onClick={() => copyText(text)}
          >
            <Copy size={14} />
          </button>
          <button
            type="button"
            className="studio__iconbtn"
            aria-label="Download TOML"
            onClick={() => downloadText(text, `${fileBase}.toml`)}
          >
            <Download size={14} />
          </button>
        </span>
      </div>

      <div className="studio__toml-editor">
        <CodeMirrorEditor
          value={text}
          onChange={setTomlText}
          onFocusChange={setTomlFocused}
          diagnostics={invalid ? diagnostics : []}
        />
      </div>

      {invalid && focused ? (
        <div className="studio__toml-banner">
          TOML has errors. The canvas and estimates show the last valid state.
        </div>
      ) : null}

      {/* Estimates live in the status strip now; this footer stays minimal. */}
      <div className="studio__toml-footer">
        <span className="studio__toml-footer-item">
          {invalid ? (
            <span className="studio__toml-err">invalid</span>
          ) : (
            <span className="studio__toml-ok">valid</span>
          )}
        </span>
        {summary ? (
          <span className="studio__toml-footer-item">
            grid <b>{summary.nx}×{summary.ny}</b> · bands <b>{summary.n_bands}</b> · <b>{summary.precision}</b>
          </span>
        ) : null}
      </div>
    </div>
  );
}
