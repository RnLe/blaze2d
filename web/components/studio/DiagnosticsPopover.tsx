'use client';

import React, { useEffect, useRef, useState } from 'react';
import { CircleAlert, CircleCheck } from 'lucide-react';
import { useStudioStore } from '../../lib/studio/store';
import { sectionOf } from './panels/ConfigPanel';

/**
 * The single validation surface in the top bar: a chip that reads
 * "valid" / "N problems"; clicking it lists every diagnostic, and clicking
 * a diagnostic opens + scrolls to the config section it belongs to.
 */
export function DiagnosticsPopover() {
  const diagnostics = useStudioStore((s) => s.toml.diagnostics);
  const invalid = useStudioStore((s) => s.toml.invalid);
  const setAccordion = useStudioStore((s) => s.setAccordion);
  const setLeftDrawerOpen = useStudioStore((s) => s.setLeftDrawerOpen);
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement | null>(null);

  const problems = diagnostics.length;

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener('pointerdown', onDown);
    return () => window.removeEventListener('pointerdown', onDown);
  }, [open]);

  useEffect(() => {
    if (problems === 0) setOpen(false);
  }, [problems]);

  const jumpTo = (path: string) => {
    setOpen(false);
    const section = sectionOf(path);
    if (!section) return;
    setAccordion(section, true);
    setLeftDrawerOpen(true);
    requestAnimationFrame(() => {
      document
        .querySelector(`[data-section="${section}"]`)
        ?.scrollIntoView({ block: 'start', behavior: 'smooth' });
    });
  };

  if (problems === 0 && !invalid) {
    return (
      <span className="studio__status-chip studio__status-chip--ok" title="Configuration is valid">
        <CircleCheck size={12} /> valid
      </span>
    );
  }

  return (
    <div className="studio__diag-wrap" ref={wrapRef}>
      <button
        type="button"
        className="studio__status-chip studio__status-chip--error studio__status-chip--btn"
        onClick={() => setOpen((v) => !v)}
        title="Show configuration problems"
      >
        <CircleAlert size={12} />
        {problems > 0 ? `${problems} problem${problems === 1 ? '' : 's'}` : 'invalid'}
      </button>
      {open ? (
        <div className="studio__diag-pop">
          {diagnostics.length === 0 ? (
            <div className="studio__diag-item studio__diag-item--static">
              The TOML text does not parse. Fix it in the editor or revert.
            </div>
          ) : (
            diagnostics.map((d, i) => (
              <button key={i} type="button" className="studio__diag-item" onClick={() => jumpTo(d.path)}>
                <span className="studio__diag-path">{d.path}</span>
                <span className="studio__diag-msg">{d.message}</span>
              </button>
            ))
          )}
        </div>
      ) : null}
    </div>
  );
}
