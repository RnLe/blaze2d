'use client';

import React from 'react';
import { Play, Square, Maximize2, Minimize2, PanelRight, FolderOpen } from 'lucide-react';
import { useStudioStore, useTemporalStore, resyncDerived } from '../../lib/studio/store';
import { Undo2, Redo2 } from 'lucide-react';

export function TopBar({
  onRun,
  onAbort,
  onToggleFullscreen,
  onOpenProjects,
}: {
  onRun: () => void;
  onAbort: () => void;
  onToggleFullscreen: () => void;
  onOpenProjects: () => void;
}) {
  const projectName = useStudioStore((s) => s.project.name);
  const dirty = useStudioStore((s) => s.project.dirty);
  const setProjectName = useStudioStore((s) => s.setProjectName);
  const status = useStudioStore((s) => s.live.status);
  const invalid = useStudioStore((s) => s.toml.invalid);
  const solverType = useStudioStore((s) => s.config.solver.type);
  const tomlVisible = useStudioStore((s) => s.ui.tomlVisible);
  const setTomlVisible = useStudioStore((s) => s.setTomlVisible);
  const fullscreen = useStudioStore((s) => s.ui.fullscreen);

  const undo = useTemporalStore((s) => s.undo);
  const redo = useTemporalStore((s) => s.redo);
  const pastCount = useTemporalStore((s) => s.pastStates.length);
  const futureCount = useTemporalStore((s) => s.futureStates.length);

  const running = status === 'running' || status === 'initializing';
  const isNative = solverType === 'operator_data';
  const canRun = !running && !invalid && !isNative;

  const statusChip = (() => {
    if (running) return <span className="studio__status-chip studio__status-chip--running">running</span>;
    if (status === 'done') return <span className="studio__status-chip studio__status-chip--done">done</span>;
    if (status === 'error') return <span className="studio__status-chip studio__status-chip--error">error</span>;
    if (invalid) return <span className="studio__status-chip studio__status-chip--error">invalid config</span>;
    return null;
  })();

  return (
    <div className="studio__topbar">
      <div className="studio__brand">
        <span className="studio__brand-name">Workbench</span>
        <span className="studio__brand-sub">Blaze2D</span>
      </div>

      <div className="studio__project">
        <input
          className="studio__project-input"
          value={projectName}
          spellCheck={false}
          aria-label="Project name"
          onChange={(e) => setProjectName(e.target.value)}
        />
        <span className={`studio__dirty${dirty ? ' studio__dirty--on' : ''}`} title="Unsaved changes" />
      </div>

      <div className="studio__spacer" />

      <div className="studio__topbar-group">
        <button type="button" className="studio__btn" onClick={onOpenProjects}>
          <FolderOpen size={14} /> Projects
        </button>

        <button
          type="button"
          className="studio__btn studio__btn--icon"
          aria-label="Undo"
          title="Undo"
          disabled={pastCount === 0}
          onClick={() => {
            undo();
            resyncDerived();
          }}
        >
          <Undo2 size={15} />
        </button>
        <button
          type="button"
          className="studio__btn studio__btn--icon"
          aria-label="Redo"
          title="Redo"
          disabled={futureCount === 0}
          onClick={() => {
            redo();
            resyncDerived();
          }}
        >
          <Redo2 size={15} />
        </button>

        <button
          type="button"
          className="studio__btn studio__btn--icon"
          aria-label="Toggle TOML pane"
          title="Toggle TOML pane"
          onClick={() => setTomlVisible(!tomlVisible)}
        >
          <PanelRight size={15} />
        </button>

        <button
          type="button"
          className="studio__btn studio__btn--icon"
          aria-label="Toggle fullscreen"
          title="Fullscreen"
          onClick={onToggleFullscreen}
        >
          {fullscreen ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
        </button>

        {statusChip}

        {running ? (
          <button type="button" className="studio__btn studio__btn--abort" onClick={onAbort}>
            <Square size={13} /> Abort
          </button>
        ) : (
          <button
            type="button"
            className="studio__btn studio__btn--run"
            disabled={!canRun}
            onClick={onRun}
            title={isNative ? 'Operator-data runs on the native driver only' : undefined}
          >
            <Play size={13} /> Run
          </button>
        )}
      </div>
    </div>
  );
}
