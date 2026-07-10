'use client';

import React, { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import {
  ArrowLeft,
  EllipsisVertical,
  FolderOpen,
  Maximize2,
  Minimize2,
  PanelLeft,
  PanelRight,
  Play,
  Redo2,
  Square,
  Undo2,
} from 'lucide-react';
import { useStudioStore, useTemporalStore, resyncDerived } from '../../lib/studio/store';
import { useLayoutMode } from '../../lib/studio/useLayoutMode';
import { DiagnosticsPopover } from './DiagnosticsPopover';

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
  const leftDrawerOpen = useStudioStore((s) => s.ui.leftDrawerOpen);
  const setLeftDrawerOpen = useStudioStore((s) => s.setLeftDrawerOpen);

  const undo = useTemporalStore((s) => s.undo);
  const redo = useTemporalStore((s) => s.redo);
  const pastCount = useTemporalStore((s) => s.pastStates.length);
  const futureCount = useTemporalStore((s) => s.futureStates.length);

  const mode = useLayoutMode();
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!menuOpen) return;
    const onDown = (e: PointerEvent) => {
      if (!menuRef.current?.contains(e.target as Node)) setMenuOpen(false);
    };
    window.addEventListener('pointerdown', onDown);
    return () => window.removeEventListener('pointerdown', onDown);
  }, [menuOpen]);

  const running = status === 'running' || status === 'initializing';
  const isNative = solverType === 'operator_data';
  const canRun = !running && !invalid && !isNative;
  const runDisabledReason = invalid
    ? 'Fix the configuration problems first'
    : isNative
      ? 'Operator-data runs on the native driver only'
      : undefined;

  const secondaryButtons = (
    <>
      <button type="button" className="studio__btn" onClick={onOpenProjects}>
        <FolderOpen size={14} /> Projects
      </button>

      <button
        type="button"
        className="studio__btn studio__btn--icon"
        aria-label="Undo"
        title="Undo (config changes)"
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
        title="Fullscreen (f)"
        onClick={onToggleFullscreen}
      >
        {fullscreen ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
      </button>
    </>
  );

  return (
    <div className="studio__topbar">
      {mode === 'narrow' ? (
        <button
          type="button"
          className="studio__btn studio__btn--icon"
          aria-label="Toggle configuration panel"
          title="Configuration"
          onClick={() => setLeftDrawerOpen(!leftDrawerOpen)}
        >
          <PanelLeft size={15} />
        </button>
      ) : null}

      <Link href="/" className="studio__brand" title="Back to the Blaze2D site">
        <ArrowLeft size={13} className="studio__brand-back" />
        <span className="studio__brand-name">Workbench</span>
        <span className="studio__brand-sub">Blaze2D</span>
      </Link>

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
        {mode === 'narrow' ? (
          <div className="studio__diag-wrap" ref={menuRef}>
            <button
              type="button"
              className="studio__btn studio__btn--icon"
              aria-label="More actions"
              title="More actions"
              onClick={() => setMenuOpen((v) => !v)}
            >
              <EllipsisVertical size={15} />
            </button>
            {menuOpen ? (
              <div className="studio__addmenu studio__addmenu--topbar" onClick={() => setMenuOpen(false)}>
                {secondaryButtons}
              </div>
            ) : null}
          </div>
        ) : (
          secondaryButtons
        )}

        <DiagnosticsPopover />

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
            title={runDisabledReason ?? 'Run (Ctrl+Enter)'}
          >
            <Play size={13} /> Run
          </button>
        )}
      </div>
    </div>
  );
}
