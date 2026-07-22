'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import './studio.css';

import { useStudioStore } from '../../lib/studio/store';
import { useStudioRunner } from '../../lib/studio/useStudioRunner';
import { useLayoutMode } from '../../lib/studio/useLayoutMode';
import { serializeConfig } from '../../lib/studio/tomlSerialize';
import { parseToml } from '../../lib/studio/tomlParse';
import { loadLastOpen, saveLastOpen } from '../../lib/studio/projects';
import { TopBar } from './TopBar';
import { ConfigPanel } from './panels/ConfigPanel';
import { CenterTabs } from './tabs/CenterTabs';
import { GeometryTab } from './tabs/GeometryTab';
import { ReciprocalTab } from './tabs/ReciprocalTab';
import { PlotsTab } from './tabs/PlotsTab';
import { DataTab } from './tabs/DataTab';
import { StatusStrip } from './tabs/StatusStrip';
import { TomlPane } from './toml/TomlPane';
import { ProjectsDialog } from './projects/ProjectsDialog';

export default function StudioApp() {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const bodyRef = useRef<HTMLDivElement | null>(null);

  const tomlText = useStudioStore((s) => s.toml.text);
  const paneFractions = useStudioStore((s) => s.ui.paneFractions);
  const centerTab = useStudioStore((s) => s.ui.centerTab);
  const tomlVisible = useStudioStore((s) => s.ui.tomlVisible);
  const setPaneFractions = useStudioStore((s) => s.setPaneFractions);
  const setCenterTab = useStudioStore((s) => s.setCenterTab);
  const setFullscreen = useStudioStore((s) => s.setFullscreen);
  const setConfig = useStudioStore((s) => s.setConfig);
  const applyParsedConfig = useStudioStore((s) => s.applyParsedConfig);
  const setProjectName = useStudioStore((s) => s.setProjectName);

  const { run, abort, validate } = useStudioRunner();
  const [projectsOpen, setProjectsOpen] = useState(false);
  const mode = useLayoutMode();
  const leftDrawerOpen = useStudioStore((s) => s.ui.leftDrawerOpen);
  const setLeftDrawerOpen = useStudioStore((s) => s.setLeftDrawerOpen);
  const setTomlVisible = useStudioStore((s) => s.setTomlVisible);

  // --- restore last-open project on mount ---
  useEffect(() => {
    const last = loadLastOpen();
    if (last) {
      setConfig(last.config, { markClean: true });
      setProjectName(last.name);
    }
    // Debug/testing affordance: the store is reachable from the console.
    (window as unknown as Record<string, unknown>).__studio = useStudioStore;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- debounced authoritative validation + two-way sync ---
  // While the TOML pane has focus, TOML wins: parse edits back into the config
  // so the canvas and controls track the text. When unfocused, the config is
  // authoritative and the text was derived from it (no parse-back needed).
  useEffect(() => {
    const id = setTimeout(() => {
      validate(tomlText);
      const state = useStudioStore.getState();
      if (state.toml.focused) {
        const parsed = parseToml(tomlText);
        if (parsed && serializeConfig(parsed) !== serializeConfig(state.config)) {
          applyParsedConfig(parsed);
        }
      }
    }, 250);
    return () => clearTimeout(id);
  }, [tomlText, validate, applyParsedConfig]);

  // --- debounced autosave (name + toml) ---
  const projectNameLive = useStudioStore((s) => s.project.name);
  useEffect(() => {
    const id = setTimeout(() => {
      saveLastOpen({ name: projectNameLive, toml: tomlText });
    }, 800);
    return () => clearTimeout(id);
  }, [tomlText, projectNameLive]);

  // In overlay-drawer layouts the TOML pane starts closed (it covers content).
  useEffect(() => {
    if (mode !== 'wide') setTomlVisible(false);
  }, [mode, setTomlVisible]);

  // --- fullscreen ---
  const toggleFullscreen = useCallback(() => {
    const el = rootRef.current;
    if (!el) return;
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => {});
    } else {
      el.requestFullscreen?.().catch(() => {});
    }
  }, []);

  useEffect(() => {
    const onFsChange = () => setFullscreen(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', onFsChange);
    return () => document.removeEventListener('fullscreenchange', onFsChange);
  }, [setFullscreen]);

  // --- run current config ---
  const handleRun = useCallback(() => {
    run(serializeConfig(useStudioStore.getState().config));
  }, [run]);

  // --- keyboard shortcuts ---
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      const typing =
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable ||
        Boolean(target.closest?.('.cm-editor'));
      const plainKey = !e.metaKey && !e.ctrlKey && !e.altKey;

      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        handleRun();
        return;
      }
      if (typing || !plainKey) return;

      if (e.key === 'f' || e.key === 'F') {
        e.preventDefault();
        toggleFullscreen();
      } else if (e.key >= '1' && e.key <= '4') {
        const tabs = ['geometry', 'reciprocal', 'plots', 'data'] as const;
        setCenterTab(tabs[Number(e.key) - 1]);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [toggleFullscreen, setCenterTab, handleRun]);

  // --- horizontal divider dragging ---
  const dragH = useRef<{ which: 'left' | 'right' } | null>(null);
  const onHDown = (which: 'left' | 'right') => (e: React.PointerEvent) => {
    dragH.current = { which };
    (e.target as Element).setPointerCapture?.(e.pointerId);
  };
  const onHMove = (e: React.PointerEvent) => {
    if (!dragH.current || !bodyRef.current) return;
    const rect = bodyRef.current.getBoundingClientRect();
    const [l, , r] = paneFractions;
    if (dragH.current.which === 'left') {
      const frac = Math.min(0.45, Math.max(0.15, (e.clientX - rect.left) / rect.width));
      setPaneFractions([frac, 1 - frac - r, r]);
    } else {
      const frac = Math.min(0.5, Math.max(0.16, (rect.right - e.clientX) / rect.width));
      setPaneFractions([l, 1 - l - frac, frac]);
    }
  };
  const onHUp = (e: React.PointerEvent) => {
    dragH.current = null;
    (e.target as Element).releasePointerCapture?.(e.pointerId);
  };

  const [leftFrac, , rightFrac] = paneFractions;
  const rightWidth = tomlVisible ? `${rightFrac * 100}%` : '0';

  return (
    <div className="studio" ref={rootRef}>
      <div className="studio__mobile-note">
        The workbench is designed for a wide screen. Some panels are stacked on narrow displays.
      </div>

      <TopBar
        onRun={handleRun}
        onAbort={abort}
        onToggleFullscreen={toggleFullscreen}
        onOpenProjects={() => setProjectsOpen(true)}
      />

      {projectsOpen ? <ProjectsDialog onClose={() => setProjectsOpen(false)} /> : null}

      <div
        className="studio__body"
        ref={bodyRef}
        onPointerMove={(e) => {
          onHMove(e);
        }}
        onPointerUp={onHUp}
      >
        {/* Left: config editor (column when wide/mid, drawer when narrow) */}
        {mode !== 'narrow' ? (
          <>
            <div className="studio__col studio__col--left" style={{ flex: `0 0 ${leftFrac * 100}%` }}>
              <ConfigPanel />
            </div>
            <div className="studio__divider" onPointerDown={onHDown('left')} onPointerUp={onHUp} />
          </>
        ) : null}

        {/* Center: full-height tab views over the status strip */}
        <div className="studio__col studio__col--center" style={{ flex: 1 }}>
          <CenterTabs />
          <div className="studio__tabhost">
            {centerTab === 'geometry' ? <GeometryTab /> : null}
            {centerTab === 'reciprocal' ? <ReciprocalTab /> : null}
            {centerTab === 'plots' ? <PlotsTab onRun={handleRun} /> : null}
            {centerTab === 'data' ? <DataTab /> : null}
          </div>
          <StatusStrip />
        </div>

        {/* Right: TOML pane (column when wide, drawer otherwise) */}
        {mode === 'wide' && tomlVisible ? (
          <>
            <div className="studio__divider" onPointerDown={onHDown('right')} onPointerUp={onHUp} />
            <div
              className="studio__col studio__col--right"
              style={{ flex: `0 0 ${rightWidth}` }}
            >
              <TomlPane />
            </div>
          </>
        ) : null}
      </div>

      {/* Overlay drawers for narrow layouts */}
      {mode !== 'wide' && tomlVisible ? (
        <>
          <div className="studio__scrim" onClick={() => setTomlVisible(false)} />
          <div className="studio__drawer studio__drawer--right">
            <TomlPane />
          </div>
        </>
      ) : null}
      {mode === 'narrow' && leftDrawerOpen ? (
        <>
          <div className="studio__scrim" onClick={() => setLeftDrawerOpen(false)} />
          <div className="studio__drawer studio__drawer--left">
            <ConfigPanel />
          </div>
        </>
      ) : null}
    </div>
  );
}
