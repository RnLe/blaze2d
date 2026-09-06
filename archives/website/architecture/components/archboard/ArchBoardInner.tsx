'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { getAssetPath } from '../../lib/paths';
import { BoardApp } from './BoardApp';
import { useBoardStore, type BoardView } from './store';
import Toolbar from './Toolbar';
import Legend from './Legend';
import Tooltip from './Tooltip';
import DetailPanel from './DetailPanel';
import './archboard.css';

export default function ArchBoardInner() {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const canvasHostRef = useRef<HTMLDivElement>(null);
  const boardRef = useRef<BoardApp | null>(null);
  const [error, setError] = useState<string | null>(null);
  const ready = useBoardStore((s) => s.ready);
  const view = useBoardStore((s) => s.view);
  const stageLabel = useBoardStore((s) => s.stageLabel);
  const caption = useBoardStore((s) => s.caption);
  const [fullscreenEnabled, setFullscreenEnabled] = useState(false);

  useEffect(() => {
    setFullscreenEnabled(typeof document !== 'undefined' && Boolean(document.fullscreenEnabled));
  }, []);

  /* ------------------------------- lifecycle ------------------------------ */
  useEffect(() => {
    const host = canvasHostRef.current;
    if (!host) return;
    let cancelled = false;
    let board: BoardApp | null = null;
    let resizeObserver: ResizeObserver | null = null;
    let intersection: IntersectionObserver | null = null;

    (async () => {
      let benchJson: unknown = null;
      try {
        const res = await fetch(getAssetPath('/data/benchmarks/series4-iterations.json'));
        if (res.ok) benchJson = await res.json();
      } catch {
        /* synthetic fallback inside BoardApp */
      }
      if (cancelled) return;

      const debugGrid =
        typeof window !== 'undefined' && new URLSearchParams(window.location.search).get('grid') === '1';

      try {
        board = await BoardApp.create(host, { benchJson, debugGrid });
      } catch (e) {
        if (!cancelled) setError(String(e));
        return;
      }
      if (cancelled) {
        board.destroy();
        return;
      }
      boardRef.current = board;

      resizeObserver = new ResizeObserver(() => board?.resize());
      resizeObserver.observe(host);

      intersection = new IntersectionObserver(
        (entries) => board?.setRunning(entries[0]?.isIntersecting ?? true),
        { threshold: 0.02 }
      );
      intersection.observe(host);

      // Deep links: ?view=precision&node=lob.upcast
      const params = new URLSearchParams(window.location.search);
      const viewParam = params.get('view') as BoardView | null;
      if (viewParam && ['layers', 'run', 'precision'].includes(viewParam)) board.setView(viewParam);
      const nodeParam = params.get('node');
      if (nodeParam) board.selectNode(nodeParam, true);
    })();

    const onFullscreenChange = () => {
      useBoardStore.getState().setFullscreen(Boolean(document.fullscreenElement));
      requestAnimationFrame(() => boardRef.current?.resize());
    };
    document.addEventListener('fullscreenchange', onFullscreenChange);

    return () => {
      cancelled = true;
      document.removeEventListener('fullscreenchange', onFullscreenChange);
      resizeObserver?.disconnect();
      intersection?.disconnect();
      board?.destroy();
      boardRef.current = null;
      useBoardStore.getState().setReady(false);
      useBoardStore.getState().setSelected(null);
      useBoardStore.getState().setHover(null);
    };
  }, []);

  /* -------------------------------- handlers ------------------------------ */
  const handlePlayPause = useCallback(() => {
    const board = boardRef.current;
    if (!board) return;
    if (useBoardStore.getState().playing) board.pause();
    else board.play();
  }, []);

  const handleFullscreen = useCallback(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;
    if (document.fullscreenElement) void document.exitFullscreen();
    else void wrapper.requestFullscreen();
  }, []);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      const board = boardRef.current;
      if (!board) return;
      switch (e.key) {
        case 'Escape':
          if (useBoardStore.getState().selected) board.selectNode(null);
          break;
        case '0':
          board.fitAll();
          break;
        case '+':
        case '=':
          board.zoomBy(1.25);
          break;
        case '-':
          board.zoomBy(0.8);
          break;
        case 'ArrowLeft':
          board.panBy(-80, 0);
          e.preventDefault();
          break;
        case 'ArrowRight':
          board.panBy(80, 0);
          e.preventDefault();
          break;
        case 'ArrowUp':
          board.panBy(0, -80);
          e.preventDefault();
          break;
        case 'ArrowDown':
          board.panBy(0, 80);
          e.preventDefault();
          break;
        case 'f':
          handleFullscreen();
          break;
        case ' ':
          if (useBoardStore.getState().view === 'run') {
            handlePlayPause();
            e.preventDefault();
          }
          break;
      }
    },
    [handleFullscreen, handlePlayPause]
  );

  return (
    <div ref={wrapperRef} className="archboard" tabIndex={0} onKeyDown={handleKeyDown}>
      <div ref={canvasHostRef} className="archboard__canvas" />

      {!ready && !error && <div className="archboard__skeleton">Loading the architecture board…</div>}
      {error && <div className="archboard__skeleton">Board failed to start: {error}</div>}

      {ready && (
        <>
          <Toolbar
            onView={(v) => boardRef.current?.setView(v)}
            onPlayPause={handlePlayPause}
            onSeek={(ms) => boardRef.current?.seek(ms)}
            onSpeed={(s) => boardRef.current?.setSpeed(s)}
            onReset={() => boardRef.current?.fitAll()}
            onFullscreen={handleFullscreen}
            fullscreenEnabled={fullscreenEnabled}
          />
          <Legend />
          {view === 'run' && (stageLabel || caption) && (
            <div className="archboard__caption">
              {stageLabel && <div className="archboard__captiontitle">{stageLabel}</div>}
              {caption && <div className="archboard__captiontext">{caption}</div>}
              <div className="archboard__captionnote">
                Stage lengths scaled from measured wall-times; intra-iteration split modeled from op counts.
              </div>
            </div>
          )}
          <Tooltip />
          <DetailPanel onClose={() => boardRef.current?.selectNode(null)} />
        </>
      )}
    </div>
  );
}
