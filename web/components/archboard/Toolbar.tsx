'use client';

import { useBoardStore, type BoardView } from './store';

interface ToolbarProps {
  onView: (view: BoardView) => void;
  onPlayPause: () => void;
  onSeek: (ms: number) => void;
  onSpeed: (speed: number) => void;
  onReset: () => void;
  onFullscreen: () => void;
  fullscreenEnabled: boolean;
}

const VIEWS: { id: BoardView; label: string }[] = [
  { id: 'layers', label: 'Layers' },
  { id: 'run', label: 'Run' },
  { id: 'precision', label: 'Precision' },
];

export default function Toolbar(props: ToolbarProps) {
  const view = useBoardStore((s) => s.view);
  const playing = useBoardStore((s) => s.playing);
  const speed = useBoardStore((s) => s.speed);
  const playbackMs = useBoardStore((s) => s.playbackMs);
  const totalMs = useBoardStore((s) => s.totalMs);
  const fullscreen = useBoardStore((s) => s.fullscreen);

  return (
    <div className="archboard__toolbar">
      <div className="archboard__views">
        {VIEWS.map((v) => (
          <button
            key={v.id}
            className={`archboard__viewbtn${view === v.id ? ' archboard__viewbtn--active' : ''}`}
            onClick={() => props.onView(v.id)}
          >
            {v.label}
          </button>
        ))}
      </div>

      {view === 'run' ? (
        <div className="archboard__runbar">
          <button className="archboard__iconbtn" onClick={props.onPlayPause} aria-label={playing ? 'Pause' : 'Play'}>
            {playing ? '⏸' : '▶'}
          </button>
          <input
            className="archboard__scrubber"
            type="range"
            min={0}
            max={totalMs}
            step={100}
            value={Math.min(playbackMs, totalMs)}
            onChange={(e) => props.onSeek(Number(e.target.value))}
          />
          <select
            className="archboard__speed"
            value={speed}
            onChange={(e) => props.onSpeed(Number(e.target.value))}
            aria-label="Playback speed"
          >
            {[0.25, 0.5, 1, 2, 4].map((s) => (
              <option key={s} value={s}>
                {s}×
              </option>
            ))}
          </select>
        </div>
      ) : (
        <button className="archboard__runbtn" onClick={props.onPlayPause}>
          ▶ Run a solve
        </button>
      )}

      <div className="archboard__spacer" />
      <button className="archboard__iconbtn" onClick={props.onReset} title="Reset view (0)" aria-label="Reset view">
        ⤢
      </button>
      {props.fullscreenEnabled && (
        <button
          className="archboard__iconbtn"
          onClick={props.onFullscreen}
          title="Fullscreen (f)"
          aria-label="Toggle fullscreen"
        >
          {fullscreen ? '⤡' : '⛶'}
        </button>
      )}
    </div>
  );
}
