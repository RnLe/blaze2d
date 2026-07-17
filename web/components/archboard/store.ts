'use client';

/**
 * Shared UI state between the pixi BoardApp (imperative, subscribes/writes via
 * the vanilla API) and the React chrome (toolbar, tooltip, detail panel).
 */

import { create } from 'zustand';

export type BoardView = 'layers' | 'run' | 'precision';

export interface HoverState {
  id: string;
  /** Screen coordinates relative to the board wrapper. */
  x: number;
  y: number;
}

interface BoardState {
  ready: boolean;
  view: BoardView;
  hover: HoverState | null;
  selected: string | null;
  playing: boolean;
  speed: number;
  /** Throttled playback position for the scrubber (ms). */
  playbackMs: number;
  totalMs: number;
  stageLabel: string | null;
  caption: string | null;
  fullscreen: boolean;

  setReady: (ready: boolean) => void;
  setView: (view: BoardView) => void;
  setHover: (hover: HoverState | null) => void;
  setSelected: (id: string | null) => void;
  setPlaying: (playing: boolean) => void;
  setSpeed: (speed: number) => void;
  setPlayback: (ms: number, stageLabel: string | null, caption: string | null) => void;
  setTotalMs: (ms: number) => void;
  setFullscreen: (fullscreen: boolean) => void;
}

export const useBoardStore = create<BoardState>((set) => ({
  ready: false,
  view: 'layers',
  hover: null,
  selected: null,
  playing: false,
  speed: 1,
  playbackMs: 0,
  totalMs: 1,
  stageLabel: null,
  caption: null,
  fullscreen: false,

  setReady: (ready) => set({ ready }),
  setView: (view) => set({ view }),
  setHover: (hover) => set({ hover }),
  setSelected: (selected) => set({ selected }),
  setPlaying: (playing) => set({ playing }),
  setSpeed: (speed) => set({ speed }),
  setPlayback: (playbackMs, stageLabel, caption) => set({ playbackMs, stageLabel, caption }),
  setTotalMs: (totalMs) => set({ totalMs }),
  setFullscreen: (fullscreen) => set({ fullscreen }),
}));
