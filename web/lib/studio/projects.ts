/**
 * Studio project persistence (localStorage).
 *
 * The durable payload is the TOML text (the Rust parser validates it) plus a
 * little UI state; results are never persisted. Named projects live under
 * `blaze.studio.project.<id>`, indexed by `blaze.studio.projects.index`. A
 * separate `blaze.studio.lastOpen` autosave restores the working document on
 * reload even before it is saved as a named project.
 */

import { parseToml } from './tomlParse';
import type { StudioConfig } from './configModel';

const LAST_OPEN_KEY = 'blaze.studio.lastOpen';
const INDEX_KEY = 'blaze.studio.projects.index';
const PROJECT_PREFIX = 'blaze.studio.project.';

const PROJECT_WARN_COUNT = 50;

export interface UiState {
  paneFractions?: [number, number, number];
  centerTab?: 'geometry' | 'reciprocal' | 'plots' | 'data';
  /** Legacy (pre-tab layout); read for migration, never written. */
  centerSplit?: number;
  /** Legacy; mapped cell -> geometry, bz -> reciprocal on load. */
  canvasTab?: 'cell' | 'bz';
}

/** Migrate a stored/imported UiState to the current shape. */
export function migrateUiState(ui: UiState | undefined): {
  paneFractions?: [number, number, number];
  centerTab?: 'geometry' | 'reciprocal' | 'plots' | 'data';
} {
  if (!ui) return {};
  const centerTab =
    ui.centerTab ??
    (ui.canvasTab === 'bz' ? 'reciprocal' : ui.canvasTab === 'cell' ? 'geometry' : undefined);
  return {
    paneFractions:
      Array.isArray(ui.paneFractions) && ui.paneFractions.length === 3
        ? ui.paneFractions
        : undefined,
    centerTab,
  };
}

/** The on-disk / exported project envelope. */
export interface ProjectEnvelope {
  format: 'blazeproj';
  version: 1;
  name: string;
  createdAt: number;
  modifiedAt: number;
  toml: string;
  uiState?: UiState;
}

export interface ProjectIndexEntry {
  id: string;
  name: string;
  modifiedAt: number;
}

// ---------------------------------------------------------------------------
// last-open autosave
// ---------------------------------------------------------------------------

export interface LastOpen {
  name: string;
  toml: string;
}

export function saveLastOpen(v: LastOpen): void {
  try {
    localStorage.setItem(LAST_OPEN_KEY, JSON.stringify(v));
  } catch {
    // best-effort
  }
}

export function loadLastOpen(): { name: string; config: StudioConfig } | null {
  try {
    const raw = localStorage.getItem(LAST_OPEN_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as LastOpen;
    if (!parsed?.toml) return null;
    const config = parseToml(parsed.toml);
    if (!config) return null;
    return { name: parsed.name || 'Untitled crystal', config };
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// named projects
// ---------------------------------------------------------------------------

function readIndex(): ProjectIndexEntry[] {
  try {
    const raw = localStorage.getItem(INDEX_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as ProjectIndexEntry[]) : [];
  } catch {
    return [];
  }
}

function writeIndex(entries: ProjectIndexEntry[]): void {
  localStorage.setItem(INDEX_KEY, JSON.stringify(entries));
}

/** A stable-ish id without Math.random/Date (both fine in the browser). */
function newId(): string {
  return `p_${Date.now().toString(36)}_${Math.floor(Math.random() * 1e6).toString(36)}`;
}

export function listProjects(): ProjectIndexEntry[] {
  return readIndex().sort((a, b) => b.modifiedAt - a.modifiedAt);
}

export function projectCount(): number {
  return readIndex().length;
}

/** Save (create or update) a named project. Returns the id, or null on quota. */
export function saveProject(
  id: string | null,
  name: string,
  toml: string,
  uiState?: UiState,
): { id: string } | { error: 'quota' | 'limit' } {
  const index = readIndex();
  const now = Date.now();

  let realId = id;
  if (!realId) {
    if (index.length >= PROJECT_WARN_COUNT * 4) {
      return { error: 'limit' };
    }
    realId = newId();
  }

  const existing = index.find((e) => e.id === realId);
  const createdAt = existing
    ? (loadProjectRaw(realId)?.createdAt ?? now)
    : now;

  const envelope: ProjectEnvelope = {
    format: 'blazeproj',
    version: 1,
    name,
    createdAt,
    modifiedAt: now,
    toml,
    uiState,
  };

  try {
    localStorage.setItem(PROJECT_PREFIX + realId, JSON.stringify(envelope));
  } catch {
    return { error: 'quota' };
  }

  const nextEntry: ProjectIndexEntry = { id: realId, name, modifiedAt: now };
  const nextIndex = existing
    ? index.map((e) => (e.id === realId ? nextEntry : e))
    : [...index, nextEntry];
  writeIndex(nextIndex);

  return { id: realId };
}

function loadProjectRaw(id: string): ProjectEnvelope | null {
  try {
    const raw = localStorage.getItem(PROJECT_PREFIX + id);
    if (!raw) return null;
    return JSON.parse(raw) as ProjectEnvelope;
  } catch {
    return null;
  }
}

/** Load a named project, parsing its TOML back into a config. */
export function loadProject(
  id: string,
): { name: string; config: StudioConfig; toml: string; uiState?: UiState } | null {
  const env = loadProjectRaw(id);
  if (!env?.toml) return null;
  const config = parseToml(env.toml);
  if (!config) return null;
  return { name: env.name, config, toml: env.toml, uiState: env.uiState };
}

export function deleteProject(id: string): void {
  localStorage.removeItem(PROJECT_PREFIX + id);
  writeIndex(readIndex().filter((e) => e.id !== id));
}

export function duplicateProject(id: string): { id: string } | null {
  const env = loadProjectRaw(id);
  if (!env) return null;
  const result = saveProject(null, `${env.name} copy`, env.toml, env.uiState);
  return 'id' in result ? { id: result.id } : null;
}

export function renameProject(id: string, name: string): void {
  const env = loadProjectRaw(id);
  if (!env) return;
  saveProject(id, name, env.toml, env.uiState);
}

// ---------------------------------------------------------------------------
// import / export
// ---------------------------------------------------------------------------

/** Build a .blazeproj envelope for download. */
export function makeEnvelope(name: string, toml: string, uiState?: UiState): ProjectEnvelope {
  const now = Date.now();
  return {
    format: 'blazeproj',
    version: 1,
    name,
    createdAt: now,
    modifiedAt: now,
    toml,
    uiState,
  };
}

/** Parse an imported .blazeproj.json file. Returns null if not a valid envelope. */
export function parseEnvelope(text: string): ProjectEnvelope | null {
  try {
    const parsed = JSON.parse(text) as ProjectEnvelope;
    if (parsed?.format !== 'blazeproj' || typeof parsed.toml !== 'string') return null;
    // Confirm the TOML is at least parseable structurally.
    if (!parseToml(parsed.toml)) return null;
    return parsed;
  } catch {
    return null;
  }
}
