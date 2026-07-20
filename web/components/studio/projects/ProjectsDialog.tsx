'use client';

import React, { useRef, useState } from 'react';
import { X, Upload, Download, Copy, Trash2, Save, FolderOpen } from 'lucide-react';
import { useStudioStore } from '../../../lib/studio/store';
import { parseToml } from '../../../lib/studio/tomlParse';
import { downloadJson } from '../../../lib/util/download';
import {
  listProjects,
  saveProject,
  loadProject,
  deleteProject,
  duplicateProject,
  renameProject,
  makeEnvelope,
  migrateUiState,
  parseEnvelope,
  type ProjectIndexEntry,
  type UiState,
} from '../../../lib/studio/projects';

/** The id of the last project opened/saved this session (save-in-place). */
let activeProjectId: string | null = null;
let activeProjectName: string | null = null;

/** Snapshot the persistable UI state at CALL time (never at dialog mount). */
function currentUiState(): UiState {
  const ui = useStudioStore.getState().ui;
  return { paneFractions: ui.paneFractions, centerTab: ui.centerTab };
}

function timeAgo(ts: number): string {
  const s = Math.floor((Date.now() - ts) / 1000);
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export function ProjectsDialog({ onClose }: { onClose: () => void }) {
  const projectName = useStudioStore((s) => s.project.name);
  const tomlText = useStudioStore((s) => s.toml.text);
  const setConfig = useStudioStore((s) => s.setConfig);
  const setProjectName = useStudioStore((s) => s.setProjectName);
  const markClean = useStudioStore((s) => s.markClean);

  const [projects, setProjects] = useState<ProjectIndexEntry[]>(() => listProjects());
  const [saveName, setSaveName] = useState(projectName);
  const [toast, setToast] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const refresh = () => setProjects(listProjects());
  const flash = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 1800);
  };

  const handleSaveCurrent = () => {
    const name = saveName.trim() || 'Untitled crystal';
    // Save in place when re-saving the same project under the same name;
    // a changed name (or no active project) creates a new entry.
    const inPlaceId = activeProjectId && activeProjectName === name ? activeProjectId : null;
    const result = saveProject(inPlaceId, name, tomlText, currentUiState());
    if ('error' in result) {
      flash(result.error === 'quota' ? 'Storage full' : 'Too many projects');
      return;
    }
    activeProjectId = result.id;
    activeProjectName = name;
    setProjectName(name);
    markClean();
    refresh();
    flash(inPlaceId ? 'Saved' : 'Saved as new project');
  };

  const handleOpen = (id: string) => {
    const loaded = loadProject(id);
    if (!loaded) {
      flash('Could not open project');
      return;
    }
    setConfig(loaded.config, { markClean: true });
    setProjectName(loaded.name);
    activeProjectId = id;
    activeProjectName = loaded.name;
    // Restore persisted layout (with legacy-field migration).
    const ui = migrateUiState(loaded.uiState);
    const store = useStudioStore.getState();
    if (ui.paneFractions) store.setPaneFractions(ui.paneFractions);
    if (ui.centerTab) store.setCenterTab(ui.centerTab);
    onClose();
  };

  const handleExport = () => {
    const name = (projectName.trim() || 'crystal').replace(/[^\w-]+/g, '_').toLowerCase();
    downloadJson(makeEnvelope(projectName, tomlText, currentUiState()), `${name}.blazeproj.json`);
  };

  const handleImportFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const env = parseEnvelope(String(reader.result));
      if (!env) {
        flash('Not a valid .blazeproj file');
        return;
      }
      const config = parseToml(env.toml);
      if (!config) {
        flash('Project TOML could not be parsed');
        return;
      }
      setConfig(config, { markClean: true });
      setProjectName(env.name);
      // Also save it as a named project so it persists.
      const saved = saveProject(null, env.name, env.toml, env.uiState);
      if ('id' in saved) {
        activeProjectId = saved.id;
        activeProjectName = env.name;
      }
      refresh();
      onClose();
    };
    reader.readAsText(file);
  };

  return (
    <div className="studio__modal-overlay" onClick={onClose}>
      <div className="studio__modal" onClick={(e) => e.stopPropagation()}>
        <div className="studio__modal-head">
          <span className="studio__modal-title">Projects</span>
          <button type="button" className="studio__iconbtn" aria-label="Close" onClick={onClose}>
            <X size={16} />
          </button>
        </div>

        {/* Save current */}
        <div className="studio__modal-section">
          <span className="studio__label">Save current crystal</span>
          <div style={{ display: 'flex', gap: 8 }}>
            <input
              className="studio__input"
              value={saveName}
              onChange={(e) => setSaveName(e.target.value)}
              placeholder="Project name"
            />
            <button type="button" className="studio__btn studio__btn--run" onClick={handleSaveCurrent}>
              <Save size={14} /> Save
            </button>
          </div>
          <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
            <button type="button" className="studio__btn" onClick={handleExport}>
              <Download size={14} /> Export .blazeproj
            </button>
            <button
              type="button"
              className="studio__btn"
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload size={14} /> Import
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".json,.blazeproj,application/json"
              style={{ display: 'none' }}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleImportFile(f);
                e.target.value = '';
              }}
            />
          </div>
        </div>

        {/* Saved list */}
        <div className="studio__modal-section" style={{ flex: 1, minHeight: 0 }}>
          <span className="studio__label">Saved ({projects.length})</span>
          <div className="studio__project-list subtle-scroll">
            {projects.length === 0 ? (
              <div className="studio__hint" style={{ padding: '12px 0' }}>
                No saved projects yet. Save the current crystal or import a .blazeproj file.
              </div>
            ) : (
              projects.map((p) => (
                <div key={p.id} className="studio__project-row">
                  <button
                    type="button"
                    className="studio__project-open"
                    onClick={() => handleOpen(p.id)}
                    title="Open"
                  >
                    <FolderOpen size={14} />
                    <span className="studio__project-name-text">{p.name}</span>
                    <span className="studio__project-time">{timeAgo(p.modifiedAt)}</span>
                  </button>
                  <button
                    type="button"
                    className="studio__iconbtn"
                    aria-label="Rename"
                    title="Rename"
                    onClick={() => {
                      const name = prompt('Rename project', p.name);
                      if (name && name.trim()) {
                        renameProject(p.id, name.trim());
                        refresh();
                      }
                    }}
                  >
                    <span style={{ fontSize: 11 }}>Aa</span>
                  </button>
                  <button
                    type="button"
                    className="studio__iconbtn"
                    aria-label="Duplicate"
                    title="Duplicate"
                    onClick={() => {
                      duplicateProject(p.id);
                      refresh();
                    }}
                  >
                    <Copy size={13} />
                  </button>
                  <button
                    type="button"
                    className="studio__iconbtn"
                    aria-label="Delete"
                    title="Delete"
                    onClick={() => {
                      deleteProject(p.id);
                      refresh();
                    }}
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              ))
            )}
          </div>
        </div>

        {toast ? <div className="studio__toast">{toast}</div> : null}
      </div>
    </div>
  );
}
