'use client';
import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from 'react';
import Link from 'next/link';
import type { Config, Diagnostic, Object as CrystalObject } from '@/lib/contract/generated';
import { ExecutionController, Validator, type Applied, type ExecutionState } from '@/lib/compute/controller';
import { diagnostic, EXECUTION_BUDGET, type Preview } from '@/lib/compute/protocol';
import { download } from '@/lib/compute/export';
import { examples, type Example } from '@/lib/examples/registry';
import { BookOpen, Braces, Check, Columns2, History, PanelLeft, Play, Square, Upload } from 'lucide-react';
import { getAssetPath } from '@/lib/paths';
import { ModelControls, type EditConfig } from './ModelControls';
import { StudyControls } from './StudyControls';
import { GeometryPreview } from './GeometryPreview';
import { ResultsView } from './ResultsView';
import { HistoryDrawer } from './HistoryDrawer';
import { TomlEditor } from './TomlEditor';
import { FieldDrafts } from './Fields';
import { EditorWorkspace, editorViews, type EditorViewName, type EditorGroup } from './EditorWorkspace';
import { ExamplesPanel } from './ExamplesPanel';
import { StudyOverview } from './StudyOverview';
import { Choices } from './Choices';
import type { CoordinateSystem } from './coordinates';
import './workbench.css';

const EMPTY_STATE: ExecutionState = { run: null, live: [] };
const DRAFT_KEY = 'blaze2d-draft-v1';
export default function Workbench() {
  const [validator, setValidator] = useState<Validator>(), [controller, setController] = useState<ExecutionController>();
  const [applied, setApplied] = useState<Applied>(), [draft, setDraft] = useState(''), [validation, setValidation] = useState<Applied>();
  const [template, setTemplate] = useState<CrystalObject>(), [preview, setPreview] = useState<Preview>();
  const [primary, setPrimary] = useState<EditorViewName>('Geometry'), [secondary, setSecondary] = useState<EditorViewName | null>(null);
  const [focused, setFocused] = useState<EditorGroup>('primary'), [sidebarOpen, setSidebarOpen] = useState(true), [setup, setSetup] = useState('Crystal');
  const [historyOpen, setHistoryOpen] = useState(false), [inspectSlug, setInspectSlug] = useState<string>(), [loadedSlug, setLoadedSlug] = useState<string>();
  const operation = useRef(false);
  const [busy, setBusy] = useState(false), [error, setError] = useState<Diagnostic>(), [validating, setValidating] = useState(false);
  const [runTitle, setRunTitle] = useState('Calculation');
  const [coordinates, setCoordinates] = useState<CoordinateSystem>('fractional');
  const [fieldRevision, setFieldRevision] = useState(0), [dirtyFields, setDirtyFields] = useState<Set<string>>(new Set());
  const reportField = useCallback((id: string, dirty: boolean) => setDirtyFields(previous => {
    if (previous.has(id) === dirty) return previous; const next = new Set(previous); if (dirty) next.add(id); else next.delete(id); return next;
  }), []);
  const file = useRef<HTMLInputElement>(null), latestApplied = useRef(applied), latestDraft = useRef(draft);
  latestApplied.current = applied; latestDraft.current = draft;
  const subscribe = useCallback((listener: () => void) => controller?.subscribe(listener) ?? (() => {}), [controller]);
  const state = useSyncExternalStore(subscribe, controller?.snapshot ?? (() => EMPTY_STATE), () => EMPTY_STATE);
  const pending = !!applied && draft !== applied.source, running = state.run?.header.status === 'running';
  const overBudget = (applied?.report.summary?.estimated_peak_bytes ?? 0) > EXECUTION_BUDGET;
  const blocked = !applied || pending || busy || !!error || overBudget || dirtyFields.size > 0;

  useEffect(() => {
    const base = process.env.NEXT_PUBLIC_BASE_PATH ?? '', validator = new Validator(base), controller = new ExecutionController(base);
    let current = true;
    setValidator(validator); setController(controller);
    void (async () => {
      const defaults = await validator.validate();
      if (!current) return;
      setTemplate(defaults.report.config?.geometry.objects?.[0]);
      const params = new URLSearchParams(window.location.search);
      const requestedView = editorViews.find(view => view.toLowerCase() === params.get('view'));
      if (requestedView) setPrimary(requestedView);
      setInspectSlug(params.get('inspect') ?? undefined);
      const example = params.get('example');
      const selected = examples.find(entry => entry.slug === example);
      let source = selected?.source as string | undefined, savedDraft: string | undefined;
      if (!source) {
        try { const saved = JSON.parse(localStorage.getItem(DRAFT_KEY) ?? 'null');
          if (saved?.schema === 'blaze2d/draft/1' && typeof saved.applied === 'string' && typeof saved.source === 'string') {
            source = saved.applied; savedDraft = saved.source; if (typeof saved.title === 'string') setRunTitle(saved.title);
          }
        } catch { /* A denied storage area must not prevent editing or calculation. */ }
      }
      const next = source === undefined ? defaults : await validator.validate(source);
      if (!current) return;
      if (!next.report.ok) { setApplied(defaults); setDraft(source ?? defaults.source); setValidation(next); setPrimary('TOML'); }
      else { setApplied(next); setDraft(savedDraft ?? next.source); setValidation(next); if (savedDraft !== undefined && savedDraft !== next.source && !requestedView) setPrimary('TOML'); }
      if (selected) {
        setRunTitle(selected.title); setLoadedSlug(selected.slug);
        const url = new URL(window.location.href); url.searchParams.delete('example'); window.history.replaceState(null, '', url);
      }
    })().catch(error => { if (current) setError(diagnostic(error)); });
    return () => { current = false; validator.dispose(); controller.dispose(); };
  }, []);

  useEffect(() => {
    if (!validator || !applied) return;
    let current = true;
    setValidating(true);
    const timer = setTimeout(() => {
      void validator.validate(draft).then(report => { if (current) { setValidation(report); setValidating(false); } })
        .catch(error => { if (current) { setError(diagnostic(error)); setValidating(false); } });
    }, 250);
    try { localStorage.setItem(DRAFT_KEY, JSON.stringify({ schema: 'blaze2d/draft/1', applied: applied.source, source: draft, title: runTitle })); } catch {}
    return () => { current = false; clearTimeout(timer); };
  }, [draft, validator, applied, runTitle]);

  useEffect(() => {
    if (!validator || !applied || overBudget) { setPreview(undefined); return; }
    let current = true;
    setPreview(undefined);
    void validator.request({ action: 'preview', source: applied.source }).then(reply => {
      if (current && reply.kind === 'preview') setPreview(reply.preview);
    }).catch(error => { if (current) setError(diagnostic(error)); });
    return () => { current = false; };
  }, [applied, validator, overBudget]);

  function accept(next: Applied, resetFields = false) {
    if (!next.report.ok) { setError(next.report.errors[0]); return false; }
    latestApplied.current = next; latestDraft.current = next.source;
    setApplied(next); setDraft(next.source); setValidation(next); setError(undefined); if (resetFields) setFieldRevision(value => value + 1);
    return true;
  }
  const edit: EditConfig = mutate => {
    if (!validator || !latestApplied.current || latestDraft.current !== latestApplied.current.source || operation.current) return;
    const source = latestDraft.current;
    const config = structuredClone(latestApplied.current.report.config!); mutate(config); operation.current = true; setBusy(true);
    void validator.edit(latestApplied.current, config).then(next => { if (latestDraft.current === source) accept(next); }).catch(error => setError(diagnostic(error))).finally(() => { operation.current = false; setBusy(false); });
  };
  async function selectTask(task: Config['task']) {
    if (!applied || !validator || operation.current) return;
    operation.current = true; setBusy(true);
    try { const reply = await validator.request({ action: 'task', source: applied.source, task }); if (reply.kind === 'validated') accept(reply); }
    catch (error) { setError(diagnostic(error)); } finally { operation.current = false; setBusy(false); }
  }
  function activate(view: EditorViewName, group: EditorGroup = focused) {
    if (group === 'secondary' && secondary) {
      if (primary === view) setPrimary(secondary);
      setSecondary(view);
    } else {
      if (secondary === view) setSecondary(primary);
      setPrimary(view); group = 'primary';
    }
    setFocused(group);
    if (view === 'Geometry') setSetup('Crystal');
    if (view === 'Study') setSetup('Calculation');
  }
  function closeSplit() { setSecondary(null); setFocused('primary'); }
  function reveal(view: EditorViewName) {
    activate(view, primary === view ? 'primary' : secondary === view ? 'secondary' : focused);
  }
  async function loadExample(example: Example) {
    if (!validator || pending || busy || dirtyFields.size || operation.current) return;
    operation.current = true; setBusy(true);
    try {
      const next = await validator.validate(example.source);
      if (!accept(next, true)) return;
      setRunTitle(example.title); setLoadedSlug(example.slug); reveal('Geometry');
      // A loaded deep link must not silently replace later edits on reload.
      const url = new URL(window.location.href); url.searchParams.delete('example'); url.searchParams.delete('inspect'); url.searchParams.delete('view');
      window.history.replaceState(null, '', url);
    } catch (error) { setError(diagnostic(error)); } finally { operation.current = false; setBusy(false); }
  }
  const hasSetupView = [primary, secondary].some(view => view && !['Examples', 'TOML'].includes(view));
  const showSidebar = sidebarOpen && hasSetupView;

  const diagnostics = validation?.source === draft ? validation.report.errors : [];
  return <FieldDrafts.Provider value={reportField}><div className="workbench-shell">
    <header className="wb-header">
      <div className="wb-brand"><Link href="/" aria-label="Blaze2D home"><img src={getAssetPath('/icons/blaze_bw.svg')} alt="" width={25} height={25} /></Link><strong>Workbench</strong><span className="wb-version">{applied?.build.version}</span></div>
      <span className="wb-document-title" title={runTitle}>{runTitle}<span> / {applied?.report.config?.task === 'operators' ? 'operators' : 'bands'}</span></span>
      <Link className="wb-guide-link" href="/workbench-guide">Guide ↗</Link>
    </header>
    <div className="wb-commandbar">
      <div className="wb-actions">
        <button aria-pressed={showSidebar} aria-controls="wb-setup" onClick={() => { if (!hasSetupView) { activate('Geometry'); setSidebarOpen(true); } else setSidebarOpen(open => !open); }}><PanelLeft size={16} /><span>Setup</span></button>
        <button onClick={() => reveal('Examples')}><BookOpen size={16} /><span>Examples</span></button>
        <button onClick={() => file.current?.click()} disabled={busy}><Upload size={16} /><span>Import TOML</span></button>
        <button onClick={() => setHistoryOpen(true)}><History size={16} /><span>History</span></button>
      </div>
      <div className="wb-actions">
        <button aria-pressed={!!secondary} onClick={() => { if (secondary) closeSplit(); else { setSecondary(primary === 'Geometry' ? 'Results' : 'Geometry'); setFocused('primary'); } }}><Columns2 size={16} /><span>Split view</span></button>
        {running ? <button className="wb-run wb-cancel" onClick={() => controller?.cancel()}><Square size={13} />Cancel</button>
          : <button className="wb-run" disabled={blocked} onClick={() => {
            if (!applied) return; try { controller?.start(applied, runTitle); reveal('Results'); } catch (error) { setError(diagnostic(error)); }
          }}><Play size={14} />Run</button>}
      </div>
      <input ref={file} type="file" accept=".toml,text/plain" hidden onChange={async event => {
        const input = event.currentTarget, selected = input.files?.[0]; if (!selected) return;
        if (selected.size > 2 * 1024 * 1024) { setError(diagnostic(new Error('TOML files must be smaller than 2 MiB.'))); input.value = ''; return; }
        const text = await selected.text(); latestDraft.current = text; setDraft(text); reveal('TOML'); setError(undefined); setRunTitle(selected.name.replace(/\.toml$/, '')); setLoadedSlug(undefined); input.value = '';
      }} />
    </div>
    <main className="wb-main">
      <div className="wb-messages">
        {!applied && !error && <p role="status">Loading the browser solver…</p>}
        {error && <div className="wb-error" role="alert"><p>{error.path && `${error.path}: `}{error.message}</p>
          {applied ? <button onClick={() => accept(applied, true)}>Revert to applied configuration</button> : <button onClick={() => window.location.reload()}>Reload</button>}
        </div>}
        {overBudget && <p className="wb-error" role="alert">Estimated memory exceeds 512 MiB. Reduce the study or run it through Python.</p>}
        {pending && <p className="wb-notice" role="status">TOML has unapplied changes. <button onClick={() => reveal('TOML')}>Review draft</button> before editing controls or running.</p>}
        {dirtyFields.size > 0 && !error && <p className="wb-notice" role="status">Finish editing with Enter or by leaving the field. Press Escape to restore its applied value.</p>}
      </div>
      <div className="wb-workspace">
        {applied && <aside id="wb-setup" className="wb-setup" aria-label="Configuration controls" hidden={!showSidebar}>
          <div className="wb-setup-heading"><span>Configuration</span><span className="wb-muted">{applied.report.config!.polarization}</span></div>
          <Choices label="Setup controls" compact value={setup} options={[{ value: 'Crystal', label: 'Crystal' }, { value: 'Calculation', label: 'Calculation' }]} onChange={setSetup} />
          <div className="wb-controls-scroll">
            <fieldset key={`model-${fieldRevision}`} className="wb-controls" hidden={setup !== 'Crystal'} disabled={pending || busy}>
              <ModelControls applied={applied} edit={edit} template={template} coordinates={coordinates} />
            </fieldset>
            <fieldset key={`study-${fieldRevision}`} className="wb-controls" hidden={setup !== 'Calculation'} disabled={pending || busy}>
              <StudyControls applied={applied} edit={edit} selectTask={selectTask} />
            </fieldset>
          </div>
          <div className="wb-setup-footer"><Check size={13} />{pending ? 'Applied configuration shown' : 'Controls update the TOML'}</div>
        </aside>}
        <EditorWorkspace primary={primary} secondary={secondary} focused={focused} onActivate={activate} onFocus={setFocused} onCloseSplit={closeSplit} pending={pending}>
          {{
            Geometry: applied && <div className="wb-geometry-view"><div className="wb-preview-heading"><div><span className="wb-eyebrow">Real space</span><h1>{applied.report.config!.geometry.lattice.type} lattice</h1></div>
              <Choices label="Object coordinate system" compact value={coordinates} disabled={dirtyFields.size > 0 || busy} options={[{ value: 'fractional', label: 'Fractional' }, { value: 'cartesian', label: 'Cartesian' }]} onChange={setCoordinates} /></div>
              <GeometryPreview preview={preview} geometry={applied.report.config!.geometry} coordinates={coordinates} />
              <p className="wb-preview-caption">{coordinates === 'fractional' ? 'Centers use direct lattice coordinates u, v.' : 'Centers use real-space x, y in reference-length units.'} The outlined cell repeats periodically.</p>
            </div>,
            Study: applied && <StudyOverview applied={applied} />,
            Results: <><ResultsView key={state.run?.header.id ?? 'empty'} state={state} />{state.run && applied && state.run.header.source !== applied.source && <p className="wb-result-snapshot wb-notice">These results belong to the saved run configuration. Your current setup has changed.</p>}</>,
            TOML: <section className="wb-toml" aria-label="TOML editor">
              <div className="wb-editor-heading"><span><Braces size={15} />calculation.toml</span><span className="wb-muted">{pending ? 'Unapplied changes' : 'Applied configuration'}</span></div>
              <div className="wb-toml-actions wb-actions">
                <button className="wb-primary" disabled={!pending || validating || !validation?.report.ok || validation.source !== draft || busy} onClick={() => validation && accept(validation, true)}>Apply</button>
                <button disabled={!pending || !applied || busy} onClick={() => applied && accept(applied, true)}>Revert</button>
                <button title="Expand defaults and apply this valid TOML" disabled={!applied || busy || validating || !validation?.report.ok || validation.source !== draft || dirtyFields.size > 0} onClick={async () => {
                  if (!validator || !applied || operation.current) return;
                  const source = latestDraft.current; operation.current = true; setBusy(true);
                  try { const reply = await validator.request({ action: 'normalize', source }); if (reply.kind === 'validated' && latestDraft.current === source) accept(reply, true); }
                  catch (error) { setError(diagnostic(error)); } finally { operation.current = false; setBusy(false); }
                }}>Normalize</button>
                <button onClick={() => download(new Blob([draft], { type: 'text/plain' }), 'calculation.toml')}>Download draft</button>
              </div>
              <TomlEditor value={draft} diagnostics={diagnostics} onChange={text => { latestDraft.current = text; setDraft(text); setError(undefined); }} />
              <div className="wb-validation" role="status">{validating ? 'Validating…' : diagnostics.length ? diagnostics.map(item => <p key={item.message}>{item.path && `${item.path}: `}{item.message}</p>) : pending ? 'Valid draft. Apply to use it.' : 'Configuration applied. Normalize expands defaults and removes comments.'}</div>
            </section>,
            Examples: <ExamplesPanel initialSlug={inspectSlug} loadedSlug={loadedSlug} disabled={!validator || busy || pending || dirtyFields.size > 0} onLoad={loadExample} />,
          }}
        </EditorWorkspace>
      </div>
    </main>
    <footer className="wb-statusbar"><span className={running ? 'wb-status-running' : ''}><span className="wb-status-dot" />{error ? 'Configuration error' : pending ? 'Unapplied draft' : running ? 'Running' : busy ? 'Validating' : dirtyFields.size ? 'Editing controls' : applied ? 'Ready' : 'Loading'}</span>
      <span>{applied ? `${applied.report.summary?.jobs} jobs · ${applied.report.resolved?.resolution.join(' × ')} grid · f64` : 'Loading solver'}</span><span>Browser worker</span></footer>
    <HistoryDrawer open={historyOpen} onClose={() => setHistoryOpen(false)} running={!!running} onSelect={run => { controller?.select(run); reveal('Results'); }} />
  </div></FieldDrafts.Provider>;
}
