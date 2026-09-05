'use client';
import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from 'react';
import Link from 'next/link';
import type { Config, Diagnostic, Object as CrystalObject } from '../../lib/contract/generated';
import { ExecutionController, Validator, type Applied, type ExecutionState } from '../../lib/compute/controller';
import { diagnostic, EXECUTION_BUDGET, type Preview } from '../../lib/compute/protocol';
import { download } from '../../lib/compute/export';
import { examples } from '../../lib/examples/catalog.generated';
import { getAssetPath } from '../../lib/paths';
import { ModelControls, type EditConfig } from './ModelControls';
import { StudyControls } from './StudyControls';
import { GeometryPreview } from './GeometryPreview';
import { ResultsView } from './ResultsView';
import { HistoryDrawer } from './HistoryDrawer';
import { TomlEditor } from './TomlEditor';
import { FieldDrafts } from './Fields';
import './workbench.css';

const EMPTY_STATE: ExecutionState = { run: null, live: [] };
const DRAFT_KEY = 'blaze2d-draft-v1';
const tabs = ['Model', 'Study', 'Results'] as const;
export default function Workbench({ initialSource, title = 'Calculation', embedded = false }: { initialSource?: string; title?: string; embedded?: boolean }) {
  const [validator, setValidator] = useState<Validator>(), [controller, setController] = useState<ExecutionController>();
  const [applied, setApplied] = useState<Applied>(), [draft, setDraft] = useState(''), [validation, setValidation] = useState<Applied>();
  const [template, setTemplate] = useState<CrystalObject>(), [preview, setPreview] = useState<Preview>();
  const [tab, setTab] = useState<typeof tabs[number]>('Model'), [editorOpen, setEditorOpen] = useState(false), [historyOpen, setHistoryOpen] = useState(false);
  const [busy, setBusy] = useState(false), [error, setError] = useState<Diagnostic>(), [validating, setValidating] = useState(false);
  const [runTitle, setRunTitle] = useState(title);
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
      const example = new URLSearchParams(window.location.search).get('example');
      const selected = examples.find(entry => entry.slug === example);
      let source = initialSource ?? selected?.source, savedDraft: string | undefined;
      if (!source && !embedded) {
        try { const saved = JSON.parse(localStorage.getItem(DRAFT_KEY) ?? 'null');
          if (saved?.schema === 'blaze2d/draft/1' && typeof saved.applied === 'string' && typeof saved.source === 'string') { source = saved.applied; savedDraft = saved.source; }
        } catch { /* A denied storage area must not prevent editing or calculation. */ }
      }
      const next = source === undefined ? defaults : await validator.validate(source);
      if (!current) return;
      if (!next.report.ok) { setApplied(defaults); setDraft(source ?? defaults.source); setValidation(next); setEditorOpen(true); }
      else { setApplied(next); setDraft(savedDraft ?? next.source); setValidation(next); setEditorOpen(savedDraft !== undefined && savedDraft !== next.source); }
      if (selected) setRunTitle(selected.title);
    })().catch(error => { if (current) setError(diagnostic(error)); });
    return () => { current = false; validator.dispose(); controller.dispose(); };
  }, [initialSource, embedded]);

  useEffect(() => {
    if (!validator || !applied) return;
    let current = true;
    setValidating(true);
    const timer = setTimeout(() => {
      void validator.validate(draft).then(report => { if (current) { setValidation(report); setValidating(false); } })
        .catch(error => { if (current) { setError(diagnostic(error)); setValidating(false); } });
    }, 250);
    if (!embedded) { try { localStorage.setItem(DRAFT_KEY, JSON.stringify({ schema: 'blaze2d/draft/1', applied: applied.source, source: draft })); } catch {} }
    return () => { current = false; clearTimeout(timer); };
  }, [draft, validator, applied, embedded]);

  useEffect(() => {
    if (!validator || !applied || overBudget) { setPreview(undefined); return; }
    let current = true;
    void validator.request({ action: 'preview', source: applied.source }).then(reply => {
      if (current && reply.kind === 'preview') setPreview(reply.preview);
    }).catch(error => { if (current) setError(diagnostic(error)); });
    return () => { current = false; };
  }, [applied, validator, overBudget]);

  function accept(next: Applied, resetFields = false) {
    if (!next.report.ok) { setError(next.report.errors[0]); return; }
    setApplied(next); setDraft(next.source); setValidation(next); setError(undefined); if (resetFields) setFieldRevision(value => value + 1);
  }
  const edit: EditConfig = mutate => {
    if (!validator || !latestApplied.current || latestDraft.current !== latestApplied.current.source || busy) return;
    const config = structuredClone(latestApplied.current.report.config!); mutate(config); setBusy(true);
    void validator.edit(latestApplied.current, config).then(next => accept(next)).catch(error => setError(diagnostic(error))).finally(() => setBusy(false));
  };
  async function selectTask(task: Config['task']) {
    if (!applied || !validator) return;
    setBusy(true);
    try { const reply = await validator.request({ action: 'task', source: applied.source, task }); if (reply.kind === 'validated') accept(reply); }
    catch (error) { setError(diagnostic(error)); } finally { setBusy(false); }
  }
  function changeTab(next: typeof tab) { setTab(next); }
  const diagnostics = validation?.source === draft ? validation.report.errors : [];
  return <FieldDrafts.Provider value={reportField}><div className={`workbench-shell${embedded ? ' workbench-embedded' : ''}`}>
    <header className="wb-header">
      <div className="wb-brand"><Link href="/" aria-label="Blaze2D home"><img src={getAssetPath('/icons/blaze_bw.svg')} alt="" width={28} height={28} /></Link><strong>Workbench</strong><span className="wb-version">{applied?.build.version}</span></div>
      <div className="wb-actions">
        <button onClick={() => file.current?.click()}>Import TOML</button>
        <button aria-pressed={editorOpen} onClick={() => setEditorOpen(open => !open)}>TOML</button>
        <button onClick={() => setHistoryOpen(true)}>History</button>
        {running ? <button className="wb-run wb-cancel" onClick={() => controller?.cancel()}>Cancel</button>
          : <button className="wb-run" disabled={blocked} onClick={() => {
            if (!applied) return; try { controller?.start(applied, runTitle); setTab('Results'); } catch (error) { setError(diagnostic(error)); }
          }}>Run</button>}
      </div>
      <input ref={file} type="file" accept=".toml,text/plain" hidden onChange={async event => {
        const selected = event.target.files?.[0]; if (!selected) return;
        if (selected.size > 2 * 1024 * 1024) { setError(diagnostic(new Error('TOML files must be smaller than 2 MiB.'))); return; }
        setDraft(await selected.text()); setEditorOpen(true); setError(undefined); setRunTitle(selected.name.replace(/\.toml$/, '')); event.target.value = '';
      }} />
      <nav className="wb-tabs" role="tablist" aria-label="Calculation workflow">{tabs.map((name, index) => <button role="tab" id={`wb-tab-${name}`} aria-controls={`wb-panel-${name}`} aria-selected={tab === name}
        tabIndex={tab === name ? 0 : -1} key={name} onClick={() => changeTab(name)} onKeyDown={event => {
          const next = event.key === 'ArrowRight' ? (index + 1) % tabs.length : event.key === 'ArrowLeft' ? (index + tabs.length - 1) % tabs.length : event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : -1;
          if (next >= 0) { event.preventDefault(); changeTab(tabs[next]); document.getElementById(`wb-tab-${tabs[next]}`)?.focus(); }
        }}>{name}</button>)}</nav>
    </header>
    <main className="wb-main">
      {!applied && !error && <p role="status">Loading the browser solver…</p>}
      {error && <div className="wb-error" role="alert"><p>{error.path && `${error.path}: `}{error.message}</p>
        {applied ? <button onClick={() => { setError(undefined); setDraft(applied.source); setValidation(applied); setFieldRevision(value => value + 1); }}>Revert to applied configuration</button> : <button onClick={() => window.location.reload()}>Reload</button>}
      </div>}
      {overBudget && <p className="wb-error" role="alert">Estimated memory exceeds 512 MiB. Reduce the study or run it through Python.</p>}
      {pending && <p className="wb-notice" role="status">TOML has unapplied changes. Apply or revert them before editing controls or running.</p>}
      {dirtyFields.size > 0 && !error && <p className="wb-notice" role="status">Finish editing with Enter or by leaving the field. Press Escape to restore its applied value.</p>}
      {editorOpen && <section className="wb-toml" aria-label="TOML editor">
        <div className="wb-row"><h2>Calculation TOML</h2><div className="wb-actions">
          <button disabled={!pending || validating || !validation?.report.ok || validation.source !== draft || busy} onClick={() => validation && accept(validation, true)}>Apply</button>
          <button disabled={!pending || !applied} onClick={() => applied && accept(applied, true)}>Revert</button>
          <button disabled={pending || !applied || busy} onClick={async () => {
            if (!validator || !applied) return; setBusy(true);
            try { const reply = await validator.request({ action: 'normalize', source: applied.source }); if (reply.kind === 'validated') accept(reply); }
            catch (error) { setError(diagnostic(error)); } finally { setBusy(false); }
          }}>Normalize</button>
          <button onClick={() => download(new Blob([draft], { type: 'text/plain' }), 'calculation.toml')}>Download draft</button>
        </div></div>
        <TomlEditor value={draft} diagnostics={diagnostics} onChange={text => { setDraft(text); setError(undefined); }} />
        <div className="wb-validation" role="status">{validating ? 'Validating…' : diagnostics.length ? diagnostics.map(item => <p key={item.message}>{item.path && `${item.path}: `}{item.message}</p>) : pending ? 'Valid draft. Apply to use it.' : 'Configuration applied.'}</div>
      </section>}
      {applied && <>
        <div role="tabpanel" id="wb-panel-Model" aria-labelledby="wb-tab-Model" hidden={tab !== 'Model'}>
          <div className="wb-model-layout"><fieldset key={`model-${fieldRevision}`} className="wb-controls" disabled={pending || busy}>
            <label className="wb-field"><span>Choose a model</span><select value="" onChange={async event => {
              const selected = examples.find(example => example.slug === event.target.value); if (!selected || !validator) return;
              setBusy(true); try { accept(await validator.validate(selected.source), true); setRunTitle(selected.title); }
              catch (error) { setError(diagnostic(error)); } finally { setBusy(false); }
            }}><option value="">{runTitle}</option>{examples.map(example => <option key={example.slug} value={example.slug}>{example.title}</option>)}</select></label>
            <ModelControls applied={applied} edit={edit} template={template} />
          </fieldset><aside className="wb-model-preview"><h2>Primitive cell</h2><GeometryPreview preview={preview} />
            <p className="wb-muted">Sampled dielectric distribution. Object centers use direct fractional coordinates.</p>
            <Link href="/workbench-guide">Workbench guide</Link>
          </aside></div>
        </div>
        <div role="tabpanel" id="wb-panel-Study" aria-labelledby="wb-tab-Study" hidden={tab !== 'Study'}>
          <fieldset key={`study-${fieldRevision}`} className="wb-study-layout" disabled={pending || busy}><StudyControls applied={applied} edit={edit} selectTask={selectTask} /></fieldset>
        </div>
      </>}
      <div role="tabpanel" id="wb-panel-Results" aria-labelledby="wb-tab-Results" hidden={tab !== 'Results'}><ResultsView state={state} /></div>
    </main>
    <HistoryDrawer open={historyOpen} onClose={() => setHistoryOpen(false)} running={!!running} onSelect={run => { controller?.select(run); setTab('Results'); }} />
  </div></FieldDrafts.Provider>;
}
