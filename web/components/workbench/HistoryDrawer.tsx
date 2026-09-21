'use client';
import { useEffect, useRef, useState } from 'react';
import { deleteRun, listRuns, loadRun, type RunHeader } from '@/lib/compute/storage';
import type { BrowserRun } from '@/lib/compute/controller';
import { diagnostic } from '@/lib/compute/protocol';

export function HistoryDrawer({ open, onClose, onSelect, running }: { open: boolean; onClose: () => void; onSelect: (run: BrowserRun) => void; running: boolean }) {
  const dialog = useRef<HTMLDialogElement>(null), [runs, setRuns] = useState<RunHeader[]>([]), [error, setError] = useState('');
  useEffect(() => {
    // Captured now so the cleanup closes the dialog this effect opened, even if
    // the ref has moved on by the time it runs.
    const element = dialog.current;
    if (!open) { element?.close(); return; }
    const previous = document.activeElement as HTMLElement | null;
    element?.showModal();
    void listRuns().then(setRuns).catch(error => setError(diagnostic(error).message));
    return () => { element?.close(); previous?.focus(); };
  }, [open]);
  return <dialog className="wb-history" ref={dialog} onCancel={event => { event.preventDefault(); onClose(); }} aria-labelledby="wb-history-title">
    <div className="wb-row"><h2 id="wb-history-title">Run history</h2><button onClick={onClose} autoFocus>Close</button></div>
    <p className="wb-muted">Stored on this device. Older runs are removed when the cache fills.</p>
    {error && <p className="wb-error" role="alert">{error}</p>}
    {!runs.length && <p>No stored runs.</p>}
    {runs.map(run => <article key={run.id} className="wb-history-entry">
      <button disabled={running} className="wb-history-open" onClick={async () => {
        try { onSelect(await loadRun(run.id)); onClose(); } catch (error) { setError(diagnostic(error).message); }
      }}><strong>{run.title}</strong><span>{new Date(run.startedAt).toLocaleString()}</span><span>{run.status.replaceAll('_', ' ')} · {run.completed} completed{run.cacheComplete === false ? ' · partial cache' : ''}</span></button>
      <button disabled={run.status === 'running'} aria-label={`Delete ${run.title} from history`} onClick={async () => {
        try { await deleteRun(run.id); setRuns(await listRuns()); } catch (error) { setError(diagnostic(error).message); }
      }}>Delete</button>
    </article>)}
    {running && <p>Cancel the current run before opening another.</p>}
  </dialog>;
}
