'use client';
import { useState } from 'react';
import type { ExecutionState } from '../../lib/compute/controller';
import { object } from '../../lib/contract/records';
import { diagnostic } from '../../lib/compute/protocol';
import { download, exportRun } from '../../lib/compute/export';
import { ArrayInspector } from './ArrayInspector';
import { BandPlot } from './BandPlot';

export function ResultsView({ state }: { state: ExecutionState }) {
  const [selectedJob, setJob] = useState(0), [sampleIndex, setSample] = useState(0);
  const [exporting, setExporting] = useState(false), [exportError, setExportError] = useState('');
  const run = state.run;
  if (!run) return <div className="wb-empty"><h2>Results</h2><p>Run a calculation to inspect bands, operators, and convergence.</p></div>;
  const records = [...run.results, ...run.errors.flatMap(error => error.partial_result ? [error.partial_result] : [])].sort((a, b) => a.job_index - b.job_index);
  const result = records.find(result => result.job_index === selectedJob) ?? records[0];
  const sample = result?.samples?.[sampleIndex] ?? result?.samples?.[0];
  const arrays = sample?.arrays ?? result?.arrays;
  const metadata = object(sample?.metadata ?? result?.metadata), parentMetadata = object(result?.metadata);
  const certification = object(metadata.certification);
  const labels = (metadata.labels ?? []) as string[], labelIndices = (metadata.label_indices ?? []) as number[];
  const sweep = object(parentMetadata.sweep_parameters);
  return <div className="wb-results">
    <div className="wb-row"><div><h2>{run.header.title}</h2><p className="wb-muted">{run.header.status.replaceAll('_', ' ')} · {run.results.length} completed · {run.errors.length} failed</p></div>
      <div className="wb-actions"><select aria-label="Export results" value="" disabled={exporting || !records.length} onChange={async event => {
        const format = event.target.value as 'json' | 'ndjson' | 'npz'; setExporting(true); setExportError('');
        try { download(await exportRun(run, format), `blaze-${run.header.id.slice(0, 8)}.${format}`); }
        catch (error) { setExportError(diagnostic(error).message); } finally { setExporting(false); }
      }}><option value="">{exporting ? 'Exporting…' : 'Export results'}</option><option value="npz">NumPy archive (.npz)</option><option value="json">JSON</option><option value="ndjson">NDJSON</option></select>
      <button onClick={() => download(new Blob([run.header.source], { type: 'text/plain' }), 'calculation.toml')}>Run TOML</button></div>
    </div>
    {state.storageError && <p className="wb-notice" role="status">{state.storageError}</p>}
    {exportError && <p className="wb-error" role="alert">{exportError}</p>}
    {run.failure && <p className="wb-error" role="alert">{run.failure.message}</p>}
    {run.errors.map(error => <p className="wb-error" role="alert" key={error.job_index}>Job {error.job_index}: {error.diagnostic.message}</p>)}
    {state.progress && run.header.status === 'running' && <p role="status">Job {state.progress.job_index}: {state.progress.completed} / {state.progress.total} samples. {state.progress.iterations} iterations in the latest solve.</p>}
    {state.live.length > 0 && state.job && <BandPlot title={`Job ${state.job.index} in progress`} data={{
      bands: state.live[0].frequencies.length, frequencies: state.live.flatMap(point => point.frequencies), distances: state.live.map(point => point.distance),
      labels: state.job.resolved.k_labels, labelIndices: state.job.resolved.k_label_indices.map(index => state.live.findIndex(point => point.sampleIndex === index)),
    }} />}
    {result && arrays && <>
      <div className="wb-fields">
        <label className="wb-field"><span>Configuration sample</span><select value={result.job_index} onChange={event => { setJob(Number(event.target.value)); setSample(0); }}>
          {records.map(result => <option key={result.job_index} value={result.job_index}>Job {result.job_index} · {Object.entries(object(object(result.metadata).sweep_parameters)).map(([name, value]) => `${name}=${JSON.stringify(value)}`).join(', ') || 'single configuration'}</option>)}
        </select></label>
        {!!result.samples?.length && <label className="wb-field"><span>Stencil sample</span><select value={sampleIndex} onChange={event => setSample(Number(event.target.value))}>
          {result.samples.map((sample, index) => <option key={sample.sample_index} value={index}>Sample {sample.sample_index} · {JSON.stringify(object(sample.metadata).k_point ?? object(sample.metadata).k_cartesian ?? '')}</option>)}</select></label>}
      </div>
      <p className={metadata.converged === true ? 'wb-certified' : 'wb-warning'}>{metadata.converged === true ? 'Eigenvalue stopping criterion met' : 'Iteration limit reached before convergence'}
        {typeof certification.max_residual === 'number' && ` · maximum residual ${(certification.max_residual as number).toExponential(3)}`}
        {Object.keys(sweep).length > 0 && ` · ${Object.entries(sweep).map(([key, value]) => `${key}: ${JSON.stringify(value)}`).join(', ')}`}
        {parentMetadata.registry_index != null && ` · registry ${parentMetadata.registry_index}: ${JSON.stringify(parentMetadata.registry)}`}
      </p>
      {arrays.frequencies && arrays.distances && <BandPlot data={{ frequencies: arrays.frequencies.data, bands: arrays.frequencies.shape[1], distances: arrays.distances.data, labels, labelIndices }} />}
      <details open={result.task === 'operators'}><summary>Arrays and dimensions</summary><ArrayInspector key={`${run.header.id}:${result.job_index}:${sampleIndex}`} arrays={arrays} /></details>
      <details><summary>Certification, quantities, and provenance</summary><pre>{JSON.stringify({ ...parentMetadata, ...metadata }, null, 2)}</pre></details>
    </>}
    <details><summary>Immutable run configuration</summary><pre>{run.header.source}</pre></details>
  </div>;
}
