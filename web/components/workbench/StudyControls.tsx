'use client';
import { useState } from 'react';
import type { Config, KCoordinates, Quantity, Sweep } from '../../lib/contract/generated';
import type { Applied } from '../../lib/compute/controller';
import { Field, JsonField, PairField, Section, ValueField } from './Fields';
import type { EditConfig } from './ModelControls';

const quantityLabels: Record<Quantity, string> = {
  velocity: 'Velocity', mass_tensor: 'Inverse mass', r_derivatives: 'Registry derivatives', born_huang: 'Born-Huang',
  slow_coefficient: 'Slow coefficients', exact_tm: 'Exact TM blocks', overlap: 'Reference overlaps',
};
export function StudyControls({ applied, edit, selectTask }: { applied: Applied; edit: EditConfig; selectTask: (task: Config['task']) => void }) {
  const config = applied.report.config!;
  const bands = config.bands, operators = config.operators;
  const [axis, setAxis] = useState({ name: '', target: '', values: '' });
  const [axisError, setAxisError] = useState('');
  const [stencil, setStencil] = useState({ count: '', width: '' });
  return <>
    <Section title="Calculation">
      <Field label="Task"><select value={config.task} onChange={event => selectTask(event.target.value as Config['task'])}>
        <option value="bands">Band structure</option><option value="operators">Operator extraction</option>
      </select></Field>
      <PairField label="Grid" axes={['nx', 'ny']} value={applied.report.resolved!.resolution} onCommit={value => edit(c => { c.grid = { resolution: value as number[] }; })} />
      {bands && <>
        <ValueField label="Band count" numeric value={bands.count} onCommit={value => edit(c => { c.bands!.count = Number(value); })} />
        <Field label="Path preset"><select value={bands.path!.preset ?? ''} onChange={event => {
          if (event.target.value) edit(c => { c.bands!.path = { preset: event.target.value, basis: 'reciprocal_fractional', intervals_per_segment: c.bands!.path!.intervals_per_segment }; });
        }}><option value="" disabled>Custom path</option><option value="square">Square: Γ, X, M, Γ</option><option value="triangular">Triangular: Γ, M, K, Γ</option><option value="rectangular">Rectangular: Γ, X, S, Y, Γ</option></select></Field>
        {!bands.path!.points && <ValueField label="Intervals per segment" numeric value={bands.path!.intervals_per_segment}
          onCommit={value => edit(c => { c.bands!.path!.intervals_per_segment = Number(value); })} />}
        <details><summary>Custom path</summary><JsonField label="Band path" value={bands.path} hint="Use vertices with intervals_per_segment, or sampled points without interpolation. State the coordinate basis."
          onCommit={value => edit(c => { c.bands!.path = value as NonNullable<Config['bands']>['path']; })} /></details>
        <label className="wb-check"><input type="checkbox" checked={!!bands.tracking} onChange={event => edit(c => { c.bands!.tracking = event.target.checked; })} />Track bands along the path</label>
      </>}
      {operators && <>
        <div className="wb-fields">
          <ValueField label="First band (zero-based)" numeric value={operators.band_lo} onCommit={value => edit(c => { c.operators!.band_lo = Number(value); })} />
          <ValueField label="Retained bands" numeric value={operators.retained_bands} onCommit={value => edit(c => { c.operators!.retained_bands = Number(value); })} />
          <ValueField label="Upper remote bands" numeric value={operators.remote_bands} onCommit={value => edit(c => { c.operators!.remote_bands = Number(value); })} />
          <PairField label="Carrier point" value={operators.k_point.value} onCommit={value => edit(c => { c.operators!.k_point.value = value as number[]; })} />
          <Field label="Carrier coordinate basis"><select value={operators.k_point.basis} onChange={event => edit(c => { c.operators!.k_point.basis = event.target.value as KCoordinates; })}>
            <option value="reciprocal_fractional">Reciprocal fractional</option><option value="cartesian_angular">Cartesian angular (1/reference length)</option>
          </select></Field>
        </div>
        <fieldset className="wb-quantities"><legend>Requested quantities</legend>{Object.entries(quantityLabels).map(([key, label]) =>
          <label className="wb-check" key={key}><input type="checkbox" checked={operators.quantities?.includes(key as Quantity) ?? false}
            onChange={event => edit(c => { c.operators!.quantities = event.target.checked
              ? [...c.operators!.quantities ?? [], key as Quantity] : c.operators!.quantities?.filter(value => value !== key); })} />{label}</label>)}</fieldset>
        <details open={!!operators.registry}><summary>Registry sampling</summary>
          <label className="wb-check"><input type="checkbox" checked={!!operators.registry} disabled={!config.geometry.objects?.length}
            onChange={event => edit(c => { if (event.target.checked) c.operators!.registry = { object: c.geometry.objects![0].name }; else delete c.operators!.registry; })} />Translate one object periodically</label>
          {operators.registry && <>
            <Field label="Moving object"><select value={operators.registry.object} onChange={event => edit(c => { c.operators!.registry!.object = event.target.value; })}>
              {config.geometry.objects?.map(object => <option key={object.name} value={object.name}>{object.name}</option>)}
            </select></Field>
            <JsonField label="Registry displacements" value={operators.registry.points} hint="Fractional [x, y] pairs, measured from the base geometry."
              onCommit={value => edit(c => { c.operators!.registry!.points = value as number[][]; })} />
            <ValueField label="Registry derivative step" numeric value={operators.registry.fd_step} onCommit={value => edit(c => { c.operators!.registry!.fd_step = Number(value); })} />
          </>}
        </details>
        <details open={!!operators.k_stencil}><summary>k-stencil</summary>
          <p className="wb-muted">A square sampling grid around the carrier. Width uses its coordinate basis.</p>
          {operators.k_stencil ? <>
            <ValueField label="Stencil points per axis" numeric value={operators.k_stencil.points_per_axis} onCommit={value => edit(c => { c.operators!.k_stencil!.points_per_axis = Number(value); })} />
            <ValueField label="Stencil half-width" numeric value={operators.k_stencil.half_width} onCommit={value => edit(c => { c.operators!.k_stencil!.half_width = Number(value); })} />
            <button onClick={() => edit(c => { delete c.operators!.k_stencil; })}>Remove stencil</button>
          </> : <>
            <Field label="Stencil points per axis"><input inputMode="numeric" value={stencil.count} onChange={event => setStencil({ ...stencil, count: event.target.value })} /></Field>
            <Field label="Stencil half-width"><input inputMode="decimal" value={stencil.width} onChange={event => setStencil({ ...stencil, width: event.target.value })} /></Field>
            <button disabled={!stencil.count || !stencil.width} onClick={() => edit(c => { c.operators!.k_stencil = { points_per_axis: Number(stencil.count), half_width: Number(stencil.width) }; })}>Add stencil</button>
          </>}
        </details>
      </>}
    </Section>
    <Section title="Parameter sweeps">
      <p className="wb-muted">Axes run in order. The last axis varies fastest.</p>
      {(config.sweeps ?? []).map((sweep, index) => <div className="wb-object" key={index}>
        <div className="wb-row"><h3>{sweep.name}</h3><button onClick={() => edit(c => { c.sweeps!.splice(index, 1); })} aria-label={`Remove sweep ${sweep.name}`}>Remove</button></div>
        <ValueField label={`Axis ${index + 1} name`} value={sweep.name} onCommit={value => edit(c => { c.sweeps![index].name = value; })} />
        <ValueField label={`Axis ${index + 1} target`} value={sweep.target} onCommit={value => edit(c => { c.sweeps![index].target = value; })} />
        {sweep.linspace ? <div className="wb-fields">{(['start', 'stop', 'count'] as const).map(key =>
          <ValueField key={key} label={`${sweep.name} ${key}`} numeric value={sweep.linspace![key]} onCommit={value => edit(c => { c.sweeps![index].linspace![key] = Number(value); })} />)}</div>
          : <JsonField label={`${sweep.name} values`} value={sweep.values} onCommit={value => edit(c => { c.sweeps![index].values = value as unknown[]; })} />}
        <details><summary>Change axis definition</summary><JsonField label={`Axis ${index + 1} definition`} value={sweep} hint="Choose values or linspace with start, stop, count."
          onCommit={value => edit(c => { c.sweeps![index] = value as Sweep; })} /></details>
      </div>)}
      <details><summary>Add an axis</summary>
        <Field label="Axis name"><input value={axis.name} onChange={event => setAxis({ ...axis, name: event.target.value })} /></Field>
        <Field label="Target"><input list="wb-sweep-targets" value={axis.target} placeholder="geometry.objects.rod.radius" onChange={event => setAxis({ ...axis, target: event.target.value })} /></Field>
        <datalist id="wb-sweep-targets"><option value="polarization" /><option value="grid.resolution" />
          {(config.geometry.objects ?? []).flatMap(object => ['radius', 'epsilon', 'center'].map(field => <option key={`${object.name}.${field}`} value={`geometry.objects.${object.name}.${field}`} />))}
        </datalist>
        <Field label="Values"><input value={axis.values} placeholder='[0.16, 0.20, 0.24]' onChange={event => setAxis({ ...axis, values: event.target.value })} /></Field>
        <button onClick={() => { try { const values: unknown = JSON.parse(axis.values); if (!Array.isArray(values)) throw new Error();
          edit(c => { (c.sweeps ??= []).push({ name: axis.name, target: axis.target, values }); }); setAxisError('');
        } catch { setAxisError('Enter a JSON array of sweep values.'); } }}>Add axis</button>
        {axisError && <p role="alert">{axisError}</p>}
      </details>
    </Section>
    <Section title="Numerical settings">
      <p className="wb-muted">f64 browser backend. {applied.report.summary?.solves} eigensolver calls in {applied.report.summary?.jobs} jobs.</p>
      <label className="wb-check"><input type="checkbox" checked={!!config.results?.eigenvectors} onChange={event => edit(c => { c.results = { eigenvectors: event.target.checked }; })} />Retain eigenvectors</label>
      <details><summary>Advanced settings</summary>
        <div className="wb-fields">
          <ValueField label="Tolerance" numeric value={config.eigensolver?.tolerance} onCommit={value => edit(c => { c.eigensolver!.tolerance = Number(value); })} />
          <ValueField label="Iteration limit" numeric value={config.eigensolver?.max_iterations} onCommit={value => edit(c => { c.eigensolver!.max_iterations = Number(value); })} />
          <ValueField label="Block size (0 = automatic)" numeric value={config.eigensolver?.block_size} onCommit={value => edit(c => { c.eigensolver!.block_size = Number(value); })} />
          {operators && <ValueField label="Residual acceptance limit" numeric optional value={operators.fail_on_residual} hint="Leave empty to retain results without a residual gate." onCommit={value => edit(c => { if (value === '') delete c.operators!.fail_on_residual; else c.operators!.fail_on_residual = Number(value); })} />}
        </div>
        <Field label="Dielectric smoothing"><select value={config.dielectric?.smoothing} onChange={event => edit(c => { c.dielectric!.smoothing = event.target.value as NonNullable<Config['dielectric']>['smoothing']; })}>
          <option value="analytic">Analytic interface</option><option value="subgrid">Subgrid sampling</option><option value="none">None</option>
        </select></Field>
        <div className="wb-fields">
          <ValueField label="Dielectric mesh size" numeric value={config.dielectric?.mesh_size} onCommit={value => edit(c => { c.dielectric!.mesh_size = Number(value); })} />
          <ValueField label="Interface tolerance" numeric value={config.dielectric?.interface_tolerance} onCommit={value => edit(c => { c.dielectric!.interface_tolerance = Number(value); })} />
        </div>
      </details>
    </Section>
  </>;
}
