'use client';
import type { Config, Lattice, Object as CrystalObject } from '../../lib/contract/generated';
import type { Applied } from '../../lib/compute/controller';
import { Field, JsonField, PairField, Section, ValueField } from './Fields';

export type EditConfig = (mutate: (config: Config) => void) => void;
export function ModelControls({ applied, edit, template }: { applied: Applied; edit: EditConfig; template?: CrystalObject }) {
  const config = applied.report.config!;
  const { lattice, objects = [] } = config.geometry;
  return <>
    <Section title="Lattice">
      <p className="wb-muted">{lattice.type[0].toUpperCase() + lattice.type.slice(1)} primitive cell. Lengths share one reference unit.</p>
      <div className="wb-fields">
        {lattice.type !== 'custom' && <ValueField label="a" numeric value={lattice.a} onCommit={value => edit(c => { c.geometry.lattice.a = Number(value); })} />}
        {['rectangular', 'oblique'].includes(lattice.type) && <ValueField label="b" numeric value={lattice.b} onCommit={value => edit(c => { c.geometry.lattice.b = Number(value); })} />}
        {lattice.type === 'oblique' && <ValueField label="Angle (degrees)" numeric value={lattice.angle_deg} onCommit={value => edit(c => { c.geometry.lattice.angle_deg = Number(value); })} />}
      </div>
      <details><summary>Change lattice definition</summary><JsonField label="Lattice" value={lattice} hint='Presets use type and a. Rectangular adds b; oblique adds b and angle_deg. Custom uses vectors.'
        onCommit={value => edit(c => { c.geometry.lattice = value as Lattice; })} /></details>
    </Section>
    <Section title="Materials and objects">
      <ValueField label="Background ε" value={config.geometry.background_epsilon} numeric onCommit={value => edit(c => { c.geometry.background_epsilon = Number(value); })} />
      {objects.map((object, index) => <div className="wb-object" key={object.name}>
        <div className="wb-row"><h3>{object.name}</h3><button onClick={() => edit(c => { c.geometry.objects!.splice(index, 1); })} aria-label={`Remove ${object.name}`}>Remove</button></div>
        <div className="wb-fields">
          <ValueField label="Name" value={object.name} onCommit={value => edit(c => { c.geometry.objects![index].name = value; })} />
          <ValueField label="Circle radius" numeric value={object.radius} onCommit={value => edit(c => { c.geometry.objects![index].radius = Number(value); })} />
          <ValueField label="Object ε" numeric value={object.epsilon} onCommit={value => edit(c => { c.geometry.objects![index].epsilon = Number(value); })} />
          <PairField label="Center (fractional)" value={object.center!} hint="Direct lattice coordinates" onCommit={value => edit(c => { c.geometry.objects![index].center = value as number[]; })} />
        </div>
      </div>)}
      {!objects.length && <p className="wb-muted">Homogeneous medium.</p>}
      <button disabled={!template} onClick={() => edit(c => {
        let number = 1; while (objects.some(object => object.name === `rod${number}`)) number++;
        (c.geometry.objects ??= []).push({ ...structuredClone(template!), name: `rod${number}` });
      })}>Add circle</button>
      <Field label="Polarization"><select value={config.polarization} onChange={event => edit(c => { c.polarization = event.target.value as Config['polarization']; })}>
        <option value="TM">TM (out-of-plane electric field)</option><option value="TE">TE (out-of-plane magnetic field)</option>
      </select></Field>
    </Section>
  </>;
}
