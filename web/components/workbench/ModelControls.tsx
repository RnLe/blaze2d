'use client';
import type { Config, Lattice, Object as CrystalObject } from '@/lib/contract/generated';
import type { Applied } from '@/lib/compute/controller';
import { toCartesian, toFractional, type CoordinateSystem } from './coordinates';
import { JsonField, PairField, Section, ValueField } from './Fields';
import { Choices, LatticeIcon } from './Choices';
import { CirclePlus, Trash2 } from 'lucide-react';

export type EditConfig = (mutate: (config: Config) => void) => void;
export function ModelControls({ applied, edit, template, coordinates = 'fractional' }: { applied: Applied; edit: EditConfig; template?: CrystalObject; coordinates?: CoordinateSystem }) {
  const config = applied.report.config!;
  const { lattice, objects = [] } = config.geometry;
  const vectors = applied.report.resolved!.lattice_vectors;
  return <>
    <Section title="Lattice">
      <Choices label="Lattice type" value={lattice.type} options={(['square', 'triangular', 'rectangular', 'oblique', 'custom'] as const).map(type => ({ value: type, label: type[0].toUpperCase() + type.slice(1), icon: <LatticeIcon type={type} /> }))} onChange={type => edit(c => {
        const a = Math.hypot(...vectors[0]), b = Math.hypot(...vectors[1]);
        const angle = Math.acos(Math.max(-1, Math.min(1, (vectors[0][0] * vectors[1][0] + vectors[0][1] * vectors[1][1]) / (a * b)))) * 180 / Math.PI;
        c.geometry.lattice = type === 'custom' ? { type, vectors: vectors.map(row => [...row]) }
          : type === 'oblique' ? { type, a, b, angle_deg: angle }
          : type === 'rectangular' ? { type, a, b } : { type, a };
      })} />
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
        <div className="wb-row"><h3><span className="wb-object-dot" />{object.name}</h3><button onClick={() => edit(c => { c.geometry.objects!.splice(index, 1); })} aria-label={`Remove ${object.name}`}><Trash2 size={14} /></button></div>
        <div className="wb-fields">
          <ValueField label="Name" value={object.name} onCommit={value => edit(c => { c.geometry.objects![index].name = value; })} />
          <ValueField label="Circle radius" numeric value={object.radius} onCommit={value => edit(c => { c.geometry.objects![index].radius = Number(value); })} />
          <ValueField label="Object ε" numeric value={object.epsilon} onCommit={value => edit(c => { c.geometry.objects![index].epsilon = Number(value); })} />
          <PairField key={coordinates} label={`Center (${coordinates})`} axes={coordinates === 'fractional' ? ['u', 'v'] : ['x', 'y']} value={coordinates === 'fractional' ? object.center! : toCartesian(object.center!, vectors)} onCommit={value => edit(c => { c.geometry.objects![index].center = coordinates === 'fractional' ? value : toFractional(value, vectors); })} />
        </div>
      </div>)}
      {!objects.length && <p className="wb-muted">Homogeneous medium.</p>}
      <button disabled={!template} onClick={() => edit(c => {
        let number = 1; while (objects.some(object => object.name === `rod${number}`)) number++;
        (c.geometry.objects ??= []).push({ ...structuredClone(template!), name: `rod${number}` });
      })}><CirclePlus size={14} />Add circle</button>
      <Choices<'TM' | 'TE'> label="Polarization" value={config.polarization!} options={[
        { value: 'TM', label: 'TM', description: 'Electric field out of plane' },
        { value: 'TE', label: 'TE', description: 'Magnetic field out of plane' },
      ]} onChange={value => edit(c => { c.polarization = value; })} />
    </Section>
  </>;
}
