'use client';

import React from 'react';
import { Plus, Trash2 } from 'lucide-react';
import { useStudioStore } from '../../../../lib/studio/store';
import { LIMITS, clamp, type LatticeKind } from '../../../../lib/studio/configModel';
import { Section } from '../Section';
import { SliderField, SelectField, NumberField } from '../Controls';

const LATTICE_OPTIONS: { value: LatticeKind; label: string }[] = [
  { value: 'square', label: 'Square' },
  { value: 'rectangular', label: 'Rectangular' },
  { value: 'triangular', label: 'Triangular / hexagonal' },
  { value: 'oblique', label: 'Oblique' },
  { value: 'custom', label: 'Custom (a1, a2)' },
];

export function GeometrySection({ hasError }: { hasError?: boolean }) {
  const geometry = useStudioStore((s) => s.config.geometry);
  const selectedAtom = useStudioStore((s) => s.ui.selectedAtom);
  const selectAtom = useStudioStore((s) => s.selectAtom);
  const patch = useStudioStore((s) => s.patchConfig);
  const lat = geometry.lattice;

  return (
    <Section
      id="geometry"
      title="Geometry"
      badge={hasError ? { kind: 'error', text: '!' } : undefined}
    >
      <SliderField
        label="Background ε"
        value={geometry.eps_bg}
        min={1}
        max={20}
        step={0.1}
        format={(v) => v.toFixed(1)}
        onChange={(v) => patch((d) => (d.geometry.eps_bg = clamp(v, LIMITS.epsMin, LIMITS.epsMax)))}
      />

      <SelectField<LatticeKind>
        label="Lattice"
        value={lat.kind}
        options={LATTICE_OPTIONS}
        onChange={(v) =>
          patch((d) => {
            d.geometry.lattice.kind = v;
            // Fill required fields with sane defaults when switching type.
            if (v === 'rectangular' && d.geometry.lattice.b === null) d.geometry.lattice.b = 1.5;
            if (v === 'oblique') {
              if (d.geometry.lattice.b === null) d.geometry.lattice.b = 1.2;
              if (d.geometry.lattice.alpha_deg === null) d.geometry.lattice.alpha_deg = 75;
            }
            if (v === 'custom') {
              if (d.geometry.lattice.a1 === null) d.geometry.lattice.a1 = [1, 0];
              if (d.geometry.lattice.a2 === null) d.geometry.lattice.a2 = [0, 1];
            }
          })
        }
      />

      {lat.kind !== 'custom' ? (
        <SliderField
          label="a"
          value={lat.a}
          min={0.5}
          max={2}
          step={0.05}
          format={(v) => v.toFixed(2)}
          onChange={(v) => patch((d) => (d.geometry.lattice.a = v))}
        />
      ) : null}

      {lat.kind === 'rectangular' || lat.kind === 'oblique' ? (
        <SliderField
          label="b"
          value={lat.b ?? 1.5}
          min={0.5}
          max={2.5}
          step={0.05}
          format={(v) => v.toFixed(2)}
          onChange={(v) => patch((d) => (d.geometry.lattice.b = v))}
        />
      ) : null}

      {lat.kind === 'oblique' ? (
        <SliderField
          label="α"
          value={lat.alpha_deg ?? 75}
          min={LIMITS.alphaMin}
          max={LIMITS.alphaMax}
          step={1}
          unit="deg"
          format={(v) => v.toFixed(0)}
          onChange={(v) => patch((d) => (d.geometry.lattice.alpha_deg = v))}
        />
      ) : null}

      {lat.kind === 'custom' ? (
        <div className="studio__field" style={{ gap: 8 }}>
          <span className="studio__label">Basis vectors</span>
          <NumberField
            label="a1.x"
            value={lat.a1?.[0] ?? 1}
            step={0.05}
            onChange={(v) => patch((d) => (d.geometry.lattice.a1 = [v, d.geometry.lattice.a1?.[1] ?? 0]))}
          />
          <NumberField
            label="a1.y"
            value={lat.a1?.[1] ?? 0}
            step={0.05}
            onChange={(v) => patch((d) => (d.geometry.lattice.a1 = [d.geometry.lattice.a1?.[0] ?? 1, v]))}
          />
          <NumberField
            label="a2.x"
            value={lat.a2?.[0] ?? 0}
            step={0.05}
            onChange={(v) => patch((d) => (d.geometry.lattice.a2 = [v, d.geometry.lattice.a2?.[1] ?? 1]))}
          />
          <NumberField
            label="a2.y"
            value={lat.a2?.[1] ?? 1}
            step={0.05}
            onChange={(v) => patch((d) => (d.geometry.lattice.a2 = [d.geometry.lattice.a2?.[0] ?? 0, v]))}
          />
        </div>
      ) : null}

      {/* Atoms */}
      <div className="studio__field" style={{ gap: 8 }}>
        <span className="studio__label">Atoms ({geometry.atoms.length})</span>
        {geometry.atoms.map((atom, i) => (
          <div
            key={i}
            className={`studio__atom${selectedAtom === i ? ' studio__atom--selected' : ''}`}
            onClick={() => selectAtom(i)}
          >
            <div className="studio__atom-head">
              <span className="studio__atom-title">Atom {i}</span>
              {geometry.atoms.length > 1 ? (
                <button
                  type="button"
                  className="studio__iconbtn"
                  aria-label={`Remove atom ${i}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    patch((d) => {
                      d.geometry.atoms.splice(i, 1);
                    });
                    selectAtom(0);
                  }}
                >
                  <Trash2 size={13} />
                </button>
              ) : null}
            </div>
            <SliderField
              label="Radius"
              value={atom.radius}
              min={LIMITS.radiusMin}
              max={LIMITS.radiusMax}
              step={0.01}
              format={(v) => v.toFixed(2)}
              onChange={(v) => patch((d) => (d.geometry.atoms[i].radius = v))}
            />
            <SliderField
              label="ε inside"
              value={atom.eps_inside}
              min={1}
              max={20}
              step={0.1}
              format={(v) => v.toFixed(1)}
              onChange={(v) => patch((d) => (d.geometry.atoms[i].eps_inside = v))}
            />
            <NumberField
              label="pos x"
              value={atom.pos[0]}
              min={0}
              max={0.999}
              step={0.01}
              onChange={(v) =>
                patch((d) => (d.geometry.atoms[i].pos[0] = clamp(v, LIMITS.posMin, LIMITS.posMax)))
              }
            />
            <NumberField
              label="pos y"
              value={atom.pos[1]}
              min={0}
              max={0.999}
              step={0.01}
              onChange={(v) =>
                patch((d) => (d.geometry.atoms[i].pos[1] = clamp(v, LIMITS.posMin, LIMITS.posMax)))
              }
            />
          </div>
        ))}
        <button
          type="button"
          className="studio__add-btn"
          onClick={() =>
            patch((d) => {
              d.geometry.atoms.push({ pos: [0.25, 0.25], radius: 0.15, eps_inside: 1.0 });
            })
          }
        >
          <Plus size={14} /> Add atom
        </button>
      </div>
    </Section>
  );
}
