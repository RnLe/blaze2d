'use client';

import React from 'react';
import { useStudioStore } from '../../../../lib/studio/store';
import type { PathPreset } from '../../../../lib/studio/configModel';
import { resolvePreset } from '../../../../lib/studio/geometry';
import { Section } from '../Section';
import { SliderField, SelectField, Segmented } from '../Controls';

export function PathSection({ hasError }: { hasError?: boolean }) {
  const path = useStudioStore((s) => s.config.path);
  const latticeKind = useStudioStore((s) => s.config.geometry.lattice.kind);
  const patch = useStudioStore((s) => s.patchConfig);
  const setCenterTab = useStudioStore((s) => s.setCenterTab);

  const resolved = resolvePreset(path.preset, latticeKind);

  return (
    <Section id="path" title="K-path" badge={hasError ? { kind: 'error', text: '!' } : undefined}>
      <div className="studio__field">
        <span className="studio__label">Mode</span>
        <Segmented<'preset' | 'points'>
          value={path.mode}
          options={[
            { value: 'preset', label: 'Preset' },
            { value: 'points', label: 'Custom points' },
          ]}
          onChange={(v) => {
            patch((d) => (d.path.mode = v));
            if (v === 'points') setCenterTab('reciprocal');
          }}
        />
      </div>

      {path.mode === 'preset' ? (
        <>
          <SelectField<PathPreset>
            label="Preset"
            value={path.preset}
            options={[
              { value: 'auto', label: 'Auto (from lattice)' },
              { value: 'square', label: 'Square: Γ-X-M-Γ' },
              { value: 'triangular', label: 'Triangular: Γ-M-K-Γ' },
              { value: 'rectangular', label: 'Rectangular: Γ-X-S-Y-Γ' },
            ]}
            onChange={(v) => patch((d) => (d.path.preset = v))}
          />
          <SliderField
            label="Points per segment"
            value={path.pointsPerSegment}
            min={2}
            max={40}
            step={1}
            format={(v) => `${v}`}
            onChange={(v) => patch((d) => (d.path.pointsPerSegment = Math.round(v)))}
          />
          {path.preset === 'auto' && resolved === null ? (
            <div className="studio__hint">
              {latticeKind} lattices have no standard high-symmetry path. Switch to custom points.
            </div>
          ) : (
            <div className="studio__hint">Resolves to the {resolved ?? 'square'} path.</div>
          )}
        </>
      ) : (
        <div className="studio__field">
          <div className="studio__field-row">
            <span className="studio__label">{path.points.length} k-points</span>
            <button
              type="button"
              className="studio__iconbtn"
              style={{ color: '#9a9a9a', fontSize: 11 }}
              onClick={() => setCenterTab('reciprocal')}
            >
              edit on zone →
            </button>
          </div>
          <div className="studio__hint">
            On the Brillouin zone view, click to add a k-point, drag to move, click a point to remove
            it. Points are fractional reciprocal coordinates.
          </div>
        </div>
      )}
    </Section>
  );
}
