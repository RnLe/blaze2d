'use client';

import React from 'react';
import { useStudioStore } from '../../../../lib/studio/store';
import type { SmoothingKind } from '../../../../lib/studio/configModel';
import { Section } from '../Section';
import { SelectField, SliderField } from '../Controls';

export function DielectricSection({ hasError }: { hasError?: boolean }) {
  const diel = useStudioStore((s) => s.config.dielectric);
  const patch = useStudioStore((s) => s.patchConfig);

  return (
    <Section
      id="dielectric"
      title="Dielectric"
      badge={hasError ? { kind: 'error', text: '!' } : undefined}
    >
      <SelectField<SmoothingKind>
        label="Interface smoothing"
        value={diel.smoothing}
        options={[
          { value: 'analytic', label: 'Analytic (MPB-style)' },
          { value: 'subgrid', label: 'Subgrid sampling' },
          { value: 'none', label: 'None' },
        ]}
        onChange={(v) => patch((d) => (d.dielectric.smoothing = v))}
      />
      {diel.smoothing === 'subgrid' ? (
        <SliderField
          label="Mesh size"
          value={diel.mesh_size}
          min={2}
          max={8}
          step={1}
          format={(v) => `${v}`}
          onChange={(v) => patch((d) => (d.dielectric.mesh_size = Math.round(v)))}
        />
      ) : null}
      <div className="studio__hint">
        Analytic smoothing uses exact filling fractions at circular interfaces (recommended). Subgrid
        integrates over a mesh; none samples pixels directly.
      </div>
    </Section>
  );
}
