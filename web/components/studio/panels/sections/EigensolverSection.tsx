'use client';

import React from 'react';
import { useStudioStore } from '../../../../lib/studio/store';
import { LIMITS, clamp } from '../../../../lib/studio/configModel';
import { Section } from '../Section';
import { SliderField, NumberField, CheckboxField } from '../Controls';

export function EigensolverSection({ hasError }: { hasError?: boolean }) {
  const eig = useStudioStore((s) => s.config.eigensolver);
  const patch = useStudioStore((s) => s.patchConfig);

  return (
    <Section
      id="eigensolver"
      title="Eigensolver"
      badge={hasError ? { kind: 'error', text: '!' } : undefined}
    >
      <SliderField
        label="Bands"
        value={eig.n_bands}
        min={LIMITS.nBandsMin}
        max={30}
        step={1}
        format={(v) => `${v}`}
        onChange={(v) =>
          patch((d) => (d.eigensolver.n_bands = clamp(Math.round(v), LIMITS.nBandsMin, LIMITS.nBandsMax)))
        }
      />
      <NumberField
        label="Max iterations"
        value={eig.max_iter}
        min={1}
        step={50}
        onChange={(v) => patch((d) => (d.eigensolver.max_iter = Math.max(1, Math.round(v))))}
      />
      <div className="studio__field-row">
        <span className="studio__label">Tolerance</span>
        <select
          className="studio__select"
          style={{ width: 100 }}
          value={String(eig.tol)}
          onChange={(e) => patch((d) => (d.eigensolver.tol = parseFloat(e.target.value)))}
        >
          <option value="0.01">1e-2</option>
          <option value="0.001">1e-3</option>
          <option value="0.0001">1e-4</option>
          <option value="0.000001">1e-6</option>
          <option value="0.00000001">1e-8</option>
        </select>
      </div>
      <CheckboxField
        label="Record convergence diagnostics"
        checked={eig.record_diagnostics}
        onChange={(v) => patch((d) => (d.eigensolver.record_diagnostics = v))}
      />
    </Section>
  );
}
