'use client';

import React from 'react';
import { useStudioStore } from '../../../../lib/studio/store';
import type { Polarization, Precision, SolverType } from '../../../../lib/studio/configModel';
import { Section } from '../Section';
import { Segmented, SelectField } from '../Controls';

export function SolverSection() {
  const solver = useStudioStore((s) => s.config.solver);
  const patch = useStudioStore((s) => s.patchConfig);

  return (
    <Section id="solver" title="Solver">
      <div className="studio__field">
        <span className="studio__label">Polarization</span>
        <Segmented<Polarization>
          value={solver.polarization}
          options={[
            { value: 'TM', label: 'TM (E out of plane)' },
            { value: 'TE', label: 'TE (H out of plane)' },
          ]}
          onChange={(v) => patch((d) => (d.solver.polarization = v))}
        />
      </div>

      <div className="studio__field">
        <span className="studio__label">Storage precision</span>
        <Segmented<Precision>
          value={solver.precision}
          options={[
            { value: 'f64', label: 'f64' },
            { value: 'f32', label: 'f32' },
          ]}
          onChange={(v) => patch((d) => (d.solver.precision = v))}
        />
        {solver.precision === 'f32' ? (
          <div className="studio__hint">
            f32 storage, f64 accumulation. In this single-threaded browser build the benefit is
            memory bandwidth, not parallelism.
          </div>
        ) : null}
      </div>

      <SelectField<SolverType>
        label="Solver type"
        value={solver.type}
        options={[
          { value: 'maxwell', label: 'Maxwell (band structure)' },
          { value: 'operator_data', label: 'Operator data (native only)' },
        ]}
        onChange={(v) => patch((d) => (d.solver.type = v))}
      />
      {solver.type === 'operator_data' ? (
        <div className="studio__hint">
          Operator-data extraction runs on the native driver only. Configure it here and export the
          TOML; in-browser runs are disabled for this solver.
        </div>
      ) : null}
    </Section>
  );
}
