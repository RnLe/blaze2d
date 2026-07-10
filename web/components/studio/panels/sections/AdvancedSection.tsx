'use client';

import React from 'react';
import { useStudioStore } from '../../../../lib/studio/store';
import { Section } from '../Section';
import { CheckboxField, NumberField, SliderField } from '../Controls';

/**
 * Run flags plus the operator-data (EA) extraction surface. Operator-data is
 * config-only in the browser: the run panel disables in-browser runs for it and
 * marks it native-only. Everything here still generates valid TOML for the CLI
 * and Python driver.
 */
export function AdvancedSection() {
  const run = useStudioStore((s) => s.config.run);
  const solverType = useStudioStore((s) => s.config.solver.type);
  const od = useStudioStore((s) => s.config.operatorData);
  const patch = useStudioStore((s) => s.patchConfig);

  const isOperatorData = solverType === 'operator_data';

  return (
    <Section
      id="advanced"
      title="Advanced"
      badge={isOperatorData ? { kind: 'native', text: 'native' } : undefined}
    >
      <div className="studio__field" style={{ gap: 8 }}>
        <span className="studio__label">Run flags</span>
        <CheckboxField
          label="Skip final Γ (reuse initial)"
          checked={run.skip_final_gamma}
          onChange={(v) => patch((d) => (d.run.skip_final_gamma = v))}
        />
        <CheckboxField
          label="Disable band tracking"
          checked={run.disable_band_tracking}
          onChange={(v) => patch((d) => (d.run.disable_band_tracking = v))}
        />
        <CheckboxField
          label="Verbose logging (native)"
          checked={run.verbose}
          onChange={(v) => patch((d) => (d.run.verbose = v))}
        />
      </div>

      {isOperatorData ? (
        <div className="studio__field" style={{ gap: 10 }}>
          <span className="studio__label">Operator-data extraction</span>
          <div className="studio__hint">
            Extracts velocity matrices, the mass tensor, and the Born-Huang potential at one (R, k₀)
            point. Runs on the native driver only; export the TOML to run it.
          </div>
          <SliderField
            label="Retained bands"
            value={od.n_retained}
            min={1}
            max={12}
            step={1}
            format={(v) => `${v}`}
            onChange={(v) => patch((d) => (d.operatorData.n_retained = Math.round(v)))}
          />
          <SliderField
            label="Remote bands"
            value={od.n_remote}
            min={0}
            max={24}
            step={1}
            format={(v) => `${v}`}
            onChange={(v) => patch((d) => (d.operatorData.n_remote = Math.round(v)))}
          />
          <NumberField
            label="k0.x"
            value={od.k0[0]}
            step={0.05}
            onChange={(v) => patch((d) => (d.operatorData.k0 = [v, d.operatorData.k0[1]]))}
          />
          <NumberField
            label="k0.y"
            value={od.k0[1]}
            step={0.05}
            onChange={(v) => patch((d) => (d.operatorData.k0 = [d.operatorData.k0[0], v]))}
          />
          <CheckboxField
            label="Compute mass tensor"
            checked={od.compute_mass_tensor}
            onChange={(v) => patch((d) => (d.operatorData.compute_mass_tensor = v))}
          />
          <CheckboxField
            label="Compute Born-Huang potential"
            checked={od.compute_born_huang}
            onChange={(v) => patch((d) => (d.operatorData.compute_born_huang = v))}
          />
          <CheckboxField
            label="Compute R-derivatives"
            checked={od.compute_r_derivatives}
            onChange={(v) => patch((d) => (d.operatorData.compute_r_derivatives = v))}
          />
        </div>
      ) : null}
    </Section>
  );
}
