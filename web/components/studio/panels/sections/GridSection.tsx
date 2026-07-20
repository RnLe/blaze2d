'use client';

import React from 'react';
import { useStudioStore } from '../../../../lib/studio/store';
import { LIMITS, clamp } from '../../../../lib/studio/configModel';
import { Section } from '../Section';
import { SliderField, CheckboxField, NumberField } from '../Controls';

export function GridSection({ hasError }: { hasError?: boolean }) {
  const grid = useStudioStore((s) => s.config.grid);
  const patch = useStudioStore((s) => s.patchConfig);
  const linked = grid.ny === null;

  return (
    <Section id="grid" title="Grid" badge={hasError ? { kind: 'error', text: '!' } : undefined}>
      <SliderField
        label="Resolution (nx)"
        value={grid.nx}
        min={LIMITS.resolutionMin}
        max={128}
        step={2}
        format={(v) => `${v}`}
        onChange={(v) =>
          patch((d) => (d.grid.nx = clamp(Math.round(v), LIMITS.resolutionMin, LIMITS.resolutionMax)))
        }
      />
      <CheckboxField
        label="ny follows nx (square grid)"
        checked={linked}
        onChange={(v) =>
          patch((d) => {
            d.grid.ny = v ? null : d.grid.nx;
          })
        }
      />
      {!linked ? (
        <NumberField
          label="ny"
          value={grid.ny ?? grid.nx}
          min={LIMITS.resolutionMin}
          max={LIMITS.resolutionMax}
          step={2}
          onChange={(v) => patch((d) => (d.grid.ny = clamp(Math.round(v), LIMITS.resolutionMin, LIMITS.resolutionMax)))}
        />
      ) : null}
      <CheckboxField
        label="Centered coordinates"
        checked={grid.centered}
        onChange={(v) => patch((d) => (d.grid.centered = v))}
      />
      <div className="studio__hint">
        Higher resolution is more accurate but slower. 32 is good for interactive runs; 64 to 128
        for production.
      </div>
    </Section>
  );
}
