'use client';

import React from 'react';
import { useStudioStore } from '../../../../lib/studio/store';
import type { OutputMode } from '../../../../lib/studio/configModel';
import { Section } from '../Section';
import { Segmented } from '../Controls';

/** Parse a comma-separated list of integers. */
function parseInts(text: string): number[] {
  return text
    .split(',')
    .map((s) => parseInt(s.trim(), 10))
    .filter((v) => !Number.isNaN(v));
}

export function OutputSection({ hasError }: { hasError?: boolean }) {
  const output = useStudioStore((s) => s.config.output);
  const patch = useStudioStore((s) => s.patchConfig);

  return (
    <Section
      id="output"
      title="Output"
      badge={hasError ? { kind: 'error', text: '!' } : undefined}
    >
      <div className="studio__field">
        <span className="studio__label">Mode</span>
        <Segmented<OutputMode>
          value={output.mode}
          options={[
            { value: 'full', label: 'Full' },
            { value: 'selective', label: 'Selective' },
          ]}
          onChange={(v) => patch((d) => (d.output.mode = v))}
        />
      </div>

      {output.mode === 'selective' ? (
        <>
          <div className="studio__field">
            <span className="studio__label">k-indices (0-based)</span>
            <input
              className="studio__input studio__input--num"
              value={output.selective.k_indices.join(', ')}
              placeholder="0, 12, 24"
              onChange={(e) => patch((d) => (d.output.selective.k_indices = parseInts(e.target.value)))}
            />
          </div>
          <div className="studio__field">
            <span className="studio__label">Bands (1-based)</span>
            <input
              className="studio__input studio__input--num"
              value={output.selective.bands.join(', ')}
              placeholder="1, 2, 3, 4"
              onChange={(e) => patch((d) => (d.output.selective.bands = parseInts(e.target.value)))}
            />
          </div>
          <div className="studio__hint">
            Selective output writes one merged CSV of the chosen k-points and bands. This affects the
            native CSV writer; the in-browser run always streams all bands.
          </div>
        </>
      ) : (
        <div className="studio__hint">
          Full mode writes one CSV per job (native driver). In the browser, results are shown live
          and exported from the results panel.
        </div>
      )}
    </Section>
  );
}
