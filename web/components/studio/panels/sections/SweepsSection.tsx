'use client';

import React from 'react';
import { Plus, Trash2, ArrowUp, ArrowDown } from 'lucide-react';
import { useStudioStore } from '../../../../lib/studio/store';
import {
  sweepParameterOptions,
  type StudioSweep,
  type StudioConfig,
} from '../../../../lib/studio/configModel';
import { Section } from '../Section';

/** Count values for one sweep (mirrors the Rust SweepSpec::count). */
function sweepCount(s: StudioSweep): number {
  if (s.mode === 'values') return s.values.length;
  if (s.step <= 0 || s.max < s.min) return 0;
  return Math.floor((s.max - s.min) / s.step + 1e-9) + 1;
}

function totalJobs(sweeps: StudioSweep[]): number {
  return sweeps.reduce((acc, s) => acc * Math.max(1, sweepCount(s)), 1);
}

/** Is this parameter discrete-only (string-valued)? */
function isDiscreteParam(param: string): boolean {
  return param === 'polarization' || param === 'lattice_type';
}

export function SweepsSection({ hasError }: { hasError?: boolean }) {
  const config = useStudioStore((s) => s.config);
  const sweeps = config.sweeps;
  const patch = useStudioStore((s) => s.patchConfig);
  const options = sweepParameterOptions(config);

  const total = totalJobs(sweeps);

  const addSweep = () => {
    // Pick a parameter not already swept.
    const used = new Set(sweeps.map((s) => s.parameter));
    const next = options.find((o) => !used.has(o)) ?? options[0];
    patch((d) => {
      d.sweeps.push({
        parameter: next,
        mode: isDiscreteParam(next) ? 'values' : 'range',
        min: 0.2,
        max: 0.4,
        step: 0.05,
        values: next === 'polarization' ? ['TM', 'TE'] : next === 'lattice_type' ? ['square', 'triangular'] : [],
      });
    });
  };

  const move = (i: number, dir: -1 | 1) => {
    const j = i + dir;
    if (j < 0 || j >= sweeps.length) return;
    patch((d) => {
      const tmp = d.sweeps[i];
      d.sweeps[i] = d.sweeps[j];
      d.sweeps[j] = tmp;
    });
  };

  return (
    <Section
      id="sweeps"
      title="Sweeps"
      badge={
        hasError
          ? { kind: 'error', text: '!' }
          : sweeps.length > 0
            ? { kind: 'native', text: `${total} jobs` }
            : undefined
      }
    >
      {sweeps.length === 0 ? (
        <div className="studio__hint">
          Add a sweep to run a nested loop over one or more parameters. The first sweep is the
          outermost loop. Base values come from the sections above.
        </div>
      ) : null}

      {sweeps.map((sweep, i) => (
        <SweepRow
          key={i}
          index={i}
          sweep={sweep}
          count={sweepCount(sweep)}
          isFirst={i === 0}
          isLast={i === sweeps.length - 1}
          options={options}
          config={config}
          onMove={move}
          onPatch={patch}
        />
      ))}

      <button type="button" className="studio__add-btn" onClick={addSweep}>
        <Plus size={14} /> Add sweep
      </button>

      {sweeps.length > 0 ? (
        <div className="studio__hint">
          Total: <b style={{ color: '#cfcfcf' }}>{total}</b> job{total === 1 ? '' : 's'}. Sweeps run
          sequentially on one browser core.
        </div>
      ) : null}
    </Section>
  );
}

function SweepRow({
  index,
  sweep,
  count,
  isFirst,
  isLast,
  options,
  onMove,
  onPatch,
}: {
  index: number;
  sweep: StudioSweep;
  count: number;
  isFirst: boolean;
  isLast: boolean;
  options: string[];
  config: StudioConfig;
  onMove: (i: number, dir: -1 | 1) => void;
  onPatch: (m: (d: StudioConfig) => void) => void;
}) {
  const discreteOnly = isDiscreteParam(sweep.parameter);
  const loopLabel = isFirst ? 'outer' : isLast ? 'inner' : 'middle';

  return (
    <div className="studio__atom">
      <div className="studio__atom-head">
        <span className="studio__atom-title">
          {index + 1}. {loopLabel} loop · {count} value{count === 1 ? '' : 's'}
        </span>
        <span style={{ display: 'flex', gap: 2 }}>
          <button
            type="button"
            className="studio__iconbtn"
            aria-label="Move up"
            disabled={isFirst}
            onClick={() => onMove(index, -1)}
          >
            <ArrowUp size={13} />
          </button>
          <button
            type="button"
            className="studio__iconbtn"
            aria-label="Move down"
            disabled={isLast}
            onClick={() => onMove(index, 1)}
          >
            <ArrowDown size={13} />
          </button>
          <button
            type="button"
            className="studio__iconbtn"
            aria-label="Remove sweep"
            onClick={() =>
              onPatch((d) => {
                d.sweeps.splice(index, 1);
              })
            }
          >
            <Trash2 size={13} />
          </button>
        </span>
      </div>

      <select
        className="studio__select"
        value={sweep.parameter}
        onChange={(e) =>
          onPatch((d) => {
            const p = e.target.value;
            d.sweeps[index].parameter = p;
            if (isDiscreteParam(p)) {
              d.sweeps[index].mode = 'values';
              if (d.sweeps[index].values.length === 0) {
                d.sweeps[index].values =
                  p === 'polarization' ? ['TM', 'TE'] : ['square', 'triangular'];
              }
            }
          })
        }
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>

      {!discreteOnly ? (
        <div className="studio__segmented">
          <button
            type="button"
            className={`studio__segbtn${sweep.mode === 'range' ? ' studio__segbtn--active' : ''}`}
            onClick={() => onPatch((d) => (d.sweeps[index].mode = 'range'))}
          >
            Range
          </button>
          <button
            type="button"
            className={`studio__segbtn${sweep.mode === 'values' ? ' studio__segbtn--active' : ''}`}
            onClick={() => onPatch((d) => (d.sweeps[index].mode = 'values'))}
          >
            Values
          </button>
        </div>
      ) : null}

      {sweep.mode === 'range' && !discreteOnly ? (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
          {(['min', 'max', 'step'] as const).map((key) => (
            <div key={key} className="studio__field" style={{ gap: 3 }}>
              <span className="studio__label" style={{ fontSize: 11 }}>
                {key}
              </span>
              <input
                className="studio__input studio__input--num"
                type="number"
                step="0.01"
                value={sweep[key]}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  if (!Number.isNaN(v)) onPatch((d) => (d.sweeps[index][key] = v));
                }}
              />
            </div>
          ))}
        </div>
      ) : (
        <ValuesEditor
          param={sweep.parameter}
          values={sweep.values}
          onChange={(vals) => onPatch((d) => (d.sweeps[index].values = vals))}
        />
      )}
    </div>
  );
}

function ValuesEditor({
  param,
  values,
  onChange,
}: {
  param: string;
  values: (number | string)[];
  onChange: (vals: (number | string)[]) => void;
}) {
  if (param === 'polarization') {
    return (
      <ToggleSet options={['TM', 'TE']} selected={values.map(String)} onChange={onChange} />
    );
  }
  if (param === 'lattice_type') {
    return (
      <ToggleSet
        options={['square', 'rectangular', 'triangular']}
        selected={values.map(String)}
        onChange={onChange}
      />
    );
  }
  // Numeric discrete list as comma-separated text.
  return (
    <input
      className="studio__input studio__input--num"
      value={values.join(', ')}
      placeholder="0.2, 0.3, 0.4"
      onChange={(e) => {
        const parts = e.target.value
          .split(',')
          .map((s) => parseFloat(s.trim()))
          .filter((v) => !Number.isNaN(v));
        onChange(parts);
      }}
    />
  );
}

function ToggleSet({
  options,
  selected,
  onChange,
}: {
  options: string[];
  selected: string[];
  onChange: (vals: string[]) => void;
}) {
  const toggle = (opt: string) => {
    const has = selected.includes(opt);
    const next = has ? selected.filter((s) => s !== opt) : [...selected, opt];
    onChange(next.length > 0 ? next : [opt]);
  };
  return (
    <div className="studio__segmented">
      {options.map((opt) => (
        <button
          key={opt}
          type="button"
          className={`studio__segbtn${selected.includes(opt) ? ' studio__segbtn--active' : ''}`}
          onClick={() => toggle(opt)}
        >
          {opt}
        </button>
      ))}
    </div>
  );
}
