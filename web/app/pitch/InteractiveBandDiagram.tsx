'use client';
import Link from 'next/link';
import { useCallback, useEffect, useId, useState, useSyncExternalStore } from 'react';
import { Columns2, Layers2 } from 'lucide-react';
import { ExecutionController, Validator, type Applied, type ExecutionState } from '@/lib/compute/controller';
import { diagnostic } from '@/lib/compute/protocol';
import { CRYSTALS, DEFAULT_SETTINGS, demoConfig, demoSeries, type DemoSettings, type Polarization } from './demo-model';
import DemoGeometry from './DemoGeometry';
import DemoBandPlot from './DemoBandPlot';
import '@/components/workbench/workbench.css';

const EMPTY: ExecutionState = { run: null, live: [] };
const POLARIZATIONS: Polarization[] = ['TM', 'TE'];
const crystalName = (crystal: string) => crystal[0].toUpperCase() + crystal.slice(1);

function Slider({ label, value, min, max, step, onChange, disabled = false, digits = 2 }: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (value: number) => void; disabled?: boolean; digits?: number;
}) {
  const id = useId();
  return <div className={`pitch-demo-slider${disabled ? ' is-inactive' : ''}`}>
    <label htmlFor={id}><span>{label}</span><output htmlFor={id}>{value.toFixed(digits)}</output></label>
    <input id={id} aria-label={label} type="range" min={min} max={max} step={step} value={value} disabled={disabled}
      onChange={event => onChange(event.currentTarget.valueAsNumber)} />
  </div>;
}

export default function InteractiveBandDiagram() {
  const [validator, setValidator] = useState<Validator>(), [controller, setController] = useState<ExecutionController>();
  const [base, setBase] = useState<Applied>();
  const [settings, setSettings] = useState<DemoSettings>(DEFAULT_SETTINGS);
  const [ready, setReady] = useState<{ key: string; applied: Applied }>();
  const [layout, setLayout] = useState<'overlaid' | 'separate'>('overlaid');
  const [error, setError] = useState(''), [initializationError, setInitializationError] = useState('');
  const key = JSON.stringify(settings);
  const subscribe = useCallback((listener: () => void) => controller?.subscribe(listener) ?? (() => {}), [controller]);
  const state = useSyncExternalStore(subscribe, controller?.snapshot ?? (() => EMPTY), () => EMPTY);
  const running = state.run?.header.status === 'running';

  useEffect(() => {
    const validator = new Validator(process.env.NEXT_PUBLIC_BASE_PATH ?? ''), controller = new ExecutionController(process.env.NEXT_PUBLIC_BASE_PATH ?? '');
    let current = true;
    setValidator(validator); setController(controller);
    void validator.validate().then(base => {
      if (!base.report.ok) throw base.report.errors[0];
      if (current) setBase(base);
    }).catch(error => { if (current) setInitializationError(diagnostic(error).message); });
    return () => { current = false; validator.dispose(); controller.dispose(); };
  }, []);

  useEffect(() => {
    if (!validator || !base) return;
    let current = true;
    // Coalesce slider movement, but never disable the sliders or drop the final value.
    const timer = setTimeout(() => {
      void (async () => {
        const applied = await validator.edit(base, demoConfig(settings));
        if (!current) return;
        if (!applied.report.ok) throw applied.report.errors[0];
        if (current) { setReady({ key, applied }); setError(''); }
      })().catch(error => { if (current) setError(diagnostic(error).message); });
    }, 120);
    return () => { current = false; clearTimeout(timer); };
  }, [validator, base, settings, key]);

  function change<K extends keyof DemoSettings>(name: K, value: DemoSettings[K]) {
    if (running) return;
    setError(''); setSettings(previous => ({ ...previous, [name]: value }));
  }
  const current = ready?.key === key;
  const matchingRun = current && state.run?.header.source === ready.applied.source;
  const resolved = ready?.applied.report.resolved ?? undefined;
  const series = matchingRun ? POLARIZATIONS.flatMap(polarization => {
    const data = demoSeries(state, polarization, settings.bands);
    return data ? [data] : [];
  }) : [];
  const largestFrequency = series.reduce((largest, data) => data.frequencies.reduce((largest, value) => Number.isFinite(value) ? Math.max(largest, value) : largest, largest), 0);
  const maximumY = largestFrequency ? Math.ceil(largestFrequency * 1.06 * 10) / 10 : 1;
  const status = !current ? 'Preparing crystal…' : !matchingRun ? 'Ready to calculate TM and TE.' : running
    ? state.progress ? `${state.job?.resolved.config.polarization ?? 'TM'} · ${state.progress.completed}/${state.progress.total} k-points` : 'Starting calculation…'
    : state.run!.header.status.replaceAll('_', ' ');

  return <div className="workbench-shell pitch-demo" data-crystal={settings.crystal} data-ready={current ?? false}
    data-shown-bands={settings.bands} data-computed-bands={resolved?.solved_bands}>
    <div className="pitch-demo-top"><span>Live browser calculation <small>f64</small></span><Link href="/workbench/">Open Workbench ↗</Link></div>
    <div className="pitch-demo-grid">
      <div className="pitch-demo-model">
        <DemoGeometry settings={settings} />
        <fieldset disabled={running}>
          <div className="pitch-demo-crystals" role="group" aria-label="Crystal model">{CRYSTALS.map(crystal =>
            <button type="button" key={crystal} aria-pressed={settings.crystal === crystal} onClick={() => change('crystal', crystal)}>{crystalName(crystal)}</button>)}</div>
          <div className="pitch-demo-sliders">
            <Slider label="Radius r/a" value={settings.radius} min={0.05} max={0.45} step={0.01} onChange={value => change('radius', value)} />
            <Slider label="Background ε" value={settings.background} min={1} max={13} step={0.1} digits={1} onChange={value => change('background', value)} />
            <Slider label="Object ε" value={settings.inclusion} min={1} max={13} step={0.1} digits={1} onChange={value => change('inclusion', value)} />
            <Slider label="Rectangular a/b" value={settings.ratio} min={0.5} max={2} step={0.05} disabled={settings.crystal !== 'rectangular'} onChange={value => change('ratio', value)} />
            <Slider label="Bands shown" value={settings.bands} min={4} max={12} step={1} digits={0} onChange={value => change('bands', value)} />
          </div>
        </fieldset>
        {running ? <button type="button" className="wb-run" onClick={() => controller?.cancel()}>Cancel calculation</button>
          : <button type="button" className="wb-run" disabled={!current || !!error || !!initializationError} onClick={() => {
            if (!ready || ready.key !== key) return;
            try { controller?.start(ready.applied, `${crystalName(settings.crystal)} crystal`); }
            catch (error) { setError(diagnostic(error).message); }
          }}>Calculate TM &amp; TE</button>}
        <p className="pitch-demo-resolution">32 × 32 grid · {resolved?.k_points_cartesian.length ?? 46} k-points</p>
      </div>
      <div className="pitch-demo-results">
        <div className="pitch-demo-results-top">
          <div className="pitch-demo-layout" role="group" aria-label="Polarization layout">
            <button type="button" aria-pressed={layout === 'overlaid'} onClick={() => setLayout('overlaid')}><Layers2 size={16} aria-hidden="true" />Overlaid</button>
            <button type="button" aria-pressed={layout === 'separate'} onClick={() => setLayout('separate')}><Columns2 size={16} aria-hidden="true" />Separate</button>
          </div>
          <div className="pitch-demo-polarizations" aria-label="Polarization colors"><span className="is-tm">TM</span><span className="is-te">TE</span></div>
        </div>
        <p className="pitch-demo-status" role="status" aria-live="polite">{status}</p>
        {(initializationError || error) && <p className="wb-error" role="alert">{initializationError || error}</p>}
        {matchingRun && state.run?.failure && <p className="wb-error" role="alert">{state.run.failure.message}</p>}
        {matchingRun && state.run?.errors.map(error => <p className="wb-error" role="alert" key={error.job_index}>{error.diagnostic.message}</p>)}
        <div className={`pitch-demo-charts${layout === 'separate' ? ' is-separated' : ''}`}>
          {layout === 'overlaid' ? <DemoBandPlot path={resolved} series={series} title="TM & TE bands" maximumY={maximumY} pending={running} />
            : POLARIZATIONS.map(polarization => <DemoBandPlot key={polarization} path={resolved} series={series.filter(data => data.polarization === polarization)} title={`${polarization} bands`} maximumY={maximumY} pending={running} />)}
        </div>
        {state.storageError && <p className="wb-muted" role="status">{state.storageError}</p>}
      </div>
    </div>
  </div>;
}
