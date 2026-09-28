#!/usr/bin/env python3
"""Consolidate admitted results without selecting attempts or re-estimating gains."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from qualitative_timing import qualitative_decision
from replay_quiet import HERE, digest


def read(path):
    return json.loads(Path(path).read_text())


def local(path):
    return str(Path(path).resolve().relative_to(HERE))


def primary_summary(audit, evidence):
    summary = audit.get('summary', audit.get('new_summary'))
    if 'geometric_speedup' in summary:
        contrasts = dict(paired=dict(geometric_ratio=summary['geometric_speedup'], ci95=summary['ci95']))
    else:
        contrasts = {k: dict(geometric_ratio=v.get('geometric_ratio', v.get('geometric_value')), ci95=v['ci95'])
                     for k, v in summary.items() if isinstance(v, dict)
                     and ('geometric_ratio' in v or 'geometric_value' in v)}
    if not contrasts:
        raise ValueError('No recorded admitted contrast: '+str(evidence))
    if 'median_seconds' in summary:
        seconds, rss = summary['median_seconds'], summary['median_rss_kib']
    else:
        samples = read(evidence/'clock-normalized.json')['samples']
        grouped = {}
        for sample in samples:
            if sample.get('pair', sample.get('block')) < 0:
                continue
            grouped.setdefault(sample.get('label', sample.get('mode')), []).append(sample['result'])
        seconds = {k: statistics.median(s['seconds'] for s in v) for k, v in grouped.items()}
        rss = {k: statistics.median(s['peak_rss_kib'] for s in v)
               if all(s.get('peak_rss_kib') is not None for s in v) else None for k, v in grouped.items()}
    return dict(contrasts=contrasts, median_seconds=seconds, median_rss_kib=rss)



def failure_examples(evidence):
    path = evidence/'samples.json'
    if not path.exists():
        return []
    examples = {}
    for sample in read(path)['samples']:
        if sample.get('valid') is True:
            continue
        label = sample.get('label', sample.get('mode'))
        if label in examples:
            continue
        value = sample.get('result', {})
        points = value.get('points', [])
        def maximum(values):
            values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
            return max(values) if values else None
        relative = [r for point in points for r in point.get('residuals', [])[int(point.get('k') == [0.0, 0.0]):]]
        zero = [point['absolute_residuals'][0] for point in points
                if point.get('k') == [0.0, 0.0] and point.get('absolute_residuals')]
        examples[label] = dict(label=label, sample_block=sample.get('pair', sample.get('block')),
                               returncode=sample.get('returncode'), requested_path_valid=sample.get('requested_path_valid'),
                               rejection=value.get('rejection'), completed_points=len(points),
                               expected_points=value.get('expected_points'),
                               maximum_nonzero_relative_residual=maximum(relative),
                               maximum_gamma_kernel_absolute_residual=maximum(zero),
                               maximum_mass_orthogonality_defect=maximum(p.get('b_orthogonality_defect') for p in points),
                               unconverged_points=[i for i, p in enumerate(points) if p.get('converged') is not True])
    return list(examples.values())

def summarize(cohort, coverage_file='coverage.json'):
    inventory = read(cohort/'inventory.json')
    coverage = read(cohort/coverage_file)
    declarations = {s['name']: s for s in inventory['studies']}
    states = [r['study'] for r in coverage['studies']]
    if len(states) != len(set(states)) or not set(states) <= set(declarations):
        raise ValueError('Duplicated or undeclared comparison')
    results = []
    roots = [Path(p) for p in inventory['completed_roots']]+[cohort]
    for row in coverage['studies']:
        declared = declarations[row['study']]
        result = dict(study=row['study'], family=declared['family'], state=row['state'],
                      attempt_count=row['attempts'], original=local(declared['original']),
                      original_sha256=declared['source_sha256'], instrumented=declared['instrumented'],
                      accepted_timing=False)
        evidence = Path(row['evidence']) if 'evidence' in row else None
        if evidence is None:
            attempts = []
            for root in roots:
                for directory in root.glob(row['study']+'*'):
                    if (directory.name == row['study'] or directory.name.startswith(row['study']+'-attempt-')) and (directory/'audit.json').is_file():
                        audit = read(directory/'audit.json')
                        attempts.append(dict(evidence=local(directory), audit_sha256=digest(directory/'audit.json'),
                                             phase=audit.get('phase'), complete=audit.get('complete'),
                                             numerical_qualified=audit.get('numerical_qualified'),
                                             clock_qualified=audit.get('clock_qualified'),
                                             environment_qualified=audit.get('environment_qualified')))
            result['unselected_attempts'] = attempts
            results.append(result)
            continue
        audit = read(evidence/'audit.json')
        manifest = read(evidence/'manifest.json')
        if row.get('audit_sha256') and digest(evidence/'audit.json') != row['audit_sha256']:
            raise ValueError('Selected audit changed: '+row['study'])
        result.update(evidence=local(evidence), audit_sha256=digest(evidence/'audit.json'),
                      manifest_sha256=digest(evidence/'manifest.json'),
                      numerical_qualified=audit.get('numerical_qualified'),
                      clock_qualified=audit.get('clock_qualified'), complete=audit.get('complete'))
        if declared['family'] == 'record':
            variants = {}
            with (evidence/'records.jsonl').open() as stream:
                for line in stream:
                    record = json.loads(line)
                    label = record['mode'] if manifest['design']['kind'] == 'interaction' else record['label']
                    variants.setdefault(label, dict(arguments=record['command'][4:], affinity=record['command'][2],
                                                     executable_sha256=manifest['artifacts'][record['command'][3]]))
                    if record['pair'] >= 0:
                        break
            result['contract'] = dict(design=manifest['design'], variants=variants,
                                      numerical_gate=manifest['numerical_gate'])
        else:
            contract = manifest['original_manifest']
            result['contract'] = {k: v for k, v in contract.items() if k not in
                                  ('output', 'baseline', 'candidate', 'binary', 'snapshot', 'load_average',
                                   'timestamp', 'harness_sha256', 'source', 'builds')}
        result['measurement_kind'] = ('workflow' if declared['family'] in ('record', 'coarse') else
                                      'frozen_trial_algebra' if 'columns' in result['contract'] else
                                      'operator_kernel' if 'repetitions' in result['contract'] else 'solver_or_interface')
        if row['state'] == 'primary qualified':
            if not audit.get('primary_latency_qualified'):
                raise ValueError('Coverage claims timing without primary admission')
            result['retained_blocks'] = list(range(20))
            result['excluded_blocks'] = []
            result['filtered'] = primary_summary(audit, evidence)
            result['all_blocks'] = result['filtered']
        elif row['state'] == 'qualitative qualified':
            analysis = read(row['analysis'])
            if digest(row['analysis']) != row['analysis_sha256'] or not analysis['qualitative_timing_qualified']:
                raise ValueError('Secondary admission changed')
            if analysis['policy_sha256'] != digest(HERE/'qualitative-protocol.md') or any(
                    digest(evidence/name) != sha for name, sha in analysis['source_sha256'].items()):
                raise ValueError('Secondary evidence or policy changed: '+row['study'])
            result.update(analysis=local(row['analysis']), analysis_sha256=row['analysis_sha256'])
            for key in ('retained_blocks', 'excluded_blocks', 'exclusions', 'filtered', 'all_blocks'):
                result[key] = analysis[key]
        else:
            result['failed_gates'] = {k: v for k, v in audit.items() if k in
                                      ('numerical_signature_mismatches', 'physical_checks', 'issues')}
            if row['state'] == 'numerical failure':
                result['first_failed_sample_per_variant'] = failure_examples(evidence)
        if row['state'] in ('primary qualified', 'qualitative qualified'):
            if not result['numerical_qualified'] or not result['clock_qualified'] or result['instrumented']:
                raise ValueError('Invalid admitted timing')
            result['accepted_timing'] = True
            result['decisions'] = {k: qualitative_decision(v, result['all_blocks']['contrasts'][k],
                                                         k == 'interaction_time_factor')
                                   for k, v in result['filtered']['contrasts'].items()}
        results.append(result)
    return dict(schema='blaze2d/symmetry-remeasurement-summary/1',
                scope='Fixed-input, same-control experiments; ratios are reference time / candidate time. '
                      'Interaction time factors have separate direction. Different studies are not pooled or multiplied. '
                      'Times are the declared internal solver/interface/kernel durations, normalized to raw clock; '
                      'process windows separately witness clocks and load. Browser loading is outside this comparison.',
                host='Intel Core i9-13900K, WSL2, f64 unless explicitly a precision comparison',
                clocks='Linux CLOCK_MONOTONIC_RAW with independent Windows QPC witness; adjusted durations normalized separately',
                policy='qualitative-protocol.md', policy_sha256=digest(HERE/'qualitative-protocol.md'),
                cohort=local(cohort), inventory_sha256=digest(cohort/'inventory.json'),
                coverage_sha256=digest(cohort/coverage_file),
                declared_comparisons=len(declarations), addressed_comparisons=len(results),
                complete=set(states) == set(declarations), state_counts=dict(Counter(r['state'] for r in results)),
                uncertainty='Intervals describe paired repeatability on this desktop, not machine-independent accuracy. '
                            'Filtered pairs are selected by host measurements alone; all numerical samples must pass. '
                            'Strict admitted estimates retain their original bootstrap; secondary analyses use the declared amendment.',
                limitations=['No exclusive hardware reservation', 'No direct DRAM bandwidth or energy measurement',
                             'Instrumented diagnostics and numerical failures have no accepted latency ratio'],
                comparisons=results)


def markdown(data):
    lines = ['# Symmetry remeasurement', '',
             'This is the current replacement for the historical timing claims. Raw samples, '
             'original strict audits and failed attempts remain unchanged. The '
             '[desktop protocol](qualitative-protocol.md) defines whole-block exclusions and the 5% practical band.', '',
             f"Addressed {data['addressed_comparisons']} of {data['declared_comparisons']} declared comparisons. "
             'A completed experiment can retain numerical evidence without an admitted timing.', '',
             '| Evidence state | Comparisons |', '|---|---:|']
    lines += [f'| {state} | {count} |' for state, count in data['state_counts'].items()]
    lines += ['', 'Ratios are reference time / candidate time; **above one is faster**. '
              'Interaction time factors are excluded from the table below because their direction differs. '
              'Each row retains its own control, input, output contract and source hashes in '
              '[the JSON report](remeasurement-summary.json). Do not multiply or pool unrelated gains.', '',
              '| Comparison | Contrast | Retained blocks | Ratio [95% interval] | All-block ratio | Practical result |',
              '|---|---|---:|---|---:|---|']
    for row in data['comparisons']:
        label = f"[{row['study']}]({row.get('evidence', row['original'])})"
        if not row['accepted_timing']:
            lines.append(f"| {label} | — | — | — | — | {row['state']} |")
            continue
        for key, contrast in row['filtered']['contrasts'].items():
            if key == 'interaction_time_factor':
                continue
            gain = contrast['geometric_ratio']; lo, hi = contrast['ci95']
            full = row['all_blocks']['contrasts'][key]['geometric_ratio']
            decision = row['decisions'][key].replace('_', ' ')
            lines.append(f"| {label} | {key} | {len(row['retained_blocks'])} | {gain:.3f} [{lo:.3f}, {hi:.3f}] | {full:.3f} | {decision} |")
    lines += ['', 'Separate diagnostic reproductions retain work counts, convergence failures and profiled observer cost. '
              'They are not additional primary timing samples. RSS is process resident memory; '
              'FFT element counts and modeled payloads are not measured memory bandwidth.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--coverage', default='coverage.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    data = summarize(args.cohort.resolve(), args.coverage)
    if args.require_complete and not data['complete']:
        raise ValueError('Inventory is not completely addressed')
    args.output.write_text(json.dumps(data, indent=2)+'\n')
    args.output.with_suffix('.md').write_text(markdown(data))
    print(json.dumps({k: data[k] for k in ('complete', 'addressed_comparisons', 'state_counts')}))


if __name__ == '__main__':
    main()
