"""Separate clock-normalized estimates from immutable reported solver timings."""
import copy
import math
import random
import statistics

from quiet_host import elapsed_clocks


def normalize(study, windows, legacy_harness):
    if len(study['samples']) != len(windows):
        raise ValueError('Each sample requires its enclosing process clock window')
    samples = copy.deepcopy(study['samples'])
    scales, issues = [], []
    for sample, window in zip(samples, windows):
        try:
            clocks = elapsed_clocks(window['start'], window['end'])
        except ValueError as error:
            issues.append(str(error))
            continue
        scale = clocks['adjusted_to_raw_scale']
        scales.append(scale)
        sample['clock_projection'] = clocks
        result = sample.get('result')
        if result is not None:
            reported = result['seconds']
            sample['reported_solver_seconds'] = reported
            result['seconds'] = reported * scale
    # A changing clock rate could bias the shorter internal interval differently
    # from its enclosing process. Do not accept a timing claim in that case.
    spread = max(scales)/min(scales)-1 if scales else math.inf
    if spread > .001:
        issues.append('Adjusted/raw clock scale varies by more than 0.1%')
    result = dict(schema='blaze2d/clock-normalized-timing/1',
                  method='Scale the reported internal duration by enclosing RAW/MONOTONIC elapsed ratio',
                  scope='Estimated original timer scope; raw process durations are recorded separately',
                  maximum_scale_relative_spread=.001,
                  scale_relative_spread=spread if math.isfinite(spread) else None,
                  issues=issues, clock_qualified=not issues, samples=samples,
                  raw_samples_unchanged=True)
    if issues:
        result['summary'] = {}
        return result
    blocks = {}
    for sample in samples:
        key = sample.get('pair', sample.get('block'))
        if key is not None and key >= 0:
            blocks.setdefault(key, []).append(sample)
    if legacy_harness == 'run_factorial':
        from run_factorial import summarize
        result['summary'] = summarize(list(blocks.values()), study['manifest']['blocks'])
        return result
    names = {'run_pairs': ('baseline', 'candidate'),
             'run_coarse_workflow_pairs': ('baseline', 'candidate'),
             'run_kernel_pairs': ('full', 'reduced'),
             'run_trial_pairs': ('full', 'blocks')}
    if legacy_harness not in names:
        raise ValueError('Unknown timing sampler: ' + legacy_harness)
    left, right = names[legacy_harness]
    rng = random.Random(study['manifest']['seed'])
    logs = []
    for block in blocks.values():
        order = [left, right]
        rng.shuffle(order)  # Preserve the original bootstrap RNG state.
        if len(block) != 2 or not all(s.get('valid') for s in block):
            continue
        times = {s.get('label', s.get('mode')): s['result']['seconds'] for s in block}
        logs.append(math.log(times[left]/times[right]))
    summary = copy.deepcopy(study['summary'])
    if logs:
        boot = sorted(math.exp(statistics.mean(rng.choices(logs, k=len(logs))))
                      for _ in range(10000))
        summary.update(geometric_speedup=math.exp(statistics.mean(logs)),
                       ci95=[boot[250], boot[9749]])
        decision = ('quality_failure' if len(logs) != len(blocks) else
                    'incomplete' if len(blocks) != study['manifest']['pairs'] else
                    'accepted' if boot[250] > 1.02 else
                    'regression' if boot[9749] < 1 else 'inconclusive')
        key = 'latency_decision' if 'latency_decision' in summary else 'decision'
        summary[key] = decision
    result['summary'] = summary
    return result
