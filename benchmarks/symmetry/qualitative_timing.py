"""Whole-block, load-only secondary analysis; strict audits remain untouched."""
import argparse
import json
import math
from pathlib import Path
import random
import statistics

from quiet_host import MEASUREMENT_PROTOCOL
from replay_quiet import digest

HERE = Path(__file__).resolve().parent
SEED = 2026092801
MIN_BLOCKS = 12


def contrasts(variants):
    variants = set(variants)
    if variants == {'off', 'pt', 'unitary', 'pt-unitary'}:
        return {
            'pt_speedup': {'off': 1, 'pt': -1},
            'reuse_speedup': {'off': 1, 'unitary': -1},
            'combined_speedup': {'off': 1, 'pt-unitary': -1},
            'reuse_after_pt_speedup': {'pt': 1, 'pt-unitary': -1},
            'pt_after_reuse_speedup': {'unitary': 1, 'pt-unitary': -1},
            'interaction_time_factor': {'pt-unitary': 1, 'off': 1, 'pt': -1, 'unitary': -1}}
    if variants == {'off', 'pt', 'sectors', 'fourier-sectors', 'pt-sectors'}:
        return {
            'pt': {'off': 1, 'pt': -1}, 'sectors': {'off': 1, 'sectors': -1},
            'combined': {'off': 1, 'pt-sectors': -1},
            'sectors_after_pt': {'pt': 1, 'pt-sectors': -1},
            'pt_after_sectors': {'sectors': 1, 'pt-sectors': -1},
            'fourier_projection': {'sectors': 1, 'fourier-sectors': -1},
            'pt_after_fourier_sectors': {'fourier-sectors': 1, 'pt-sectors': -1},
            'interaction_time_factor': {'pt-sectors': 1, 'off': 1, 'pt': -1, 'sectors': -1}}
    for left, right in [('baseline', 'candidate'), ('full', 'reduced'), ('full', 'blocks')]:
        if variants == {left, right}:
            return {'paired': {left: 1, right: -1}}
    raise ValueError('Unknown comparison variants')


def summarize(blocks, ids, resamples=10000):
    variants = set(blocks[ids[0]])
    if any(set(blocks[i]) != variants for i in ids):
        raise ValueError('Incomplete matched block')
    definitions = contrasts(variants)
    vectors = [[sum(weight*math.log(blocks[i][v]['seconds']) for v, weight in terms.items())
                for terms in definitions.values()] for i in ids]
    rng = random.Random(SEED)
    boots = [[] for _ in definitions]
    for _ in range(resamples):
        chosen = rng.choices(vectors, k=len(vectors))
        for j in range(len(definitions)):
            boots[j].append(math.exp(sum(v[j] for v in chosen)/len(chosen)))
    result = {}
    for j, name in enumerate(definitions):
        ordered = sorted(boots[j])
        result[name] = dict(geometric_ratio=math.exp(sum(v[j] for v in vectors)/len(vectors)),
                            ci95=[ordered[int(.025*resamples)], ordered[int(.975*resamples)-1]])
    return dict(contrasts=result,
                median_seconds={v: statistics.median(blocks[i][v]['seconds'] for i in ids)
                                for v in sorted(variants)},
                median_rss_kib={v: statistics.median(blocks[i][v]['peak_rss_kib'] for i in ids)
                               if all(blocks[i][v]['peak_rss_kib'] is not None for i in ids) else None
                                for v in sorted(variants)})


def qualitative_decision(filtered, unfiltered, interaction=False):
    ratio = filtered['geometric_ratio']; all_ratio = unfiltered['geometric_ratio']
    if abs(math.log(ratio/all_ratio)) > math.log(1.1):
        return 'sensitive_to_environment_exclusions'
    lo, hi = filtered['ci95']
    if lo > 1.05 and all_ratio > 1.05:
        return 'above_multiplicative_time_prediction' if interaction else 'improvement'
    if hi < 1/1.05 and all_ratio < 1/1.05:
        return 'below_multiplicative_time_prediction' if interaction else 'regression'
    return 'small_or_inconclusive'


def analyze(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    audit = json.loads((directory/'audit.json').read_text())
    source_files = ['manifest.json', 'audit.json']
    result = dict(schema='blaze2d/qualitative-timing/1', evidence=str(directory.resolve()),
                  selection='Whole blocks selected solely by the original recorded host-load gate',
                  minimum_blocks=MIN_BLOCKS, bootstrap_seed=SEED, bootstrap_resamples=10000,
                  original_strict_primary_qualified=audit.get('primary_latency_qualified', False),
                  policy_sha256=digest(HERE/'qualitative-protocol.md'),
                  analyzer_sha256=digest(__file__), qualitative_timing_qualified=False, issues=[])
    if manifest.get('measurement_protocol') != MEASUREMENT_PROTOCOL:
        result['issues'].append('Superseded or missing independent clock protocol')
    if audit.get('phase') == 'preflight':
        result['issues'].append('No measured blocks')
    if not audit.get('clock_qualified'):
        result['issues'].append('Independent clock gate failed')
    if not audit.get('numerical_qualified'):
        result['issues'].append('Numerical/path gate failed or unavailable')
    if not audit.get('complete'):
        result['issues'].append('Incomplete study')
    if audit.get('numerical_signature_mismatches'):
        result['issues'].append('Frozen numerical output changed')
    record_study = manifest['schema'] == 'blaze2d/record-replay/1'
    if record_study and not audit.get('artifacts_and_contracts_unchanged'):
        result['issues'].append('Provenance changed')
    if result['issues']:
        result['source_sha256'] = {name: digest(directory/name) for name in source_files}
        return result
    if record_study:
        source_files.append('records.jsonl')
        if digest(directory/'records.jsonl') != audit['records_sha256']:
            raise ValueError('Raw records changed')
        samples = [json.loads(line) for line in (directory/'records.jsonl').read_text().splitlines()]
        interaction = manifest['design']['kind'] == 'interaction'
        extracted = [(s['pair'], s['mode'] if interaction else s['label'],
                      s['normalized_seconds'], s['measurement']['peak_rss_kib']) for s in samples]
    else:
        source_files.extend(['samples.json', 'clock-normalized.json', 'environment.jsonl'])
        projection = json.loads((directory/'clock-normalized.json').read_text())
        if not projection['clock_qualified']:
            raise ValueError('Conflicting clock audit')
        samples = projection['samples']
        if any(s.get('valid') is not True for s in samples):
            raise ValueError('Conflicting numerical audit')
        extracted = [(s.get('pair', s.get('block')), s.get('label', s.get('mode')),
                      s['result']['seconds'], s['result'].get('peak_rss_kib')) for s in samples]
    blocks = {}
    for key, variant, seconds, rss in extracted:
        if key < 0:
            continue
        if variant in blocks.setdefault(key, {}):
            raise ValueError('Duplicate variant in block')
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Invalid internal duration')
        blocks[key][variant] = dict(seconds=seconds, peak_rss_kib=rss)
    if set(blocks) != set(range(20)):
        raise ValueError('Expected all 20 original measured blocks')
    environments = {r['block']: r for r in audit['environment_blocks'] if r['block'] >= 0}
    if set(environments) != set(blocks) or len(environments) != 20:
        raise ValueError('Missing whole-block environment evidence')
    retained = [i for i in sorted(blocks) if environments[i]['accepted']]
    excluded = [i for i in sorted(blocks) if not environments[i]['accepted']]
    result.update(retained_blocks=retained, excluded_blocks=excluded,
                  exclusions={str(i): environments[i]['issues'] for i in excluded},
                  instrumented=bool(audit.get('instrumented', False)),
                  source_sha256={name: digest(directory/name) for name in source_files})
    # Freeze the environment-only selection before evaluating any elapsed ratios.
    if len(retained) < MIN_BLOCKS:
        result['issues'].append('Fewer than 12 clean complete blocks')
        return result
    result['filtered'] = summarize(blocks, retained)
    result['all_blocks'] = summarize(blocks, sorted(blocks))
    result['qualitative_timing_qualified'] = not result['instrumented']
    result['decisions'] = {name: qualitative_decision(
        value, result['all_blocks']['contrasts'][name], name == 'interaction_time_factor')
        for name, value in result['filtered']['contrasts'].items()}
    if result['instrumented']:
        result['issues'].append('Instrumented diagnostic is not primary latency evidence')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.directory), indent=2)+'\n')


if __name__ == '__main__':
    main()
