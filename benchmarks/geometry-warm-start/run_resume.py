"""Measure result-only checkpoint restart, including cancellation and repeated work."""
import argparse
import gzip
import json
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks/symmetry'))
from quiet_host import QuietHost, assess
from run_public_sweep import estimate


def main():
    import blaze
    import numpy as np
    from run_paths import configuration
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    source, _ = configuration('square', 'TE', 32, 'f64', False, 'radius', 'secant', 'gamma', 'same_k', 2)
    source += '\n[bands.symmetry]\nk_reuse = "identical"\n'
    (a.output / 'input.toml').write_text(source)
    config = blaze.Config.from_toml(source)
    rng = random.Random(2026092812)
    blocks = []
    with QuietHost() as host, tempfile.TemporaryDirectory(prefix='blaze-resume-') as folder:
        for pair in range(-1, 3):
            label = 'warmup' if pair < 0 else f'pair-{pair:02d}'
            modes = ['continuous', 'resume']; rng.shuffle(modes)
            block, frequencies = {}, {}
            for mode in modes:
                checkpoint = Path(folder) / f'{label}-{mode}.ndjson'
                stored = 0

                def interrupt(event):
                    nonlocal stored
                    if event['event'] == 'result':
                        stored += 1
                        if stored == 4:
                            raise InterruptedError('four results stored')

                before = host.snapshot()
                start = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
                if mode == 'resume':
                    try:
                        blaze.run_checkpointed(config, checkpoint, threads=2, progress=interrupt)
                    except InterruptedError as error:
                        assert str(error) == 'four results stored'
                study = blaze.run_checkpointed(config, checkpoint, threads=2)
                elapsed = time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - start
                after = host.snapshot()
                passed = not study['errors'] and len(study['results']) == 8
                points = [p for r in study['results'] for p in r['metadata']['geometry_warm_start']['points'] if p]
                passed &= all(r['metadata']['accuracy']['accepted'] for r in study['results'])
                frequencies[mode] = np.array([np.sort(r['frequencies'], axis=1) for r in study['results']])
                block[mode] = dict(elapsed_seconds=elapsed, host=assess(before, after, limit=2, windows_limit=4),
                    start=before, end=after, passed=bool(passed), roots=sum(p['cold_root'] for p in points),
                    iterations=sum(sum(r['metadata']['iterations']) for r in study['results']),
                    statistics=study['statistics'], build=study['results'][0]['metadata']['build'])
                (a.output / f'{label}-{mode}.ndjson.gz').write_bytes(gzip.compress(checkpoint.read_bytes(), mtime=0))
            block['frequency_difference'] = float(np.max(np.abs(frequencies['continuous']-frequencies['resume'])))
            block['passed'] = all(block[m]['passed'] for m in modes) and block['frequency_difference'] < 2e-7
            block['admitted'] = all(block[m]['host']['accepted'] for m in modes)
            blocks.append(block)
            (a.output / 'blocks.json').write_text(json.dumps(blocks, indent=2) + '\n')
            print(label, block['passed'], block['admitted'], {m: round(block[m]['elapsed_seconds'], 3) for m in modes}, flush=True)
    ratios = [b['continuous']['elapsed_seconds']/b['resume']['elapsed_seconds'] for b in blocks[1:] if b['passed'] and b['admitted']]
    (a.output / 'summary.json').write_text(json.dumps(dict(numerical_pass=all(b['passed'] for b in blocks),
        paired_median_ratio=statistics.median(ratios) if ratios else None, ratio=estimate(ratios),
        scope='Python checkpoint API, including fsync, cancellation join and restart; excludes interpreter import; '
              'unconsumed cancelled work is charged in time/CPU, not included in completed-result iteration totals'), indent=2) + '\n')
    if not all(b['passed'] for b in blocks):
        raise SystemExit('Resume qualification failed')


if __name__ == '__main__':
    main()
