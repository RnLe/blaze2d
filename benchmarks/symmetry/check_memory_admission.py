"""Matched resource-admission regression; complete fields survive for FFT audit."""
import argparse
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

from run_pairs import quality_passes
from support_contract import certificate_issues, independent_certificate

ROOT = Path(__file__).resolve().parents[2]
CASES = [(name, pol, mode) for name, mode in [('gamma', 'partners'), ('mirror-line', 'orbit_mirror')]
         for pol in ['TM', 'TE']]
SEED = 2026092402


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numerics(result):
    value = copy.deepcopy(result)
    for key in ('seconds', 'peak_rss_kib', 'profile', 'counters', 'stage_counters'):
        del value[key]
    for point in value['points']:
        if point['symmetry']['sectors'] is not None:
            point['symmetry']['sectors'].pop('workspace', None)
    return value


def audit_fields(value):
    issues = []
    checks = []
    for i, point in enumerate(value['points']):
        n = dict(operator_identity=point['operator_identity'], grid=value['grid'], polarization=value['polarization'])
        document = dict(field_bits=point['field_bits'], spectrum=[x*x for x in value['spectrum'][i]])
        c = independent_certificate(n, document, projected=False)
        issues += [f'point {i}: {x}' for x in certificate_issues(c, value['bands'], point['k'] == [0.0, 0.0], 'independent FFT')]
        checks.append(c)
    return dict(checks=checks, issues=issues)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--builds', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    builds = json.loads(args.builds.read_text())
    assert builds.get('complete'), 'all isolated builds must have completed'
    subprocess.run(['git', 'diff', 'HEAD', '--exit-code'], cwd=ROOT, check=True, capture_output=True)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == builds['candidate_revision']
    for artifact in builds['artifacts'].values():
        assert sha(artifact['path']) == artifact['sha256']
    assert len({a['target_directory'] for a in builds['artifacts'].values()}) == 4
    for kind in ('unprofiled', 'profiled'):
        assert builds['artifacts']['baseline_' + kind]['sha256'] != builds['artifacts']['candidate_' + kind]['sha256'], \
            'identical parent/candidate executable: refusing invalid comparison'
    assert sha(ROOT / 'crates/core/examples/symmetry_memory_bench.rs') == builds['harness_sha256']
    args.output.mkdir(exist_ok=False)
    (args.output / 'objects').mkdir()
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', RAYON_NUM_THREADS='1')
    manifest = dict(schema='blaze2d/memory-admission/1', builds=builds, cases=CASES, pairs=20,
                    seed=SEED, cpu=0, started=time.time(), runner_sha256=sha(__file__),
                    protocol_sha256=sha(ROOT/'benchmarks/symmetry/memory-admission-protocol.md'),
                    thread_environment={k: env[k] for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS')})
    (args.output/'manifest.json').write_bytes(canonical(manifest)+b'\n')
    records, objects, signatures, issues = [], {}, {}, []

    def execute(case, label, pair, phase):
        name, pol, mode = case
        artifact = label + ('_profiled' if phase in ('profile', 'preflight_profiled') else '_unprofiled')
        command = ['taskset', '-c', '0', builds['artifacts'][artifact]['path'], name, '31', '8', pol, mode, '1e-14']
        row = dict(case=list(case), label=label, pair=pair, phase=phase, command=command, time=time.time(), load_before=os.getloadavg())
        try:
            process = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, timeout=600)
            row.update(exit_code=process.returncode, stderr=process.stderr.decode(errors='replace'))
            if process.returncode:
                raise RuntimeError('benchmark process failed')
            value = json.loads(process.stdout)
            n = numerics(value)
            payload = canonical(n); digest = hashlib.sha256(payload).hexdigest()
            if digest not in objects:
                objects[digest] = n
                (args.output/'objects'/f'{digest}.json.gz').write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
            passed = quality_passes(value, 8) and len(value['points']) == (1 if name == 'gamma' else 5)
            passed = passed and all(p['symmetry']['sectors'] is not None for p in value['points'])
            admission = [p['symmetry']['sectors'].get('workspace') if p['symmetry']['sectors'] else None for p in value['points']]
            if label == 'candidate':
                passed = passed and all(m and m['model'] == 'rectangular-sector-payload/1' and
                    m['peak_estimated_payload_bytes'] <= m['budget_bytes'] for m in admission)
            else:
                passed = passed and all(m is None for m in admission)
            row.update(numerics_sha256=digest, passed=passed, seconds=value['seconds'], peak_rss_kib=value['peak_rss_kib'],
                       admission=admission, counters=value['counters'], stage_counters=value['stage_counters'], profile=value['profile'])
            reference = signatures.setdefault(case, digest)
            if digest != reference:
                issues.append(f'{case} {label} {phase}/{pair}: numerical signature differs')
            if not passed:
                issues.append(f'{case} {label} {phase}/{pair}: quality/path gate failed')
        except Exception as e:
            row['error'] = str(e); issues.append(f'{case} {label} {phase}/{pair}: {e}')
        records.append(row)
        with (args.output/'records.jsonl').open('ab') as f:
            f.write(canonical(row)+b'\n')

    # Gate actual execution of the intended code before collecting timing pairs.
    # Keep these separate from the declared warmups and primary randomization.
    for case in CASES:
        for label in ('baseline', 'candidate'):
            for phase in ('preflight_unprofiled', 'preflight_profiled'):
                execute(case, label, 0, phase)
    if issues:
        audit = dict(schema='blaze2d/memory-admission-audit/1', records=len(records),
                     objects=len(objects), summaries=[], issues=issues, phase='preflight',
                     records_sha256=sha(args.output/'records.jsonl'), finished=time.time())
        (args.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        print(json.dumps({'phase':'preflight', 'issues':issues}),flush=True)
        return True
    print(json.dumps({'preflight_records':len(records),'issues':[]}),flush=True)
    rng = random.Random(SEED)
    for case in CASES:
        for pair in range(-1, 20):
            order = ['baseline', 'candidate']; rng.shuffle(order)
            for label in order:
                execute(case, label, pair, 'warmup' if pair < 0 else 'primary')
        print(json.dumps({'completed_case': case, 'records': len(records), 'issue_count': len(issues)}), flush=True)
    for case in CASES:
        for label in ('baseline', 'candidate'):
            execute(case, label, 0, 'profile')

    # All sampling is complete before independent NumPy/SciPy work begins.
    audits = {digest: audit_fields(n) for digest, n in objects.items()}
    for digest, a in audits.items():
        issues += [digest + ': ' + x for x in a['issues']]
    summaries = []
    for index, case in enumerate(CASES):
        if issues:
            break  # No timing conclusion is valid when correctness or provenance fails.
        primary = [r for r in records if r['case'] == list(case) and r['phase'] == 'primary']
        if len(primary) != 40 or any('seconds' not in r for r in primary):
            issues.append(f'{case}: incomplete timing records'); continue
        by = {label: sorted([r for r in primary if r['label'] == label], key=lambda r:r['pair']) for label in ('baseline', 'candidate')}
        logs = [math.log(a['seconds']/b['seconds']) for a,b in zip(by['baseline'],by['candidate'])]
        bootstrap = random.Random(SEED+index)
        estimates = sorted(math.exp(statistics.mean(bootstrap.choices(logs,k=20))) for _ in range(10_000))
        ci = [estimates[250], estimates[9749]]
        profiles = {r['label']:r for r in records if r['case'] == list(case) and r['phase'] == 'profile'}
        if len(profiles) != 2 or any('counters' not in r for r in profiles.values()):
            issues.append(f'{case}: missing profile'); continue
        if profiles['baseline']['counters'] != profiles['candidate']['counters']:
            issues.append(f'{case}: work counters changed')
        n = objects[signatures[case]]
        summaries.append(dict(case=case, pairs=20, geometric_speedup=math.exp(statistics.mean(logs)), ci95=ci,
            equivalent_within_2_percent=ci[0] >= 1/1.02 and ci[1] <= 1.02,
            practical_regression=ci[1] < 1/1.02,
            median_seconds={k:statistics.median(r['seconds'] for r in v) for k,v in by.items()},
            median_peak_rss_kib={k:statistics.median(r['peak_rss_kib'] for r in v) for k,v in by.items()},
            iterations=sum(p['iterations'] for p in n['points']),
            profile_time_over_unprofiled_median={k:v['seconds']/statistics.median(r['seconds'] for r in by[k]) for k,v in profiles.items()},
            profile_counters=profiles['candidate']['counters']))
    audit = dict(schema='blaze2d/memory-admission-audit/1', records=len(records), objects=len(objects),
                 exact_signatures=len(signatures), independent=audits, summaries=summaries, issues=issues,
                 records_sha256=sha(args.output/'records.jsonl'), finished=time.time())
    (args.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'records':len(records),'objects':len(objects),'summaries':summaries,'issues':issues}),flush=True)
    return bool(issues)


if __name__ == '__main__':
    raise SystemExit(main())
