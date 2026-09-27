"""Freeze public coarse-start inputs and independently audit full output fields.

Correctness collection only: no speed inference or concurrent primary sampling.
The resulting pilot JSONs can be passed to replay_quiet.py without changing the
existing randomized paired sampler or host thresholds.
"""
import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import subprocess
import time

from audit_coarse_workflow import audit
from replay_quiet import HERE, ROOT, digest


def design():
    studies = [dict(case=case, grid=n, bands=q, polarization=pol, baseline_mode='off', candidate_mode='coarse')
        for n, q in [(31, 8), (65, 16)] for case in ['square-gamma', 'square-generic'] for pol in ['TM', 'TE']]
    studies += [dict(case='square-return', grid=31, bands=8, polarization=pol,
                     baseline_mode='off', candidate_mode='coarse') for pol in ['TM', 'TE']]
    studies += [dict(case='square-gamma', grid=31, bands=8, polarization=pol,
                     baseline_mode='pt', candidate_mode='pt-coarse') for pol in ['TM', 'TE']]
    for study in studies:
        study['name'] = f"{study['case']}-{study['grid']}-{study['bands']}-{study['polarization']}-{study['baseline_mode']}"
    return studies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--builds', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    builds = json.loads(args.builds.read_text())
    assert builds['complete'] and all(digest(a['path']) == a['sha256'] for a in builds['artifacts'].values())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'raw').mkdir()
    (out/'pilots').mkdir()
    binary = builds['artifacts']['unprofiled']['path']
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', RAYON_NUM_THREADS='1')
    studies = design()
    manifest = dict(scope='public coarse correctness and frozen future primary inputs; no timing claim',
        builds=builds, build_manifest_sha256=digest(args.builds), cases=studies, started=time.time(),
        harness_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        contracts={name: digest(HERE/name) for name in ['prepare_coarse_workflow.py', 'audit_coarse_workflow.py',
            'coarse_workflow_contract.py', 'run_coarse_workflow_pairs.py', 'coarse-workflow-protocol.md',
            'audit_wasm_pt_sectors.py', 'interface_contract.py', 'run_pairs.py']})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    results = []
    for spec in studies:
        values, records, issues, checks = {}, [], [], {}
        for label in ['baseline', 'candidate']:
            command = ['taskset', '-c', '0', binary, spec['case'], str(spec['grid']), str(spec['bands']),
                       spec['polarization'], spec[label+'_mode'], '1e-14', 'audit']
            process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, timeout=600)
            path = out/'raw'/(spec['name']+'-'+label+'.json.gz')
            path.write_bytes(gzip.compress(process.stdout, mtime=0))
            entry = dict(label=label, command=command, returncode=process.returncode,
                         stdout=str(path.relative_to(out)), stdout_sha256=digest(path),
                         stderr=process.stderr.decode(errors='replace'))
            records.append(entry)
            if process.returncode != 0:
                issues.append(label+': process failure')
                continue
            value = json.loads(process.stdout)
            values[label] = value
            if value['build']['source_revision'] != builds['source_revision']:
                issues.append(label+': embedded source revision differs from frozen build')
            if value['counters_available'] or value['counters'] or value['timings']:
                issues.append(label+': primary artifact has profiling enabled')
            checks[label] = audit(value, values.get('baseline') if label == 'candidate' else None)
            issues += [label+': '+issue for issue in checks[label]['issues']]
        result = dict(name=spec['name'], records=records, independent=checks, issues=issues)
        results.append(result)
        if not issues:
            samples = []
            for label, value in values.items():
                projected = copy.deepcopy(value)
                projected['audit_arrays'] = projected['audit_material'] = None
                samples.append(dict(pair=0, label=label, valid=True, result=projected))
            pilot = dict(manifest=dict(**{k:v for k,v in spec.items() if k != 'name'},
                baseline=binary, candidate=binary, baseline_sha256=digest(binary), candidate_sha256=digest(binary),
                comparison_contract='public_coarse', pairs=20, cpu=0, tolerance=1e-14,
                require_interface_contract=True, require_requested_path=True, stop_on_quality_failure=True,
                output_contract='fields', build_manifest_sha256=digest(args.builds),
                source_revision=builds['source_revision'], pilot_scope='independently audited numerical signature; not a timing comparison',
                diagnostic_projection='audit_arrays and audit_material omitted in primary output; full originals retained'), samples=samples)
            (out/'pilots'/(spec['name']+'.json')).write_text(json.dumps(pilot, indent=2)+'\n')
        (out/'audit.json').write_text(json.dumps(dict(cases=results, complete=False), indent=2)+'\n')
        print(json.dumps(dict(case=spec['name'], admitted_for_timing=not issues, issues=issues)), flush=True)
    (out/'audit.json').write_text(json.dumps(dict(cases=results, complete=True), indent=2)+'\n')
    print(json.dumps(dict(cases=len(results), admitted=sum(not r['issues'] for r in results))), flush=True)


if __name__ == '__main__':
    main()
