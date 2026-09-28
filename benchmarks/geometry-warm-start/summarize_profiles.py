"""Extract nonoverlapping phase questions and complete work from explanatory profiles."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    records = []
    for source in sorted(a.directory.rglob('*.profile.json')):
        d = json.loads(source.read_text())
        manifest = json.loads(source.with_name(source.name.replace('.profile.json', '.manifest.json')).read_text())
        threads = int(manifest['command'][manifest['command'].index('--threads')+1])
        counters, timings, stages = d['counters'], d['timings'], d['stages']

        def ns(name):
            return timings.get(name, {}).get('nanos', 0)

        def actions(c):
            return {axis: c.get(f'operator_{axis}.columns', 0) + c.get(f'geometry.delta_{axis}.columns', 0)
                    for axis in ['A', 'B']}

        # These two final-check scopes are separate implementations, not nested
        # parents and children. Lock qualification belongs to correction work.
        certificates = ['final_certification', 'geometry.final_certification']
        certificate_ns = sum(ns(s) for s in certificates)
        run_ns = ns('run.wall')
        point_ns = ns('run.gamma_point') + ns('run.nonzero_point')
        makespan_ns = ns('workflow.collect_makespan')
        row = dict(case=str(source.relative_to(a.directory)), threads=threads, actions=actions(counters),
            rounds=counters.get('iterations', 0),
            fft_2d_transforms=sum(counters.get(f'fft.{direction}.vectors', 0) for direction in ['forward', 'inverse']),
            field_allocations=counters.get('field.allocations', 0),
            traffic={k: v for k, v in counters.items() if k.endswith('_bytes')},
            worker_run_seconds=run_ns/1e9, collect_makespan_seconds=makespan_ns/1e9,
            fraction_of_worker_time_in_band_solves=run_ns/(threads*makespan_ns) if makespan_ns else None,
            solver_inclusive_seconds=ns('eigensolve')/1e9,
            initialization_seconds=ns('initialization')/1e9,
            predictor_seconds=(ns('geometry.predictor') + ns('geometry.source_response'))/1e9,
            model_prepare_seconds=ns('geometry.model_prepare')/1e9,
            model_build_seconds=ns('geometry.model_build')/1e9,
            model_evaluate_seconds=ns('geometry.model_evaluate')/1e9,
            shape_prepare_seconds=ns('geometry.shape_prepare')/1e9,
            shape_interpolation_seconds=ns('geometry.shape_transport')/1e9,
            path_prediction_seconds=ns('geometry.path_prediction')/1e9,
            final_certificate_seconds=certificate_ns/1e9,
            final_certificate_worker_fraction=certificate_ns/run_ns if run_ns else None,
            diagnostic_worker_seconds_without_final_check=(run_ns-certificate_ns)/1e9,
            lock_certificate_seconds=ns('eigensolve.lock_certification')/1e9,
            gamma_point_seconds=ns('run.gamma_point')/1e9,
            nonzero_point_seconds=ns('run.nonzero_point')/1e9,
            gamma_point_fraction=ns('run.gamma_point')/point_ns if point_ns else None,
            stage_actions={s: actions(stages.get(s, {})) for s in certificates +
                          ['eigensolve.lock_certification', 'run.gamma_point', 'run.nonzero_point',
                           'geometry.predictor', 'geometry.source_response', 'geometry.model_prepare',
                           'geometry.model_build', 'geometry.model_evaluate', 'geometry.shape_prepare',
                           'geometry.shape_transport', 'geometry.path_prediction']})
        records.append(row)
    if not records:
        raise SystemExit('No profiles found')
    (a.directory/'summary.json').write_text(json.dumps(dict(records=records,
        scope='Instrumented explanation only. Worker sums are not wall time; inclusive phases overlap. '
              'Diagnostic subtraction does not disable final certification and is not a speedup claim. '
              'A/B counts include coefficient differences but exclude exact-zero differences and identity aliases. '
              'Traffic channels are separate software counters, not a sum or physical DRAM bytes.'), indent=2)+'\n')
    # Worker count changes the geometry chains even when input TOML is identical.
    inputs = {}
    for manifest in a.directory.parent.glob('*/*/manifest.json'):
        d = json.loads(manifest.read_text())
        if 'inputs' not in d or not manifest.with_name('blocks.json').exists():
            continue
        blocks = json.loads(manifest.with_name('blocks.json').read_text())
        for block in blocks:
            if not block['valid']:
                continue
            for side in ['reference', 'candidate']:
                key = (d['inputs'][side], d.get('threads', d.get('case', {}).get('threads', 2)))
                inputs.setdefault(key, set()).add(block[side]['validation']['work']['iterations'])
    matches = []
    for manifest in sorted(a.directory.rglob('*.manifest.json')):
        d = json.loads(manifest.read_text())
        threads = int(d['command'][d['command'].index('--threads')+1])
        expected = inputs.get((d['input_sha256'], threads), set())
        rounds = d['validation']['work']['iterations']
        matches.append(dict(profile=str(manifest.relative_to(a.directory)), threads=threads,
            rounds=rounds, unprofiled_rounds=sorted(expected), pass_=expected == {rounds}))
    (a.directory/'work-equivalence.json').write_text(json.dumps(matches, indent=2)+'\n')


if __name__ == '__main__':
    main()
