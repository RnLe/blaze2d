"""Unit tests for case rendering, record reduction, load verdicts, statistics and lab pages."""
import copy
import gzip
import json
import math
import tempfile
import tomllib
import unittest
from pathlib import Path

import cases
import loadmon
import runlib
import run_timing
import summarize

FIXTURE = Path(__file__).with_name('fixtures') / 'e054876-sq-TM-016-pt.json.gz'


class CaseTests(unittest.TestCase):
    def test_subset_sizes(self):
        sizes = {name: len(cases.subset(name)) for name in cases.SUBSETS}
        self.assertEqual(sizes, dict(full768=768, dev96=96, screen32=32, timing48=48, pilot8=8, pt128=16, odd=8, sample96=96))

    def test_ids_and_labels_are_unique_and_stable(self):
        matrix = cases.full_matrix()
        self.assertEqual(len({c.id for c in matrix}), 768)
        self.assertEqual(len({c.label for c in matrix}), 768)
        case = cases.Case('square', 'TE', 64, 0.30, 1.0, 13.0)
        self.assertEqual(case.label, 'sq-TE-064-r30-e01-13')
        self.assertEqual(case.id, cases.Case('square', 'TE', 64, 0.3, 1.0, 13.0).id)

    def test_timing_roles_follow_work_then_contrast(self):
        roles = cases.timing_roles()
        self.assertEqual(len(roles), 48)
        hard = [c for c in cases.full_matrix() if roles.get(c.id) == ('hard', 'contrast')]
        self.assertTrue(all((c.eps_rod, c.eps_bg, c.radius) == (1.0, 13.0, 0.30) for c in hard))
        stratum = [c for c in cases.full_matrix() if c.stratum == ('square', 'TM', 16)]
        work = {c.id: index for index, c in enumerate(stratum)}
        chosen = {roles[0]: c for c in stratum if (roles := cases.timing_roles(work).get(c.id))}
        self.assertEqual(chosen['easy'], stratum[0])
        self.assertEqual(chosen['hard'], stratum[-1])
        self.assertEqual(chosen['median'], stratum[(len(stratum) - 1) // 2])

    def test_render_round_trips_and_merges_variants(self):
        variants = cases.load_variants()
        case = cases.Case('triangular', 'TM', 32, 0.45, 7.0, 11.0)
        legacy = tomllib.loads(cases.render(case, variants['frozen-legacy']))
        self.assertEqual(legacy, json.loads(json.dumps(case.config())))
        self.assertEqual(legacy['bands']['path'], {'preset': 'triangular', 'intervals_per_segment': 20})
        self.assertNotIn('eigensolver', legacy)
        tight = tomllib.loads(cases.render(case, variants['frozen-pt-tight']))
        self.assertEqual(tight['bands']['symmetry'], {'arithmetic': 'pt_real', 'k_reuse': 'identical'})
        self.assertEqual(tight['eigensolver'], {'tolerance': 1e-13, 'max_iterations': 600})
        self.assertEqual(tight['bands']['count'], 12)
        self.assertNotIn('binary', tight)

    def test_merge_semantics(self):
        base = {'a': {'b': 1, 'c': [1, 2]}, 'd': 2}
        merged = cases.merge(base, {'a': {'c': [3], 'e': {'f': 'x'}}, 'd': None})
        self.assertEqual(merged, {'a': {'b': 1, 'c': [3], 'e': {'f': 'x'}}})
        self.assertEqual(base, {'a': {'b': 1, 'c': [1, 2]}, 'd': 2})
        text = cases.toml_text({'x': 1e-13, 'q': 'a"b', 'inline': [{'k': 1}, 2], 'table': {'odd key': True}})
        self.assertEqual(tomllib.loads(text), {'x': 1e-13, 'q': 'a"b', 'inline': [{'k': 1}, 2],
                                               'table': {'odd key': True}})


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.document = json.loads(gzip.decompress(FIXTURE.read_bytes()))

    def record(self, document, **fields):
        record = dict(dict(returncode=0, timed_out=False, output_sha256='0' * 64), **fields)
        record.update(runlib.reduce_output(document))
        record['validation'] = runlib.validate(record)
        return record

    def test_reduces_saved_frozen_output(self):
        record = self.record(self.document)
        self.assertEqual(record['iterations'], [30, 13, 15, 0])
        self.assertEqual(record['iterations_total'], 58)
        self.assertEqual(record['points'], 4)
        self.assertEqual(record['execution']['resolved_arithmetic'], 'pt_real')
        self.assertEqual(record['symmetry']['methods'], {'fourier': 3})
        self.assertEqual(record['symmetry']['reused_points'], 1)
        self.assertEqual(record['symmetry']['representations'], {'pt_real_te': 3})
        self.assertEqual(len(record['frequencies']), 4)
        self.assertEqual(record['frequencies'][0][0], 0.0)
        self.assertEqual(record['residuals']['zero_modes'], 2)
        self.assertEqual(record['source_revision'], 'e054876609f710d076af6c5c39520fbb67b283d3')
        self.assertEqual(record['optional'], {})
        self.assertEqual(record['validation'], {'ok': True, 'completed': True, 'flags': []})
        self.assertEqual(record['frequencies_sha256'],
                         runlib.array_digest(self.document['results'][0]['arrays']['frequencies']))

    def test_optional_metadata_is_tolerated(self):
        document = copy.deepcopy(self.document)
        metadata = document['results'][0]['metadata']
        metadata.update(point_seconds=[0.1, 0.2, 0.3, 0.0], eigensolver={'stopping': 'residual'},
                        work=[{'iterations': 30, 'a_columns': 400}, {'iterations': 13, 'a_columns': 100}],
                        trace=[{'k': 0}], preconditioner={'kind': 'diagonal'})
        optional = self.record(document)['optional']
        self.assertEqual(optional['work'], {'iterations': 43, 'a_columns': 500})
        self.assertEqual(optional['trace'], {'present': True, 'length': 1})
        self.assertEqual(optional['eigensolver'], {'stopping': 'residual'})

    def test_validation_flags(self):
        document = copy.deepcopy(self.document)
        result = document['results'][0]
        result['metadata']['converged_per_point'][1] = False
        result['arrays']['residuals']['data'][5] = 1e-3
        result['metadata']['certification']['b_orthogonality_defect'] = 1e-6
        flags = self.record(document)['validation']
        self.assertFalse(flags['ok'])
        self.assertFalse(flags['completed'])
        self.assertEqual(set(flags['flags']), {'nonconverged:1', 'residual>1e-5', 'b_orthogonality>1e-8'})
        failed = self.record(None, returncode=2, output_sha256=None)['validation']
        self.assertEqual(failed['flags'], ['exit:2', 'no_output'])


def linux(t, background=0.0, bench=0.0, self_cpu=0.0, **extra):
    """Cumulative reading with constant per-second rates on 4 CPUs; CPU 2 is the benchmark."""
    ticks = runlib.os.sysconf('SC_CLK_TCK')
    cpu = [round(background / 3 * t * ticks)] * 4
    cpu[2] = round(bench * t * ticks)
    row = dict(kind='linux', t=t, total_busy=sum(cpu), cpu_busy=cpu, psi_cpu_some_us=0, psi_memory_some_us=0,
               psi_memory_full_us=0, loadavg=[0, 0, 0], pswpin=0, pswpout=0, mem_available=8 * 1024**3,
               self_cpu=self_cpu * t)
    row.update(extra)
    return row


def windows(t, busy):
    return dict(kind='windows', t=t, latency=0.001, busy=busy * t, collector=0.0, qpc=round(t * 1e7),
                qpc_frequency=10**7)


def samples(background=0.0, windows_busy=1.0, spike=None, **extra):
    rows = [dict(kind='header', t=0.0, affinity=[3])]
    rows += [linux(i / 4, background, bench=1.0, **extra) for i in range(41)]
    if spike:
        for row in rows[1:]:
            if row['t'] > spike:
                row['cpu_busy'][0] += 300
                row['total_busy'] += 300
    rows += [windows(float(i), windows_busy) for i in range(11)]
    return sorted(rows, key=lambda r: r['t'])


class VerdictTests(unittest.TestCase):
    def verdict(self, rows, **kwargs):
        options = dict(child_cpu_seconds=7.9, benchmark_cpus=[2])
        options.update(kwargs)
        return loadmon.verdict(rows, 1.0, 9.0, **options)

    def test_quiet_window_is_clean(self):
        result = self.verdict(samples(0.2, windows_busy=2.0))
        self.assertEqual(result['verdict'], 'clean', result['reasons'])
        self.assertAlmostEqual(result['metrics']['linux_background_mean'], 0.2, places=2)
        self.assertAlmostEqual(result['metrics']['windows_background_mean'], 1.0, places=2)

    def test_background_and_spikes(self):
        self.assertIn('linux_mean', self.verdict(samples(0.8))['reasons'][0])
        spike = self.verdict(samples(0.0, spike=5.0))
        self.assertTrue(any(r.startswith('linux_peak') for r in spike['reasons']))
        self.assertTrue(loadmon.background_only(spike))
        heavy = self.verdict(samples(0.0, windows_busy=3.0))
        self.assertTrue(any(r.startswith('windows_mean') for r in heavy['reasons']))

    def test_memory_cpu_pressure_and_on_cpu(self):
        rows = samples()
        for row in rows:
            if row['kind'] == 'linux' and row['t'] > 5:
                row['pswpout'] = 3
                row['psi_cpu_some_us'] = 2e6
        result = self.verdict(rows)
        codes = {r.split(':')[0] for r in result['reasons']}
        self.assertEqual(codes, {'memory', 'psi_cpu'})
        self.assertFalse(loadmon.background_only(result))
        starved = self.verdict(samples(), child_cpu_seconds=7.0)
        self.assertTrue(any(r.startswith('on_cpu') for r in starved['reasons']))
        low = self.verdict(samples(mem_available=1024**3))
        self.assertEqual(low['verdict'], 'contaminated')

    def test_sampler_cpu_is_not_background(self):
        result = self.verdict(samples(0.4, self_cpu=0.4))
        self.assertAlmostEqual(result['metrics']['linux_background_mean'], 0.0, places=2)
        rows = samples(0.4, self_cpu=0.4)
        rows[0]['affinity'] = [2]  # A sampler on the benchmark CPU is not in the background sum.
        self.assertAlmostEqual(self.verdict(rows)['metrics']['linux_background_mean'], 0.4, places=2)

    def test_short_runs_fall_back_or_are_unknown(self):
        rows = samples()
        self.assertEqual(loadmon.verdict(rows, 1.01, 1.2, 0.19, [2])['verdict'], 'unknown')
        snap = dict(monotonic=1.0, monotonic_raw=1.0, owned_cpu_seconds=0.0, linux_busy_seconds=0.0,
                    windows=dict(busy_seconds=0.0, collector_cpu_seconds=0.0, qpc_ticks=0, qpc_frequency=1000),
                    swap_in=0, swap_out=0, memory_stall_us=0, memory_available_bytes=4 * 1024**3)
        end = copy.deepcopy(snap)
        end.update(monotonic=1.2, monotonic_raw=1.2, owned_cpu_seconds=0.197, linux_busy_seconds=0.2)
        end['windows'].update(busy_seconds=0.3, qpc_ticks=200)
        result = loadmon.verdict(rows, 1.0, 1.2, 0.197, [2], snapshots=[snap, end])
        self.assertEqual((result['verdict'], result['metrics']['method']), ('clean', 'assess'))
        end['linux_busy_seconds'] = 0.5
        self.assertEqual(loadmon.verdict(rows, 1.0, 1.2, 0.197, [2], snapshots=[snap, end])['verdict'],
                         'contaminated')

    def test_missing_windows_evidence_is_unknown(self):
        rows = [r for r in samples() if r['kind'] != 'windows']
        self.assertEqual(self.verdict(rows)['verdict'], 'unknown')


class StatisticsTests(unittest.TestCase):
    def test_uniform_speedup(self):
        per_case = {f'c{i}': ([2.0 * (i + 1)] * 3, [1.0 * (i + 1)] * 3) for i in range(5)}
        stats = run_timing.ratio_statistics(per_case, resamples=1000)
        self.assertAlmostEqual(stats['geometric_mean'], 2.0)
        self.assertAlmostEqual(stats['time_weighted'], 2.0)
        self.assertEqual(stats['ci95_geometric'], [2.0, 2.0])
        self.assertEqual(stats['decision'], 'improvement')
        self.assertEqual(stats['pairs'], 15)

    def test_mixed_ratios_and_seeded_bootstrap(self):
        per_case = {'a': ([1.0], [1.0]), 'b': ([4.0], [1.0]), 'c': ([1.0], [4.0]), 'd': ([10.0], [5.0])}
        stats = run_timing.ratio_statistics(per_case, resamples=2000)
        self.assertAlmostEqual(stats['geometric_mean'], 2 ** 0.25)
        self.assertAlmostEqual(stats['median_case'], (1.0 + 2.0) / 2)
        self.assertAlmostEqual(stats['time_weighted'], 16.0 / 11.0)
        low, high = stats['ci95_geometric']
        self.assertTrue(low < stats['geometric_mean'] < high)
        self.assertEqual(stats, run_timing.ratio_statistics(per_case, resamples=2000))
        self.assertEqual(stats['decision'], 'inconclusive')
        self.assertIsNone(run_timing.ratio_statistics({}))
        self.assertEqual(run_timing.interval(list(range(10000)))[0], 250)
        self.assertEqual(run_timing.interval(list(range(10000)))[1], 9749)

    def test_matched_admission(self):
        def run(state, background, reasons=()):
            return dict(load=dict(verdict=state, reasons=list(reasons),
                                  metrics=dict(linux_background_mean=background, windows_background_mean=1.0)))
        self.assertTrue(run_timing.matched([run('contaminated', 1.2, ['linux_mean: 1.2 > 0.5']), run('clean', 0.9)]))
        self.assertFalse(run_timing.matched([run('contaminated', 1.4, ['linux_mean: x']), run('clean', 0.2)]))
        self.assertFalse(run_timing.matched([run('contaminated', 0.3, ['memory: swap']), run('clean', 0.2)]))
        self.assertFalse(run_timing.matched([run('unknown', 0.3, ['evidence: x']), run('clean', 0.2)]))


def timing_study(seconds):
    """Entries and runs of an admitted pair study with seconds = {case: (a, b)} and a fixed clock."""
    entries, runs = [], []
    for case, (a, b) in seconds.items():
        params = dict(lattice='square', polarization='TM', resolution=16, radius=0.3, eps_rod=1.0, eps_bg=13.0)
        ids = {}
        for name, value in (('A', a), ('B', b)):
            ids[name] = f'{case}:0:0:{name}'
            start = dict(monotonic=0.0, monotonic_raw=0.0, windows=dict(qpc_ticks=0, qpc_frequency=1000))
            end = dict(monotonic=value, monotonic_raw=1.1 * value,
                       windows=dict(qpc_ticks=round(1100 * value), qpc_frequency=1000))
            runs.append(dict(run_id=ids[name], case_id=case, variant=name, tag='timing', job_seconds=value,
                             wall_seconds=1.1 * value + 0.01, snapshots=[start, end], load=dict(verdict='clean'),
                             validation=dict(completed=True)))
        entries.append(dict(key=f'{case}:0', case_id=case, label=case, params=params, rep=0, tier='strict',
                            quality=True, run_ids=ids))
    return entries, runs


class SummaryTests(unittest.TestCase):
    def test_timing_summary_normalizes_and_bootstraps(self):
        entries, runs = timing_study({'c0': (2.0, 1.0), 'c1': (3.0, 1.0)})
        summary = run_timing.summarize('demo', ['A', 'B'], entries, runs, resamples=500)
        self.assertTrue(summary['clock']['clock_qualified'])
        self.assertEqual(summary['primary_metric'], 'solver')
        strict = summary['tiers']['strict']['solver']
        self.assertAlmostEqual(strict['geometric_mean'], math.sqrt(6))
        self.assertAlmostEqual(strict['time_weighted'], 5.0 / 2.0)
        self.assertAlmostEqual(summary['per_case'][0]['seconds']['A'], 2.2)
        self.assertFalse(summary['sensitive_to_environment'])
        self.assertEqual(summary['counts']['clean_runs'], 4)

    def test_comparison_pages_from_a_timing_study(self):
        entries, runs = timing_study({'c0': (2.0, 1.0), 'c1': (3.0, 1.0)})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'tuning-demo'
            root.mkdir()
            binary = dict(name='bin', sha256='0' * 64, revision='r')
            (root / 'manifest.json').write_text(json.dumps(dict(
                kind='timing', stage='tuning', status='complete', started='2026-10-01T00:00:00+00:00',
                subset='dev96', variants={'A': {}, 'B': {}}, binaries={'A': binary, 'B': binary})))
            (root / 'summary.json').write_text(json.dumps(run_timing.summarize('tuning-demo', ['A', 'B'], entries,
                                                                               runs, resamples=500)))
            (root / 'runs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in runs))
            pages = summarize.build(Path(directory), [])
        tuning = pages['tuning']
        self.assertEqual(tuning['status'], 'complete')
        self.assertEqual([c['kind'] for c in tuning['charts']], ['ratio_bars', 'bars'])
        self.assertAlmostEqual(tuning['tiles'][0]['value'], math.sqrt(6))
        self.assertIn('Speed-up B over A', [t['label'] for t in pages['index']['tiles']])
        self.assertEqual(pages['runs']['runs'][0]['clean'], 4)
        self.assertEqual(pages['baseline']['status'], 'empty')
        json.dumps(pages, allow_nan=False)

    def test_empty_results_produce_empty_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            pages = summarize.build(Path(directory) / 'missing', [])
        self.assertEqual(set(pages), {'index', 'baseline', 'conditioning', 'tuning', 'multigrid',
                                      'ablation', 'ml', 'runs'})
        keys = {'schema', 'page', 'title', 'subtitle', 'generated_at', 'source', 'status', 'tiles',
                'charts', 'tables', 'notes'}
        for name, document in pages.items():
            self.assertTrue(keys <= set(document), name)
            self.assertEqual(document['status'], 'empty', name)
            self.assertEqual(document['schema'], 'blaze2d-lab/preconditioner/1')
            self.assertTrue(document['notes'], name)
            json.dumps(document, allow_nan=False)
        self.assertEqual([s['status'] for s in pages['index']['stages']], ['pending'] * 6)
        self.assertEqual(pages['index']['timeline'], [])
        self.assertEqual(pages['runs']['runs'], [])

    def test_finite_replaces_nonfinite_numbers(self):
        self.assertEqual(summarize.finite({'a': [math.nan, 1.0, math.inf]}), {'a': [None, 1.0, None]})


if __name__ == '__main__':
    unittest.main()
