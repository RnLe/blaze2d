import copy
import unittest
from replay_quiet import audit_replay, numerical, practical, signature


class ReplayTests(unittest.TestCase):
    def study(self):
        result = dict(seconds=1., spectrum=[[1.]], iterations=7, peak_rss_kib=100,
                      counters={'fft': 8}, profile={})
        samples = [dict(pair=0, label=label, valid=True, result=copy.deepcopy(result))
                   for label in ['baseline', 'candidate']]
        d = dict(manifest={'pairs': 1}, samples=samples, summary={'ci95': [1.05, 1.15]})
        a = dict(monotonic=0, owned_cpu_seconds=0, linux_busy_seconds=0,
                 windows={'busy_seconds': 0, 'collector_cpu_seconds': 0},
                 swap_in=0, swap_out=0, memory_stall_us=0, memory_available_bytes=4*1024**3)
        b = copy.deepcopy(a); b.update(monotonic=1, owned_cpu_seconds=1, linux_busy_seconds=1)
        b['windows']['busy_seconds'] = 1
        c = copy.deepcopy(b); c.update(monotonic=2, owned_cpu_seconds=2, linux_busy_seconds=2)
        c['windows']['busy_seconds'] = 2
        return d, copy.deepcopy(d), [{'start': a, 'end': b}, {'start': b, 'end': c}]

    def test_runtime_changes_do_not_change_numerical_signature(self):
        a, b, w = self.study(); b['samples'][0]['result'].update(seconds=.1, peak_rss_kib=80)
        self.assertTrue(audit_replay(a,b,w)['timing_qualified'])

    def test_changed_work_or_frequency_rejects_fast_result(self):
        for key in ['iterations', 'spectrum', 'counters']:
            a,b,w = self.study(); b['samples'][0]['result'][key] = 999
            self.assertFalse(audit_replay(a,b,w)['timing_qualified'])

    def test_profiler_call_counts_are_preserved(self):
        self.assertEqual(numerical({'profile': {'fft': {'calls': 3, 'nanos': 200}}}),
                         {'profile': {'fft': {'calls': 3}}})

    def test_contaminated_block_overrides_apparent_win(self):
        a,b,w = self.study(); w[-1]['end']['windows']['busy_seconds'] += 8
        audit = audit_replay(a,b,w)
        self.assertFalse(audit['timing_qualified']); self.assertEqual(audit['practical_decisions'], {})

    def test_incomplete_or_failed_study_has_no_gain(self):
        a,b,w = self.study(); b['manifest']['pairs'] = 20
        self.assertFalse(audit_replay(a,b,w)['timing_qualified'])
        b['manifest']['pairs'] = 1; b['samples'][0]['valid'] = False
        self.assertFalse(audit_replay(a,b,w)['timing_qualified'])

    def test_instrumented_latency_is_not_primary(self):
        a,b,w = self.study()
        for d in [a,b]:
            for s in d['samples']:s['result']['profile']={'fft':{'calls':8,'nanos':1}}
        audit = audit_replay(a,b,w)
        self.assertTrue(audit['timing_qualified']); self.assertFalse(audit['primary_latency_qualified'])

    def test_practical_threshold_is_symmetric(self):
        self.assertEqual(practical([.981, .999]), 'equivalent')
        self.assertEqual(practical([.97, .99]), 'inconclusive')
        self.assertEqual(practical([.9, .97]), 'regression')


if __name__ == '__main__':
    unittest.main()
