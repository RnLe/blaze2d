import copy
import unittest

from normalized_timing import normalize


class ClockProjectionTests(unittest.TestCase):
    def study(self):
        samples, windows = [], []
        for pair in range(20):
            for label, seconds in [('baseline', 2.), ('candidate', 1.)]:
                samples.append(dict(pair=pair, label=label, valid=True,
                                    result={'seconds': 1.1*seconds, 'iterations': 7}))
                windows.append(dict(start={'monotonic': 0, 'monotonic_raw': 0,
                    'windows': {'qpc_ticks': 0, 'qpc_frequency': 1000}},
                    end={'monotonic': 1.1*seconds, 'monotonic_raw': seconds,
                         'windows': {'qpc_ticks': seconds*1000, 'qpc_frequency': 1000}}))
        return dict(samples=samples, manifest={'pairs': 20, 'seed': 20260923},
                    summary={'valid_pairs': 20, 'latency_decision': 'accepted'}), windows

    def test_uniform_clock_error_changes_seconds_but_preserves_gain(self):
        study, windows = self.study(); original = copy.deepcopy(study)
        result = normalize(study, windows, 'run_pairs')
        self.assertTrue(result['clock_qualified'])
        self.assertEqual(result['summary']['ci95'], [2.,2.])
        self.assertEqual(result['samples'][0]['result']['seconds'], 2.)
        self.assertEqual(result['samples'][0]['reported_solver_seconds'], 2.2)
        self.assertEqual(study, original)

    def test_variant_dependent_clock_change_prevents_claim(self):
        study, windows = self.study(); windows[1]['end']['monotonic'] = 1.2
        result = normalize(study, windows, 'run_pairs')
        self.assertFalse(result['clock_qualified'])
        self.assertEqual(result['summary'], {})

    def test_clock_gap_does_not_hide_numerical_failure(self):
        study, windows = self.study(); study['samples'][0]['valid'] = False
        result = normalize(study, windows, 'run_pairs')
        self.assertEqual(result['summary']['latency_decision'], 'quality_failure')
        del windows[0]['end']['windows']['qpc_ticks']
        self.assertFalse(normalize(study, windows, 'run_pairs')['clock_qualified'])


if __name__ == '__main__':
    unittest.main()
