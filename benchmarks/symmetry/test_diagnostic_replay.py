import copy
import unittest

from replay_diagnostics import validate
from replay_record_study import payload_sha


class DiagnosticReplayTests(unittest.TestCase):
    def test_root_profile_preserves_failed_convergence_and_inner_work(self):
        value = dict(seconds=1, converged=False, rejection='fresh residual',
                     counters=dict(inner_iterations=768), points=[dict(residual=2e-4)])
        old = dict(result=copy.deepcopy(value))
        value['seconds'] = 2
        self.assertEqual(validate(value, old, 'root-profiles', None), [])
        value['converged'] = True
        self.assertTrue(validate(value, old, 'root-profiles', None))
        value['converged'] = False
        value['counters']['inner_iterations'] = 0
        self.assertTrue(validate(value, old, 'root-profiles', None))

    def test_diagnostic_clock_variation_cannot_hide_changed_work_or_certificates(self):
        value = dict(seconds=1, peak_rss_kib=100, spectrum=[1, 2], report=dict(residual=1e-9),
                     counters=dict(iterations=10), profile=dict(solve=dict(calls=1, nanos=10, exclusive_nanos=8)))
        old = dict(result=copy.deepcopy(value))
        value.update(seconds=2, peak_rss_kib=150)
        value['profile']['solve']['nanos'] = 20
        self.assertEqual(validate(value, old, 'D04', None), [])
        value['counters']['iterations'] = 11
        self.assertTrue(validate(value, old, 'D04', None))
        value['counters']['iterations'] = 10
        value['report']['residual'] = 1e-8
        self.assertTrue(validate(value, old, 'D04', None))

    def test_negative_control_must_preserve_the_actual_rejection(self):
        nvalue = dict(accepted=False, rejection='material covariance', defect=.1)
        measurement = dict(seconds=.001, peak_rss_kib=100, counters={}, stage_counters={}, profile={})
        old = dict(numerics_sha256=payload_sha(nvalue), measurement=copy.deepcopy(measurement))
        value = dict(numerics=nvalue, measurement=dict(measurement, references=None))
        self.assertEqual(validate(value, old, 'D05', None), [])
        nvalue['accepted'] = True
        self.assertTrue(validate(value, old, 'D05', None))

    def test_new_coarse_profile_requires_counters_and_identical_physical_output(self):
        value = dict(seconds=1, points=[dict(iterations=20, field_bits=[1, 2])],
                     counters={}, stage_counters={}, timings={}, counters_available=False)
        old = dict(result=copy.deepcopy(value))
        self.assertTrue(validate(value, old, 'coarse', None))
        value.update(counters=dict(iterations=20), counters_available=True)
        self.assertEqual(validate(value, old, 'coarse', None), [])
        value['points'][0]['field_bits'][0] = 3
        self.assertTrue(validate(value, old, 'coarse', None))


if __name__ == '__main__':
    unittest.main()
