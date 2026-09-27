"""Admission history must not select fast results or reset failed retries."""
import json
from pathlib import Path
import tempfile
import unittest

from rerun_critical_studies import previous_attempts
from replay_quiet import interaction_decision


class PriorAttempts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.study = dict(name='case-TE', original_sha256='frozen-input')

    def tearDown(self):
        self.temporary.cleanup()

    def attempt(self, cohort, suffix='', *, admitted=False, numerical=True, timestamp=1, mismatch=False):
        out = self.root/cohort/(self.study['name'] + suffix)
        out.mkdir(parents=True)
        (out/'manifest.json').write_text(json.dumps(dict(original_sha256='frozen-input', timestamp=timestamp)))
        (out/'audit.json').write_text(json.dumps(dict(environment_qualified=admitted,
            numerical_qualified=numerical, timing_qualified=admitted and numerical,
            primary_latency_qualified=admitted and numerical,
            numerical_signature_mismatches=[0] if mismatch else [])))
        return out

    def test_suffixed_failures_count_once_across_roots(self):
        self.attempt('one')
        self.attempt('two', '-attempt-2')
        self.attempt('two', '-attempt-3')
        prior, failures = previous_attempts(self.study, [self.root/'one', self.root/'two', self.root/'two'])
        self.assertIsNone(prior)
        self.assertEqual(failures, 3)

    def test_first_admitted_numerical_failure_is_preserved(self):
        first = self.attempt('one', '-attempt-2', admitted=True, numerical=False, timestamp=2)
        self.attempt('two', admitted=True, timestamp=3)
        prior, failures = previous_attempts(self.study, [self.root/'two', self.root/'one'])
        self.assertEqual(prior['referenced_attempt'], str(first))
        self.assertFalse(prior['timing_qualified'])
        self.assertFalse(prior['numerical_qualified'])
        self.assertEqual(failures, 0)

    def test_signature_failure_cannot_be_hidden_by_a_later_result(self):
        self.attempt('one', mismatch=True)
        self.attempt('two', admitted=True)
        with self.assertRaisesRegex(ValueError, 'Numerical signature changed'):
            previous_attempts(self.study, [self.root/'one', self.root/'two'])

    def test_unfinished_attempt_needs_investigation(self):
        (self.root/'one'/self.study['name']).mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'Incomplete prior attempt'):
            previous_attempts(self.study, [self.root/'one'])

    def test_interaction_is_a_time_factor_not_a_speedup(self):
        self.assertEqual(interaction_decision([1.1, 1.2]), 'above_multiplicative_time_prediction')
        self.assertEqual(interaction_decision([0.8, 0.9]), 'below_multiplicative_time_prediction')
        self.assertEqual(interaction_decision([0.99, 1.01]), 'within_two_percent_of_multiplicative_prediction')


if __name__ == '__main__':
    unittest.main()
