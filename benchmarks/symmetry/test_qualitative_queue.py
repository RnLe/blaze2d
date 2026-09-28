import json
from pathlib import Path
import tempfile
import unittest

from continue_qualitative_studies import attempts, choose, new_allowance_used


class QualitativeQueueTests(unittest.TestCase):
    def test_new_inventory_cannot_reset_the_additional_attempt_allowance(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path/'manifest.json').write_text(json.dumps(dict(preflight_limits=dict(linux=.25, windows=1.5))))
            prior = [(1, path, {})]
            self.assertFalse(new_allowance_used(prior))
            (path/'manifest.json').write_text(json.dumps(dict(preflight_limits=dict(linux=1.0, windows=3.0))))
            self.assertTrue(new_allowance_used(prior))

    def test_first_strict_numerical_failure_survives_later_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prior = []
            for number, success in enumerate([False, True]):
                path = root/str(number); path.mkdir()
                audit = dict(environment_qualified=True, numerical_qualified=success, primary_latency_qualified=success)
                (path/'audit.json').write_text(json.dumps(audit))
                prior.append((number, path, audit))
            result = choose({}, prior, root)
            self.assertEqual(result['state'], 'numerical failure')
            self.assertEqual(result['evidence'], str(root/'0'))

    def test_old_clock_cannot_enter_through_the_strict_reference_shortcut(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); path = root/'x'; path.mkdir()
            (path/'manifest.json').write_text(json.dumps(dict(measurement_protocol='old-clock')))
            (path/'audit.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'superseded clock'):
                attempts(dict(name='x'), [root])


if __name__ == '__main__':
    unittest.main()
