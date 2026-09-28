import json
from pathlib import Path
import tempfile
import unittest

from quiet_host import MEASUREMENT_PROTOCOL
from replay_quiet import digest
from rerun_record_studies import prior_attempts


class RecordQueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.study = dict(name='D05-fixture', study='fixture', original_files={'manifest.json': 'source-hash'})

    def tearDown(self):
        self.temporary.cleanup()

    def attempt(self, number, environment=False, numerical=True, root=None):
        root = root or self.root
        directory = root/('D05-fixture' if number == 1 else f'D05-fixture-attempt-{number}')
        directory.mkdir(parents=True)
        manifest = dict(study='fixture', original_files=self.study['original_files'],
                        measurement_protocol=MEASUREMENT_PROTOCOL, timestamp=number)
        (directory/'manifest.json').write_text(json.dumps(manifest))
        (directory/'records.jsonl').write_text(json.dumps(dict(issues=[]))+'\n')
        audit = dict(complete=True, artifacts_and_contracts_unchanged=True,
                     records_sha256=digest(directory/'records.jsonl'), environment_qualified=environment,
                     numerical_qualified=numerical, primary_latency_qualified=environment and numerical)
        (directory/'audit.json').write_text(json.dumps(audit))
        return directory

    def test_suffixed_attempts_across_roots_count_once(self):
        self.attempt(1)
        self.attempt(2)
        other = self.root/'later'
        self.attempt(3, root=other)
        prior, count = prior_attempts(self.study, [self.root, self.root, other])
        self.assertIsNone(prior)
        self.assertEqual(count, 3)

    def test_first_admitted_numerical_failure_cannot_be_replaced_by_success(self):
        first = self.attempt(1, environment=True, numerical=False)
        self.attempt(2, environment=True, numerical=True)
        prior, count = prior_attempts(self.study, [self.root])
        self.assertEqual(prior['referenced_attempt'], str(first))
        self.assertFalse(prior['numerical_qualified'])
        self.assertEqual(count, 2)

    def test_raw_record_corruption_requires_investigation(self):
        directory = self.attempt(1)
        (directory/'records.jsonl').write_text(json.dumps(dict(issues=['changed fields']))+'\n')
        with self.assertRaisesRegex(ValueError, 'raw records changed'):
            prior_attempts(self.study, [self.root])

    def test_output_change_is_not_an_environmental_retry(self):
        directory = self.attempt(1)
        (directory/'records.jsonl').write_text(json.dumps(dict(issues=['changed fields']))+'\n')
        audit = json.loads((directory/'audit.json').read_text())
        audit['records_sha256'] = digest(directory/'records.jsonl')
        (directory/'audit.json').write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError, 'output mismatch'):
            prior_attempts(self.study, [self.root])

    def test_old_clock_protocol_is_not_reused(self):
        directory = self.attempt(1)
        manifest = json.loads((directory/'manifest.json').read_text())
        manifest['measurement_protocol'] = 'old-clock'
        (directory/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'superseded clock'):
            prior_attempts(self.study, [self.root])


if __name__ == '__main__':
    unittest.main()
