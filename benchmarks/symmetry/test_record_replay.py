import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from replay_record_study import (
    archived_physical_gate, canonical, grouping, load_object, payload_sha,
    select, verify_result, unpack,
)


class RecordReplayTests(unittest.TestCase):
    def fixture(self, family):
        numerics = {'field_bits': [[1, 2], [3, 4]], 'iterations': 12}
        measurement = dict(seconds=1.0, peak_rss_kib=100, counters={}, stage_counters={}, profile={})
        original = dict(numerics_sha256=payload_sha(numerics))
        if family in ('D05', 'dispatch'):
            original['measurement'] = measurement.copy()
            value = dict(numerics=numerics, measurement=dict(measurement, references=None))
        else:
            original.update(measurement)
            value = dict(numerics, **measurement)
        return original, value

    def test_runtime_variation_is_allowed_but_fields_and_work_must_repeat(self):
        for family in ['D05', 'dispatch', 'PT']:
            with self.subTest(family=family):
                original, value = self.fixture(family)
                measurement = value['measurement'] if family in ('D05', 'dispatch') else value
                measurement.update(seconds=2.0, peak_rss_kib=200)
                self.assertEqual(verify_result(value, original, family)[2], [])
                changed = copy.deepcopy(value)
                numerics = changed['numerics'] if family in ('D05', 'dispatch') else changed
                numerics['field_bits'][0][0] += 1
                self.assertIn('Full numerical snapshot changed', verify_result(changed, original, family)[2])
                numerics['field_bits'][0][0] -= 1
                numerics['iterations'] += 1
                self.assertIn('Full numerical snapshot changed', verify_result(changed, original, family)[2])

    def test_instrumentation_and_reference_execution_cannot_enter_primary(self):
        for family in ['D05', 'dispatch', 'PT']:
            original, value = self.fixture(family)
            measurement = value['measurement'] if family in ('D05', 'dispatch') else value
            measurement['counters'] = {'fft': 14}
            self.assertIn('Instrumented output in primary sampling', verify_result(value, original, family)[2])
        original, value = self.fixture('D05')
        value['measurement']['references'] = {'projected': {}}
        with self.assertRaisesRegex(ValueError, 'references'):
            verify_result(value, original, 'D05')

    def test_corrupted_archive_object_is_rejected(self):
        value = {'field_bits': [123]}
        sha = payload_sha(value)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/'objects').mkdir()
            path = root/'objects'/f'{sha}.json.gz'
            path.write_bytes(gzip.compress(canonical(value)))
            self.assertEqual(load_object(root, sha), value)
            path.write_bytes(gzip.compress(canonical({'field_bits': [124]})))
            with self.assertRaisesRegex(ValueError, 'changed'):
                load_object(root, sha)

    def test_dispatch_parent_object_is_still_hash_verified(self):
        value = {'field_bits': [123]}
        sha = payload_sha(value)
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary); (parent/'objects').mkdir()
            root = parent/'dispatch'; root.mkdir()
            (root/'manifest.json').write_text(json.dumps(
                dict(schema='blaze2d/support-dispatch/1', parent='..')))
            path = parent/'objects'/f'{sha}.json.gz'
            path.write_bytes(gzip.compress(canonical(value)))
            self.assertEqual(load_object(root, sha), value)
            path.write_bytes(gzip.compress(canonical({'field_bits': [124]})))
            with self.assertRaisesRegex(ValueError, 'changed'):
                load_object(root, sha)

    def test_memory_workspace_metadata_cannot_be_hidden_by_field_comparison(self):
        value = dict(seconds=1.0, peak_rss_kib=100, counters={}, stage_counters={}, profile={},
                     points=[dict(field_bits=[1, 2], symmetry=dict(sectors=dict(
                         workspace=dict(budget_bytes=2048, peak_estimated_payload_bytes=1024))))])
        nvalue, measurement = unpack(value, 'memory')
        original = dict(copy.deepcopy(measurement), numerics_sha256=payload_sha(nvalue))
        self.assertEqual(verify_result(value, original, 'memory')[2], [])
        value['points'][0]['symmetry']['sectors']['workspace']['budget_bytes'] = 4096
        issues = verify_result(value, original, 'memory')[2]
        self.assertEqual(issues, ['Non-timing measurement metadata or work counters changed'])
        value['points'][0]['field_bits'][0] = 3
        self.assertIn('Full numerical snapshot changed', verify_result(value, original, 'memory')[2])

    def test_incomplete_or_duplicate_blocks_are_never_summarized(self):
        rows = [dict(pair=i, label=label) for i in range(-1, 20) for label in ['baseline', 'candidate']]
        self.assertEqual(len(grouping(rows, {'kind': 'paired'}, 'D05')), 21)
        with self.assertRaisesRegex(ValueError, 'complete blocks'):
            grouping(rows[:-1], {'kind': 'paired'}, 'D05')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            grouping(rows+[rows[0]], {'kind': 'paired'}, 'D05')

    def test_selection_preserves_order_and_excludes_profile_records(self):
        rows = [dict(study='x', pair=i, label=label, phase=phase)
                for i, phase in [(-1, 'warmup'), (0, 'primary'), (0, 'profile')]
                for label in ['candidate', 'baseline']]
        self.assertEqual(select(rows, 'x', {'kind': 'paired'}, 'D05'), rows[:4])

    def test_archived_physical_failure_is_not_erased_by_a_replay(self):
        audit = dict(issues=[], comparisons=[dict(study='x', valid=False)])
        self.assertFalse(archived_physical_gate(audit, 'x', {}, 'D05'))
        audit['comparisons'][0]['valid'] = True
        self.assertTrue(archived_physical_gate(audit, 'x', {}, 'D05'))
        audit['issues'].append('incomplete reference ledger')
        self.assertFalse(archived_physical_gate(audit, 'x', {}, 'D05'))
        with self.assertRaisesRegex(ValueError, 'Missing or ambiguous'):
            archived_physical_gate(dict(issues=[], comparisons=[]), 'x', {}, 'D05')


if __name__ == '__main__':
    unittest.main()
