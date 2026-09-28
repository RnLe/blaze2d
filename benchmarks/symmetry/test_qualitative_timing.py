import copy
import json
from pathlib import Path
import tempfile
import unittest

from qualitative_timing import analyze, qualitative_decision, summarize
from quiet_host import MEASUREMENT_PROTOCOL


class QualitativeTimingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = dict(schema='blaze2d/quiet-replay/1', measurement_protocol=MEASUREMENT_PROTOCOL)
        self.audit = dict(clock_qualified=True, numerical_qualified=True, complete=True,
                          numerical_signature_mismatches=[], environment_blocks=[
                              dict(block=i, accepted=i < 16, issues=[] if i < 16 else ['background CPU'])
                              for i in range(-1, 20)])
        self.projection = dict(clock_qualified=True, samples=[
            dict(pair=i, label=v, valid=True, result=dict(seconds=t, peak_rss_kib=100))
            for i in range(-1, 20) for v, t in [('baseline', 2.0), ('candidate', 1.0)]])

    def tearDown(self):
        self.temporary.cleanup()

    def evaluate(self):
        for name, value in [('manifest.json', self.manifest), ('audit.json', self.audit),
                            ('clock-normalized.json', self.projection), ('samples.json', {})]:
            (self.root/name).write_text(json.dumps(value))
        (self.root/'environment.jsonl').write_text('')
        return analyze(self.root)

    def test_whole_block_selection_is_independent_of_elapsed_ratio(self):
        first = self.evaluate()
        self.assertEqual(first['retained_blocks'], list(range(16)))
        for sample in self.projection['samples']:
            if sample['pair'] >= 16:
                sample['result']['seconds'] *= 100 if sample['label'] == 'candidate' else .1
        second = self.evaluate()
        self.assertEqual(first['retained_blocks'], second['retained_blocks'])
        self.assertEqual(first['filtered'], second['filtered'])
        self.assertNotEqual(first['all_blocks'], second['all_blocks'])
        self.assertEqual(second['decisions']['paired'], 'sensitive_to_environment_exclusions')

    def test_excluded_numerical_failure_still_invalidates_all_timing(self):
        self.audit['numerical_qualified'] = False
        self.projection['samples'][-1]['valid'] = False
        result = self.evaluate()
        self.assertFalse(result['qualitative_timing_qualified'])
        self.assertNotIn('filtered', result)

    def test_too_few_clean_blocks_are_not_summarized(self):
        for block in self.audit['environment_blocks']:
            block['accepted'] = block['block'] < 11
        result = self.evaluate()
        self.assertFalse(result['qualitative_timing_qualified'])
        self.assertNotIn('filtered', result)

    def test_superseded_clocks_are_not_rehabilitated(self):
        self.manifest['measurement_protocol'] = 'adjusted-only'
        self.assertFalse(self.evaluate()['qualitative_timing_qualified'])
        self.manifest['measurement_protocol'] = MEASUREMENT_PROTOCOL
        self.audit['clock_qualified'] = False
        self.assertFalse(self.evaluate()['qualitative_timing_qualified'])

    def test_instrumented_runs_stay_diagnostic(self):
        self.audit['instrumented'] = True
        self.assertFalse(self.evaluate()['qualitative_timing_qualified'])

    def test_missing_variant_is_not_a_valid_pair(self):
        self.projection['samples'].pop(2)
        with self.assertRaisesRegex(ValueError, 'Incomplete matched block'):
            self.evaluate()

    def test_factorial_resamples_complete_vectors_and_labels_time_factor(self):
        blocks = {i: {v: dict(seconds=t, peak_rss_kib=None) for v,t in
                       [('off', 12), ('pt', 6), ('unitary', 4), ('pt-unitary', 3)]} for i in range(12)}
        result = summarize(blocks, list(blocks), resamples=100)
        ratio = result['contrasts']['interaction_time_factor']
        self.assertAlmostEqual(ratio['geometric_ratio'], 1.5)
        self.assertEqual(qualitative_decision(ratio, ratio, True), 'above_multiplicative_time_prediction')
        self.assertIsNone(result['median_rss_kib']['off'])


if __name__ == '__main__':
    unittest.main()
