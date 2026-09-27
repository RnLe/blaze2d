"""Reject hidden fine-problem changes and omitted initializer work."""
import copy
import json
from pathlib import Path
import unittest

from coarse_workflow_contract import path_issues, same_jobs


class CoarseContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        archived = json.loads(Path(__file__).with_name('L08-native-square-star-TM-factorial.json').read_text())
        cls.reference = next(sample['result'] for sample in archived['samples'] if sample['mode'] == 'off')

    def pair(self):
        baseline = copy.deepcopy(self.reference)
        candidate = copy.deepcopy(baseline)
        candidate['mode'] = 'coarse'
        options = dict(resolution=[15, 15], max_seed_field_bytes=67108864)
        candidate['configuration']['bands']['symmetry']['coarse_start'] = options
        point = candidate['points'][0]
        point['iterations'] += 7
        report = point['symmetry']
        report['attempted_iterations'] += 7
        report['solver_attempts'].insert(0, dict(method='coarse_initializer', iterations=7, retained=False))
        report['coarse_start'] = dict(**options, used_as_seed=True, rejection=None,
            reserved_seed_field_bytes=274816, iterations=7)
        return baseline, candidate

    def test_initializer_only_changes_the_seed_contract(self):
        a, b = self.pair()
        self.assertTrue(same_jobs(a, b))
        self.assertEqual(path_issues(a), [])
        self.assertEqual(path_issues(b), [])

    def test_arithmetic_and_fine_solver_cannot_change_with_the_seed(self):
        a, b = self.pair()
        b['configuration']['bands']['symmetry']['arithmetic'] = 'pt_real'
        self.assertFalse(same_jobs(a, b))
        a, b = self.pair()
        b['configuration']['eigensolver']['block_size'] += 1
        self.assertFalse(same_jobs(a, b))

    def test_omitted_source_work_is_rejected(self):
        _, b = self.pair()
        b['points'][0]['symmetry']['solver_attempts'].pop(0)
        self.assertTrue(path_issues(b))

    def test_successful_fallback_is_not_initializer_acceptance(self):
        _, b = self.pair()
        b['points'][0]['symmetry']['coarse_start']['used_as_seed'] = False
        self.assertTrue(path_issues(b))


if __name__ == '__main__':
    unittest.main()
