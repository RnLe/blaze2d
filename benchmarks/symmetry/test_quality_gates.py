import copy
import unittest
from run_pairs import quality_passes


class PhysicalQualityGate(unittest.TestCase):
    def setUp(self):
        self.result = {"spectrum": [[0.1, 0.2]], "points": [{
            "k": [0.1, 0.2], "converged": True, "field_count": 2,
            "residuals": [1e-7, 1e-7], "absolute_residuals": [1e-6, 1e-6],
            "b_orthogonality_defect": 1e-12}]}

    def test_late_nonfinite_certificate_cannot_hide_behind_maximum(self):
        self.assertTrue(quality_passes(self.result, 2))
        for bad in (float('nan'), float('inf'), None, -1.0):
            for key in ('residuals', 'absolute_residuals'):
                result = copy.deepcopy(self.result)
                result['points'][0][key][1] = bad
                self.assertFalse(quality_passes(result, 2))
            result = copy.deepcopy(self.result)
            result['points'].append(copy.deepcopy(result['points'][0]))
            result['spectrum'].append([0.1, 0.2])
            result['points'][1]['b_orthogonality_defect'] = bad
            self.assertFalse(quality_passes(result, 2))

    def test_gamma_requires_finite_absolute_certificate_and_complete_shapes(self):
        result = self.result
        point = result['points'][0]
        point['k'] = [0.0, 0.0]
        point['residuals'][0] = 1.0
        point['absolute_residuals'][0] = 1e-12
        self.assertTrue(quality_passes(result, 2))
        point['absolute_residuals'] = []
        self.assertFalse(quality_passes(result, 2))
        point['k'] = [0.1, 0.2]
        point['residuals'] = [1e-7]
        self.assertFalse(quality_passes(result, 2))
        self.assertFalse(quality_passes({'spectrum': [], 'points': []}, 2))


if __name__ == '__main__':
    unittest.main()
