import copy
import gzip
import json
import math
from pathlib import Path
import unittest

from pt_sector_contract import independent, path_issues, same_pencil, spectrum_error
from run_pairs import quality_passes
from run_pt_sectors import interval


class PtSectorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).with_name('pt-sector-fixtures')/'native.json.gz'
        cls.fixtures=json.loads(gzip.decompress(path.read_bytes()))

    def test_actual_small_modes_pass_both_reported_and_independent_gates(self):
        reference=self.fixtures['gamma-off']
        for name,value in self.fixtures.items():
            with self.subTest(name=name):
                self.assertEqual(path_issues(value),[])
                self.assertTrue(same_pencil(reference,value))
                self.assertLess(spectrum_error(reference,value),1e-6)
                self.assertEqual(independent(value,reference)['issues'],[])

    def test_ignored_representation_and_unexplained_complex_fallback_fail(self):
        for failure in ('ignored','unexplained','forced-real'):
            value=copy.deepcopy(self.fixtures['rotation-pt-sectors'])
            sectors=value['points'][0]['symmetry']['sectors']['sectors']
            complex_sector=next(s for s in sectors if s.get('coordinate_rejection'))
            if failure=='ignored':
                for sector in sectors:sector.pop('coordinates')
            elif failure=='unexplained':
                complex_sector.pop('coordinate_rejection')
            else:
                complex_sector['coordinates']['representation']='pt_real_te'
            self.assertTrue(path_issues(value),failure)

    def test_whole_solve_fallback_is_not_accepted_as_a_sector_speedup(self):
        value=copy.deepcopy(self.fixtures['gamma-pt-sectors'])
        report=value['points'][0]['symmetry']
        report['sectors']=None
        report['solver_attempts'][0]['method']='unrestricted'
        self.assertTrue(quality_passes(value,4))
        self.assertTrue(path_issues(value))

    def test_discarded_iterations_cannot_disappear_from_the_ledger(self):
        value=copy.deepcopy(self.fixtures['gamma-pt-sectors'])
        point=value['points'][0]
        point['symmetry']['attempted_iterations']+=1
        self.assertTrue(any('work ledger' in x for x in path_issues(value)))

    def test_returned_fields_are_checked_independently_of_solver_flags(self):
        value=copy.deepcopy(self.fixtures['gamma-pt-sectors'])
        value['points'][0]['field_bits'][2][0][0]=0x4000000000000000
        self.assertTrue(quality_passes(value,4))
        self.assertEqual(path_issues(value),[])
        self.assertTrue(independent(value)['issues'])

    def test_material_change_and_missing_multiplicity_are_rejected(self):
        reference=self.fixtures['gamma-off']
        value=copy.deepcopy(self.fixtures['gamma-pt-sectors'])
        value['points'][0]['operator_identity']['operator_bits'][-1]^=1
        self.assertFalse(same_pencil(reference,value))
        value['points'][0]['symmetry']['sectors']['sum_sector_dimensions']-=1
        self.assertTrue(path_issues(value))

    def test_nonfinite_bits_cannot_hide_behind_finite_quality_flags(self):
        value=copy.deepcopy(self.fixtures['gamma-pt-sectors'])
        value['points'][0]['field_bits'][1][0][0]=0x7ff8000000000000
        self.assertTrue(quality_passes(value,4))
        self.assertTrue(path_issues(value))

    def test_practical_decisions_use_the_declared_two_percent_threshold(self):
        for ratio,expected in [(2.,'gain'),(.5,'regression'),(.99,'equivalent_within_2_percent')]:
            result=interval([math.log(ratio)]*20,123)
            self.assertAlmostEqual(result['geometric_ratio'],ratio)
            self.assertEqual(result['practical_decision'],expected)


if __name__=='__main__':unittest.main()
