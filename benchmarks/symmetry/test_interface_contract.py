import copy
import unittest
from interface_contract import physical_configuration, same_interface_jobs


class MatchedInterfaceContract(unittest.TestCase):
    def setUp(self):
        self.result = {
            "configuration": {
                "geometry": {"epsilon": 8.9}, "grid": {"resolution": [3, 5]},
                "bands": {"count": 2, "symmetry": {"k_reuse": "off", "arithmetic": "spatial_complex"}},
                "polarization": "TM", "eigensolver": {"precision": "f64", "tolerance": 1e-14},
                "results": {"eigenvectors": True}},
            "grid": [3, 5], "bands": 2, "expected_points": 2, "field_output": True,
            "polarization": "TM", "precision": "f64", "tolerance": 1e-14,
            "points": [{"k": [0.1, 0.2]}, {"k": [-0.2, 0.1]}],
            "arrays": {name: {"dtype": "float64", "shape": [2, 2], "payload_bytes": 32}
                       for name in ("frequencies", "k_points", "residuals", "absolute_residuals")}}
        self.result["arrays"]["eigenvectors"] = {
            "dtype": "complex128", "shape": [2, 2, 5, 3], "payload_bytes": 960}

    def test_policy_change_is_allowed_but_geometry_or_point_order_is_not(self):
        a = self.result
        self.assertIsNotNone(physical_configuration(a))
        b = copy.deepcopy(a)
        b["configuration"]["bands"]["symmetry"].update(k_reuse="unitary", arithmetic="pt_real")
        self.assertTrue(same_interface_jobs(a, b))
        b["configuration"]["geometry"]["epsilon"] = 9.0
        self.assertFalse(same_interface_jobs(a, b))
        b = copy.deepcopy(a)
        b["points"].reverse()
        self.assertFalse(same_interface_jobs(a, b))

    def test_fields_values_and_transposed_shapes_are_distinct_contracts(self):
        a = self.result
        b = copy.deepcopy(a)
        b["arrays"]["eigenvectors"]["shape"] = [2, 2, 3, 5]
        self.assertIsNone(physical_configuration(b))
        b = copy.deepcopy(a)
        b["field_output"] = False
        b["configuration"]["results"]["eigenvectors"] = False
        del b["arrays"]["eigenvectors"]
        self.assertIsNotNone(physical_configuration(b))
        self.assertFalse(same_interface_jobs(a, b))
        b["arrays"]["frequencies"]["payload_bytes"] = 31
        self.assertIsNone(physical_configuration(b))

    def test_missing_nonfinite_or_inconsistent_configuration_fails(self):
        for key in ("configuration", "arrays", "field_output"):
            b = copy.deepcopy(self.result)
            del b[key]
            self.assertIsNone(physical_configuration(b))
        b = copy.deepcopy(self.result)
        b["configuration"]["geometry"]["epsilon"] = float("nan")
        self.assertIsNone(physical_configuration(b))
        b = copy.deepcopy(self.result)
        b["grid"] = [5, 3]
        self.assertIsNone(physical_configuration(b))


if __name__ == "__main__":
    unittest.main()
