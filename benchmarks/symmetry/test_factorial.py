import copy
import unittest
from run_factorial import MODES, summarize


class FactorialAttribution(unittest.TestCase):
    def block(self, times):
        return [{"mode": mode, "valid": True, "result": {"seconds": value}}
                for mode, value in zip(MODES, times)]

    def test_exact_multiplicative_and_antagonistic_cases(self):
        a = self.block([8, 4, 2, 1])
        s = summarize([a]*20, 20, resamples=100)
        self.assertAlmostEqual(s["pt_speedup"]["geometric_value"], 2)
        self.assertAlmostEqual(s["reuse_speedup"]["geometric_value"], 4)
        self.assertAlmostEqual(s["combined_speedup"]["geometric_value"], 8)
        self.assertAlmostEqual(s["interaction_time_factor"]["geometric_value"], 1)
        b = self.block([8, 4, 2, 2])
        s = summarize([b]*20, 20, resamples=100)
        self.assertAlmostEqual(s["interaction_time_factor"]["geometric_value"], 2)
        self.assertAlmostEqual(s["combined_speedup"]["geometric_value"], 4)

    def test_one_failed_cell_prevents_whole_experiment_acceptance(self):
        a = self.block([8, 4, 2, 1])
        b = copy.deepcopy(a)
        b[2]["valid"] = False
        s = summarize([a, b], 20, resamples=100)
        self.assertEqual(s["valid_blocks"], 1)
        self.assertEqual(s["failed_blocks"], 1)
        self.assertEqual(s["quality_decision"], "fail")
        self.assertEqual(s["combined_speedup"]["latency_decision"], "quality_failure")

    def test_a_smoke_block_cannot_claim_statistical_acceptance(self):
        s = summarize([self.block([8, 4, 2, 1])], 1, resamples=100)
        self.assertEqual(s["quality_decision"], "pass")
        self.assertIsNone(s["combined_speedup"]["ci95"])
        self.assertEqual(s["combined_speedup"]["latency_decision"], "insufficient_samples")


if __name__ == "__main__":
    unittest.main()
