from __future__ import annotations

import unittest

try:
    from artifacts.formal_readiness.tools.formal_data_pilot import (
        _project_core_factorial,
    )
except ModuleNotFoundError as error:
    raise unittest.SkipTest(str(error)) from error


class FormalDataPilotProjectionTests(unittest.TestCase):
    def test_two_gpu_projection_matches_formal_round_robin_scheduler(self):
        rates = {
            "endpoint": 2.0,
            "alignment": 1.0,
            "response": 4.0,
            "full": 0.5,
        }
        benchmarks = {
            arm: {"optimizer_steps_per_second": rate}
            for arm, rate in rates.items()
        }
        config = {
            "seeds": [101, 102, 103],
            "factorial": {"steps": 20},
        }
        projection = _project_core_factorial(benchmarks, config, gpu_count=2)

        self.assertEqual(projection["job_count"], 12)
        self.assertEqual(projection["device_queue_seconds"], [45.0, 180.0])
        self.assertEqual(
            projection["estimated_factorial_optimizer_wall_seconds"], 180.0
        )
        self.assertAlmostEqual(
            projection["estimated_factorial_optimizer_gpu_hours"], 225.0 / 3600.0
        )
        self.assertIn("external and representation baselines", projection["excluded"])


if __name__ == "__main__":
    unittest.main()
