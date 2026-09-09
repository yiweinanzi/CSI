import copy
import unittest

from validate_complete_D import validate_counts


class DCompletionContractTests(unittest.TestCase):
    def setUp(self):
        self.bounded = {"status": "PAUSED_AT_UNIT_BOUNDARY", "sota_ready": False, "completed_units": 1, "total_units": 1489}
        self.progress = {"completed_units": 1, "total_units": 1489}
        phases = [{"phase": "training_complete", "probe": probe, "family": family} for probe in ("compatibility", "csi_only", "map_only", "scene_id_only", "edit_status_xor", "variant_id_matcher") for family in ("linear", "mlp2")]
        phases += [{"phase": "training_complete", "probe": probe, "family": None} for probe in ("response", "without_map", "edit_only", "csi_only", "oracle_x")]
        self.telemetry = {
            "workers": {"cuda:0": {"arm": "endpoint", "seed": 20270001, "checkpoint_index": 0, "unit_index": 0, "completed_training_models": 17, "total_training_models": 17}},
            "observed_probe_timelines": {"cuda:0/checkpoint-00": {"phase_timings": phases}},
        }

    def test_exact_original_counts_accepted(self):
        self.assertEqual(validate_counts(self.bounded, self.progress, self.telemetry)["completed_units"], 1)

    def test_partial_or_altered_official_count_rejected(self):
        for count in (0, 17, True):
            with self.assertRaises(RuntimeError):
                validate_counts({**self.bounded, "completed_units": count}, self.progress, self.telemetry)
        with self.assertRaises(RuntimeError):
            validate_counts(self.bounded, {**self.progress, "total_units": 17}, self.telemetry)

    def test_one_probe_is_not_a_unit(self):
        self.telemetry["workers"]["cuda:0"]["completed_training_models"] = 1
        with self.assertRaises(RuntimeError):
            validate_counts(self.bounded, self.progress, self.telemetry)

    def test_duplicate_model_does_not_replace_missing_model(self):
        phases = self.telemetry["observed_probe_timelines"]["cuda:0/checkpoint-00"]["phase_timings"]
        phases[-1] = copy.deepcopy(phases[0])
        with self.assertRaises(RuntimeError):
            validate_counts(self.bounded, self.progress, self.telemetry)

    def test_different_unit_identity_rejected(self):
        self.telemetry["workers"]["cuda:0"]["arm"] = "full"
        with self.assertRaises(RuntimeError):
            validate_counts(self.bounded, self.progress, self.telemetry)

    def test_probe_completion_cannot_approve_sota(self):
        with self.assertRaises(RuntimeError):
            validate_counts({**self.bounded, "sota_ready": True}, self.progress, self.telemetry)


if __name__ == "__main__":
    unittest.main()
