from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from formal_v2.external_adapters.controlled_map_adapter import (
    CHECKPOINT_SCHEMA,
    load_controlled_map_config,
    validate_controlled_map_checkpoint,
)
from formal_v2.external_adapters.pmnet_adapter import load_pmnet_config
from formal_v2.external_adapters.wigatr_adapter import (
    _deterministic_pyg_attention_mask,
)
from formal_v2.external_adapters.scene_id_sigmap import (
    _load_authenticated_checkpoint,
    _validate_checkpoint,
)
from formal_v2.external_adapters.shuffled_pair_control import (
    InsufficientDerangementSupport,
    _bucket_count,
    _validate_pilot,
)
from formal_v2.external_adapters.wigatr_protocol import load_wigatr_config
from formal_v2.formal_external import (
    _validate_manifest,
    require_dataset_adapter_profiles,
)
from formal_v2.formal_claim_controls import (
    SHUFFLED_FIXTURE_SUPPORT_RESULT_SCHEMA,
    SHUFFLED_FIXTURE_SUPPORT_STATUS,
    _validate_fixture_support_result,
    _validate_formal_checkpoint,
)
from formal_v2.formal_io import read_strict_json, sha256_file
from formal_v2.formal_model import CSIPairsFormalModel


ROOT = Path(__file__).resolve().parents[2]


def _checkpoint_payload(dataset_sha256: str) -> dict:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "model_name": "SigMap",
        "method": "sigmap",
        "implementation_status": "style-controlled-implementation",
        "source_revision": "frozen-v6-sigmap-spec-2026-08-06",
        "dataset_sha256": dataset_sha256,
        "train_role": "source_encoder_train",
        "selection_role": "source_method_selection",
        "target_roles_read": [],
        "selected_step": 1,
        "selection_loss": 0.5,
        "normalizer": {
            "csi_mean": [0.0],
            "csi_std": [1.0],
            "map_mean": [0.0],
            "map_std": [1.0],
            "context_mean": [0.0],
            "context_std": [1.0],
        },
        "model_metadata": {"antennas": 1, "subcarriers": 1},
        "effective_batch_size": 2,
        "microbatch_size": 1,
        "gradient_accumulation_steps": 2,
        "configured_precision": "float32",
        "executed_precision": "float32",
        "autocast_enabled": False,
        "state_dict": {"weight": torch.ones(1)},
    }


class ControlledMapCheckpointContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset_path = self.root / "fixture.npz"
        self.dataset_path.write_bytes(b"strict-checkpoint-binding")
        self.dataset = SimpleNamespace(source_path=self.dataset_path)
        self.dataset_sha256 = sha256_file(self.dataset_path)
        self.execution = {
            "source_revision": "frozen-v6-sigmap-spec-2026-08-06",
            "dataset_sha256": self.dataset_sha256,
            "checkpoint_sha256": "a" * 64,
        }

    def tearDown(self):
        self.temporary.cleanup()

    def test_scene_id_accepts_the_shared_v2_checkpoint_contract(self):
        payload = _checkpoint_payload(self.dataset_sha256)
        self.assertIs(validate_controlled_map_checkpoint(payload), payload)
        _validate_checkpoint(payload, self.dataset, self.execution)

    def test_checkpoint_contract_rejects_missing_and_extra_fields(self):
        missing = _checkpoint_payload(self.dataset_sha256)
        missing.pop("microbatch_size")
        with self.assertRaisesRegex(RuntimeError, "fields must be exact"):
            validate_controlled_map_checkpoint(missing)
        extra = _checkpoint_payload(self.dataset_sha256)
        extra["legacy_field"] = True
        with self.assertRaisesRegex(RuntimeError, "fields must be exact"):
            validate_controlled_map_checkpoint(extra)

    def test_checkpoint_contract_rejects_schema_and_binding_drift(self):
        legacy = _checkpoint_payload(self.dataset_sha256)
        legacy["schema_version"] = "csi-pairs-v6-controlled-map-checkpoint-v1"
        with self.assertRaisesRegex(RuntimeError, "schema mismatch"):
            validate_controlled_map_checkpoint(legacy)
        wrong_dataset = _checkpoint_payload("b" * 64)
        with self.assertRaisesRegex(RuntimeError, "source-only contract"):
            _validate_checkpoint(wrong_dataset, self.dataset, self.execution)
        wrong_source = copy.deepcopy(self.execution)
        wrong_source["source_revision"] = "different-source"
        with self.assertRaisesRegex(RuntimeError, "source-only contract"):
            _validate_checkpoint(
                _checkpoint_payload(self.dataset_sha256), self.dataset, wrong_source
            )

    def test_checkpoint_loader_rejects_sha_mismatch(self):
        checkpoint = self.root / "checkpoint.pt"
        torch.save(_checkpoint_payload(self.dataset_sha256), checkpoint)
        execution = {"checkpoint_sha256": sha256_file(checkpoint)}
        loaded = _load_authenticated_checkpoint(checkpoint, execution)
        self.assertEqual(loaded["schema_version"], CHECKPOINT_SCHEMA)
        execution["checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "differs from external-baseline"):
            _load_authenticated_checkpoint(checkpoint, execution)


class ExternalSmokeManifestTests(unittest.TestCase):
    def test_smoke_manifest_is_strict_and_uses_only_small_model_specific_doses(self):
        manifest = read_strict_json(
            ROOT / "formal_v2/external_adapters/all_map_adapters_smoke_v1.json"
        )
        _validate_manifest(manifest)
        self.assertEqual(
            {row["model_name"] for row in manifest["adapters"]},
            {"SigMap", "Wi-GATr", "PMNet", "WiSER", "RFIR"},
        )
        profiles = []
        for row in manifest["adapters"]:
            path = ROOT / row["adapter_config_path"]
            payload = read_strict_json(path)
            profiles.append(payload["profile"])
            if row["model_name"] == "Wi-GATr":
                self.assertLessEqual(load_wigatr_config(path)["training"]["steps"], 2)
            elif row["model_name"] == "PMNet":
                self.assertLessEqual(load_pmnet_config(path)["training"]["epochs"], 1)
            else:
                controlled = load_controlled_map_config(path)
                self.assertEqual(controlled["model_name"], row["model_name"])
                self.assertLessEqual(controlled["training"]["steps"], 2)
        self.assertEqual(set(profiles), {"software-smoke-only"})

    def test_dataset_profile_binding_rejects_formal_dose_for_fixture(self):
        formal = read_strict_json(
            ROOT / "formal_v2/external_adapters/all_map_adapters_v1.json"
        )
        smoke = read_strict_json(
            ROOT / "formal_v2/external_adapters/all_map_adapters_smoke_v1.json"
        )
        fixture = SimpleNamespace(is_fixture=True)
        scientific = SimpleNamespace(is_fixture=False)
        self.assertEqual(require_dataset_adapter_profiles(smoke, fixture), "software-smoke-only")
        self.assertEqual(require_dataset_adapter_profiles(formal, scientific), "formal-paper-dose")
        with self.assertRaisesRegex(RuntimeError, "software-smoke-only"):
            require_dataset_adapter_profiles(formal, fixture)
        with self.assertRaisesRegex(RuntimeError, "formal-paper-dose"):
            require_dataset_adapter_profiles(smoke, scientific)


class ClaimControlSmokeContractTests(unittest.TestCase):
    def _pilot(self) -> dict:
        return {
            "schema_version": "csi-pairs-v6-frozen-pilot-v1",
            "source_roles": [
                "source_encoder_train",
                "source_method_selection",
            ],
            "checkpoint_reused_for_final_training": False,
            "alignment_scale": 0.04,
            "response_scale": 1.95,
            "alignment_null_tolerance": 0.0,
        }

    def test_zero_noop_tolerance_from_factorial_producer_is_valid(self):
        _validate_pilot(self._pilot())

    def test_pilot_rejects_nonpositive_scales_and_negative_tolerance(self):
        for key, value in (
            ("alignment_scale", 0.0),
            ("response_scale", 0.0),
            ("alignment_null_tolerance", -1e-9),
        ):
            with self.subTest(key=key):
                pilot = self._pilot()
                pilot[key] = value
                with self.assertRaisesRegex(RuntimeError, "pilot scale is invalid"):
                    _validate_pilot(pilot)

    def _insufficient_support_result(self) -> dict:
        return {
            "schema_version": SHUFFLED_FIXTURE_SUPPORT_RESULT_SCHEMA,
            "status": SHUFFLED_FIXTURE_SUPPORT_STATUS,
            "dataset_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "fixture": True,
            "scientific_use": "FORBIDDEN",
            "reason": "strict fixture stratum cannot be deranged",
            "failed_branch": "alignment_h_map_edge",
            "required_pairing_strata": [
                "scene",
                "edit_family",
                "route",
                "effect_bucket",
            ],
            "pair_registry_sha256": "3" * 64,
            "checkpoint_index_sha256": "4" * 64,
            "adapter_source_sha256": "5" * 64,
        }

    def test_insufficient_derangement_is_fixture_only_and_exact(self):
        result = self._insufficient_support_result()
        _validate_fixture_support_result(
            result, SimpleNamespace(is_fixture=True)
        )
        with self.assertRaisesRegex(RuntimeError, "only be reported for a fixture"):
            _validate_fixture_support_result(
                result, SimpleNamespace(is_fixture=False)
            )
        result["unbound_field"] = True
        with self.assertRaisesRegex(RuntimeError, "fields are not exact"):
            _validate_fixture_support_result(
                result, SimpleNamespace(is_fixture=True)
            )

    def test_formal_derangement_preflight_remains_fail_closed(self):
        with self.assertRaises(InsufficientDerangementSupport):
            _bucket_count([(0, 0, 1, 0)], "alignment_h_map_edge")


class FormalCheckpointContractTests(unittest.TestCase):
    def _payload(self) -> tuple[dict, dict]:
        model = CSIPairsFormalModel(
            patch_count=4,
            patch_rows=2,
            patch_columns=2,
            patch_dim=2,
            map_channels=3,
            action_channels=12,
            radio_dim=4,
            latent_dim=8,
            state_dim=8,
            map_dim=8,
            hidden_dim=16,
            attention_heads=2,
        )
        evidence = {
            "artifact_label": "software-fixture",
            "dataset_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "fixture": True,
            "scientific_use": "FORBIDDEN",
            "source_tree_sha256": "3" * 64,
            "requirements_lock_sha256": "4" * 64,
            "runtime_provenance_sha256": "5" * 64,
            "runtime_provenance": {"schema_version": "test-runtime"},
        }
        return {
            "schema_version": "csi-pairs-formal-checkpoint-v2.1-v6",
            "arm": "full",
            "seed": 17,
            "model_spec": {
                "patch_count": 4,
                "patch_rows": 2,
                "patch_columns": 2,
                "patch_dim": 2,
                "map_channels": 3,
                "action_channels": 12,
                "radio_dim": 4,
                "latent_dim": 8,
                "state_dim": 8,
                "map_dim": 8,
                "hidden_dim": 16,
                "attention_heads": 2,
            },
            "normalization": {},
            "teacher_checkpoint_sha256": "6" * 64,
            "checkpoint_rule": "fixed_final_step_no_target_selection",
            "state_dict": model.state_dict(),
            **evidence,
        }, evidence

    def _validate(self, payload: dict, evidence: dict) -> dict:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "full.pt"
            torch.save(payload, path)
            return _validate_formal_checkpoint(path, {"seed": 17}, evidence)

    def test_full_checkpoint_accepts_complete_runtime_evidence_contract(self):
        payload, evidence = self._payload()
        self.assertEqual(self._validate(payload, evidence)["seed"], 17)

    def test_full_checkpoint_rejects_missing_and_extra_fields(self):
        payload, evidence = self._payload()
        payload.pop("runtime_provenance")
        with self.assertRaisesRegex(RuntimeError, "frozen F/P contract"):
            self._validate(payload, evidence)
        payload, evidence = self._payload()
        payload["legacy_field"] = True
        with self.assertRaisesRegex(RuntimeError, "frozen F/P contract"):
            self._validate(payload, evidence)

    def test_full_checkpoint_rejects_runtime_binding_drift(self):
        payload, evidence = self._payload()
        payload["source_tree_sha256"] = "7" * 64
        with self.assertRaisesRegex(RuntimeError, "source_tree_sha256 mismatch"):
            self._validate(payload, evidence)


class RecordedRuntimeAuthenticationTests(unittest.TestCase):
    def test_recorded_runtime_path_is_fail_closed(self):
        from formal_v2.formal_evidence import evidence_context_from_recorded_runtime

        config = {"artifact_label": "fixture", "schema_version": "unused"}
        dataset_path = Path(tempfile.gettempdir()) / "recorded-runtime-fixture.npz"
        dataset = SimpleNamespace(source_path=dataset_path, is_fixture=True)
        dataset_path.write_bytes(b"runtime-binding")
        runtime = {
            "source_tree_sha256": "1" * 64,
            "requirements_lock_sha256": "2" * 64,
        }
        with patch(
            "formal_v2.formal_evidence.configure_reproducible_runtime"
        ), patch(
            "formal_v2.formal_evidence.validate_runtime_provenance",
            return_value=runtime,
        ) as validate, patch(
            "formal_v2.formal_evidence.config_sha256", return_value="3" * 64
        ):
            result = evidence_context_from_recorded_runtime(
                config, dataset, "CANDIDATE_NOT_CLAIM", runtime
            )
        validate.assert_called_once_with(runtime)
        self.assertEqual(result["scientific_use"], "FORBIDDEN")
        with patch(
            "formal_v2.formal_evidence.configure_reproducible_runtime"
        ), patch(
            "formal_v2.formal_evidence.validate_runtime_provenance",
            side_effect=RuntimeError("main runtime installer report provenance mismatch"),
        ):
            with self.assertRaisesRegex(RuntimeError, "installer report provenance"):
                evidence_context_from_recorded_runtime(config, dataset, "FORBIDDEN", {})
        dataset_path.unlink(missing_ok=True)


class WiGATrDeterminismTests(unittest.TestCase):
    def test_attention_mask_is_deterministic_block_diagonal_tensor(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = SimpleNamespace(
            batch=torch.tensor([0, 0, 1, 1, 1], device=device, dtype=torch.long)
        )
        mask = _deterministic_pyg_attention_mask(inputs, torch)
        self.assertEqual(mask.device.type, device)
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.sum(dim=1).tolist(), [2, 2, 3, 3, 3])

        causal = _deterministic_pyg_attention_mask(inputs, torch, causal=True)
        self.assertEqual(causal.sum(dim=1).tolist(), [1, 2, 1, 2, 3])

        repeated = _deterministic_pyg_attention_mask(
            inputs, torch, multiply_batch_sizes=2
        )
        self.assertEqual(tuple(repeated.shape), (10, 10))
        self.assertEqual(repeated.sum(dim=1).tolist(), [2, 2, 3, 3, 3] * 2)


if __name__ == "__main__":
    unittest.main()
