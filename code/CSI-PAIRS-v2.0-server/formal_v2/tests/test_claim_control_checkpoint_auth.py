from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from formal_v2 import formal_migration as migration
from formal_v2 import formal_upstream as upstream
from formal_v2.formal_config import ARMS
from formal_v2.formal_claim_controls import (
    _CHECKPOINT_EVIDENCE_LEGACY_V1,
    _CHECKPOINT_EVIDENCE_MIGRATED_LEGACY_V1,
    _CHECKPOINT_EVIDENCE_RECORDED_RUNTIME_V3,
    _CHECKPOINT_COMMON_EVIDENCE_FIELDS,
    _CHECKPOINT_PROVENANCE_PAYLOAD_FIELDS,
    _FORMAL_CHECKPOINT_BASE_FIELDS,
    _SHUFFLED_CHECKPOINT_BASE_FIELDS,
    _authenticated_qualification_gate,
    _load_control_full_checkpoint_payloads,
    _validate_checkpoint_evidence,
    _validate_formal_checkpoint,
    _validate_shuffled_checkpoint,
    _verify_checkpoint_index_binding,
)
from formal_v2.formal_evidence import (
    RUNTIME_PROVENANCE_FIELDS,
    RUNTIME_PROVENANCE_SCHEMA,
    TORCH_RUNTIME_FIELDS,
)
from formal_v2.formal_io import read_strict_json, sha256_file, write_json
from formal_v2.tests.test_formal_migration import (
    ACCEPTED_UTC,
    CONFIG_SHA256,
    NOW,
    FormalMigrationFixture,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _runtime_digest(runtime: dict) -> str:
    encoded = json.dumps(
        runtime,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ClaimControlCheckpointAuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.source_sha256 = _digest("fixed-source")
        self.requirements_sha256 = _digest("requirements")
        self.runtime = self._runtime_record()
        self.runtime_sha256 = _runtime_digest(self.runtime)
        self.common_evidence = {
            "artifact_label": "csi-pairs-formal-v2.1-v6",
            "dataset_sha256": _digest("dataset"),
            "config_sha256": _digest("config"),
            "fixture": False,
            "scientific_use": "CANDIDATE_NOT_CLAIM",
        }
        self.current_evidence = {
            **self.common_evidence,
            "source_tree_sha256": self.source_sha256,
            "requirements_lock_sha256": self.requirements_sha256,
            "runtime_provenance_sha256": self.runtime_sha256,
            "runtime_provenance": self.runtime,
        }
        self.legacy_row = {"seed": 20270001}
        self.current_row = {
            "seed": 20270001,
            "source_tree_sha256": self.source_sha256,
            "requirements_lock_sha256": self.requirements_sha256,
            "runtime_provenance_sha256": self.runtime_sha256,
            "teacher_checkpoint_sha256": _digest("teacher"),
        }

    def test_recorded_runtime_formal_shape_is_accepted_exactly(self):
        payload = self._formal_payload(current=True)
        self.assertEqual(
            set(payload),
            set(_FORMAL_CHECKPOINT_BASE_FIELDS).union(
                _CHECKPOINT_PROVENANCE_PAYLOAD_FIELDS
            ),
        )
        version = _validate_checkpoint_evidence(
            payload,
            base_fields=_FORMAL_CHECKPOINT_BASE_FIELDS,
            provenance_binding=self.current_row,
            evidence=self.current_evidence,
            label="claim-control checkpoint",
        )
        self.assertEqual(version, _CHECKPOINT_EVIDENCE_RECORDED_RUNTIME_V3)

    def test_legacy_shape_is_accepted_only_with_an_explicit_legacy_binding(self):
        payload = self._formal_payload(current=False)
        version = _validate_checkpoint_evidence(
            payload,
            base_fields=_FORMAL_CHECKPOINT_BASE_FIELDS,
            provenance_binding=self.legacy_row,
            evidence=self.common_evidence,
            label="claim-control checkpoint",
        )
        self.assertEqual(version, _CHECKPOINT_EVIDENCE_LEGACY_V1)
        with self.assertRaisesRegex(RuntimeError, "recorded-runtime-v3"):
            _validate_checkpoint_evidence(
                payload,
                base_fields=_FORMAL_CHECKPOINT_BASE_FIELDS,
                provenance_binding=self.current_row,
                evidence=self.current_evidence,
                label="claim-control checkpoint",
            )

    def test_current_binding_rejects_each_missing_provenance_field_and_downgrade(self):
        for missing in [
            (field,) for field in sorted(_CHECKPOINT_PROVENANCE_PAYLOAD_FIELDS)
        ] + [tuple(sorted(_CHECKPOINT_PROVENANCE_PAYLOAD_FIELDS))]:
            with self.subTest(missing=missing):
                payload = self._formal_payload(current=True)
                for field in missing:
                    payload.pop(field)
                with self.assertRaisesRegex(RuntimeError, "fields are not exact"):
                    _validate_checkpoint_evidence(
                        payload,
                        base_fields=_FORMAL_CHECKPOINT_BASE_FIELDS,
                        provenance_binding=self.current_row,
                        evidence=self.current_evidence,
                        label="claim-control checkpoint",
                    )

    def test_unknown_checkpoint_field_is_rejected_for_both_versions(self):
        for current, binding, evidence in (
            (False, self.legacy_row, self.common_evidence),
            (True, self.current_row, self.current_evidence),
        ):
            with self.subTest(current=current):
                payload = self._formal_payload(current=current)
                payload["unknown_future_or_forged_field"] = True
                with self.assertRaisesRegex(RuntimeError, "fields are not exact"):
                    _validate_checkpoint_evidence(
                        payload,
                        base_fields=_FORMAL_CHECKPOINT_BASE_FIELDS,
                        provenance_binding=binding,
                        evidence=evidence,
                        label="claim-control checkpoint",
                    )

    def test_partial_provenance_binding_is_rejected_not_treated_as_legacy(self):
        partial = {
            "seed": 20270001,
            "source_tree_sha256": self.source_sha256,
        }
        with self.assertRaisesRegex(RuntimeError, "binding is incomplete"):
            _validate_checkpoint_evidence(
                self._formal_payload(current=True),
                base_fields=_FORMAL_CHECKPOINT_BASE_FIELDS,
                provenance_binding=partial,
                evidence=self.current_evidence,
                label="claim-control checkpoint",
            )

    def test_runtime_provenance_tampering_is_rejected_before_model_loading(self):
        payload = self._formal_payload(current=True)
        payload["runtime_provenance"] = copy.deepcopy(payload["runtime_provenance"])
        payload["runtime_provenance"]["python_version"] = "3.12.tampered"
        with self.assertRaisesRegex(RuntimeError, "digest or identity mismatch"):
            _validate_checkpoint_evidence(
                payload,
                base_fields=_FORMAL_CHECKPOINT_BASE_FIELDS,
                provenance_binding=self.current_row,
                evidence=self.current_evidence,
                label="claim-control checkpoint",
            )

        payload["runtime_provenance_sha256"] = _runtime_digest(
            payload["runtime_provenance"]
        )
        with self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
            _validate_checkpoint_evidence(
                payload,
                base_fields=_FORMAL_CHECKPOINT_BASE_FIELDS,
                provenance_binding=self.current_row,
                evidence=self.current_evidence,
                label="claim-control checkpoint",
            )

    def test_fixed_formal_checkpoint_shape_passes_the_full_loader(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable")

        class FakeFormalModel:
            def __init__(self, **_model_spec):
                self._state = {"weight": torch.ones(1)}

            def load_state_dict(self, state_dict, strict):
                if strict is not True or set(state_dict) != set(self._state):
                    raise RuntimeError("state mismatch")

            def state_dict(self):
                return self._state

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "full.pt"
            torch.save(self._formal_payload(current=True, torch_module=torch), checkpoint)
            with patch(
                "formal_v2.formal_model.CSIPairsFormalModel", FakeFormalModel
            ):
                loaded = _validate_formal_checkpoint(
                    checkpoint,
                    self.current_row,
                    self.current_evidence,
                    teacher_checkpoint_sha256=_digest("teacher"),
                )
        self.assertEqual(loaded["runtime_provenance_sha256"], self.runtime_sha256)

    def test_claim_control_preflight_accepts_current_runtime_checkpoints(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable")

        class FakeFormalModel:
            def __init__(self, **_model_spec):
                self._state = {"weight": torch.ones(1)}

            def load_state_dict(self, state_dict, strict):
                if strict is not True or set(state_dict) != set(self._state):
                    raise RuntimeError("state mismatch")

            def state_dict(self):
                return self._state

        with tempfile.TemporaryDirectory() as temporary:
            factorial = Path(temporary)
            checkpoint = factorial / "checkpoints" / "full.pt"
            checkpoint.parent.mkdir()
            torch.save(
                self._formal_payload(current=True, torch_module=torch), checkpoint
            )
            row = {
                **self.current_row,
                "arm": "full",
                "path": checkpoint.relative_to(factorial).as_posix(),
                "sha256": sha256_file(checkpoint),
            }
            with patch(
                "formal_v2.formal_model.CSIPairsFormalModel", FakeFormalModel
            ):
                loaded = _load_control_full_checkpoint_payloads(
                    {"seeds": [20270001]},
                    factorial,
                    {"checkpoints": [row]},
                    self.current_evidence,
                    _digest("teacher"),
                    legacy_runtime=False,
                )
        self.assertEqual(tuple(loaded), (20270001,))
        self.assertEqual(loaded[20270001][1]["seed"], 20270001)

    def test_claim_control_preflight_rejects_missing_checkpoint_provenance(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable")

        for missing in sorted(_CHECKPOINT_PROVENANCE_PAYLOAD_FIELDS):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temporary:
                factorial = Path(temporary)
                checkpoint = factorial / "full.pt"
                payload = self._formal_payload(current=True, torch_module=torch)
                payload.pop(missing)
                torch.save(payload, checkpoint)
                row = {
                    **self.current_row,
                    "arm": "full",
                    "path": checkpoint.name,
                    "sha256": sha256_file(checkpoint),
                }
                with self.assertRaisesRegex(RuntimeError, "fields are not exact"):
                    _load_control_full_checkpoint_payloads(
                        {"seeds": [20270001]},
                        factorial,
                        {"checkpoints": [row]},
                        self.current_evidence,
                        _digest("teacher"),
                        legacy_runtime=False,
                    )

    def test_claim_control_preflight_rejects_row_payload_provenance_mismatch(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable")

        with tempfile.TemporaryDirectory() as temporary:
            factorial = Path(temporary)
            checkpoint = factorial / "full.pt"
            torch.save(
                self._formal_payload(current=True, torch_module=torch), checkpoint
            )
            row = {
                **self.current_row,
                "source_tree_sha256": _digest("different-source"),
                "arm": "full",
                "path": checkpoint.name,
                "sha256": sha256_file(checkpoint),
            }
            with self.assertRaisesRegex(RuntimeError, "source_tree_sha256 provenance mismatch"):
                _load_control_full_checkpoint_payloads(
                    {"seeds": [20270001]},
                    factorial,
                    {"checkpoints": [row]},
                    self.current_evidence,
                    _digest("teacher"),
                    legacy_runtime=False,
                )

    def test_claim_control_adapters_preflight_before_expensive_work(self):
        from formal_v2.external_adapters import retention_control, shuffled_pair_control

        config = {
            "seeds": [20270001],
            "data": {
                "require_clean_csi": True,
                "minimum_repeats": 1,
                "minimum_target_cities": 1,
                "minimum_source_cities": 1,
                "minimum_banks_per_target_city": 1,
                "minimum_independent_base_map_clusters_per_target_city": 1,
                "minimum_banks_per_source_role": 1,
                "minimum_independent_source_final_unseen_clusters": 1,
                "minimum_independent_external_validation_clusters": 1,
            },
        }
        dataset = SimpleNamespace(is_fixture=False, validate=lambda **_kwargs: None)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_run = root / "output-run"
            upstream = root / "upstream"
            factorial = upstream / "factorial"
            output_run.mkdir()
            factorial.mkdir(parents=True)
            write_json(factorial / "checkpoint_index.json", {"checkpoints": []})
            teacher = root / "teacher.pt"
            teacher.write_bytes(b"teacher")
            context = root / "context.json"
            write_json(
                context,
                {
                    "schema_version": "csi-pairs-v6-claim-control-context-v1",
                    "config": config,
                    **self.current_evidence,
                },
            )
            authenticated = SimpleNamespace(migrated=True)
            qualification = {
                "teacher_checkpoint": str(teacher),
                "teacher_checkpoint_sha256": sha256_file(teacher),
            }

            cases = (
                (shuffled_pair_control, "_build_corpus", True),
                (retention_control, "_fit_probes", False),
            )
            for module, expensive_name, shuffled in cases:
                with (
                    self.subTest(module=module.__name__),
                    patch.object(module, "configure_reproducible_runtime"),
                    patch.object(module.FormalDataset, "load", return_value=dataset),
                    patch.object(
                        module, "evidence_context", return_value=self.current_evidence
                    ),
                    patch.object(
                        module,
                        "_resolve_run_roots",
                        return_value=(output_run, upstream, authenticated),
                    ),
                    patch.object(
                        module,
                        "_authenticated_qualification_gate",
                        return_value=qualification,
                    ),
                    patch.object(
                        module,
                        "_load_control_full_checkpoint_payloads",
                        side_effect=RuntimeError("checkpoint preflight failed"),
                    ) as preflight,
                    patch.object(module, "load_teacher_bundle") as load_teacher,
                    patch.object(module, expensive_name) as expensive,
                    self.assertRaisesRegex(RuntimeError, "checkpoint preflight failed"),
                ):
                    arguments = {
                        "dataset_path": root / "dataset.npz",
                        "run_root": None,
                        "output_root": output_run / "control-output",
                        "context_path": context,
                        "output_run_root": output_run,
                        "upstream_root": upstream,
                    }
                    if shuffled:
                        arguments["control_seed"] = 20270807
                    module.run_shuffled_pair_control(**arguments) if shuffled else (
                        module.run_retention_control(**arguments)
                    )
                preflight.assert_called_once()
                load_teacher.assert_not_called()
                expensive.assert_not_called()

    def test_formal_checkpoint_rejects_teacher_that_only_matches_its_index_row(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable")

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "full.pt"
            torch.save(self._formal_payload(current=True, torch_module=torch), checkpoint)
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                _validate_formal_checkpoint(
                    checkpoint,
                    self.current_row,
                    self.current_evidence,
                    teacher_checkpoint_sha256=_digest("qualification-teacher"),
                )

    def test_shuffled_checkpoint_requires_current_adapter_provenance(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable")

        class FakeFormalModel:
            def __init__(self, **_model_spec):
                self._state = {"weight": torch.ones(1)}

            def load_state_dict(self, state_dict, strict):
                if strict is not True or set(state_dict) != set(self._state):
                    raise RuntimeError("state mismatch")

        payload = self._shuffled_payload(torch)
        self.assertEqual(
            set(payload),
            set(_SHUFFLED_CHECKPOINT_BASE_FIELDS).union(
                _CHECKPOINT_PROVENANCE_PAYLOAD_FIELDS
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "shuffled.pt"
            torch.save(payload, checkpoint)
            with patch(
                "formal_v2.formal_model.CSIPairsFormalModel", FakeFormalModel
            ):
                _validate_shuffled_checkpoint(
                    checkpoint,
                    {"seed": 20270001},
                    self.current_evidence,
                    _digest("training-provenance"),
                    _digest("teacher"),
                )

    def test_local_qualification_uses_the_expected_manifested_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            qualification_root = Path(temporary) / "qualification"
            teacher = qualification_root / "checkpoints" / "teacher.pt"
            teacher.parent.mkdir(parents=True)
            teacher.write_bytes(b"local-teacher")
            gate_path = qualification_root / "gate.json"
            gate = {
                "teacher_checkpoint": str(teacher.resolve()),
                "teacher_checkpoint_sha256": sha256_file(teacher),
            }
            write_json(gate_path, gate)
            resolved = SimpleNamespace(
                qualification_gate=gate_path,
                qualification_root=qualification_root,
                migrated=False,
            )
            dataset = SimpleNamespace(is_fixture=False)
            with (
                patch(
                    "formal_v2.formal_evidence.require_formal_qualification",
                    return_value=gate,
                ) as require_qualification,
                patch(
                    "formal_v2.formal_evidence.require_stage_manifested_gate",
                    return_value=gate,
                ) as require_manifest,
            ):
                authenticated = _authenticated_qualification_gate(
                    {}, dataset, resolved
                )

            self.assertEqual(
                authenticated["teacher_checkpoint_sha256"], sha256_file(teacher)
            )
            require_qualification.assert_called_once_with(
                gate,
                {},
                dataset,
                allow_nonscientific_fixture=True,
            )
            self.assertEqual(require_manifest.call_args.args[0], gate_path)
            self.assertIs(require_manifest.call_args.args[1], gate)

    def test_accepted_migration_checkpoint_index_uses_legacy_runtime_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = FormalMigrationFixture(Path(temporary))
            config = {"seeds": [20270001, 20270002, 20270003]}
            dataset = SimpleNamespace(source_path=fixture.dataset, is_fixture=False)
            common_evidence = {
                key: fixture.factorial_evidence[key]
                for key in _CHECKPOINT_COMMON_EVIDENCE_FIELDS
            }
            with (
                patch.object(migration, "_RUNNING_SOURCE_ROOT", fixture.new_source),
                patch.object(upstream, "_RUNNING_SOURCE_ROOT", fixture.new_source),
                patch.object(upstream, "config_sha256", return_value=CONFIG_SHA256),
            ):
                request = migration.write_migration_request(
                    **fixture.request_arguments()
                )
                approval = fixture.write_migration_approval(request)
                migration.accept_migration_request(
                    request["request_path"],
                    approval,
                    **fixture.acceptance_arguments(),
                    accepted_utc=ACCEPTED_UTC,
                    now=NOW,
                )
                with patch(
                    "formal_v2.formal_claim_controls.evidence_context",
                    return_value=common_evidence,
                ):
                    checkpoints = _verify_checkpoint_index_binding(
                        config,
                        dataset,
                        fixture.new_run,
                        {
                            "checkpoint_index_sha256": sha256_file(
                                fixture.checkpoint_index
                            )
                        },
                    )

            index = read_strict_json(fixture.checkpoint_index)
            self.assertEqual(set(checkpoints), set(config["seeds"]))
            self.assertEqual(
                set(checkpoints.values()),
                {
                    row["sha256"]
                    for row in index["checkpoints"]
                    if row["arm"] == "full"
                },
            )
            loaded = _load_control_full_checkpoint_payloads(
                config,
                fixture.factorial_root,
                index,
                common_evidence,
                sha256_file(fixture.teacher),
                legacy_runtime=True,
            )
            self.assertEqual(tuple(loaded), tuple(config["seeds"]))
            self.assertTrue(
                all(payload["arm"] == "full" for _, payload in loaded.values())
            )

            row = next(
                row
                for row in index["checkpoints"]
                if row["arm"] == "full"
            )
            import torch

            payload = torch.load(
                fixture.factorial_root / row["path"],
                map_location="cpu",
                weights_only=False,
            )
            version = _validate_checkpoint_evidence(
                payload,
                base_fields=_FORMAL_CHECKPOINT_BASE_FIELDS,
                provenance_binding=row,
                evidence=common_evidence,
                label="claim-control checkpoint",
                legacy_runtime=True,
            )
            self.assertEqual(version, _CHECKPOINT_EVIDENCE_MIGRATED_LEGACY_V1)

    def test_checkpoint_index_and_rows_reject_unknown_fields(self):
        scalar_evidence = {
            key: value
            for key, value in self.current_evidence.items()
            if not isinstance(value, (dict, list))
        }
        row = {
            "seed": 20270001,
            "arm": "endpoint",
            "path": "checkpoints/endpoint.pt",
            "sha256": _digest("checkpoint"),
            "parameters": 1,
            "measured_flops_per_step": 1,
            "execution_device": "cpu",
            "teacher_checkpoint_sha256": _digest("teacher"),
            **scalar_evidence,
        }
        base = {
            "schema_version": "csi-pairs-formal-checkpoint-index-v2.1-v6",
            **self.current_evidence,
            "checkpoints": [],
        }
        config = {"seeds": [20270001, 20270002, 20270003]}
        dataset = SimpleNamespace(is_fixture=False)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint_index.json"
            resolved = SimpleNamespace(checkpoint_index=path, migrated=False)
            for label, index, message in (
                ("index", {**base, "unknown": True}, "index fields"),
                (
                    "row",
                    {**base, "checkpoints": [{**row, "unknown": True}]},
                    "row fields",
                ),
            ):
                with self.subTest(label=label):
                    write_json(path, index)
                    with (
                        patch(
                            "formal_v2.formal_upstream.resolve_authenticated_upstream",
                            return_value=resolved,
                        ),
                        patch(
                            "formal_v2.formal_claim_controls.evidence_context",
                            return_value=self.current_evidence,
                        ),
                        self.assertRaisesRegex(RuntimeError, message),
                    ):
                        _verify_checkpoint_index_binding(
                            config,
                            dataset,
                            temporary,
                            {"checkpoint_index_sha256": sha256_file(path)},
                        )

    def test_local_checkpoint_index_rejects_teacher_outside_qualification(self):
        scalar_evidence = {
            key: value
            for key, value in self.current_evidence.items()
            if not isinstance(value, (dict, list))
        }
        config = {"seeds": [20270001, 20270002, 20270003]}
        rows = []
        for seed in config["seeds"]:
            for arm in ARMS:
                rows.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "path": f"checkpoints/{seed}_{arm}.pt",
                        "sha256": _digest(f"checkpoint-{seed}-{arm}"),
                        "parameters": 1,
                        "measured_flops_per_step": 1,
                        "execution_device": "cpu",
                        "teacher_checkpoint_sha256": _digest("teacher"),
                        **scalar_evidence,
                    }
                )
        rows[0]["teacher_checkpoint_sha256"] = _digest("other-teacher")
        index = {
            "schema_version": "csi-pairs-formal-checkpoint-index-v2.1-v6",
            **self.current_evidence,
            "checkpoints": rows,
        }
        dataset = SimpleNamespace(is_fixture=False)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint_index.json"
            write_json(path, index)
            resolved = SimpleNamespace(checkpoint_index=path, migrated=False)
            with (
                patch(
                    "formal_v2.formal_upstream.resolve_authenticated_upstream",
                    return_value=resolved,
                ),
                patch(
                    "formal_v2.formal_claim_controls.evidence_context",
                    return_value=self.current_evidence,
                ),
                patch(
                    "formal_v2.formal_claim_controls._authenticated_qualification_gate",
                    return_value={
                        "teacher_checkpoint_sha256": _digest("teacher")
                    },
                ),
                self.assertRaisesRegex(RuntimeError, "teacher hash mismatch"),
            ):
                _verify_checkpoint_index_binding(
                    config,
                    dataset,
                    temporary,
                    {"checkpoint_index_sha256": sha256_file(path)},
                )

    def _formal_payload(self, *, current: bool, torch_module=None):
        state_value = 1 if torch_module is None else torch_module.ones(1)
        payload = {
            "schema_version": "csi-pairs-formal-checkpoint-v2.1-v6",
            "arm": "full",
            "seed": 20270001,
            "model_spec": {},
            "normalization": {},
            "teacher_checkpoint_sha256": _digest("teacher"),
            "checkpoint_rule": "fixed_final_step_no_target_selection",
            "state_dict": {"weight": state_value},
            **self.common_evidence,
        }
        if current:
            payload.update(
                {
                    "source_tree_sha256": self.source_sha256,
                    "requirements_lock_sha256": self.requirements_sha256,
                    "runtime_provenance_sha256": self.runtime_sha256,
                    "runtime_provenance": copy.deepcopy(self.runtime),
                }
            )
        return payload

    def _shuffled_payload(self, torch_module):
        return {
            **self._formal_payload(current=True, torch_module=torch_module),
            "schema_version": "csi-pairs-v6-shuffled-formal-checkpoint-v1",
            "pairing_breaks": [
                "alignment_h_map_edge",
                "response_action_target",
            ],
            "training_provenance_sha256": _digest("training-provenance"),
        }

    def _runtime_record(self):
        runtime = {key: None for key in RUNTIME_PROVENANCE_FIELDS}
        runtime.update(
            {
                "schema_version": RUNTIME_PROVENANCE_SCHEMA,
                "source_tree_sha256": self.source_sha256,
                "requirements_lock_sha256": self.requirements_sha256,
                "installer_report_path": "/runtime/install-report.json",
                "installer_report_sha256": _digest("install-report"),
                "reviewed_wheelhouse_path": "/runtime/wheels",
                "reviewed_wheelhouse_sha256": _digest("wheelhouse"),
                "reviewed_wheel_manifest_path": "/runtime/wheel-manifest.json",
                "reviewed_wheel_manifest_sha256": _digest("wheel-manifest"),
                "python_version": "3.12.13",
                "python_implementation": "CPython",
                "python_executable": "/runtime/bin/python3.12",
                "python_executable_name": "python3.12",
                "python_prefix": "/runtime",
                "python_dont_write_bytecode": True,
                "platform_system": "Linux",
                "platform_release": "test",
                "platform_machine": "x86_64",
                "platform_mac_version": None,
                "platform_libc_name": "glibc",
                "platform_libc_version": "2.31",
                "cublas_workspace_config": ":4096:8",
                "torch": self._torch_record(),
                "installed_distributions": {
                    "torch": {
                        "version": "2.5.1+cu121",
                        "record_sha256": _digest("torch-record"),
                        "wheel_sha256": _digest("torch-wheel"),
                    }
                },
            }
        )
        return runtime

    @staticmethod
    def _torch_record():
        record = {key: None for key in TORCH_RUNTIME_FIELDS}
        record.update(
            {
                "version": "2.5.1+cu121",
                "cuda_version": "12.1",
                "cuda_available": True,
                "gpu_names": ["NVIDIA A100-SXM4-40GB"],
                "deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cudnn_allow_tf32": False,
                "cuda_matmul_allow_tf32": False,
                "float32_matmul_precision": "highest",
            }
        )
        return record


if __name__ == "__main__":
    unittest.main()
