from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from formal_v2.external_adapters import (
    controlled_map_adapter,
    pmnet_adapter,
    resource_control,
    retention_control,
    scene_id_sigmap,
    shuffled_pair_control,
    wigatr_adapter,
)
from formal_v2.external_adapters.scene_id_sigmap import _validate_checkpoint
from formal_v2.formal_claim_controls import _require_explicit_root_bindings as claim_roots
from formal_v2.formal_controls import (
    _require_explicit_root_bindings as resource_roots,
    _validate_manifest as validate_resource_manifest,
)
from formal_v2.formal_external import (
    _resolve_adapter_command,
    _validate_manifest as validate_external_manifest,
)
from formal_v2.formal_io import sha256_file
from formal_v2.formal_scene_id import _require_explicit_root_bindings as scene_roots


ROOT = Path(__file__).resolve().parents[2]


class AdapterRootSeparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        base = Path(self.temporary.name)
        self.output_run = (base / "new-run").resolve()
        self.upstream = (base / "legacy-run").resolve()
        self.output = self.output_run / "stage" / "adapter"
        self.output.mkdir(parents=True)
        self.upstream.mkdir()
        self.authenticated = SimpleNamespace(
            run_root=self.output_run,
            migrated=True,
            migration=SimpleNamespace(legacy_run_root=self.upstream),
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_all_adapter_root_resolvers_accept_only_the_authenticated_pair(self):
        direct_resolvers = (
            retention_control._resolve_run_roots,
            shuffled_pair_control._resolve_run_roots,
            resource_control._resolve_run_roots,
            scene_id_sigmap._resolve_run_roots,
        )
        with patch(
            "formal_v2.formal_upstream.resolve_authenticated_upstream",
            return_value=self.authenticated,
        ):
            for resolver in direct_resolvers:
                with self.subTest(resolver=resolver.__module__):
                    resolved = resolver(
                        object(),
                        {},
                        self.output,
                        output_run_root=self.output_run,
                        upstream_root=self.upstream,
                    )
                    self.assertEqual(resolved[:2], (self.output_run, self.upstream))
                    with self.assertRaisesRegex(RuntimeError, "binding mismatch"):
                        resolver(
                            object(),
                            {},
                            self.output,
                            output_run_root=self.output_run,
                            upstream_root=self.output_run,
                        )

            adapter_resolvers = (
                controlled_map_adapter._resolve_run_roots,
                pmnet_adapter._resolve_run_roots,
                wigatr_adapter._resolve_run_roots,
            )
            for resolver in adapter_resolvers:
                with self.subTest(resolver=resolver.__module__):
                    args = SimpleNamespace(
                        run_root=None,
                        output_run_root=str(self.output_run),
                        upstream_root=str(self.upstream),
                    )
                    resolved = resolver(args, object(), {}, self.output)
                    self.assertEqual(resolved[:2], (self.output_run, self.upstream))
                    args.upstream_root = args.output_run_root
                    with self.assertRaisesRegex(RuntimeError, "binding mismatch"):
                        resolver(args, object(), {}, self.output)

    def test_legacy_alias_is_rejected_when_a_migration_directory_exists(self):
        (self.output_run / "migration").mkdir()
        with self.assertRaisesRegex(RuntimeError, "requires explicit"):
            retention_control._resolve_run_roots(
                object(), {}, self.output, run_root=self.output_run
            )

    def test_manifests_bind_distinct_roots_and_current_sources(self):
        external_path = (
            ROOT / "formal_v2/external_adapters/all_map_adapters_v1.json"
        )
        external = json.loads(external_path.read_text(encoding="utf-8"))
        validate_external_manifest(external)
        for adapter in external["adapters"]:
            self.assertIn("--output-run-root", adapter["command"])
            self.assertIn("--upstream-root", adapter["command"])
            self.assertNotIn("--run-root", adapter["command"])

        resource_path = (
            ROOT / "formal_v2/external_adapters/resource_controls_v3.json"
        )
        resource = json.loads(resource_path.read_text(encoding="utf-8"))
        validate_resource_manifest(resource, resource_path.parent)

        claim_roots(
            [
                "{python}",
                "{adapter_source}",
                "--output-run-root",
                "{output_run_root}",
                "--upstream-root",
                "{upstream_root}",
            ]
        )
        resource_roots(resource["controls"][0]["command"])
        scene_roots(
            [
                "{python}",
                "{adapter_source}",
                "--output-run-root",
                "{output_run_root}",
                "--upstream-root",
                "{upstream_root}",
            ]
        )

    def test_external_command_keeps_output_and_upstream_roots_distinct(self):
        manifest = json.loads(
            (
                ROOT / "formal_v2/external_adapters/all_map_adapters_v1.json"
            ).read_text(encoding="utf-8")
        )
        _digest, command = _resolve_adapter_command(
            manifest["adapters"][0],
            dataset_path=ROOT / "dataset.npz",
            output_path=self.output,
            output_run_root=self.output_run,
            upstream_root=self.upstream,
        )
        self.assertEqual(
            Path(command[command.index("--output-run-root") + 1]),
            self.output_run,
        )
        self.assertEqual(
            Path(command[command.index("--upstream-root") + 1]),
            self.upstream,
        )

    def test_scene_id_accepts_the_checkpoint_version_the_sigmap_trainer_writes(self):
        dataset_path = Path(self.temporary.name) / "dataset.npz"
        dataset_path.write_bytes(b"dataset")
        dataset = SimpleNamespace(source_path=dataset_path)
        execution = {
            "source_revision": "frozen-revision",
            "checkpoint_sha256": "a" * 64,
        }
        payload = {
            "schema_version": "csi-pairs-v6-controlled-map-checkpoint-v2",
            "model_name": "SigMap",
            "method": "sigmap",
            "implementation_status": "style-controlled-implementation",
            "source_revision": execution["source_revision"],
            "dataset_sha256": sha256_file(dataset_path),
            "train_role": "source_encoder_train",
            "selection_role": "source_method_selection",
            "target_roles_read": [],
            "selected_step": 1,
            "selection_loss": 0.1,
            "normalizer": {},
            "model_metadata": {},
            "effective_batch_size": 8,
            "microbatch_size": 2,
            "gradient_accumulation_steps": 4,
            "configured_precision": "float32",
            "executed_precision": "float32",
            "autocast_enabled": False,
            "state_dict": {"weight": object()},
        }
        _validate_checkpoint(payload, dataset, execution)
        payload["schema_version"] = "csi-pairs-v6-controlled-map-checkpoint-v1"
        with self.assertRaisesRegex(RuntimeError, "source-only contract"):
            _validate_checkpoint(payload, dataset, execution)


if __name__ == "__main__":
    unittest.main()
