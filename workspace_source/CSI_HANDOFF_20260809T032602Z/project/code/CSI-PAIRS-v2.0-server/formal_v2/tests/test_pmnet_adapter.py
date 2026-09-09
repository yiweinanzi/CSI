from __future__ import annotations

import json
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from formal_v2.external_adapters.pmnet_adapter import (
    PMNET_SOURCE_REVISION,
    _PMNetRadiomapDataset,
    _build_execution_manifest,
    _build_official_model,
    _evaluate_six_conditions,
    _fit_source_power_normalization,
    _fit_source_only_model,
    _inverse_localize_radiomap,
    _masked_mse,
    _pmnet_input,
    _radiomap_samples,
    _radiomap_target,
    _require_formal_resources,
    _verify_vendor_tree,
    load_pmnet_config,
)
from formal_v2.formal_dataset import FormalDataset, FormalDatasetError
from formal_v2.formal_external import (
    _validate_execution_manifest,
    _validate_manifest as validate_external_manifest,
)
from formal_v2.formal_fixture import write_nonscientific_fixture
from formal_v2.formal_io import parse_strict_json, sha256_file
from formal_v2.formal_resources import validate_resource_registry_structure


ROOT = Path(__file__).resolve().parents[2]


class PMNetAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "fixture.npz"
        write_nonscientific_fixture(self.path)
        self.dataset = FormalDataset.load(self.path)
        self.config = load_pmnet_config(
            ROOT / "formal_v2/configs/pmnet_official_v1.json"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_official_source_architecture_and_forward_are_frozen(self):
        self.assertEqual(
            PMNET_SOURCE_REVISION,
            "a0e0c5926de721074beeb23f630f2d313f6508dd",
        )
        vendor = ROOT / "formal_v2/external_adapters/vendor/PMNet"
        _verify_vendor_tree(vendor)
        self.assertEqual(
            sha256_file(vendor / "models/pmnet_v3.py"),
            "5ea7c5b4f0de14504c8e673a820a80452dd103f8c270a1b5d9533306d3d28c62",
        )
        self.assertEqual(self.config["model"]["n_blocks"], [3, 3, 27, 3])
        self.assertEqual(self.config["model"]["atrous_rates"], [6, 12, 18])
        self.assertEqual(self.config["model"]["multi_grids"], [1, 2, 4])
        self.assertEqual(self.config["model"]["output_stride"], 8)

        model = _build_official_model(self.config, material_category_count=3)
        model.eval()
        with torch.no_grad():
            prediction = model(torch.zeros(1, 6, 32, 32))
        self.assertEqual(tuple(prediction.shape), (1, 1, 32, 32))
        self.assertEqual(model.layer1.conv1.conv.in_channels, 6)
        self.assertEqual(model.layer1.pool.kernel_size, 2)
        self.assertEqual(model.layer1.pool.stride, 2)

    def test_input_contains_physical_map_channels_and_transmitter_only(self):
        scene = int(self.dataset.indices_for_role("source_encoder_train")[0])
        supplied = self.dataset.maps[scene, 0]
        model_input = _pmnet_input(self.dataset, scene, supplied)
        material_count = int(
            self.dataset.metadata["assets"]["material_category_count"]
        )
        self.assertEqual(
            model_input.shape[0],
            2 + material_count + 1,
        )
        self.assertTrue(np.array_equal(model_input[0], supplied[0]))
        self.assertEqual(float(np.sum(model_input[-1])), 1.0)

        changed = supplied.copy()
        changed[1, 0, 0] += 1.0
        changed_input = _pmnet_input(self.dataset, scene, changed)
        self.assertFalse(np.array_equal(model_input, changed_input))

    def test_radiomap_loss_reads_only_observed_source_positions(self):
        mean, scale = _fit_source_power_normalization(self.dataset, 1e-12)
        scene = int(self.dataset.indices_for_role("source_encoder_train")[0])
        target, mask = _radiomap_target(
            self.dataset,
            scene,
            world=0,
            power_mean=mean,
            power_scale=scale,
            power_floor=1e-12,
        )
        self.assertGreater(int(mask.sum()), 0)
        self.assertLess(int(mask.sum()), int(mask.size))
        prediction = torch.as_tensor(target[None, None], dtype=torch.float32)
        tensor_target = torch.as_tensor(target[None, None], dtype=torch.float32)
        tensor_mask = torch.as_tensor(mask[None, None])
        baseline = _masked_mse(prediction, tensor_target, tensor_mask)
        changed = prediction.clone()
        changed[~tensor_mask] = 1_000_000.0
        self.assertEqual(float(baseline), 0.0)
        self.assertEqual(float(_masked_mse(changed, tensor_target, tensor_mask)), 0.0)

    def test_fixture_batch_runs_official_model_backward(self):
        mean, scale = _fit_source_power_normalization(self.dataset, 1e-12)
        samples = _PMNetRadiomapDataset(
            self.dataset,
            "source_encoder_train",
            mean,
            scale,
            1e-12,
        )
        first = samples[0]
        second = samples[1]
        model_input = torch.stack((first[0], second[0]))
        target = torch.stack((first[1], second[1]))
        mask = torch.stack((first[2], second[2]))
        model = _build_official_model(
            self.config,
            int(self.dataset.metadata["assets"]["material_category_count"]),
        )
        loss = _masked_mse(model(model_input), target, mask)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            any(value is not None and torch.any(value != 0) for value in gradients)
        )

    def test_training_and_selection_samples_follow_the_source_ledger(self):
        train = _radiomap_samples(self.dataset, "source_encoder_train")
        selection = _radiomap_samples(self.dataset, "source_method_selection")
        self.assertTrue(train)
        self.assertTrue(selection)
        self.assertTrue(
            all(
                str(self.dataset.scene_roles[scene]) == "source_encoder_train"
                for scene, _world in train
            )
        )
        self.assertTrue(
            all(
                str(self.dataset.scene_roles[scene]) == "source_method_selection"
                for scene, _world in selection
            )
        )

    def test_dataset_rejects_receiver_coordinates_outside_the_map_frame(self):
        self.dataset.positions[0, 0] = np.asarray([1000.0, 1000.0])
        with self.assertRaisesRegex(FormalDatasetError, "outside the frozen map extent"):
            self.dataset.validate()

    def test_inverse_localization_uses_power_and_rejects_occupied_cells(self):
        radiomap = np.asarray(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0, 15.0],
            ]
        )
        supplied_map = np.zeros((3, 4, 4), dtype=np.float64)
        supplied_map[0, 1, 1] = 1.0
        predicted = _inverse_localize_radiomap(
            radiomap,
            supplied_map,
            observed_power=6.0,
            origin_xy_m=(-2.0, -2.0),
            resolution_m=1.0,
            occupancy_threshold=0.5,
        )
        self.assertTrue(np.array_equal(predicted, np.asarray([0.5, -0.5])))

    def test_registry_exposes_pmnet_as_the_second_c1_model(self):
        manifest = parse_strict_json(
            (ROOT / "formal_v2/external_adapters/all_map_adapters_v1.json").read_text()
        )
        validate_external_manifest(manifest)
        self.assertEqual(
            [row["model_name"] for row in manifest["adapters"] if row["c1_eligible"]],
            ["Wi-GATr", "PMNet"],
        )
        pmnet = next(row for row in manifest["adapters"] if row["model_name"] == "PMNet")
        self.assertEqual(pmnet["implementation_status"], "official-code-adaptation")
        self.assertEqual(pmnet["source_revision"], PMNET_SOURCE_REVISION)

        resources = validate_resource_registry_structure(
            json.loads(
                (ROOT / "formal_v2/configs/waibu_resources_v1.json").read_text()
            )
        )
        source = next(row for row in resources if row["file"] == "PMNet-a0e0c592.zip")
        self.assertEqual(source["license_url"], "https://opensource.org/license/mit")
        self.assertTrue(source["redistribution_allowed"])

    def test_formal_resource_preflight_requires_cuda_and_frozen_memory(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "requires an NVIDIA CUDA"):
                _require_formal_resources(self.config)

    def test_microbatched_training_checkpoint_six_conditions_and_manifest(self):
        class TinyRadiomap(torch.nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.projection = torch.nn.Conv2d(channels, 1, kernel_size=1)

            def forward(self, values):
                return self.projection(values)

        config = copy.deepcopy(self.config)
        config["training"].update(
            {
                "epochs": 1,
                "batch_size": 2,
                "microbatch_size": 1,
                "lr_decay_every_epochs": 1,
            }
        )
        mean, scale = _fit_source_power_normalization(self.dataset, 1e-12)
        channels = 2 + int(
            self.dataset.metadata["assets"]["material_category_count"]
        ) + 1
        model = TinyRadiomap(channels)
        output = Path(self.temporary.name) / "pmnet-smoke"
        output.mkdir()
        checkpoint, training = _fit_source_only_model(
            model, self.dataset, config, output, mean, scale
        )
        self.assertTrue(checkpoint.is_file())
        self.assertEqual(training["microbatch_size"], 1)

        routes = {}
        for scene in range(self.dataset.scene_count):
            per_source = {}
            for edge in self.dataset.directed_edges(scene):
                per_source.setdefault(edge.source_world, []).append(edge.target_world)
            for source_world, targets in per_source.items():
                for position in range(self.dataset.position_count):
                    for index, target_world in enumerate(sorted(targets)):
                        routes[(scene, source_world, target_world, position)] = (
                            2 if index == 0 else 0
                        )
        rows = _evaluate_six_conditions(
            model,
            self.dataset,
            SimpleNamespace(alignment_route=routes),
            config,
            mean,
            scale,
        )
        self.assertTrue(rows)
        self.assertEqual({row["condition"] for row in rows}, {
            "correct",
            "paired_active_alternative",
            "paired_null_alternative",
            "wrong_city",
            "geometry_destroyed",
            "empty",
        })

        config_path = output / "adapter_config.json"
        training_path = output / "training_record.json"
        result_path = output / "six_condition_results.csv"
        runtime_path = output / "runtime_provenance.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        training_path.write_text(json.dumps(training), encoding="utf-8")
        result_path.write_text("result\n1\n", encoding="utf-8")
        runtime_path.write_text("{}\n", encoding="utf-8")
        command = ["{python}", "adapter.py"]
        command_sha256 = hashlib.sha256(
            json.dumps(command, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        execution = _build_execution_manifest(
            self.dataset,
            config_path,
            training_path,
            checkpoint,
            result_path,
            command_sha256,
            runtime_path,
        )
        self.assertEqual(
            execution["schema_version"],
            "csi-pairs-v6-external-execution-v4",
        )
        self.assertEqual(execution["checkpoint_sha256"], sha256_file(checkpoint))
        (output / "execution_manifest.json").write_text(
            json.dumps(execution, sort_keys=True), encoding="utf-8"
        )
        adapter = {
            "adapter_id": "pmnet-official-csi-pairs-v1",
            "model_name": "PMNet",
            "implementation_status": "official-code-adaptation",
            "source_revision": PMNET_SOURCE_REVISION,
            "command": command,
            "adapter_config_sha256": sha256_file(config_path),
        }
        with (
            mock.patch(
                "formal_v2.formal_evidence.validate_runtime_provenance",
                side_effect=lambda value: value,
            ),
            mock.patch(
                "formal_v2.formal_evidence.runtime_provenance",
                return_value={},
            ),
        ):
            _validate_execution_manifest(
                adapter, output, result_path, self.dataset
            )

            runtime_path.write_text('{"changed":true}\n', encoding="utf-8")
            execution["runtime_provenance_sha256"] = sha256_file(runtime_path)
            (output / "execution_manifest.json").write_text(
                json.dumps(execution, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "independently probed"):
                _validate_execution_manifest(
                    adapter, output, result_path, self.dataset
                )


if __name__ == "__main__":
    unittest.main()
