from __future__ import annotations

import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile
import unittest
import zipfile


SERVER_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_CONTROL = SERVER_ROOT / "artifacts" / "a100_dataset_package_v6"
POLICY_PATH = PACKAGE_CONTROL / "PACKAGE_POLICY.json"
REGISTRY_PATH = PACKAGE_CONTROL / "external_dataset_registry/EXTERNAL_DATASET_REGISTRY.json"
CONFIG_PATH = SERVER_ROOT / "formal_v2/configs/a100_dataset_suite_v6.json"
BUILDER_PATH = PACKAGE_CONTROL / "build_a100_dataset_package_v6.py"
SPLIT_LEDGER_PATH = PACKAGE_CONTROL / "FORMAL_MAIN_SPLIT_LEDGER.json"
PACKAGE_ID = "CSI-PAIRS-A100-DATASETS-v2"


def load_builder():
    name = "a100_dataset_builder"
    spec = importlib.util.spec_from_file_location(name, BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load A100 dataset builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class A100DatasetPackageV6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        cls.registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.split_ledger = json.loads(SPLIT_LEDGER_PATH.read_text(encoding="utf-8"))
        cls.builder = load_builder()

    def test_binary_payload_stays_out_of_git_and_formal_use_is_blocked(self):
        archive_policy = self.policy["archive_policy"]
        self.assertFalse(archive_policy["commit_binary_zip_to_git"])
        self.assertFalse(archive_policy["commit_dataset_payloads_to_git"])
        readiness = self.policy["readiness"]
        self.assertEqual(readiness["FORMAL_INPUT_READY"], "BLOCKED")
        self.assertEqual(readiness["FORMAL_TRAINING_READY"], "NO")
        self.assertEqual(readiness["LAUNCH_READY"], "BLOCKED")
        self.assertEqual(readiness["SCIENTIFIC_EVIDENCE"], "NOT_ASSESSED")

    def test_package_id_external_counts_and_required_components_are_frozen(self):
        self.assertEqual(self.config["package_id"], PACKAGE_ID)
        external = self.config["external_payload"]
        self.assertEqual(external["expected_file_count"], 133)
        self.assertEqual(sum(group["expected_file_count"] for group in external["expected_groups"]), 133)
        components = {row["component_id"]: row for row in self.config["workspace_components"]}
        self.assertEqual(
            set(components),
            {
                "primary_raw_inputs",
                "cpu_llvm22_34bank_candidate",
                "cpu_same_engine_verification_evidence",
                "a100_sionna_fixture",
            },
        )
        self.assertEqual(components["cpu_llvm22_34bank_candidate"]["scientific_use"], "CANDIDATE_NOT_CLAIM")
        self.assertEqual(components["a100_sionna_fixture"]["scientific_use"], "FORBIDDEN")

    def test_main_shape_and_two_target_cities_are_frozen(self):
        contract = self.policy["formal_main_contract"]
        self.assertEqual(contract["dimensions"]["banks"], 34)
        self.assertEqual(contract["dimensions"]["worlds_per_bank"], 4)
        self.assertEqual(contract["dimensions"]["positions_per_bank"], 256)
        self.assertEqual(contract["city_banks"]["target-boston"], 8)
        self.assertEqual(contract["city_banks"]["target-seattle"], 8)
        self.assertEqual(sum(contract["city_banks"].values()), 34)

    def test_formal_split_ledger_separates_training_selection_calibration_and_test(self):
        ledger = self.split_ledger
        self.assertEqual(
            ledger["storage_policy"],
            "ONE_DATASET_WITH_ROLE_INDEXES_NO_PAYLOAD_DUPLICATION",
        )
        rows = self.builder.expected_split_assignments(ledger)
        self.assertEqual(len(rows), 34)
        self.assertEqual(len({row["scene_id"] for row in rows}), 34)
        self.assertEqual(len({row["base_map_cluster_id"] for row in rows}), 34)

        role_counts = {}
        for row in rows:
            role = row["role"]
            role_counts[role] = role_counts.get(role, 0) + 1
        for role in self.builder.SOURCE_ROLES:
            self.assertEqual(role_counts[role], 2)
        self.assertEqual(role_counts["target"], 16)
        self.assertEqual(role_counts["external_validation"], 4)

        target = ledger["target"]
        self.assertEqual(target["position_partition"]["support_pool"]["positions_per_bank"], 128)
        self.assertEqual(target["position_partition"]["query"]["positions_per_bank"], 128)
        self.assertEqual(target["label_budgets_unique_positions"], [0, 8, 32, 128])
        final_role = ledger["source"]["bank_index_to_role"][-1]
        self.assertEqual(final_role["role"], "source_final_unseen_bank")
        self.assertIn("any training", final_role["forbidden"])

    def test_public_substitutions_are_explicit_and_not_formal_main_data(self):
        self.assertTrue(self.registry["training_core_complete"])
        self.assertFalse(self.registry["all_original_sources_complete"])
        self.assertTrue(self.registry["public_substitutions_used"])
        self.assertFalse(self.registry["formal_sibling_world_compatible"])
        self.assertEqual(self.registry["silent_stage0_use"], "FORBIDDEN")
        datasets = {row["id"]: row for row in self.registry["datasets"]}
        replacement = datasets["deepsense_for_unreleased_wwm"]
        self.assertEqual(replacement["actual_included_source"], "DeepSense6G")
        self.assertIn("WWM_reproduction", replacement["forbidden_claims"])
        self.assertFalse(replacement["formal_main_data"])
        self.assertIn("IRT2HighRes", datasets["radiomapseer_public_core"]["missing_originals"])

    def test_exclusion_and_safe_path_helpers_are_fail_closed(self):
        global_exclusions = set(self.config["required_exclusions"])
        exact = {"live_regeneration/data_verification/regenerated.npz"}
        for relative in (
            "DeepMIMO/.venv/bin/python",
            "DeepMIMO/repository/.git/objects/data",
            "UrbanMIMOMap/partial_downloads/a.partial",
            "DeepMIMO/__pycache__/x.pyc",
            "live_regeneration/data_verification/regenerated.npz",
        ):
            with self.subTest(relative=relative):
                self.assertTrue(self.builder.excluded(PurePosixPath(relative), exact, global_exclusions))
        self.assertFalse(
            self.builder.excluded(PurePosixPath("RadioMapSeer/RadioMapSeer.zip"), exact, global_exclusions)
        )
        with self.assertRaises(self.builder.BuildError):
            self.builder.safe_relative("../escape", "test path")

    def test_zip64_writer_uses_fixed_stored_bytes_and_manifest_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "payload.bin"
            source.write_bytes(b"a100-package-smoke")
            entry = self.builder.entry_for(source, "data/payload.bin", "smoke")
            archive_path = root / "smoke.zip"
            with zipfile.ZipFile(archive_path, "w", allowZip64=True) as archive:
                self.builder.write_file(archive, PACKAGE_ID, entry)
            with zipfile.ZipFile(archive_path) as archive:
                member = archive.getinfo(f"{PACKAGE_ID}/data/payload.bin")
                self.assertEqual(member.compress_type, zipfile.ZIP_STORED)
                self.assertEqual(member.date_time, self.builder.ZIP_TIMESTAMP)
                self.assertEqual(archive.read(member), b"a100-package-smoke")

        manifest = json.loads(
            self.builder.manifest_for(
                {
                    "package_id": PACKAGE_ID,
                    "archive_root": PACKAGE_ID,
                    "target_platform": "Linux x86_64 with NVIDIA A100 GPUs",
                    "fixed_zip_timestamp": "2026-08-10T00:00:00Z",
                },
                [entry],
            )
        )
        self.assertEqual(manifest["statuses"]["FORMAL_TRAINING_READY"], "NO")
        self.assertFalse(manifest["statuses"]["ALL_ORIGINAL_SOURCES_COMPLETE"])

    def test_destination_script_preserves_linux_preapproval_gate(self):
        script = (PACKAGE_CONTROL / "scripts/A100_REGENERATE_AND_VERIFY.sh").read_text(encoding="utf-8")
        self.assertIn("no pre-approved Linux x86_64 libLLVM", script)
        self.assertIn("FORMAL_TRAINING_READY=NO", script)
        self.assertIn("two visible A100 GPUs are required", script)

    def test_server_and_anonymous_release_boundaries_are_explicit(self):
        server_builder = (
            SERVER_ROOT / "formal_v2/scripts/build_server_bundle.sh"
        ).read_text(encoding="utf-8")
        anonymous_builder = (
            SERVER_ROOT / "formal_v2/scripts/build_anonymous_supplement.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("artifacts/a100_dataset_package_v6", server_builder)
        self.assertIn("artifacts/dataset_suite_v6/ROLE_ASSIGNMENTS.csv", server_builder)
        self.assertIn("artifacts/dataset_suite_v6/DATA_AVAILABILITY.md", server_builder)
        self.assertIn(
            "artifacts/dataset_suite_v6/external_wireless_metadata/SHA256SUMS",
            server_builder,
        )
        self.assertIn(
            "--exclude='formal_v2/tests/test_a100_dataset_package_v6.py'",
            anonymous_builder,
        )
        self.assertIn(
            "--exclude='formal_v2/configs/a100_dataset_suite_v6.json'",
            anonymous_builder,
        )
        self.assertIn(
            "--exclude='formal_v2/external_data_bundle'",
            anonymous_builder,
        )


if __name__ == "__main__":
    unittest.main()
