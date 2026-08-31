from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from artifacts.formal_readiness.tools import freeze_v5_wideband_train32_protocol as freeze
from formal_v2 import sionna_osm_candidate as candidate
from formal_v2.sionna_osm_candidate import (
    build_engine_config,
    expected_scene_ledger,
    load_config,
)
from formal_v2.formal_io import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = PROJECT_ROOT / "formal_v2/configs/sionna_osm_formal_candidate_v5.json"
EXCLUSIONS = [
    {
        "city_id": "source-chicago",
        "exclusion_manifest_sha256": "1" * 64,
        "excluded_bs_utm_xy_m": [[446326.778396, 4637429.248346]],
    },
    {
        "city_id": "source-austin",
        "exclusion_manifest_sha256": "2" * 64,
        "excluded_bs_utm_xy_m": [[621596.537389, 3347031.605659]],
    },
]


def test_protocol_derivation_changes_only_bandwidth_and_training_coverage() -> None:
    base = load_config(BASE_CONFIG)
    derived = freeze.derive_protocol(base, EXCLUSIONS)

    assert len(expected_scene_ledger(derived)) == 227
    assert derived["schema_version"] == "csi-pairs-sionna-osm-candidate-config-v6"
    assert derived["radio"]["subcarrier_spacing_hz"] == 1_000_000.0
    for before, after in zip(base["cities"], derived["cities"], strict=True):
        if before["split_group"] == "source":
            assert after["bank_count"] == before["bank_count"] + 12
            assert after["source_role_bank_counts"]["source_encoder_train"] == 16
            unchanged_before = dict(before["source_role_bank_counts"])
            unchanged_after = dict(after["source_role_bank_counts"])
            unchanged_before.pop("source_encoder_train")
            unchanged_after.pop("source_encoder_train")
            assert unchanged_after == unchanged_before
        else:
            assert after == before

    normalized_base = json.loads(json.dumps(base))
    normalized_base["schema_version"] = "csi-pairs-sionna-osm-candidate-config-v6"
    normalized_base["radio"]["subcarrier_spacing_hz"] = 1_000_000.0
    for city in normalized_base["cities"]:
        if city["split_group"] == "source":
            city["bank_count"] += 12
            city["source_role_bank_counts"]["source_encoder_train"] = 16
    normalized_base["post_rt_exclusions"] = {
        "schema_version": "csi-pairs-v5-post-rt-candidate-exclusions-config-v1",
        "selection_policy": "exclude-bound-method-selection-rt-failed-bs-centers-v1",
        "failed_gate": "minimum_active_branching_fraction",
        "threshold": 0.8,
        "cities": sorted(EXCLUSIONS, key=lambda row: row["city_id"]),
    }
    assert derived == normalized_base
    assert freeze._diagnostic_projection(derived)["schema_version"].endswith("v5")
    assert "post_rt_exclusions" not in freeze._diagnostic_projection(derived)


def test_protocol_derivation_rejects_an_already_modified_base() -> None:
    base = load_config(BASE_CONFIG)
    base["radio"]["subcarrier_spacing_hz"] = 1_000_000.0
    with pytest.raises(ValueError, match="original 16 x 30 kHz grid"):
        freeze.derive_protocol(base, EXCLUSIONS)


def test_v6_asset_selector_excludes_bound_centers_before_bank_assignment() -> None:
    config = freeze.derive_protocol(load_config(BASE_CONFIG), EXCLUSIONS)
    candidates = [
        (1.0, 1.0, 446000.0, 4637000.0, []),
        (1.0, 1.0, 446326.778396, 4637429.248346, []),
        (1.0, 1.0, 447000.0, 4638000.0, []),
    ]

    filtered, kept_ranks, excluded, binding = (
        candidate._apply_post_rt_candidate_exclusions(
            config, "source-chicago", candidates
        )
    )

    assert filtered == [candidates[0], candidates[2]]
    assert kept_ranks == [0, 2]
    assert excluded == [
        {
            "candidate_rank": 1,
            "bs_utm_xy_m": [446326.778396, 4637429.248346],
            "reason": "bound_post_rt_physical_qualification_failure",
        }
    ]
    assert binding["exclusion_manifest_sha256"] == "1" * 64


def test_v6_asset_selector_rejects_a_stale_exclusion_center() -> None:
    config = freeze.derive_protocol(load_config(BASE_CONFIG), EXCLUSIONS)
    with pytest.raises(RuntimeError, match="centers are absent"):
        candidate._apply_post_rt_candidate_exclusions(
            config,
            "source-chicago",
            [(1.0, 1.0, 446000.0, 4637000.0, [])],
        )


def test_v6_engine_profile_and_exclusion_provenance_are_distinct(tmp_path: Path) -> None:
    config = freeze.derive_protocol(load_config(BASE_CONFIG), EXCLUSIONS)
    (tmp_path / "asset_manifest.json").write_text("{}\n", encoding="ascii")
    manifest = {
        "banks": [],
        "config_sha256": "3" * 64,
        "source_config_sha256": "4" * 64,
        "raw_sources": [],
    }

    engine = build_engine_config(tmp_path, manifest, config)

    assert engine["profile"].endswith("v6-post-rt-exclusions")
    assert engine["post_rt_exclusions"] == config["post_rt_exclusions"]


def _physical_bank_rows(count: int, role: str, offset: int = 0) -> list[dict]:
    return [
        {
            "scene_index": offset + index,
            "scene_id": f"scene-{offset + index}",
            "bank_id": f"bank-{offset + index}",
            "base_map_cluster_id": f"cluster-{offset + index}",
            "role": role,
            "sha256": f"{offset + index + 10:064x}"[-64:],
            "manifest_sha256": f"{offset + index + 100:064x}"[-64:],
            "bytes": 1000 + index,
            "manifest_bound": True,
        }
        for index in range(count)
    ]


def _write_physical_pilot(tmp_path: Path, *, eval_stride: int = 1) -> Path:
    candidate = tmp_path / "candidate.json"
    formal = tmp_path / "formal.json"
    source = tmp_path / "probe.py"
    candidate.write_text("{}\n", encoding="ascii")
    formal.write_text("{}\n", encoding="ascii")
    source.write_text("# frozen probe\n", encoding="ascii")
    fit_rows = _physical_bank_rows(32, "source_encoder_train")
    eval_rows = _physical_bank_rows(8, "source_method_selection", 32)
    seed = 12345
    checkpoint = tmp_path / "gate.pt"
    torch.save(
        {
            "schema_version": freeze.PHYSICAL_CHECKPOINT_SCHEMA,
            "classification": {
                "status": freeze.PHYSICAL_DIAGNOSTIC_STATUS,
                "scientific_use": "FORBIDDEN",
                "formal_dataset": False,
                "paper_table_eligible": False,
            },
            "contract": freeze.PHYSICAL_GATE_CONTRACT,
            "state_dicts": [{}, {}, {}],
            "training": {
                "seed": seed,
                "ensemble_size": 3,
                "fit_stride": 16,
                "world_scope": "all",
            },
            "bindings": {
                "candidate_config": str(candidate),
                "candidate_config_sha256": sha256_file(candidate),
                "formal_config": str(formal),
                "formal_config_sha256": sha256_file(formal),
                "probe_source": str(source),
                "probe_source_sha256": sha256_file(source),
                "fit_banks": fit_rows,
            },
        },
        checkpoint,
    )
    pilot = tmp_path / "pilot.json"
    pilot.write_text(
        json.dumps(
            {
                "schema_version": freeze.PHYSICAL_DIAGNOSTIC_SCHEMA,
                "status": freeze.PHYSICAL_DIAGNOSTIC_STATUS,
                "pilot": True,
                "fixture": False,
                "formal_dataset": False,
                "scientific_use": "FORBIDDEN",
                "paper_table_eligible": False,
                "simulation_not_measurement": True,
                "formal_g1_g2_claim": "NOT_EVALUATED",
                "fit_stride": 16,
                "eval_stride": eval_stride,
                "world_scope": "all",
                "learned_gate_subset_passed": True,
                "fit_banks": fit_rows,
                "eval_banks": eval_rows,
                "eval_learned_gate_summary": [
                    {
                        "bank_id": row["bank_id"],
                        "passed_subset_gate": True,
                        "active_units": 256,
                        "null_units": 256,
                        "relative_improvement_vs_copy": 0.03,
                        "relative_improvement_vs_exact_action_swap": 0.04,
                        "null_violation_rate": 0.01,
                    }
                    for row in eval_rows
                ],
                "gate_checkpoint": str(checkpoint),
                "gate_checkpoint_sha256": sha256_file(checkpoint),
                "learned_gate": {"seed": seed},
            }
        )
        + "\n",
        encoding="ascii",
    )
    return pilot


def test_physical_protocol_freeze_pilot_authenticates_full_position_scope(
    tmp_path: Path,
) -> None:
    pilot_path = _write_physical_pilot(tmp_path)
    candidate = tmp_path / "candidate.json"
    formal = tmp_path / "formal.json"
    _payload, receipt = freeze._validate_physical_response_pilot(
        pilot_path,
        candidate_sha256=sha256_file(candidate),
        formal_sha256=sha256_file(formal),
        minimum_gain=0.02,
        maximum_null_violation=0.05,
        required_eval_stride=1,
    )
    assert receipt["eval_stride"] == 1
    assert receipt["minimum_improvement_vs_copy"] == 0.03
    assert receipt["maximum_null_violation_rate"] == 0.01


def test_physical_protocol_freeze_pilot_rejects_a_failed_bank(tmp_path: Path) -> None:
    pilot_path = _write_physical_pilot(tmp_path)
    payload = json.loads(pilot_path.read_text(encoding="ascii"))
    payload["eval_learned_gate_summary"][0]["null_violation_rate"] = 0.051
    pilot_path.write_text(json.dumps(payload) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="failed bank"):
        freeze._validate_physical_response_pilot(
            pilot_path,
            candidate_sha256=sha256_file(tmp_path / "candidate.json"),
            formal_sha256=sha256_file(tmp_path / "formal.json"),
            minimum_gain=0.02,
            maximum_null_violation=0.05,
            required_eval_stride=1,
        )
