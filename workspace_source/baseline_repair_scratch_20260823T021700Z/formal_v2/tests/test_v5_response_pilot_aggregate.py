from __future__ import annotations

import argparse
import json

from artifacts.formal_readiness.tools import v5_response_pilot_aggregate as aggregate


def _classification() -> dict:
    return {
        "scientific_use": "FORBIDDEN",
        "formal_dataset": False,
        "paper_table_eligible": False,
        "formal_g1_g2_claim": "NOT_EVALUATED",
    }


def test_response_aggregate_preserves_diagnostic_classification(tmp_path) -> None:
    bank = {"bank_id": "bank-1"}
    response_paths = []
    for seed in (11, 12):
        path = tmp_path / f"seed-{seed}.json"
        path.write_text(
            json.dumps(
                {
                    **_classification(),
                    "seed": seed,
                    "candidate_config_sha256": "candidate",
                    "formal_config_sha256": "formal",
                    "eval_banks": [bank],
                    "g2_response_precursor": "PASS",
                    "evaluation_summary": {"all_banks_passed": True},
                    "bank_rows": [
                        {
                            "bank_id": "bank-1",
                            "relative_improvement_vs_copy": 0.10,
                            "relative_improvement_vs_exact_action_swap": 0.08,
                            "null_violation_rate": 0.01,
                            "passed": True,
                        }
                    ],
                }
            ),
            encoding="ascii",
        )
        response_paths.append(path)
    physical_path = tmp_path / "physical.json"
    physical_path.write_text(
        json.dumps(
            {
                **_classification(),
                "world_scope": "all",
                "eval_banks": [bank],
                "learned_gate_subset_passed": True,
                "eval_learned_gate_summary": [
                    {
                        "bank_id": "bank-1",
                        "relative_improvement_vs_copy": 0.20,
                        "relative_improvement_vs_exact_action_swap": 0.15,
                        "null_violation_rate": 0.02,
                        "geometry_matched_wrong_action_fraction": 0.875,
                        "passed_subset_gate": True,
                    }
                ],
            }
        ),
        encoding="ascii",
    )

    payload = aggregate.run(
        argparse.Namespace(
            response_pilot=response_paths,
            physical_probe=physical_path,
            output=tmp_path / "aggregate",
        )
    )

    assert payload["result"] == "PASS"
    assert payload["scientific_use"] == "FORBIDDEN"
    assert payload["formal_g1_g2_claim"] == "NOT_EVALUATED"
    assert (tmp_path / "aggregate" / "metrics.csv").read_text(
        encoding="ascii"
    ).count("\n") == 3
