from __future__ import annotations

from dataclasses import dataclass
import re


Location = tuple[str, str]


@dataclass(frozen=True)
class TraceFamily:
    expectation: str
    entry: str
    code: tuple[Location, ...]
    config: tuple[Location, ...]
    tests: tuple[Location, ...]
    dynamic: str
    status: str = "EXACT"
    blocker: str = "NONE"
    repair: str = "No code repair is pending; non-fixture scientific evidence remains separate."
    paper: str = "paper_v2/main.tex"


def _family(
    expectation: str,
    entry: str,
    code: tuple[Location, ...],
    config: tuple[Location, ...],
    tests: tuple[Location, ...],
    dynamic_test: str,
    *,
    status: str = "EXACT",
    blocker: str = "NONE",
    repair: str = "No code repair is pending; non-fixture scientific evidence remains separate.",
) -> TraceFamily:
    return TraceFamily(
        expectation=expectation,
        entry=entry,
        code=code,
        config=config,
        tests=tests,
        dynamic=(
            f"required replay: {dynamic_test}; exact-head execution status is "
            "external to this static matrix"
        ),
        status=status,
        blocker=blocker,
        repair=repair,
    )


_CLAIMS_CONFIG = (("artifacts/v2_0_claim_evidence_contract.json", '"claims"'),)
_FORMAL_CONFIG = "formal_v2/configs/formal_v2.json"


TRACE_FAMILIES = {
    "claim_boundary": _family(
        "Claims are promoted only inside the registered relative, non-causal evidence boundary.",
        "assemble-claims",
        (("formal_v2/formal_claims.py", "def _claim_state"),),
        _CLAIMS_CONFIG,
        (("formal_v2/tests/test_formal_v2.py", "def test_claim_semantics_recheck_g0_c2_c11_and_g8_evidence"),),
        "formal_v2.tests.test_formal_v2.EvidenceAndPathTests.test_claim_semantics_recheck_g0_c2_c11_and_g8_evidence",
    ),
    "shared_method": _family(
        "The retained F/P method uses the frozen observable inputs and shared encoder contract.",
        "run-factorial",
        (("formal_v2/formal_model.py", "class CSIPairsFormalModel"),),
        ((_FORMAL_CONFIG, '"factorial"'),),
        (("formal_v2/tests/test_formal_v2.py", "def test_model_api_has_no_target_route_or_position_input_and_dual_outputs"),),
        "formal_v2.tests.test_formal_v2.ProtocolTests.test_model_api_has_no_target_route_or_position_input_and_dual_outputs",
    ),
    "wrong_map": _family(
        "Each eligible external model executes the complete six-condition paired wrong-map protocol.",
        "run-wrong-map; run-external-baselines",
        (("formal_v2/formal_external.py", "def _validate_six_condition_rows"),),
        (("formal_v2/external_adapters/all_map_adapters_v1.json", '"adapters"'),),
        (("formal_v2/tests/test_pmnet_adapter.py", "def test_registry_exposes_pmnet_as_the_second_c1_model"),),
        "formal_v2.tests.test_pmnet_adapter.PMNetAdapterTests.test_registry_exposes_pmnet_as_the_second_c1_model",
        status="PARTIAL/PROXY",
        blocker="EXTERNAL_DATA_REQUIRED",
        repair="Execute both shipped C1-eligible adapters on authenticated non-fixture data and pass every per-city six-condition gate.",
    ),
    "scene_id": _family(
        "Scene-ID is assessed only through held-out positions and authenticated training provenance.",
        "run-scene-id-audit",
        (("formal_v2/formal_scene_id.py", "def run_scene_id_audit"),),
        (("formal_v2/formal_scene_id.py", '"csi-pairs-v6-scene-id-adapters-v3"'),),
        (("formal_v2/tests/test_formal_v2.py", "def test_scene_id_rows_are_exactly_paired_to_held_out_positions"),),
        "formal_v2.tests.test_formal_v2.EvidenceAndPathTests.test_scene_id_rows_are_exactly_paired_to_held_out_positions",
        status="PARTIAL/PROXY",
        blocker="EXTERNAL_DATA_REQUIRED",
        repair="Run the authenticated source-city scene-ID audit on non-fixture banks before making the mechanism claim.",
    ),
    "probes": _family(
        "CGS, intrinsic outputs, unified probes, native response metrics, and their scopes remain separate.",
        "run-evaluation",
        (("formal_v2/formal_evaluation.py", "def _compatibility_dataset"),),
        ((_FORMAL_CONFIG, '"evaluation"'),),
        (("formal_v2/tests/test_risk_path_evaluation_integrity.py", "def test_native_probe_correlation_uses_seed_bank_foundation_macro"),),
        "formal_v2.tests.test_risk_path_evaluation_integrity.RiskPathEvaluationIntegrityTests.test_native_probe_correlation_uses_seed_bank_foundation_macro",
    ),
    "paired_unit": _family(
        "Every unit binds paired worlds, receiver position, radio state, CSI, assets, and provenance.",
        "inspect-data; verify-data",
        (("formal_v2/formal_dataset.py", "class FormalDataset"),),
        ((_FORMAL_CONFIG, '"data"'),),
        (("formal_v2/tests/test_formal_v2.py", "def test_maps_radio_and_coordinates_are_typed"),),
        "formal_v2.tests.test_formal_v2.DatasetTests.test_maps_radio_and_coordinates_are_typed",
    ),
    "world_bank": _family(
        "World banks are reusable scene-level hypercubes with authenticated Hamming-one edges and foundations.",
        "inspect-data; verify-data",
        (("formal_v2/formal_dataset.py", "def _canonical_foundation_sha256"),),
        ((_FORMAL_CONFIG, '"data"'),),
        (("formal_v2/tests/test_formal_v2.py", "def test_hypercube_edges_are_bidirectional_hamming_one"),),
        "formal_v2.tests.test_formal_v2.DatasetTests.test_hypercube_edges_are_bidirectional_hamming_one",
    ),
    "common_positions": _family(
        "All siblings share legal positions and target support positions are globally excluded from query metrics.",
        "inspect-data; run-factorial; run-evaluation",
        (("formal_v2/formal_factorial.py", "def eligible_query_indices"),),
        ((_FORMAL_CONFIG, '"data"'),),
        (("formal_v2/tests/test_formal_v2.py", "def test_target_position_identity_cannot_cross_support_and_query_banks"),),
        "formal_v2.tests.test_formal_v2.DatasetTests.test_target_position_identity_cannot_cross_support_and_query_banks",
    ),
    "edit_contract": _family(
        "P0 edits are typed, signed, reversible, hash-bound, and use the frozen material semantics.",
        "inspect-data; run-factorial",
        (("formal_v2/formal_protocol.py", "def typed_signed_edit"),),
        ((_FORMAL_CONFIG, '"data"'),),
        (("formal_v2/tests/test_formal_v2.py", "def test_typed_action_has_inverse_direction_and_categorical_from_to"),),
        "formal_v2.tests.test_formal_v2.ProtocolTests.test_typed_action_has_inverse_direction_and_categorical_from_to",
    ),
    "observability": _family(
        "Model-visible maps, actions, radio state, and BS pose obey the V6 information allowlist.",
        "run-factorial",
        (("formal_v2/formal_model.py", "class CSIPairsFormalModel"),),
        ((_FORMAL_CONFIG, '"factorial"'),),
        (("formal_v2/tests/test_formal_v2.py", "def test_f_inherits_complete_teacher_encoder_and_uses_context_pose"),),
        "formal_v2.tests.test_formal_v2.ProtocolTests.test_f_inherits_complete_teacher_encoder_and_uses_context_pose",
    ),
    "splits": _family(
        "Role, city, bank, calibration, support, and query splits remain disjoint at their registered units.",
        "inspect-data; verify-data",
        (("formal_v2/formal_dataset.py", "class FormalDataset"),),
        ((_FORMAL_CONFIG, '"data"'),),
        (("formal_v2/tests/test_formal_v2.py", "def test_every_source_role_itself_must_cover_two_source_cities"),),
        "formal_v2.tests.test_formal_v2.DatasetTests.test_every_source_role_itself_must_cover_two_source_cities",
    ),
    "permissions": _family(
        "Each data role has one frozen permission set and forbidden roles cannot enter qualification or fitting.",
        "inspect-data; qualify",
        (("formal_v2/formal_dataset.py", "class FormalDataset"),),
        ((_FORMAL_CONFIG, '"data"'),),
        (("formal_v2/tests/test_formal_v2.py", "def test_qualification_blocking_allowlist_excludes_all_other_roles"),),
        "formal_v2.tests.test_formal_v2.DatasetTests.test_qualification_blocking_allowlist_excludes_all_other_roles",
    ),
    "teacher": _family(
        "The frozen CSI-only teacher uses independently resampled exact-cardinality masks and authenticated checkpoints.",
        "qualify",
        (("formal_v2/formal_teacher.py", "def train_teacher_bundle"),),
        ((_FORMAL_CONFIG, '"qualification"'),),
        (("formal_v2/tests/test_data_protocol_integrity.py", "def test_teacher_training_resamples_masks_at_every_step"),),
        "formal_v2.tests.test_data_protocol_integrity.FrozenMaskContractTests.test_teacher_training_resamples_masks_at_every_step",
    ),
    "gauge": _family(
        "Physical deltas use one sourced complex reference per scene-position with no world axis; independent regeneration authenticates the reference and CSI.",
        "inspect-data; verify-data",
        (("formal_v2/formal_dataset.py", "phase_reference_values must have world-independent shape"),),
        (("formal_v2/DATA_CONTRACT.md", 'phase_gauge_rule=shared_complex_reference'),),
        (("formal_v2/tests/test_data_protocol_integrity.py", "def test_phase_reference_fields_reject_a_per_world_axis"),),
        "formal_v2.tests.test_data_protocol_integrity.DatasetIdentityAndNoiseTests.test_phase_reference_fields_reject_a_per_world_axis",
        repair=(
            "Author decision A freezes shared_complex_reference for the formal study. "
            "A phase-invariant target would be a separately versioned future protocol."
        ),
    ),
    "routing": _family(
        "Author decision R1 keeps native full-channel response on r^A and the unified patch probe on r^{R,q}; teacher sensitivity is an independent stratum.",
        "qualify; run-factorial",
        (("formal_v2/formal_evaluation.py", "def _native_response_route_is_active"),),
        ((_FORMAL_CONFIG, '"qualification"'),),
        (("formal_v2/tests/test_route_estimand_contract.py", "def test_native_and_unified_response_keep_distinct_route_granularities"),),
        "formal_v2.tests.test_route_estimand_contract.ResponseEstimandContractTests.test_native_and_unified_response_keep_distinct_route_granularities",
    ),
    "mask_query": _family(
        "Input masks hide every output query and preserve the frozen random and axis-block families.",
        "qualify; run-factorial",
        (("formal_v2/formal_protocol.py", "def frozen_mask_query_bank"),),
        ((_FORMAL_CONFIG, '"qualification"'),),
        (("formal_v2/tests/test_formal_v2.py", "def test_mask_bank_has_three_modes_query_hidden_and_full_coverage"),),
        "formal_v2.tests.test_formal_v2.ProtocolTests.test_mask_bank_has_three_modes_query_hidden_and_full_coverage",
    ),
    "endpoint": _family(
        "All four arms optimize the same bank-macro endpoint target and differ only in registered supervision switches.",
        "run-factorial",
        (("formal_v2/formal_model.py", "def endpoint_per_sample"),),
        ((_FORMAL_CONFIG, '"arms"'),),
        (("formal_v2/tests/test_factorial_integrity.py", "def test_target_sampler_is_unconditional_and_not_fifty_fifty_reweighted"),),
        "formal_v2.tests.test_factorial_integrity.ResponsePlanMutationTests.test_target_sampler_is_unconditional_and_not_fifty_fifty_reweighted",
    ),
    "alignment": _family(
        "Alignment uses complete active quartets, null dead zones, and no privileged incident edges.",
        "run-factorial; run-evaluation",
        (("formal_v2/formal_model.py", "def alignment_active_quartet_loss"),),
        ((_FORMAL_CONFIG, '"factorial"'),),
        (("formal_v2/tests/test_factorial_integrity.py", "def test_alignment_corpus_excludes_natural_incident_edges"),),
        "formal_v2.tests.test_factorial_integrity.ArmExecutionMutationTests.test_alignment_corpus_excludes_natural_incident_edges",
    ),
    "response": _family(
        "Response predicts registered latent and physical targets with active direction and null safety contracts.",
        "run-factorial; run-evaluation",
        (("formal_v2/formal_model.py", "def response_component_losses"),),
        ((_FORMAL_CONFIG, '"factorial"'),),
        (("formal_v2/tests/test_risk_path_evaluation_integrity.py", "def test_perfect_response_has_complete_transition_metrics"),),
        "formal_v2.tests.test_risk_path_evaluation_integrity.RiskPathEvaluationIntegrityTests.test_perfect_response_has_complete_transition_metrics",
    ),
    "dose_resources": _family(
        "The 2x2 arms share data and schedules while recording real parameters, calls, updates, FLOPs, gradients, and timing.",
        "run-factorial; run-resource-controls",
        (("formal_v2/formal_factorial.py", "def _measure_execution"),),
        ((_FORMAL_CONFIG, '"factorial"'),),
        (("formal_v2/tests/test_factorial_integrity.py", "def test_disabled_branches_are_not_forwarded_or_profiled_by_proxy"),),
        "formal_v2.tests.test_factorial_integrity.ArmExecutionMutationTests.test_disabled_branches_are_not_forwarded_or_profiled_by_proxy",
    ),
    "q_comp": _family(
        "q_comp is source-fitted, active-paired, candidate-order randomized, and never given global map-truth semantics.",
        "run-risk",
        (("formal_v2/formal_risk.py", "q_comp_route_contract"),),
        ((_FORMAL_CONFIG, '"risk"'),),
        (("formal_v2/tests/test_formal_v2.py", "def test_q_comp_random_order_and_candidate_swap_complement"),),
        "formal_v2.tests.test_formal_v2.ProtocolTests.test_q_comp_random_order_and_candidate_swap_complement",
    ),
    "p_fail": _family(
        "p_fail is a separate source-only k=0 risk calibrator on frozen map proposals and common support.",
        "run-risk",
        (("formal_v2/formal_risk.py", "def fit_constrained_risk_calibrator"),),
        ((_FORMAL_CONFIG, '"risk"'),),
        (("formal_v2/tests/test_risk_path_evaluation_integrity.py", "def test_risk_ranking_ignores_observations_outside_common_support"),),
        "formal_v2.tests.test_risk_path_evaluation_integrity.RiskPathEvaluationIntegrityTests.test_risk_ranking_ignores_observations_outside_common_support",
    ),
    "theory_boundary": _family(
        "Theory and paper wording retain only registered identification, symmetry, ranking, and target-regression implications.",
        "paper protocol; assemble-claims",
        (("formal_v2/formal_claims.py", "CLAIM_DEPENDENCIES"),),
        _CLAIMS_CONFIG,
        (("formal_v2/tests/test_formal_v2.py", "def test_claim_semantics_recheck_g0_c2_c11_and_g8_evidence"),),
        "formal_v2.tests.test_formal_v2.EvidenceAndPathTests.test_claim_semantics_recheck_g0_c2_c11_and_g8_evidence",
    ),
    "no_x_action": _family(
        "Action-only, no-X, oracle-X, position, and privileged-input boundaries are explicit and fail closed.",
        "qualify; run-factorial",
        (("formal_v2/formal_model.py", "class CSIPairsFormalModel"),),
        ((_FORMAL_CONFIG, '"qualification"'),),
        (("formal_v2/tests/test_formal_v2.py", "def test_inverse_localizer_api_has_no_true_position_argument"),),
        "formal_v2.tests.test_formal_v2.WiGATrAdapterTests.test_inverse_localizer_api_has_no_true_position_argument",
    ),
    "factorial": _family(
        "The strict four-arm design changes only Alignment/Response factors and uses synchronized registered inference.",
        "run-factorial; run-evaluation",
        (("formal_v2/formal_config.py", "ARMS ="), ("formal_v2/formal_statistics.py", "def hierarchical_factorial_interval")),
        ((_FORMAL_CONFIG, '"arms"'),),
        (("formal_v2/tests/test_factorial_integrity.py", "def test_primary_g4_intervals_are_one_synchronized_family"),),
        "formal_v2.tests.test_factorial_integrity.FactorialStatisticsMutationTests.test_primary_g4_intervals_are_one_synchronized_family",
    ),
    "resource_controls": _family(
        "Equal-FLOP, parameter-matched, independent-concat, and report-only controls are executable and provenance-bound.",
        "run-resource-controls",
        (("formal_v2/formal_controls.py", "def run_resource_controls"),),
        (("formal_v2/external_adapters/resource_controls_v3.json", '"controls"'),),
        (("formal_v2/tests/test_evidence_integrity.py", "def test_resource_control_commands_must_execute_bound_source_and_spec"),),
        "formal_v2.tests.test_evidence_integrity.EvidenceIntegrityTests.test_resource_control_commands_must_execute_bound_source_and_spec",
    ),
    "branch_metrics": _family(
        "Each branch first earns its native active/null metric before any cross-capability or joint claim.",
        "run-evaluation; assemble-claims",
        (("formal_v2/formal_evaluation.py", "def _evaluation_gate"),),
        ((_FORMAL_CONFIG, '"evaluation"'),),
        (("formal_v2/tests/test_risk_path_evaluation_integrity.py", "def test_g3_gate_cannot_pool_away_target_response_control_failure"),),
        "formal_v2.tests.test_risk_path_evaluation_integrity.RiskPathEvaluationIntegrityTests.test_g3_gate_cannot_pool_away_target_response_control_failure",
    ),
    "rq_wrong_map": _family(
        "RQ1 requires authenticated non-fixture six-condition results from at least two eligible models.",
        "run-external-baselines",
        (("formal_v2/formal_external.py", "def run_external_baselines"),),
        (("formal_v2/external_adapters/all_map_adapters_v1.json", '"adapters"'),),
        (("formal_v2/tests/test_evidence_integrity.py", "def test_six_equal_conditions_cannot_pass_c1"),),
        "formal_v2.tests.test_evidence_integrity.EvidenceIntegrityTests.test_six_equal_conditions_cannot_pass_c1",
        status="PARTIAL/PROXY",
        blocker="EXTERNAL_DATA_REQUIRED",
        repair="Run the complete non-fixture RQ1 protocol with two C1-eligible models.",
    ),
    "rq_alignment": _family(
        "RQ2 requires non-fixture active/null CGS and shortcut-control evidence under the frozen probe.",
        "run-evaluation; run-shuffled-pair-control",
        (("formal_v2/formal_evaluation.py", "def _evaluation_gate"),),
        ((_FORMAL_CONFIG, '"evaluation"'),),
        (("formal_v2/tests/test_risk_path_evaluation_integrity.py", "def test_headline_alignment_excludes_privileged_natural_incident_edges"),),
        "formal_v2.tests.test_risk_path_evaluation_integrity.RiskPathEvaluationIntegrityTests.test_headline_alignment_excludes_privileged_natural_incident_edges",
        status="PARTIAL/PROXY",
        blocker="EXTERNAL_DATA_REQUIRED",
        repair="Run authenticated non-fixture RQ2 probes and negative controls.",
    ),
    "rq_response": _family(
        "RQ3 requires non-fixture target-free direction, magnitude, native-unit, and null-safety evidence.",
        "run-evaluation",
        (("formal_v2/formal_evaluation.py", "def _transition_metrics"),),
        ((_FORMAL_CONFIG, '"evaluation"'),),
        (("formal_v2/tests/test_risk_path_evaluation_integrity.py", "def test_perfect_response_has_complete_transition_metrics"),),
        "formal_v2.tests.test_risk_path_evaluation_integrity.RiskPathEvaluationIntegrityTests.test_perfect_response_has_complete_transition_metrics",
        status="PARTIAL/PROXY",
        blocker="EXTERNAL_DATA_REQUIRED",
        repair="Run authenticated non-fixture RQ3 response evaluation.",
    ),
    "rq_localization": _family(
        "RQ4 requires both target cities at strict k=0 and city-level k=8 with no support leakage.",
        "run-factorial; run-evaluation",
        (("formal_v2/formal_factorial.py", "def _run_localization"),),
        ((_FORMAL_CONFIG, '"factorial"'),),
        (("formal_v2/tests/test_formal_v2.py", "def test_target_city_cluster_minimum_uses_canonical_foundations"),),
        "formal_v2.tests.test_formal_v2.DatasetTests.test_target_city_cluster_minimum_uses_canonical_foundations",
        status="PARTIAL/PROXY",
        blocker="EXTERNAL_DATA_REQUIRED",
        repair="Supply two qualified target cities and run strict k=0/k=8 localization.",
    ),
    "rq_risk": _family(
        "RQ5 requires frozen-proposal k=0 calibration, AURC, coverage, and common-support evidence.",
        "run-risk",
        (("formal_v2/formal_risk.py", "def run_risk_contract"),),
        ((_FORMAL_CONFIG, '"risk"'),),
        (("formal_v2/tests/test_risk_path_evaluation_integrity.py", "def test_calibration_macro_and_ci_have_explicit_seed_layer"),),
        "formal_v2.tests.test_risk_path_evaluation_integrity.RiskPathEvaluationIntegrityTests.test_calibration_macro_and_ci_have_explicit_seed_layer",
        status="PARTIAL/PROXY",
        blocker="EXTERNAL_DATA_REQUIRED",
        repair="Run authenticated non-fixture RQ5 calibration and risk-coverage evaluation.",
    ),
    "path": _family(
        "Path incidence uses authenticated retraces, local edge aggregation, matched strata, and zero-path equivalence.",
        "run-path",
        (("formal_v2/formal_path.py", "def run_path_audit"),),
        ((_FORMAL_CONFIG, '"path"'),),
        (("formal_v2/tests/test_risk_path_evaluation_integrity.py", "def test_path_gate_cannot_pass_without_provenance"),),
        "formal_v2.tests.test_risk_path_evaluation_integrity.RiskPathEvaluationIntegrityTests.test_path_gate_cannot_pass_without_provenance",
    ),
    "shortcuts": _family(
        "Shortcut features, batch isolation, noise, leakage, and normalization audits cannot access forbidden labels or shared state.",
        "run-evaluation",
        (("formal_v2/formal_evaluation.py", "def _alignment_shortcut_rows"),),
        ((_FORMAL_CONFIG, '"evaluation"'),),
        (("formal_v2/tests/test_risk_path_evaluation_integrity.py", "def test_shortcut_metadata_has_no_match_label_or_forced_unk_feature"),),
        "formal_v2.tests.test_risk_path_evaluation_integrity.RiskPathEvaluationIntegrityTests.test_shortcut_metadata_has_no_match_label_or_forced_unk_feature",
    ),
    "shuffled": _family(
        "Shuffled-pair controls preserve marginals while executing distinct matched and deranged training artifacts.",
        "run-shuffled-pair-control",
        (("formal_v2/formal_claim_controls.py", "def run_shuffled_pair_control"),),
        (("formal_v2/external_adapters/shuffled_pair_control_v3.json", '"schema_version"'),),
        (("formal_v2/tests/test_evidence_integrity.py", "def test_shuffled_control_cannot_reuse_matched_checkpoint"),),
        "formal_v2.tests.test_evidence_integrity.EvidenceIntegrityTests.test_shuffled_control_cannot_reuse_matched_checkpoint",
    ),
    "retention": _family(
        "The frozen retained encoder earns map, compatibility, response, no-X, and auxiliary retention evidence independently of disposable heads.",
        "run-retention-audit",
        (("formal_v2/formal_claim_controls.py", "def run_retention_audit"),),
        (("formal_v2/external_adapters/retention_control_v3.json", '"schema_version"'),),
        (("formal_v2/tests/test_evidence_integrity.py", "def test_retention_probe_must_bind_its_seed_full_checkpoint"),),
        "formal_v2.tests.test_evidence_integrity.EvidenceIntegrityTests.test_retention_probe_must_bind_its_seed_full_checkpoint",
    ),
    "internal_baselines": _family(
        "Internal baselines and the four arms use common splits, information budgets, readouts, and implementation labels.",
        "run-representation-baselines; run-factorial",
        (("formal_v2/formal_representation_baselines.py", "def run_representation_baselines"),),
        (("formal_v2/configs/representation_baselines_v1.json", '"models"'),),
        (("formal_v2/tests/test_formal_v2.py", "def test_representation_registry_preserves_paper_labels_and_roles"),),
        "formal_v2.tests.test_formal_v2.WaibuIntegrationTests.test_representation_registry_preserves_paper_labels_and_roles",
    ),
    "external_baselines": _family(
        "External baselines retain accurate official/controlled labels, licenses, checkpoints, and information budgets.",
        "verify-waibu-resources; run-external-baselines",
        (("formal_v2/formal_external.py", "def run_external_baselines"),),
        (("formal_v2/configs/waibu_resources_v1.json", '"resources"'),),
        (("formal_v2/tests/test_formal_v2.py", "def test_every_waibu_resource_has_frozen_authentication_metadata"),),
        "formal_v2.tests.test_formal_v2.WaibuIntegrationTests.test_every_waibu_resource_has_frozen_authentication_metadata",
        status="PARTIAL/PROXY",
        blocker="LICENSE_OR_ACCESS_REQUIRED",
        repair="Fetch restricted inputs locally and supply the missing faithful checkpoint/evidence without redistributing bytes.",
    ),
    "literature": _family(
        "Novelty wording is conditional on a current authenticated literature search and contradiction audit.",
        "run-literature-resources",
        (("formal_v2/formal_literature.py", "def run_literature_resource_gate"),),
        (("formal_v2/configs/waibu_resources_v1.json", '"resources"'),),
        (("formal_v2/tests/test_formal_v2.py", "def test_literature_gate_rejects_unbound_or_contradictory_novelty_records"),),
        "formal_v2.tests.test_formal_v2.EvidenceAndPathTests.test_literature_gate_rejects_unbound_or_contradictory_novelty_records",
        status="PARTIAL/PROXY",
        blocker="EXTERNAL_DATA_REQUIRED",
        repair="Run and authenticate the submission-date literature search before using novelty-sensitive wording.",
    ),
    "statistics": _family(
        "Data roles, canonical units, city/bank/seed hierarchy, synchronized bootstrap, sign flips, Holm, and frozen thresholds match V6.",
        "run-factorial; run-evaluation; run-risk; run-path",
        (("formal_v2/formal_statistics.py", "def exact_factorial_utilities"),),
        ((_FORMAL_CONFIG, '"bootstrap_resamples"'),),
        (("formal_v2/tests/test_formal_v2.py", "def test_exact_sign_flip_does_not_use_monte_carlo_plus_one_correction"),),
        "formal_v2.tests.test_formal_v2.StatisticsTests.test_exact_sign_flip_does_not_use_monte_carlo_plus_one_correction",
    ),
    "results": _family(
        "Main-paper result cells require authenticated non-fixture rows, denominators, intervals, and gate states.",
        "paper build after formal evidence",
        (("paper_v2/main.tex", r"tab:localization"),),
        _CLAIMS_CONFIG,
        (("paper_v2/build_reproducible.sh", "latexmk"),),
        "paper_v2/build_reproducible.sh plus docs/results/实验数据.md factorial meters",
        status="PARTIAL/PROXY",
        blocker="EXTERNAL_DATA_REQUIRED",
        repair="Keep the executed 32-cell factorial as a descriptive negative; fill C1 comparison cells only after same-data adapters run.",
    ),
    "claim_gates": _family(
        "C1-C13 remain BLOCKED until every authenticated dependency and claim-specific discriminator is present.",
        "assemble-claims",
        (("formal_v2/formal_claims.py", "CLAIM_DEPENDENCIES"),),
        _CLAIMS_CONFIG,
        (("formal_v2/tests/test_formal_v2.py", "def test_gate_and_claim_identifiers_are_fixed"),),
        "formal_v2.tests.test_formal_v2.EvidenceAndPathTests.test_gate_and_claim_identifiers_are_fixed",
        status="PARTIAL/PROXY",
        blocker="EXTERNAL_DATA_REQUIRED",
        repair="Keep claims blocked until the registered non-fixture evidence dependencies exist and pass.",
    ),
    "orchestration": _family(
        "G0-G8 preflight, approval, execution, stop-on-failure, and same-run evidence authentication are ordered and fail closed.",
        "prepare-full-run; create-run-approval; all",
        (("formal_v2/formal_cli.py", "def _run_authorized_full_chain"), ("formal_v2/formal_run_approval.py", "def authenticate_prepared_run")),
        ((_FORMAL_CONFIG, '"schema_version"'),),
        (("formal_v2/tests/test_run_approval.py", "def test_authorized_chain_order_scientific_fail_continues_engineering_error_stops"),),
        "formal_v2.tests.test_run_approval.FullRunApprovalTests.test_authorized_chain_order_scientific_fail_continues_engineering_error_stops",
    ),
}


SECTION_FAMILIES = {
    0: {"overview": "claim_boundary", "1": "claim_boundary", "2": "shared_method", "3": "claim_gates", "4": "factorial", "5": "claim_boundary", "6": "claim_boundary"},
    1: {"overview": "wrong_map", "1": "wrong_map", "2": "scene_id", "3": "claim_boundary", "4": "probes"},
    2: {"overview": "paired_unit", "1": "paired_unit", "2": "world_bank", "3": "common_positions", "4": "edit_contract", "5": "observability", "6": "splits", "7": "permissions"},
    3: {"overview": "routing", "1": "teacher", "2": "routing", "3": "routing", "4": "mask_query"},
    4: {"overview": "shared_method", "1": "shared_method", "2": "endpoint", "3": "alignment", "4": "response", "5": "dose_resources", "6": "claim_boundary"},
    5: {"overview": "probes", "1": "probes", "2": "probes", "3": "q_comp", "4": "p_fail"},
    6: {"overview": "theory_boundary", "1": "theory_boundary", "2": "theory_boundary", "3": "theory_boundary", "4": "theory_boundary", "5": "no_x_action", "6": "claim_boundary", "7": "no_x_action"},
    7: {"overview": "factorial", "1": "factorial", "2": "factorial", "3": "factorial", "4": "resource_controls", "5": "branch_metrics"},
    8: {"overview": "claim_gates", "RQ1": "rq_wrong_map", "RQ2": "rq_alignment", "RQ3": "rq_response", "RQ4": "rq_localization", "RQ5": "rq_risk"},
    9: {"overview": "path", "1": "path", "2": "path", "3": "path"},
    10: {"overview": "shortcuts", "1": "shortcuts", "2": "shuffled", "3": "retention"},
    11: {"overview": "external_baselines", "1": "internal_baselines", "2": "external_baselines", "3": "literature"},
    12: {"overview": "statistics", "1": "statistics", "2": "statistics", "3": "statistics", "4": "statistics"},
    13: {"overview": "results"},
    14: {"overview": "claim_gates"},
    15: {"overview": "claim_gates", "1": "orchestration", "2": "claim_boundary"},
}


def evidence_family_for_heading(section: int, heading: str) -> str:
    if heading.startswith("RQ"):
        match = re.match(r"^(RQ[1-5])", heading)
        subsection = match.group(1) if match else ""
    else:
        match = re.match(r"^\d+\.(\d+)\s", heading)
        subsection = match.group(1) if match else "overview"
    try:
        return SECTION_FAMILIES[section][subsection]
    except KeyError as error:
        raise RuntimeError(
            f"no semantic trace family for section={section}, heading={heading!r}"
        ) from error


_CLAUSE_FAMILY_OVERRIDES = (
    (
        re.compile(
            r"(?:pair-consistent|phase[_ -]gauge|phase reference|"
            r"相位参考|相位基准|共同相位|共同时钟|相位不变|相位旋转|"
            r"分别旋转会|具体选择只能在 `source-method-selection`)"
        ),
        "gauge",
        None,
    ),
    (
        re.compile(r"(?:`?q_comp`?|q_\{\\mathrm\{comp\}\})"),
        "q_comp",
        frozenset({0, 1, 5}),
    ),
    (
        re.compile(r"(?:`?p_fail`?|p_\{\\mathrm\{fail\}\})"),
        "p_fail",
        frozenset({0, 1, 5}),
    ),
)


def evidence_family_for_clause(section: int, heading: str, clause: str) -> str:
    for pattern, family, allowed_sections in _CLAUSE_FAMILY_OVERRIDES:
        if pattern.search(clause) and (
            allowed_sections is None or section in allowed_sections
        ):
            return family
    return evidence_family_for_heading(section, heading)
