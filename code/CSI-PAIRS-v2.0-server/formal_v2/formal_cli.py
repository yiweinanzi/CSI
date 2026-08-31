from __future__ import annotations

import argparse
import fcntl
import json
import os
import secrets
import sys
from pathlib import Path

from .formal_config import load_formal_config, resolve_dataset_path
from .formal_dataset import FormalDataset, SOURCE_ROLES
from .formal_fixture import write_nonscientific_fixture
from .formal_io import artifact_manifest, read_strict_json, write_json


DEFAULT_SHUFFLED_PAIR_MANIFEST = str(
    Path(__file__).resolve().parent
    / "external_adapters"
    / "shuffled_pair_control_v3.json"
)
DEFAULT_RETENTION_MANIFEST = str(
    Path(__file__).resolve().parent / "external_adapters" / "retention_control_v3.json"
)
DEFAULT_SCENE_ID_MANIFEST = "builtin:sigmap-scene-id-v1"
DEFAULT_RESOURCE_CONTROL_MANIFEST = str(
    Path(__file__).resolve().parent
    / "external_adapters"
    / "resource_controls_v3.json"
)


COMMAND_OUTPUT_PATHS = {
    "inspect-data": "data_contract.json",
    "verify-data": "data_verification",
    "qualify": "qualification",
    "run-wrong-map": "wrong_map",
    "run-factorial": "factorial",
    "run-evaluation": "evaluation",
    "run-risk": "risk",
    "run-path": "path",
    "run-external-baselines": "external_baselines",
    "run-representation-baselines": "representation_baselines",
    "run-resource-controls": "controls",
    "run-scene-id-audit": "scene_id",
    "run-external-validity": "external_validity",
    "run-literature-resources": "literature_resources",
    "run-rt-calibration": "qualification/rt_calibration",
    "run-shuffled-pair-control": "controls/shuffled_pair",
    "run-retention-audit": "evaluation/retention",
    "assemble-claims": "claims",
}
FILE_OUTPUT_COMMANDS = frozenset({"inspect-data"})
SELF_RESERVING_DIRECTORY_COMMANDS = frozenset({"run-representation-baselines"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CSI-PAIRS V2.1 V6 evidence-gated formal experiment runner"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    fixture = subparsers.add_parser("make-fixture", help="create a permanently non-scientific code fixture")
    fixture.add_argument("--output", required=True)
    fixture.add_argument("--seed", type=int, default=20270805)
    fixture.add_argument("--positions", type=int, default=16)
    fixture.add_argument(
        "--source-banks-per-role",
        type=int,
        default=2,
        help="non-scientific fixture banks per source role; two preserves the V6 source-city contract",
    )
    resources = subparsers.add_parser("verify-waibu-resources")
    resources.add_argument("--registry", required=True)
    resources.add_argument("--waibu-root", required=True)
    resources.add_argument("--output", required=True)
    sionna_export = subparsers.add_parser("export-sionna-scenes")
    sionna_export.add_argument("--dataset", required=True)
    sionna_export.add_argument("--output", required=True)
    sionna_export.add_argument("--license-id", required=True)
    sionna_export.add_argument("--carrier-frequency-hz", type=float, required=True)
    sionna_export.add_argument("--subcarrier-spacing-hz", type=float, required=True)
    sionna_export.add_argument("--receiver-z-m", type=float, default=1.5)
    sionna_export.add_argument("--max-depth", type=int, default=5)
    sionna_export.add_argument("--refraction", action="store_true")
    approval = subparsers.add_parser(
        "create-run-approval",
        help="create an external llm-judge approval after a coding-agent review of a prepared request",
    )
    approval.add_argument("--request", required=True)
    approval.add_argument("--output", required=True)
    approval.add_argument(
        "--judge",
        required=True,
        help="llm-judge identity as family:id, family in {codex, claude-code, cursor}",
    )
    approval.add_argument("--expires-utc", required=True)
    approval.add_argument("--attest-llm-judged", action="store_true")
    migration_acceptance = subparsers.add_parser(
        "accept-migration-request",
        help="accept an exact migration request using a separately stored external LLM approval",
    )
    migration_acceptance.add_argument("--request", required=True)
    migration_approval = migration_acceptance.add_mutually_exclusive_group(
        required=True
    )
    migration_approval.add_argument(
        "--approval-manifest",
        dest="approval_manifest",
        help="external migration approval manifest",
    )
    migration_approval.add_argument(
        "--approval",
        dest="approval_manifest",
        help="alias for --approval-manifest",
    )
    migration_acceptance.add_argument("--config", required=True)
    migration_acceptance.add_argument("--dataset", required=True)
    migration_acceptance.add_argument("--protocol", required=True)
    migration_acceptance.add_argument("--output", required=True)
    operator_preflight = subparsers.add_parser(
        "operator-preflight",
        help="operator inventory only; missing optional gates are not launch blockers",
    )
    operator_preflight.add_argument("--dataset", required=True)
    operator_preflight.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "formal_v2.json"),
    )
    operator_preflight.add_argument(
        "--output",
        help="optional directory; writes operator_preflight.json; not a formal run",
    )
    operator_preflight.add_argument("--compute-plan")
    evaluation_status = subparsers.add_parser(
        "evaluation-status",
        help="read and validate streaming evaluation progress without taking a run lock",
    )
    evaluation_status.add_argument("--output", required=True)

    for command in (
        "inspect-data",
        "verify-data",
        "qualify",
        "run-wrong-map",
        "run-factorial",
        "run-evaluation",
        "run-risk",
        "run-path",
        "run-external-baselines",
        "run-representation-baselines",
        "run-resource-controls",
        "run-scene-id-audit",
        "run-external-validity",
        "run-literature-resources",
        "run-rt-calibration",
        "run-shuffled-pair-control",
        "run-retention-audit",
        "assemble-claims",
        "export-data-verification",
        "prepare-full-run",
        "all",
    ):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
        child.add_argument("--output", required=True)
        child.add_argument("--dataset", help="optional dataset override; does not mutate the config")
        if command in {"run-factorial", "prepare-full-run", "all"}:
            child.add_argument(
                "--allow-nonscientific-fixture",
                action="store_true",
                help="exercise downstream code on a fixture; outputs remain scientific_use=FORBIDDEN",
            )
        if command == "run-factorial":
            child.add_argument("--qualification-gate", help="defaults to OUTPUT/qualification/gate.json")
        if command == "qualify":
            child.add_argument("--data-verification-gate", help="defaults to OUTPUT/data_verification/gate.json")
            child.add_argument(
                "--resume",
                action="store_true",
                help="resume one incomplete qualification from its authenticated teacher checkpoint",
            )
        if command == "verify-data":
            child.add_argument("--verifier-manifest", required=True)
        if command == "export-data-verification":
            child.add_argument("--verification-root", required=True)
        if command in {"prepare-full-run", "all"}:
            child.add_argument(
                "--compute-plan",
                help="optional advisory disk/GPU/time/license budget; placeholders are accepted",
            )
            child.add_argument("--verifier-manifest")
            child.add_argument(
                "--risk-feature-manifest",
                help="deprecated; the full chain replays risk features in first-party code",
            )
            child.add_argument("--adapter-manifest", required=True)
            child.add_argument(
                "--control-manifest", default=DEFAULT_RESOURCE_CONTROL_MANIFEST
            )
            child.add_argument(
                "--scene-id-manifest", default=DEFAULT_SCENE_ID_MANIFEST
            )
            child.add_argument("--external-validity-manifest")
            child.add_argument("--literature-resource-manifest")
            child.add_argument("--rt-calibration-manifest")
            child.add_argument(
                "--shuffled-pair-manifest", default=DEFAULT_SHUFFLED_PAIR_MANIFEST
            )
            child.add_argument(
                "--retention-manifest", default=DEFAULT_RETENTION_MANIFEST
            )
            child.add_argument(
                "--representation-baseline-config",
                default=str(Path(__file__).resolve().parent / "configs" / "representation_baselines_v1.json"),
            )
        if command == "all":
            child.add_argument(
                "--approval-manifest",
                help="external llm-judge approval bound to OUTPUT/approval/request.json",
            )
            child.add_argument(
                "--approve-full-experiment",
                action="store_true",
                help="deprecated compatibility flag; it has no authorization power",
            )
        if command == "run-wrong-map":
            child.add_argument("--qualification-gate", help="defaults to OUTPUT/qualification/gate.json")
        if command == "run-evaluation":
            child.add_argument("--qualification-gate", help="defaults to OUTPUT/qualification/gate.json")
            child.add_argument("--factorial-gate", help="defaults to OUTPUT/factorial/gate.json")
        if command == "run-risk":
            child.add_argument(
                "--risk-features",
                help="deprecated; risk features are replayed first-party from OUTPUT artifacts",
            )
        if command == "run-external-baselines":
            child.add_argument("--adapter-manifest", required=True)
        if command == "run-representation-baselines":
            child.add_argument("--representation-baseline-config", required=True)
        if command == "run-resource-controls":
            child.add_argument(
                "--control-manifest", default=DEFAULT_RESOURCE_CONTROL_MANIFEST
            )
        if command == "run-scene-id-audit":
            child.add_argument(
                "--scene-id-manifest", default=DEFAULT_SCENE_ID_MANIFEST
            )
        if command == "run-external-validity":
            child.add_argument("--external-validity-manifest", required=True)
        if command == "run-literature-resources":
            child.add_argument("--literature-resource-manifest", required=True)
        if command == "run-rt-calibration":
            child.add_argument("--rt-calibration-manifest", required=True)
        if command == "run-shuffled-pair-control":
            child.add_argument(
                "--claim-control-manifest", default=DEFAULT_SHUFFLED_PAIR_MANIFEST
            )
        if command == "run-retention-audit":
            child.add_argument(
                "--claim-control-manifest", default=DEFAULT_RETENTION_MANIFEST
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    from .formal_evidence import configure_reproducible_runtime

    configure_reproducible_runtime()
    args = build_parser().parse_args(argv)
    try:
        _reject_unsafe_mutating_output_argument(args.command, args.output)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    migration_accepted = (
        args.command == "all"
        and (Path(args.output).resolve() / "migration" / "accepted.json").is_file()
    )
    if args.command == "all" and not args.approval_manifest and not migration_accepted:
        print(
            "error: all requires --approval-manifest for the exact prepared run; "
            "--approve-full-experiment is deprecated and cannot authorize execution",
            file=sys.stderr,
        )
        return 2
    output_lock: _OutputLock | None = None
    try:
        if args.command == "make-fixture":
            source_banks_per_role = int(args.source_banks_per_role)
            scene_count = len(SOURCE_ROLES) * source_banks_per_role + 4
            path = write_nonscientific_fixture(
                args.output,
                seed=int(args.seed),
                scene_count=scene_count,
                positions=int(args.positions),
                source_banks_per_role=source_banks_per_role,
            )
            print(json.dumps({"status": "success", "fixture": str(path.resolve()), "scientific_use": "FORBIDDEN"}))
            return 0
        if args.command == "verify-waibu-resources":
            from .formal_resources import verify_waibu_resources

            output = Path(args.output).resolve()
            output_lock = _acquire_output_lock(output)
            result = verify_waibu_resources(args.registry, args.waibu_root, output)
            print(json.dumps({"status": result["status"], "output": str(output)}, sort_keys=True))
            return 0 if result.get("passed") is True else 1
        if args.command == "export-sionna-scenes":
            from .sionna_scene_export import export_sionna_scenes

            output = Path(args.output).resolve()
            output_lock = _acquire_output_lock(_sionna_export_lock_root(output))
            manifest = export_sionna_scenes(
                args.dataset,
                output,
                license_id=args.license_id,
                carrier_frequency_hz=args.carrier_frequency_hz,
                subcarrier_spacing_hz=args.subcarrier_spacing_hz,
                receiver_z_m=args.receiver_z_m,
                max_depth=args.max_depth,
                refraction=bool(args.refraction),
            )
            print(json.dumps({"status": "PASS", "manifest": str(manifest)}, sort_keys=True))
            return 0
        if args.command == "create-run-approval":
            from .formal_run_approval import create_llm_judge_approval_manifest

            approval = create_llm_judge_approval_manifest(
                args.request,
                args.output,
                judge=args.judge,
                expires_utc=args.expires_utc,
                attest_llm_judged=bool(args.attest_llm_judged),
            )
            print(
                json.dumps(
                    {
                        "status": "APPROVED",
                        "approval_manifest": approval["approval_manifest_path"],
                        "request_sha256": approval["request_sha256"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "accept-migration-request":
            from .formal_evidence import config_sha256
            from .formal_migration import accept_migration_request

            output_argument = Path(args.output)
            if output_argument.is_symlink():
                raise RuntimeError("migration output root cannot be a symlink")
            output = output_argument.resolve()
            output_lock = _acquire_output_lock(output)
            config = load_formal_config(args.config)
            accepted = accept_migration_request(
                args.request,
                args.approval_manifest,
                new_run_root=output,
                new_source_root=Path(__file__).resolve().parent,
                protocol_path=args.protocol,
                dataset_path=args.dataset,
                config_sha256=config_sha256(config),
                expected_seeds=config["seeds"],
            )
            print(
                json.dumps(
                    {
                        "status": accepted["status"],
                        "accepted_path": accepted["accepted_path"],
                        "accepted_sha256": accepted["accepted_sha256"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "operator-preflight":
            from .formal_operator_preflight import run_operator_preflight

            report = run_operator_preflight(
                args.dataset,
                config_path=args.config,
                output_root=args.output,
                compute_plan_path=args.compute_plan,
            )
            print(json.dumps(report, sort_keys=True, ensure_ascii=True))
            return 0
        if args.command == "evaluation-status":
            from .formal_evaluation_resume import read_evaluation_status
            from .formal_evaluation_streaming import STATE_DIRECTORY

            state_root = Path(args.output).resolve() / STATE_DIRECTORY
            status = read_evaluation_status(state_root)
            if status is None:
                status = {
                    "status": "NOT_STARTED",
                    "state_root": str(state_root),
                }
            print(json.dumps(status, sort_keys=True, ensure_ascii=True))
            return 0
        config = load_formal_config(args.config)
        dataset_path = Path(args.dataset).resolve() if args.dataset else resolve_dataset_path(config)
        dataset = FormalDataset.load(
            dataset_path,
            require_clean_csi=bool(config["data"]["require_clean_csi"]),
        )
        dataset.validate(
            require_clean_csi=bool(config["data"]["require_clean_csi"]),
            minimum_repeats=int(config["data"]["minimum_repeats"]),
            minimum_target_cities=int(config["data"]["minimum_target_cities"]),
            minimum_source_cities=int(config["data"]["minimum_source_cities"]),
            minimum_banks_per_target_city=int(config["data"]["minimum_banks_per_target_city"]),
            minimum_independent_base_map_clusters_per_target_city=int(
                config["data"][
                    "minimum_independent_base_map_clusters_per_target_city"
                ]
            ),
            minimum_banks_per_source_role=int(config["data"]["minimum_banks_per_source_role"]),
            minimum_independent_source_final_unseen_clusters=int(
                config["data"]["minimum_independent_source_final_unseen_clusters"]
            ),
            minimum_independent_external_validation_clusters=int(
                config["data"]["minimum_independent_external_validation_clusters"]
            ),
            minimum_unique_support_positions_per_target_city=max(
                int(value) for value in config["localization"]["label_budgets"]
            ),
        )
        output_argument = Path(args.output)
        _reject_unsafe_mutating_output_argument(args.command, output_argument)
        output = output_argument.resolve()
        migration_mode = (output / "migration").exists() or (
            output / "migration"
        ).is_symlink()
        full_run_preflight = None
        if args.command in {"prepare-full-run", "all"} and not (
            args.command == "all" and migration_mode
        ):
            from .formal_run_approval import (
                full_run_input_values,
                preflight_full_run,
            )

            full_run_preflight = preflight_full_run(
                config,
                dataset,
                output,
                args.compute_plan,
                full_run_input_values(args),
                resource_registry=(
                    Path(__file__).resolve().parent
                    / "configs"
                    / "waibu_resources_v1.json"
                ),
                waibu_root=Path(__file__).resolve().parents[1] / "waibu",
                resume=args.command == "all",
            )
        output_lock = _acquire_output_lock(output)
        if args.command == "prepare-full-run":
            _reserve_full_run_output(output)
        elif args.command == "all":
            if not output.is_dir() or output.is_symlink():
                raise RuntimeError("all requires an existing prepared run root")
        elif args.command == "run-evaluation" and migration_mode:
            if not output.is_dir() or output.is_symlink():
                raise RuntimeError(
                    "migrated run-evaluation requires an existing migration run root"
                )
        elif args.command == "export-data-verification":
            pass
        elif args.command == "qualify" and args.resume:
            qualification_dir = output / "qualification"
            if not qualification_dir.is_dir() or qualification_dir.is_symlink():
                raise RuntimeError("qualify --resume requires an incomplete qualification directory")
            if (qualification_dir / "gate.json").exists() or (
                qualification_dir / "manifest.json"
            ).exists():
                raise RuntimeError("qualify --resume refuses a completed qualification")
        else:
            _reserve_command_output(args.command, output)
            output.mkdir(parents=True, exist_ok=True)
        if args.command == "prepare-full-run":
            result = _prepare_full_run(
                config,
                dataset,
                output,
                args,
                full_run_preflight,
            )
        elif args.command == "inspect-data":
            target = output / "data_contract.json"
            report = dataset.contract_report()
            write_json(target, report)
            result = report
        elif args.command == "qualify":
            from .formal_qualification import run_formal_qualification

            verification_path = (
                Path(args.data_verification_gate)
                if args.data_verification_gate
                else output / "data_verification" / "gate.json"
            )
            if verification_path.is_file() and not verification_path.is_symlink():
                verification_gate = read_strict_json(verification_path)
                verification_gate_path = verification_path
            else:
                verification_gate = None
                verification_gate_path = None
            result = run_formal_qualification(
                config,
                dataset,
                output,
                verification_gate,
                data_verification_gate_path=verification_gate_path,
                resume=bool(args.resume),
            )
        elif args.command == "verify-data":
            from .formal_data_verification import run_data_verification

            result = run_data_verification(config, dataset, args.verifier_manifest, output)
        elif args.command == "export-data-verification":
            from .formal_data_verification import export_precomputed_verification

            result = export_precomputed_verification(
                config,
                dataset,
                args.verification_root,
                output,
            )
            print(json.dumps(result, sort_keys=True))
            return _result_exit_code(result)
        elif args.command == "run-wrong-map":
            from .formal_wrong_map import run_formal_wrong_map

            gate_path = Path(args.qualification_gate) if args.qualification_gate else output / "qualification" / "gate.json"
            result = run_formal_wrong_map(config, dataset, output, read_strict_json(gate_path))
        elif args.command == "run-factorial":
            gate_path = Path(args.qualification_gate) if args.qualification_gate else output / "qualification" / "gate.json"
            gate = read_strict_json(gate_path)
            from .formal_factorial import run_formal_factorial

            result = run_formal_factorial(
                config,
                dataset,
                output,
                gate,
                allow_nonscientific_fixture=bool(args.allow_nonscientific_fixture),
            )
        elif args.command == "run-evaluation":
            result = _run_evaluation_command(config, dataset, output, args)
        elif args.command == "run-risk":
            from .formal_risk import run_risk_contract

            result = _run_with_authenticated_upstream(
                config,
                dataset,
                output,
                lambda _upstream: run_risk_contract(config, dataset, output),
            )
        elif args.command == "run-path":
            from .formal_path import run_path_audit

            result = _run_with_authenticated_upstream(
                config,
                dataset,
                output,
                lambda upstream: run_path_audit(
                    config, dataset, upstream.factorial_root, output
                ),
            )
        elif args.command == "run-external-baselines":
            from .formal_external import run_external_baselines

            result = _run_with_authenticated_upstream(
                config,
                dataset,
                output,
                lambda _upstream: run_external_baselines(
                    config, dataset, args.adapter_manifest, output
                ),
            )
        elif args.command == "run-representation-baselines":
            from .formal_representation_baselines import run_representation_baselines

            result = _run_with_authenticated_upstream(
                config,
                dataset,
                output,
                lambda _upstream: run_representation_baselines(
                    config, dataset, args.representation_baseline_config, output
                ),
            )
        elif args.command == "run-resource-controls":
            from .formal_controls import run_resource_controls

            result = _run_with_authenticated_upstream(
                config,
                dataset,
                output,
                lambda _upstream: run_resource_controls(
                    config, dataset, args.control_manifest, output
                ),
            )
        elif args.command == "run-scene-id-audit":
            from .formal_scene_id import run_scene_id_audit

            result = _run_with_authenticated_upstream(
                config,
                dataset,
                output,
                lambda _upstream: run_scene_id_audit(
                    config, dataset, args.scene_id_manifest, output
                ),
            )
        elif args.command == "run-external-validity":
            from .formal_external_validity import run_external_validity

            result = run_external_validity(
                config, dataset, args.external_validity_manifest, output
            )
        elif args.command == "run-literature-resources":
            from .formal_literature import run_literature_resource_gate

            result = run_literature_resource_gate(
                config, dataset, args.literature_resource_manifest, output
            )
        elif args.command == "run-rt-calibration":
            from .formal_rt_calibration import run_rt_calibration_gate

            result = run_rt_calibration_gate(
                config, dataset, args.rt_calibration_manifest, output
            )
        elif args.command == "run-shuffled-pair-control":
            from .formal_claim_controls import run_shuffled_pair_control

            result = _run_with_authenticated_upstream(
                config,
                dataset,
                output,
                lambda _upstream: run_shuffled_pair_control(
                    config, dataset, args.claim_control_manifest, output
                ),
            )
        elif args.command == "run-retention-audit":
            from .formal_claim_controls import run_retention_audit

            result = _run_with_authenticated_upstream(
                config,
                dataset,
                output,
                lambda _upstream: run_retention_audit(
                    config, dataset, args.claim_control_manifest, output
                ),
            )
        elif args.command == "assemble-claims":
            from .formal_claims import assemble_claim_evidence

            result = _run_with_authenticated_upstream(
                config,
                dataset,
                output,
                lambda _upstream: assemble_claim_evidence(config, dataset, output),
            )
        else:
            result = _run_authorized_full_chain(
                config,
                dataset,
                output,
                args,
                full_run_preflight,
            )
        if args.command == "prepare-full-run":
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "output": str(output),
                        "request": result["request"],
                        "request_sha256": result["request_sha256"],
                    },
                    sort_keys=True,
                )
            )
            return _result_exit_code(result)
        from .formal_evidence import evidence_context

        root_scientific_use = "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM"
        root_evidence = evidence_context(config, dataset, root_scientific_use)
        write_json(
            output / "manifest.json",
            {
                "schema_version": "csi-pairs-formal-root-manifest-v2.1-v6",
                **root_evidence,
                "files": artifact_manifest(output, evidence=root_evidence),
            },
        )
        print(json.dumps({"status": result.get("status", "success"), "output": str(output.resolve())}, sort_keys=True))
        return _result_exit_code(result)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    finally:
        if output_lock is not None:
            output_lock.release()


def _prepare_full_run(config, dataset, output, args, preflight):
    from .formal_data_verification import run_data_verification
    from .formal_external_validity import run_external_validity
    from .formal_literature import run_literature_resource_gate
    from .formal_qualification import run_formal_qualification
    from .formal_resources import verify_waibu_resources
    from .formal_rt_calibration import run_rt_calibration_gate
    from .formal_run_approval import write_approval_request

    resource_gate = verify_waibu_resources(
        Path(__file__).resolve().parent / "configs" / "waibu_resources_v1.json",
        Path(__file__).resolve().parents[1] / "waibu",
        output,
    )
    _require_preapproval_pass(resource_gate, "waibu resource verification", dataset)
    literature = _optional_arg(args, "literature_resource_manifest")
    if literature:
        _try_optional_stage(
            "G0 literature/resource gate",
            lambda: run_literature_resource_gate(config, dataset, literature, output),
        )
    rt_manifest = _optional_arg(args, "rt_calibration_manifest")
    if rt_manifest:
        _try_optional_stage(
            "independent RT calibration",
            lambda: run_rt_calibration_gate(config, dataset, rt_manifest, output),
        )
    verifier = _optional_arg(args, "verifier_manifest")
    verification = None
    if verifier:
        verification = _try_optional_stage(
            "independent data verification",
            lambda: run_data_verification(config, dataset, verifier, output),
        )
    if verification is not None and verification.get("passed") is True:
        qualification = run_formal_qualification(
            config,
            dataset,
            output,
            verification,
            data_verification_gate_path=output / "data_verification" / "gate.json",
        )
    else:
        qualification = run_formal_qualification(config, dataset, output, None)
    _require_preapproval_pass(
        qualification,
        "G1/G2 Response qualification",
        dataset,
        allow_fixture_failure=True,
    )
    external_validity = _optional_arg(args, "external_validity_manifest")
    if external_validity:
        _try_optional_stage(
            "G8 independent external validity",
            lambda: run_external_validity(
                config, dataset, external_validity, output
            ),
        )
    return write_approval_request(
        config,
        dataset,
        output,
        preflight,
        run_nonce=secrets.token_hex(32),
    )


def _run_authorized_full_chain(config, dataset, output, args, preflight):
    return _run_with_authenticated_upstream(
        config,
        dataset,
        output,
        lambda upstream: _run_authorized_full_chain_for_upstream(
            config,
            dataset,
            output,
            args,
            preflight,
            upstream,
        ),
    )


def _run_authorized_full_chain_for_upstream(
    config, dataset, output, args, preflight, upstream
):
    if upstream.migrated:
        qualification = read_strict_json(upstream.qualification_gate)
        factorial = read_strict_json(upstream.factorial_gate)
    else:
        from .formal_run_approval import (
            authenticate_prepared_run,
            mark_approval_accepted,
        )

        accepted = authenticate_prepared_run(
            config,
            dataset,
            output,
            preflight,
            args.approval_manifest,
        )
        mark_approval_accepted(output, accepted)
        qualification = read_strict_json(upstream.qualification_gate)

        from .formal_wrong_map import run_formal_wrong_map

        _require_full_stage(
            run_formal_wrong_map(config, dataset, output, qualification),
            "wrong-map control",
            dataset,
        )
        from .formal_factorial import run_formal_factorial

        factorial = run_formal_factorial(
            config,
            dataset,
            output,
            qualification,
            allow_nonscientific_fixture=bool(args.allow_nonscientific_fixture),
        )
        _require_full_stage(factorial, "four-arm factorial", dataset)

    evaluation = _run_evaluation_for_upstream(
        config,
        dataset,
        output,
        upstream,
        qualification,
        factorial,
    )

    _require_full_stage(
        evaluation,
        "formal evaluation",
        dataset,
    )
    from .formal_risk import run_risk_contract

    _require_full_stage(run_risk_contract(config, dataset, output), "risk contract", dataset)
    from .formal_path import run_path_audit

    _require_full_stage(
        run_path_audit(config, dataset, upstream.factorial_root, output),
        "path audit",
        dataset,
    )
    from .formal_external import run_external_baselines

    _require_full_stage(
        run_external_baselines(config, dataset, args.adapter_manifest, output),
        "external baselines",
        dataset,
    )
    from .formal_representation_baselines import run_representation_baselines

    _require_full_stage(
        run_representation_baselines(
            config,
            dataset,
            args.representation_baseline_config,
            output,
        ),
        "representation baselines",
        dataset,
    )
    from .formal_controls import run_resource_controls

    _require_full_stage(
        run_resource_controls(config, dataset, args.control_manifest, output),
        "resource controls",
        dataset,
    )
    from .formal_scene_id import run_scene_id_audit

    _require_full_stage(
        run_scene_id_audit(config, dataset, args.scene_id_manifest, output),
        "scene-ID audit",
        dataset,
    )
    from .formal_claim_controls import run_retention_audit, run_shuffled_pair_control

    _require_full_stage(
        run_shuffled_pair_control(
            config,
            dataset,
            args.shuffled_pair_manifest,
            output,
        ),
        "shuffled-pair control",
        dataset,
    )
    _require_full_stage(
        run_retention_audit(config, dataset, args.retention_manifest, output),
        "retention audit",
        dataset,
    )
    from .formal_claims import assemble_claim_evidence

    result = assemble_claim_evidence(config, dataset, output)
    _require_full_stage(result, "claim assembly", dataset)
    return result


def _run_evaluation_command(config, dataset, output, args):
    def invoke(upstream):
        if upstream.migrated and (args.qualification_gate or args.factorial_gate):
            raise RuntimeError("migrated evaluation forbids upstream path overrides")
        if upstream.migrated:
            qualification = None
            factorial = None
        else:
            qualification_path = (
                Path(args.qualification_gate)
                if args.qualification_gate
                else upstream.qualification_gate
            )
            factorial_path = (
                Path(args.factorial_gate)
                if args.factorial_gate
                else upstream.factorial_gate
            )
            qualification = read_strict_json(qualification_path)
            factorial = read_strict_json(factorial_path)
        return _run_evaluation_for_upstream(
            config,
            dataset,
            output,
            upstream,
            qualification,
            factorial,
        )

    return _run_with_authenticated_upstream(
        config,
        dataset,
        output,
        invoke,
    )


def _run_evaluation_for_upstream(
    config, dataset, output, upstream, qualification, factorial
):
    if upstream.migrated:
        return _run_migrated_evaluation(config, dataset, output, upstream)

    from .formal_evaluation import run_formal_evaluation

    return run_formal_evaluation(
        config,
        dataset,
        output,
        qualification,
        factorial,
    )


def _run_migrated_evaluation(config, dataset, output, upstream):
    migration = upstream.migration
    if migration is None:
        raise RuntimeError("migrated evaluation requires an authenticated migration")

    from .formal_evaluation_identity import build_migrated_evaluation_execution
    from .formal_evaluation_streaming import run_streaming_formal_evaluation

    execution = build_migrated_evaluation_execution(config, dataset, upstream)
    return run_streaming_formal_evaluation(
        config,
        dataset,
        output,
        upstream_root=migration.legacy_run_root,
        qualification_gate_path=upstream.qualification_gate,
        factorial_gate_path=upstream.factorial_gate,
        checkpoint_index_path=upstream.checkpoint_index,
        checkpoint_inventory_sha256=migration.checkpoint_inventory_sha256,
        run_identity=execution.identity,
        execution_devices=execution.devices,
        batch_size=execution.batch_size,
    )


def _run_with_legacy_read_lock(upstream, invoke):
    if not upstream.migrated:
        return invoke()
    migration = upstream.migration
    if migration is None:
        raise RuntimeError("migrated execution requires an authenticated migration")
    legacy_read_lock = _acquire_legacy_read_lock(migration.legacy_run_root)
    try:
        return invoke()
    finally:
        legacy_read_lock.release()


def _run_with_authenticated_upstream(config, dataset, output, invoke):
    from .formal_upstream import resolve_authenticated_upstream

    located = resolve_authenticated_upstream(config, dataset, output)
    if not located.migrated:
        return invoke(located)
    migration = located.migration
    if migration is None:
        raise RuntimeError("migrated execution requires an authenticated migration")
    locked_root = Path(migration.legacy_run_root).resolve()

    def reauthenticate_and_invoke():
        authenticated = resolve_authenticated_upstream(config, dataset, output)
        current = authenticated.migration
        if (
            not authenticated.migrated
            or current is None
            or Path(current.legacy_run_root).resolve() != locked_root
        ):
            raise RuntimeError(
                "migration identity changed while acquiring the legacy read lock"
            )
        return invoke(authenticated)

    return _run_with_legacy_read_lock(located, reauthenticate_and_invoke)


def _optional_arg(args, name):
    value = getattr(args, name, None)
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "None":
        return None
    return text


def _try_optional_stage(label, invoke):
    try:
        return invoke()
    except Exception as error:
        print(
            f"warning: optional {label} skipped ({type(error).__name__}: {error})",
            file=sys.stderr,
        )
        return None


def _require_preapproval_pass(
    result,
    label,
    dataset,
    *,
    allow_fixture_failure=False,
):
    if result.get("passed") is not True and not (
        dataset.is_fixture and allow_fixture_failure
    ):
        raise RuntimeError(f"{label} did not pass; no approval request was issued")


def _require_full_stage(result, label, dataset):
    if not isinstance(result, dict):
        raise RuntimeError(f"{label} returned no auditable stage result")
    status = result.get("status")
    if not isinstance(status, str) or not status:
        raise RuntimeError(f"{label} returned no auditable stage status")
    scientific_statuses = {"PASS": True, "FAIL": False}
    completion_statuses = {
        "COMPLETE",
        "DIAGNOSTIC_COMPLETE_NOT_DOMAIN_EVIDENCE",
    }
    passed = result.get("passed")
    if passed is not None and type(passed) is not bool:
        raise RuntimeError(f"{label} returned a malformed passed field")
    if status == "BLOCKED":
        if passed is not False:
            raise RuntimeError(f"{label} returned an inconsistent scientific status")
        if result.get("engineering_complete") is not True:
            raise RuntimeError(
                f"{label} is scientifically blocked and/or engineering-incomplete"
            )
        return
    if status not in {*scientific_statuses, *completion_statuses}:
        raise RuntimeError(f"{label} produced incomplete or invalid engineering output")
    if status in scientific_statuses and passed is not scientific_statuses[status]:
        raise RuntimeError(f"{label} returned an inconsistent scientific status")
    if status in completion_statuses and "passed" in result:
        raise RuntimeError(f"{label} returned an inconsistent completion status")


class _OutputLock:
    def __init__(self, path: Path, guard_path: Path, guard_handle) -> None:
        self.path = path
        self.guard_path = guard_path
        self.guard_handle = guard_handle
        self.released = False

    def is_file(self) -> bool:
        return self.path.is_file()

    def unlink(self, missing_ok: bool = False) -> None:
        self.release(missing_ok=missing_ok)

    def release(self, missing_ok: bool = True) -> None:
        if self.released:
            if not missing_ok:
                raise FileNotFoundError(self.path)
            return
        try:
            self.path.unlink(missing_ok=missing_ok)
        finally:
            fcntl.flock(self.guard_handle.fileno(), fcntl.LOCK_UN)
            self.guard_handle.close()
            self.released = True


class _LegacyReadLock:
    def __init__(self, guard_path: Path, handle) -> None:
        self.guard_path = guard_path
        self.handle = handle
        self.released = False

    def release(self) -> None:
        if self.released:
            return
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.released = True


def _acquire_legacy_read_lock(legacy_run_root: str | Path) -> _LegacyReadLock:
    root = Path(legacy_run_root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("authenticated legacy run root is missing or invalid")
    lock_path = root.parent / f".{root.name}.csi-pairs-operation.lock"
    guard_path = lock_path.with_name(f"{lock_path.name}.guard")
    if guard_path.is_symlink() or not guard_path.is_file():
        raise RuntimeError("legacy operation-lock guard is missing or invalid")
    descriptor = os.open(
        guard_path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    handle = os.fdopen(descriptor, "rb", buffering=0)
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(
            "legacy run still has an active writer; migrated execution is forbidden"
        ) from error
    return _LegacyReadLock(guard_path, handle)


def _acquire_output_lock(output_root: Path) -> _OutputLock:
    output_root = output_root.resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_root.parent / f".{output_root.name}.csi-pairs-operation.lock"
    guard_path = lock_path.with_name(f"{lock_path.name}.guard")
    if guard_path.is_symlink():
        raise RuntimeError(f"operation lock guard cannot be a symlink: {guard_path}")
    descriptor = os.open(
        guard_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    guard_handle = os.fdopen(descriptor, "r+b", buffering=0)
    try:
        fcntl.flock(guard_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        guard_handle.close()
        raise FileExistsError(
            f"refusing concurrent V2 output use; operation lock exists: {lock_path}"
        ) from error
    try:
        if lock_path.is_symlink():
            raise RuntimeError(f"operation lock cannot be a symlink: {lock_path}")
        owner = {
            "schema_version": "csi-pairs-v6-operation-lock-v2",
            "pid": os.getpid(),
            "output_root": str(output_root),
        }
        temporary = lock_path.with_name(
            f".{lock_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(
            json.dumps(owner, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="ascii",
        )
        os.replace(temporary, lock_path)
        return _OutputLock(lock_path, guard_path, guard_handle)
    except Exception:
        fcntl.flock(guard_handle.fileno(), fcntl.LOCK_UN)
        guard_handle.close()
        raise


def _reject_unsafe_mutating_output_argument(
    command: str, output_root: str | Path | None
) -> None:
    """Reject symlink roots where mutation relies on authenticated migration state."""

    if output_root is None:
        return
    candidate = Path(output_root)
    if not candidate.is_symlink():
        return
    if command == "all":
        raise RuntimeError("all output root cannot be a symlink")
    migration_root = candidate / "migration"
    if command == "run-evaluation" and (
        migration_root.exists() or migration_root.is_symlink()
    ):
        raise RuntimeError("migrated run-evaluation output root cannot be a symlink")


def _sionna_export_lock_root(output: Path) -> Path:
    output = output.resolve()
    return output.parent if output.name == "inputs" else output


def _reserve_command_output(command: str, output_root: Path) -> Path | None:
    relative_path = COMMAND_OUTPUT_PATHS.get(command)
    if relative_path is None:
        return None
    target = output_root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if command in SELF_RESERVING_DIRECTORY_COMMANDS:
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing to overwrite V2 output: {target}")
        return target
    try:
        if command in FILE_OUTPUT_COMMANDS:
            target.touch(exist_ok=False)
        else:
            target.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite V2 output: {target}") from error
    return target


def _reserve_full_run_output(output_root: Path) -> None:
    try:
        output_root.mkdir(parents=True, exist_ok=False)
        return
    except FileExistsError:
        pass
    entries = list(output_root.iterdir()) if output_root.is_dir() else []
    if (
        len(entries) != 1
        or entries[0].name != "inputs"
        or not entries[0].is_dir()
        or entries[0].is_symlink()
    ):
        raise FileExistsError(
            "refusing to reuse V2 full-run output directory except for a single "
            f"pre-staged inputs directory: {output_root}"
        )


def _result_exit_code(result: object) -> int:
    if not isinstance(result, dict):
        return 0
    if result.get("passed") is False:
        return 1
    if result.get("status") in {
        "FAIL",
        "BLOCKED",
        "INVALID",
        "QUALIFICATION_NO_GO",
        "DRY_RUN_FAIL_NOT_EVIDENCE",
        "INCOMPLETE_FAIL_CLOSED",
    }:
        return 1
    for key in ("gate_vector", "upstream_gates"):
        vector = result.get(key)
        if isinstance(vector, dict) and any(
            value in {"FAIL", "INVALID"} for value in vector.values()
        ):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
