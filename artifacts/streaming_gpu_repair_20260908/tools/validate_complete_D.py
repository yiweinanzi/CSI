"""Read-only independent validation of the one-unit frozen Streaming D output."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


EXPECTED_HEAD = "2ce24729814d1d7357fafb280ab10192ee3ace6b"
EXPECTED_DATA = "060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac"
EXPECTED_SOURCE = "0024cadc5d024175c3a6d2c3000e261d0061b70cc35480ea973c0f7c6a297fed"


def validate_counts(bounded, progress, telemetry):
    if bounded.get("status") != "PAUSED_AT_UNIT_BOUNDARY" or bounded.get("sota_ready") is not False:
        raise RuntimeError("D has no bounded, non-claim completion receipt")
    for value in (bounded, progress):
        if type(value.get("completed_units")) is not int or value["completed_units"] != 1 or value.get("total_units") != 1489:
            raise RuntimeError("D must complete exactly one original unit out of 1489")
    workers = telemetry.get("workers", {})
    if set(workers) != {"cuda:0"}:
        raise RuntimeError("D's declared single executing worker differs")
    worker = workers["cuda:0"]
    if (worker.get("arm"), worker.get("seed"), worker.get("checkpoint_index"), worker.get("unit_index")) != ("endpoint", 20270001, 0, 0):
        raise RuntimeError("D completed model telemetry belongs to another work unit")
    if worker.get("completed_training_models") != 17 or worker.get("total_training_models") != 17:
        raise RuntimeError("D does not have all 17 original model completions")
    phases = telemetry["observed_probe_timelines"]["cuda:0/checkpoint-00"]["phase_timings"]
    expected = {
        (probe, family) for probe in ("compatibility", "csi_only", "map_only", "scene_id_only", "edit_status_xor", "variant_id_matcher")
        for family in ("linear", "mlp2")
    } | {(probe, None) for probe in ("response", "without_map", "edit_only", "csi_only", "oracle_x")}
    observed = [(row.get("probe"), row.get("family")) for row in phases if row.get("phase") == "training_complete"]
    if len(observed) != 17 or set(observed) != expected:
        raise RuntimeError("D training completion timeline is incomplete or duplicated")
    return {"completed_units": 1, "total_units": 1489, "completed_training_models": 17}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "server", "config", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    root, server, output = (Path(value).resolve() for value in (args.run, args.server, args.output))
    if output.exists() or output == root or root in output.parents:
        raise RuntimeError("D validation receipt must be new and outside the evaluated run")
    bounded_path = root / "bounded_validation_result.json"
    if not bounded_path.is_file():
        print(json.dumps({"status": "NOT_READY", "reason": "D has not committed its bounded result", "sota_ready": False}))
        return 3
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=server, text=True).strip() != EXPECTED_HEAD:
        raise RuntimeError("Use the frozen D runtime for validation")
    if subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=server, text=True).strip():
        raise RuntimeError("Frozen D source has tracked changes")
    sys.path.insert(0, str(server))
    from formal_v2.formal_cli import _acquire_legacy_read_lock
    from formal_v2.formal_config import load_formal_config
    from formal_v2.formal_evaluation_repair import load_validation_corpus
    from formal_v2.formal_evaluation_resume import EvaluationResumeStore, EvaluationRunIdentity, write_atomic_json
    from formal_v2.formal_evaluation_streaming import (
        EVALUATION_TABLE_FIELDS, _contract_identity, _probe_contract_rows,
        _restore_probe_bundle, write_csv_fragment_stream,
    )
    from formal_v2.formal_io import read_strict_json, sha256_file
    from formal_v2.formal_evidence import configure_reproducible_runtime
    import formal_v2
    import numpy as np
    from types import SimpleNamespace
    from formal_v2.formal_evaluation_streaming import _dataset_response_output_dim

    if Path(formal_v2.__file__).resolve().parent != server / "formal_v2":
        raise RuntimeError("Validator imported a different evaluation runtime")
    configure_reproducible_runtime()
    lock = _acquire_legacy_read_lock(root)
    try:
        bounded = read_strict_json(bounded_path)
        progress = read_strict_json(root / "evaluation_state/evaluation_progress.json")
        telemetry = read_strict_json(root / "evaluation_state/probe_progress.json")
        counts = validate_counts(bounded, progress, telemetry)
        config = load_formal_config(args.config)
        if config["evaluation"]["probe_steps"] != 2000:
            raise RuntimeError("D probe update protocol changed")
        corpus_path = root / "validation_source_corpus/manifest.json"
        if Path(bounded["validation_corpus_manifest"]).resolve() != corpus_path:
            raise RuntimeError("D receipt points to a different corpus")
        arrays, manifest, bundle = load_validation_corpus(corpus_path, config, expected_sha256=bounded["validation_corpus_manifest_sha256"])
        identity = EvaluationRunIdentity.from_dict(read_strict_json(root / "execution_identity.json"))
        if identity.as_dict() != progress["run_identity"] or identity.as_dict() != manifest["execution_identity"]:
            raise RuntimeError("D progress, bundle, and execution identities differ")
        if identity.code_revision != EXPECTED_HEAD or identity.dataset_sha256 != EXPECTED_DATA or identity.source_tree_sha256 != EXPECTED_SOURCE:
            raise RuntimeError("D identity differs from the frozen experiment")
        if telemetry["run_id"] != identity.run_nonce:
            raise RuntimeError("D model telemetry belongs to another run")
        origin = read_strict_json(manifest["origin_receipt_path"])
        with np.load(origin["upstream"]["dataset_path"], allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
        checkpoint = manifest["checkpoint"]
        bundle = _restore_probe_bundle(
            Path(manifest["probe_bundle_path"]), config,
            seed=checkpoint["seed"], arm=checkpoint["arm"],
            expected_response_output_dim=_dataset_response_output_dim(SimpleNamespace(metadata=metadata)),
        )
        store = EvaluationResumeStore(root / "evaluation_state", identity)
        table_receipts = []
        for table, rows in _probe_contract_rows(bundle, config, seed=checkpoint["seed"], arm=checkpoint["arm"]).items():
            shard = store.load_completed_shard(_contract_identity(identity, checkpoint, table, 0), suffix=".csv.gz")
            if shard is None:
                raise RuntimeError("D has no completed contract table: " + table)
            expected = io.BytesIO()
            count = write_csv_fragment_stream(expected, rows, fieldnames=EVALUATION_TABLE_FIELDS[table], evidence=origin["evaluation_evidence"])
            if hashlib.sha256(expected.getvalue()).hexdigest() != sha256_file(shard.payload_path):
                raise RuntimeError("D saved contract table differs from its restored bundle: " + table)
            table_receipts.append({"table": table, "rows": count, "path": str(shard.payload_path), "sha256": sha256_file(shard.payload_path)})
        result = {
            "schema_version": "csi-pairs-independent-D-validation-v1",
            "status": "PASS_COMPLETE_ORIGINAL_UNIT", "validated_at": datetime.now(timezone.utc).isoformat(),
            **counts, "run": str(root), "run_identity": identity.as_dict(),
            "probe_steps": 2000, "tables": table_receipts,
            "validation_corpus_manifest_sha256": sha256_file(corpus_path),
            "probe_bundle_sha256": manifest["probe_bundle_sha256"],
            "captured_array_shapes": {key: list(value.shape) for key, value in arrays.items()},
            "scientific_use": "NON_CLAIM", "sota_ready": False,
            "original_parent_exit_code": None,
            "exit_note": "Original wrapper disappeared; this independent validation does not infer its exit code.",
            "scope": "One complete original probe-state unit and its tables; not all 1489 units or formal E.",
            "validator_source_sha256": sha256_file(__file__),
        }
        write_atomic_json(output, result)
        print(json.dumps(result, sort_keys=True))
        return 0
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
