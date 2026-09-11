import copy
import importlib
from types import SimpleNamespace

import pytest

from formal_v2 import experiment_runtime as runtime
from formal_v2.formal_cli import main
from formal_v2.formal_fixture import write_nonscientific_fixture


def test_runtime_records_versions_without_install_receipts(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.sys, "prefix", str(tmp_path))
    runtime.configure_reproducible_runtime()
    record = runtime.runtime_provenance()
    assert runtime.validate_runtime_provenance(record) == record
    assert record["dependency_record_kind"] == "installed-version-summary"
    assert "installer_report_path" not in record
    changed = copy.deepcopy(record)
    changed["installed_distributions"]["numpy"] = "changed"
    with pytest.raises(ValueError, match="hash mismatch"):
        runtime.validate_runtime_provenance(changed)
    changed["schema_version"] = "csi-pairs-runtime-provenance-v3"
    with pytest.raises(ValueError, match="historical checkout"):
        runtime.validate_runtime_provenance(changed)


def test_inspect_fixture_and_refuse_overwrite(tmp_path):
    dataset = write_nonscientific_fixture(tmp_path / "input.npz")
    output = tmp_path / "inspect"
    args = ["inspect-data", "--dataset", str(dataset), "--output", str(output), "--allow-nonscientific-fixture"]
    assert main(args) == 0
    result = (output / "data_contract.json").read_bytes()
    assert main(args) == 2
    assert (output / "data_contract.json").read_bytes() == result


def test_all_retained_experiment_backends_import():
    for name in ("paper_train", "paper_core", "paper_suite", "formal_factorial",
                 "formal_evaluation_streaming", "formal_external", "formal_controls",
                 "formal_claim_controls", "formal_risk", "formal_representation_baselines",
                 "external_adapters.wigatr_adapter", "external_adapters.pmnet_adapter",
                 "external_adapters.shuffled_pair_control", "external_adapters.retention_control"):
        importlib.import_module("formal_v2." + name)


def test_fixture_extension_cannot_bypass_overwrite_check(tmp_path):
    path = tmp_path / "fixture.npz"
    path.write_bytes(b"existing dataset")
    assert main(["make-fixture", "--output", str(tmp_path / "fixture")]) == 2
    assert path.read_bytes() == b"existing dataset"


def test_source_identity_accepts_a_dirty_worktree(monkeypatch):
    # Resume is bound to source bytes; an uncommitted checkout needs no approval.
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="a" * 40 + "\n"))
    record = runtime.source_identity()
    assert record["git_commit"] == "a" * 40
    assert len(record["source_tree_sha256"]) == 64
