from types import SimpleNamespace
import pytest
from formal_v2.formal_upstream import resolve_authenticated_upstream


def test_current_run_paths_and_no_escape(tmp_path):
    upstream = resolve_authenticated_upstream({}, SimpleNamespace(), tmp_path)
    assert upstream.factorial_root == tmp_path / "factorial"
    assert upstream.qualification_gate == tmp_path / "qualification/gate.json"
    assert not upstream.migrated
    with pytest.raises(ValueError, match="escape"):
        upstream.factorial_path("../outside")


def test_archived_migration_is_not_silently_reused(tmp_path):
    (tmp_path / "migration").mkdir()
    with pytest.raises(ValueError, match="read-only"):
        resolve_authenticated_upstream({}, SimpleNamespace(), tmp_path)
