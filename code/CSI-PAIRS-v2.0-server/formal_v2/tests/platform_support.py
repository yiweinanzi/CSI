"""Explicit test operations unavailable to unprivileged Windows accounts."""
import pytest


def symlink_or_skip(path, *args, **kwargs):
    try:
        return path.symlink_to(*args, **kwargs)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows account lacks symbolic-link privilege")
        raise
