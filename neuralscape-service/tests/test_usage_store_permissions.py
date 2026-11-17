"""The SQLite usage ledger holds tenant and task identifiers: owner-only files."""

import os
import stat

import pytest

from usage_store import SqliteUsageStore


@pytest.mark.parametrize("mask", [0o022, 0o002, 0o000])
def test_ledger_files_are_owner_only_whatever_the_umask(tmp_path, mask):
    old = os.umask(mask)
    try:
        store = SqliteUsageStore(str(tmp_path / "usage.db"))
        store.close()
        store = SqliteUsageStore(str(tmp_path / "usage.db"))
    finally:
        os.umask(old)
    names = [n for n in os.listdir(tmp_path) if n.startswith("usage.db")]
    assert "usage.db" in names
    for name in names:
        mode = stat.S_IMODE(os.stat(tmp_path / name).st_mode)
        assert mode & 0o077 == 0, (name, oct(mode))
    store.close()
