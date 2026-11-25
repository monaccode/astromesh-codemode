import threading

import pytest

from astromesh_codemode.adapters import close_sync, run_sync
from tests.adapters._helpers import make_cm


def _bridge_threads():
    return [t for t in threading.enumerate() if t.name.startswith("astromesh-codemode-")]


def test_close_sync_closes_pool_and_thread():
    cm = make_cm()
    assert "3" in run_sync(cm, cm.run("len(await get_rows(n=3))"))
    assert "3" in run_sync(cm, cm.run("len(await get_rows(n=3))"))
    assert cm._pool is not None
    assert len(_bridge_threads()) == 1
    close_sync(cm)
    assert cm._pool is None
    assert _bridge_threads() == []
    close_sync(cm)  # twice is fine
    # a later sync call starts a fresh loop and pool
    assert "3" in run_sync(cm, cm.run("len(await get_rows(n=3))"))
    close_sync(cm)
    assert _bridge_threads() == []


def test_close_sync_without_sync_use_is_noop():
    cm = make_cm()
    close_sync(cm)
    assert _bridge_threads() == []


async def test_run_sync_inside_running_loop_is_refused():
    cm = make_cm()
    with pytest.raises(RuntimeError, match="running event loop"):
        run_sync(cm, cm.run("1"))
