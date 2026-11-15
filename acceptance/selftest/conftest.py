import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
pytest_plugins = ["pytester"]

from stub import State, make_server  # noqa: E402


@pytest.fixture
def stub():
    state = State()
    srv, url = make_server(state)
    yield state, url
    srv.shutdown()
    srv.server_close()
