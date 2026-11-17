"""Tests for logging_config: rendering, level, settings-backed config, request and job ids."""

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest
import structlog
from fastapi import FastAPI
from fastapi.testclient import TestClient

import logging_config
from config import Settings, settings
from logging_config import REQUEST_ID_HEADER, RequestIdMiddleware, configure_logging


@pytest.fixture(autouse=True)
def isolated_logging(monkeypatch):
    """Give each test a clean logging/structlog state and restore it afterwards."""
    root = logging.getLogger()
    saved = (list(root.handlers), root.level, structlog.get_config())
    arq_saved = {n: list(logging.getLogger(n).handlers) for n in logging_config._ARQ_LOGGERS}
    structlog.contextvars.clear_contextvars()
    monkeypatch.setattr(settings, "log_level", "INFO")
    monkeypatch.setattr(settings, "log_format", "json")
    yield
    structlog.contextvars.clear_contextvars()
    root.handlers[:] = saved[0]
    root.setLevel(saved[1])
    structlog.configure(**saved[2])
    for name, handlers in arq_saved.items():
        logging.getLogger(name).handlers[:] = handlers


def _lines(capsys) -> list[str]:
    return [line for line in capsys.readouterr().out.splitlines() if line.strip()]


def _app_logging_in_request() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/ping")
    async def ping():
        logging.getLogger("tests.ping").info("inside the request")
        return {"ok": True}

    return app


def _in_task(coro_fn):
    """Run ``coro_fn`` in its own task, as ARQ does for every job."""

    async def runner():
        return await asyncio.create_task(coro_fn())

    return asyncio.run(runner())


# ── rendering ────────────────────────────────────────────────────


def test_json_format_renders_one_json_object_per_line(capsys):
    configure_logging()
    logging.getLogger("tests.render").info("hello json")
    structlog.get_logger("tests.render").info("hello structlog", key="v")
    stdlib, native = [json.loads(line) for line in _lines(capsys)]
    assert stdlib["event"] == "hello json"
    assert stdlib["level"] == "info"
    assert stdlib["logger"] == "tests.render"
    assert "timestamp" in stdlib
    assert native["event"] == "hello structlog" and native["key"] == "v"


def test_console_format_is_not_json(capsys):
    settings.log_format = "console"
    configure_logging()
    logging.getLogger("tests.render").info("hello console")
    (line,) = _lines(capsys)
    assert "hello console" in line
    with pytest.raises(ValueError):
        json.loads(line)


def test_unknown_format_falls_back_to_json(capsys):
    settings.log_format = "fancy"
    configure_logging()
    logging.getLogger("tests.render").info("still json")
    (line,) = _lines(capsys)
    assert json.loads(line)["event"] == "still json"


def test_level_filters_lower_severity(capsys):
    settings.log_level = "warning"
    configure_logging()
    log = logging.getLogger("tests.level")
    log.info("dropped")
    log.warning("kept")
    events = [json.loads(line)["event"] for line in _lines(capsys)]
    assert events == ["kept"]
    assert logging.getLogger().level == logging.WARNING


def test_invalid_level_falls_back_to_info():
    settings.log_level = "loud"
    configure_logging()
    assert logging.getLogger().level == logging.INFO


# ── settings-backed configuration ───────────────────────────────


def test_defaults_are_unchanged(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    fresh = Settings(_env_file=None)
    assert fresh.log_level == "INFO"
    assert fresh.log_format == "json"


def test_dotenv_only_values_are_honoured(tmp_path, monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("LOG_LEVEL=debug\nLOG_FORMAT=console\n")
    loaded = Settings(_env_file=str(env_file))
    assert loaded.log_level == "debug" and loaded.log_format == "console"

    # configure_logging reads the settings object, so a .env-only value takes effect
    monkeypatch.setattr(settings, "log_level", loaded.log_level)
    monkeypatch.setattr(settings, "log_format", loaded.log_format)
    configure_logging()
    assert logging.getLogger().level == logging.DEBUG


def test_environment_overrides_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("LOG_LEVEL=debug\nLOG_FORMAT=console\n")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    monkeypatch.setenv("LOG_FORMAT", "json")
    loaded = Settings(_env_file=str(env_file))
    assert loaded.log_level == "ERROR" and loaded.log_format == "json"


# ── request id ───────────────────────────────────────────────────


def test_request_id_is_generated_echoed_and_logged_inside_the_request(capsys):
    configure_logging()
    resp = TestClient(_app_logging_in_request()).get("/ping")
    header = resp.headers[REQUEST_ID_HEADER]
    assert len(header) == 32
    logged = [json.loads(line) for line in _lines(capsys)]
    inside = next(r for r in logged if r["event"] == "inside the request")
    assert inside["request_id"] == header


def test_incoming_request_id_is_accepted_when_sane(capsys):
    configure_logging()
    resp = TestClient(_app_logging_in_request()).get("/ping", headers={REQUEST_ID_HEADER: "trace-abc_123.4:5"})
    assert resp.headers[REQUEST_ID_HEADER] == "trace-abc_123.4:5"
    inside = next(json.loads(line) for line in _lines(capsys) if "inside the request" in line)
    assert inside["request_id"] == "trace-abc_123.4:5"


@pytest.mark.parametrize("bad", ["good-id\n", "has space", "x" * 129, "semi;colon", 'quo"te', "<script>"])
def test_unsafe_incoming_request_id_is_replaced(bad):
    resp = TestClient(_app_logging_in_request()).get("/ping", headers={REQUEST_ID_HEADER: bad})
    assert resp.headers[REQUEST_ID_HEADER] != bad
    assert len(resp.headers[REQUEST_ID_HEADER]) == 32


def test_each_request_gets_its_own_id_and_context_is_cleared():
    client = TestClient(_app_logging_in_request())
    first = client.get("/ping").headers[REQUEST_ID_HEADER]
    second = client.get("/ping").headers[REQUEST_ID_HEADER]
    assert first != second
    assert "request_id" not in structlog.contextvars.get_contextvars()


def test_request_id_is_on_auth_rejections_and_public_probes(monkeypatch):
    import main

    monkeypatch.setattr(settings, "neuralscape_api_key", "a-test-only-key")
    monkeypatch.setattr(settings, "neuralscape_auth_disabled", False)
    client = TestClient(main.app, raise_server_exceptions=False)

    denied = client.get("/v1/memories", params={"user_id": "u"})
    assert denied.status_code == 401
    assert len(denied.headers[REQUEST_ID_HEADER]) == 32

    live = client.get("/health/live", headers={REQUEST_ID_HEADER: "probe-1"})
    assert live.status_code == 200
    assert live.headers[REQUEST_ID_HEADER] == "probe-1"


def test_unhandled_500_keeps_the_request_id_in_header_and_log_line(capsys, monkeypatch):
    import main

    @main.app.get("/_boom_for_test")
    async def boom():
        raise RuntimeError("kaboom")

    monkeypatch.setattr(settings, "neuralscape_auth_disabled", True)
    monkeypatch.setattr(settings, "neuralscape_api_key", "")
    monkeypatch.setattr(settings, "neuralscape_user_token_secret", "")
    configure_logging()
    try:
        resp = TestClient(main.app, raise_server_exceptions=False).get(
            "/_boom_for_test", headers={REQUEST_ID_HEADER: "boom-1"}
        )
    finally:
        main.app.router.routes[:] = [r for r in main.app.router.routes if getattr(r, "path", "") != "/_boom_for_test"]
    assert resp.status_code == 500
    assert resp.headers[REQUEST_ID_HEADER] == "boom-1"
    logged = [json.loads(line) for line in _lines(capsys)]
    unhandled = next(r for r in logged if r["event"].startswith("Unhandled error"))
    assert unhandled["request_id"] == "boom-1"
    assert "request_id" not in structlog.contextvars.get_contextvars()


# ── tracebacks ───────────────────────────────────────────────────


def test_exceptions_render_a_traceback_string_in_json(capsys):
    configure_logging()
    try:
        raise ValueError("boom-plain")
    except ValueError:
        logging.getLogger("tests.exc").exception("stdlib failure")
    try:
        raise ValueError("boom-native")
    except ValueError:
        structlog.get_logger("tests.exc").exception("native failure")
    plain, native = [json.loads(line) for line in _lines(capsys)]
    for record, marker in ((plain, "boom-plain"), (native, "boom-native")):
        assert isinstance(record["exception"], str)
        assert "Traceback" in record["exception"] and f"ValueError: {marker}" in record["exception"]


# ── worker job context ───────────────────────────────────────────


def test_job_context_is_bound_for_the_job_and_cleared_after(capsys, monkeypatch):
    import worker

    class FakeJob:
        def __init__(self, job_id, redis, _queue_name=None):
            self.job_id = job_id

        async def info(self):
            return SimpleNamespace(function="process_memory_store")

    monkeypatch.setattr(worker, "Job", FakeJob)
    configure_logging(worker=True)
    on_start = worker._make_on_job_start("test-queue")
    on_end = worker._make_after_job_end("test-queue")

    async def job():
        await on_start({"job_id": "job-42", "job_try": 2, "redis": object()})
        logging.getLogger("worker.test").info("working on it")
        await on_end({"job_id": "job-42"})
        logging.getLogger("worker.test").info("after the job")

    _in_task(job)
    logged = [json.loads(line) for line in _lines(capsys)]
    during = next(r for r in logged if r["event"] == "working on it")
    assert during["job_id"] == "job-42"
    assert during["function"] == "process_memory_store"
    assert during["queue"] == "test-queue"
    assert during["job_try"] == 2
    after = next(r for r in logged if r["event"] == "after the job")
    assert "job_id" not in after and "function" not in after


def test_job_binding_never_fails_the_job(monkeypatch):
    import worker

    class BrokenJob:
        def __init__(self, *args, **kwargs):
            pass

        async def info(self):
            raise ConnectionError("redis down")

    monkeypatch.setattr(worker, "Job", BrokenJob)
    on_start = worker._make_on_job_start("q")

    async def go():
        await on_start({"job_id": "job-7", "redis": object()})
        return structlog.contextvars.get_contextvars()

    bound = _in_task(go)
    assert bound["job_id"] == "job-7" and "function" not in bound


def test_all_three_worker_classes_install_the_hook():
    import worker

    for cls in (worker.WorkerSettings, worker.GraphWorkerSettings, worker.IngestWorkerSettings):
        assert callable(cls.on_job_start), cls
        assert callable(cls.after_job_end), cls
