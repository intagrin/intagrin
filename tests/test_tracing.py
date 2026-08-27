import asyncio
import json

import pytest
from typer.testing import CliRunner

from intagrin.cli import app
from intagrin.tracing.console import (
    EventStreamer,
    LogLevel,
    Tracer,
    clear_trace_context,
    set_trace_context,
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def reset_tracer():
    Tracer.set_level(LogLevel.NORMAL)
    Tracer.set_json_mode(False)
    clear_trace_context()
    yield
    Tracer.set_level(LogLevel.NORMAL)
    Tracer.set_json_mode(False)
    clear_trace_context()


def _capture_error_events():
    events = []
    orig_emit = EventStreamer.emit

    def spy(event_type, data):
        if event_type == "error":
            events.append(data)
        return orig_emit(event_type, data)

    EventStreamer.emit = staticmethod(spy)
    return events, orig_emit


def test_log_error_auto_captures_traceback_inside_except_block(capsys):
    try:
        raise ValueError("boom")
    except Exception:
        Tracer.log_error("something broke")
    out = capsys.readouterr().out
    assert "Traceback" in out
    assert "ValueError: boom" in out


def test_log_error_has_no_traceback_outside_except_block(capsys):
    Tracer.log_error("just a message, no active exception")
    out = capsys.readouterr().out
    assert "Traceback" not in out


def test_log_error_exc_info_false_forces_no_traceback(capsys):
    try:
        raise ValueError("boom")
    except Exception:
        Tracer.log_error("suppressed", exc_info=False)
    out = capsys.readouterr().out
    assert "Traceback" not in out


def test_log_error_emits_to_event_streamer_regardless_of_json_mode():
    events, orig_emit = _capture_error_events()
    try:
        Tracer.log_error("visible to dashboard")
        assert len(events) == 1
        assert events[0]["message"] == "visible to dashboard"
    finally:
        EventStreamer.emit = orig_emit


def test_quiet_level_suppresses_steps_but_not_errors(capsys):
    Tracer.set_level(LogLevel.QUIET)
    Tracer.log_step("should be silent", "details")
    assert capsys.readouterr().out == ""

    Tracer.log_error("errors always print")
    assert "errors always print" in capsys.readouterr().out


def test_json_mode_emits_valid_json_lines(capsys):
    Tracer.set_json_mode(True)
    Tracer.log_step("step_name", "step details")
    out = capsys.readouterr().out.strip()
    record = json.loads(out)
    assert record["event"] == "step"
    assert record["message"] == "step details"


def test_trace_context_tags_console_output(capsys):
    set_trace_context(session_id="sess_42", agent_name="billing")
    Tracer.log_step("tagged step", "x")
    out = capsys.readouterr().out
    assert "sess_42" in out
    assert "billing" in out


def test_trace_context_propagates_to_event_streamer():
    set_trace_context(session_id="sess_99", agent_name="triage")
    q = EventStreamer.subscribe()
    try:
        Tracer.log_step("x", "y")
        payload = asyncio.run(q.get())
        assert payload["context"] == {"session_id": "sess_99", "agent": "triage"}
    finally:
        EventStreamer.unsubscribe(q)


def test_broken_config_surfaces_full_traceback_through_cli(tmp_path, monkeypatch):
    """Regression test: top-level CLI handlers used to swallow tracebacks with a bare
    `console.print(f"...{e}")`. A real config parse failure must now show the actual exception
    chain, not just its string message."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ai.yaml").write_text("not: valid: yaml: [\n")

    result = runner.invoke(app, ["run", "some_workflow"])

    assert result.exit_code != 0
    assert "Traceback" in result.stdout
    assert "ParserError" in result.stdout or "ScannerError" in result.stdout


def test_json_logs_flag_surfaces_traceback_as_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ai.yaml").write_text("not: valid: yaml: [\n")

    result = runner.invoke(app, ["--json-logs", "run", "some_workflow"])

    assert result.exit_code != 0
    json_lines = [line for line in result.stdout.splitlines() if line.strip().startswith("{")]
    assert json_lines, f"expected at least one JSON log line, got:\n{result.stdout}"
    record = json.loads(json_lines[-1])
    assert record["level"] == "error"
    assert record.get("traceback")
