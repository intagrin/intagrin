"""runtime/sandbox.py's E2B backend (SandboxToolConfig.backend: "e2b") — run_sandboxed_code_e2b
must match run_sandboxed_code's own contract (see tests/test_sandbox.py): always return a
descriptive string, never raise, and report a non-zero exit the same informational way the local
backend does, even though the e2b SDK itself raises CommandExitException on that case internally.

Skipped entirely when the optional `e2b` package isn't installed (`pip install "intagrin[e2b]"`),
matching this repo's existing pattern for psycopg-gated tests (see pyproject.toml's
per-file-ignores comment) — no live E2B account or network access is exercised here, only the
SDK's own classes mocked at the network boundary.
"""

import asyncio
import sys

import pytest

e2b = pytest.importorskip("e2b")

from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402

from intagrin.errors import IntaGrinError  # noqa: E402
from intagrin.runtime.sandbox import run_sandboxed_code_e2b  # noqa: E402


def _fake_sandbox(run_result=None, run_side_effect=None):
    sandbox = MagicMock()
    sandbox.files.write = AsyncMock()
    sandbox.commands.run = AsyncMock(return_value=run_result, side_effect=run_side_effect)
    sandbox.kill = AsyncMock()
    return sandbox


def test_raises_ig_rt_010_when_package_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "e2b", None)
    with pytest.raises(IntaGrinError) as exc_info:
        asyncio.run(run_sandboxed_code_e2b("print(1)", "python", 10, None))
    assert exc_info.value.code == "IG-RT-010"


def test_happy_path_writes_script_and_reports_success():
    sandbox = _fake_sandbox(
        run_result=MagicMock(exit_code=0, stdout="hello\n", stderr="")
    )
    with patch("e2b.AsyncSandbox.create", new_callable=AsyncMock, return_value=sandbox):
        result = asyncio.run(run_sandboxed_code_e2b("print('hello')", "python", 10, None))

    assert "Exit code: 0" in result
    assert "hello" in result
    sandbox.files.write.assert_awaited_once_with("/tmp/script.py", "print('hello')")
    sandbox.commands.run.assert_awaited_once_with("python3 /tmp/script.py", timeout=10)
    sandbox.kill.assert_awaited_once()


def test_bash_language_uses_bash_script_and_interpreter():
    sandbox = _fake_sandbox(run_result=MagicMock(exit_code=0, stdout="hi\n", stderr=""))
    with patch("e2b.AsyncSandbox.create", new_callable=AsyncMock, return_value=sandbox):
        asyncio.run(run_sandboxed_code_e2b("echo hi", "bash", 10, None))

    sandbox.files.write.assert_awaited_once_with("/tmp/script.sh", "echo hi")
    sandbox.commands.run.assert_awaited_once_with("bash /tmp/script.sh", timeout=10)


def test_nonzero_exit_is_unwrapped_from_command_exit_exception_not_raised():
    exc = e2b.CommandExitException(stdout="partial output", stderr="boom", exit_code=3, error=None)
    sandbox = _fake_sandbox(run_side_effect=exc)
    with patch("e2b.AsyncSandbox.create", new_callable=AsyncMock, return_value=sandbox):
        result = asyncio.run(run_sandboxed_code_e2b("import sys; sys.exit(3)", "python", 10, None))

    assert "Exit code: 3" in result
    assert "partial output" in result
    assert "boom" in result
    sandbox.kill.assert_awaited_once()


def test_command_timeout_reported_honestly_not_raised():
    sandbox = _fake_sandbox(run_side_effect=e2b.TimeoutException("timed out"))
    with patch("e2b.AsyncSandbox.create", new_callable=AsyncMock, return_value=sandbox):
        result = asyncio.run(run_sandboxed_code_e2b("import time; time.sleep(100)", "python", 5, None))

    assert "timed out" in result.lower()
    sandbox.kill.assert_awaited_once()


def test_authentication_failure_reported_honestly_not_raised():
    with patch(
        "e2b.AsyncSandbox.create",
        new_callable=AsyncMock,
        side_effect=e2b.AuthenticationException("bad key"),
    ):
        result = asyncio.run(run_sandboxed_code_e2b("print(1)", "python", 10, None))

    assert "authentication failed" in result.lower()
    assert "E2B_API_KEY" in result


def test_sandbox_create_failure_reported_honestly_not_raised():
    with patch(
        "e2b.AsyncSandbox.create",
        new_callable=AsyncMock,
        side_effect=e2b.NotFoundException("template not found"),
    ):
        result = asyncio.run(run_sandboxed_code_e2b("print(1)", "python", 10, "bad-template"))

    assert "e2b sandbox error" in result.lower()


def test_sandbox_is_always_killed_even_when_files_write_raises():
    sandbox = _fake_sandbox()
    sandbox.files.write = AsyncMock(side_effect=e2b.SandboxException("disk full"))
    with patch("e2b.AsyncSandbox.create", new_callable=AsyncMock, return_value=sandbox):
        result = asyncio.run(run_sandboxed_code_e2b("print(1)", "python", 10, None))

    assert "e2b sandbox error" in result.lower()
    sandbox.kill.assert_awaited_once()


def test_template_is_passed_through_to_sandbox_create():
    sandbox = _fake_sandbox(run_result=MagicMock(exit_code=0, stdout="", stderr=""))
    with patch(
        "e2b.AsyncSandbox.create", new_callable=AsyncMock, return_value=sandbox
    ) as create_mock:
        asyncio.run(run_sandboxed_code_e2b("print(1)", "python", 10, "my-custom-template"))

    create_mock.assert_awaited_once_with(template="my-custom-template", timeout=20)
