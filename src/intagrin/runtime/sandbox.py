"""Isolated execution for agent-generated code (see `SandboxToolConfig` / a `tools[].type:
"sandbox"` entry) — the missing piece between the `coding-agent` template's coder/verifier loop
(which writes and reviews code) and actually *running* code an LLM produced.

Be precise about what this does and doesn't protect against, matching this codebase's existing
"say so directly rather than rounding up" posture (see the fuzzer's own keyword-heuristic
disclosure in testing/fuzzer.py):

WHAT THIS PROVIDES (reliable, stdlib-only, no new dependency):
  - A separate OS process, not code run in-process — a crash or infinite loop in the executed
    code can't take down the engine.
  - Resource limits: wall-clock timeout, plus CPU-time and address-space (memory) limits on POSIX
    via the `resource` module (a no-op on Windows, where that module doesn't exist).
  - A secret-free environment: the subprocess gets an explicit minimal env (PATH only), never a
    copy of the engine's own os.environ — so a leaked API key isn't sitting in the sandboxed
    process's environment for the executed code to read.
  - An ephemeral, isolated working directory, deleted afterward.
  - Argument-list subprocess invocation (never shell=True) — no shell-injection surface from the
    code string itself.

WHAT THIS DOES NOT PROVIDE (be honest about this, don't let a project think otherwise):
  - Filesystem isolation. The subprocess runs as the same OS user as the engine and can read/
    write anything that user can outside its own working directory. This is NOT a chroot/
    container/microVM boundary.
  - Network isolation. There is no firewall/namespace here — the process can make outbound
    network calls like any other process this user runs.
  - Protection against a truly adversarial payload. This reduces blast radius for a buggy or
    runaway script (the common case for LLM-generated code), not a security boundary for content
    you'd call hostile. For that, run this behind requires_approval and/or swap in a real
    container/microVM-based executor — this module is intentionally a single, easily-replaced
    function so that swap doesn't ripple through the rest of the codebase.
"""

import asyncio
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_LANGUAGE_COMMAND = {
    "python": lambda script_path: [sys.executable, str(script_path)],
    "bash": lambda script_path: ["/bin/bash", str(script_path)],
}

_MAX_OUTPUT_CHARS = 4000


def _preexec_limits(max_memory_mb: int | None, cpu_seconds: int):
    """Returns a preexec_fn applying POSIX rlimits in the child process before exec — None on
    platforms without the `resource` module (Windows), where this is silently skipped rather than
    failing the whole tool call over a best-effort limit."""
    try:
        import resource
    except ImportError:
        return None

    def _apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        if max_memory_mb is not None:
            limit_bytes = max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

    return _apply


def _truncate(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return text[:_MAX_OUTPUT_CHARS] + f"\n...[truncated, {len(text)} total chars]"


def _run_sync(
    code: str, language: str, timeout_seconds: int, max_memory_mb: int | None
) -> str:
    build_command = _LANGUAGE_COMMAND[language]
    workdir = tempfile.mkdtemp(prefix="intagrin-sandbox-")
    try:
        script_path = Path(workdir) / ("script.py" if language == "python" else "script.sh")
        script_path.write_text(code)

        try:
            proc = subprocess.run(
                build_command(script_path),
                cwd=workdir,
                env={"PATH": "/usr/bin:/bin:/usr/local/bin"},
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                preexec_fn=_preexec_limits(max_memory_mb, timeout_seconds),
            )
        except subprocess.TimeoutExpired:
            return f"Sandbox timed out after {timeout_seconds}s — the process was killed."

        exit_note = ""
        if proc.returncode < 0:
            exit_note = (
                f" (terminated by signal {-proc.returncode} — likely the CPU-time or "
                "memory limit, not a normal exit)"
            )
        parts = [f"Exit code: {proc.returncode}{exit_note}"]
        if proc.stdout:
            parts.append(f"stdout:\n{_truncate(proc.stdout)}")
        if proc.stderr:
            parts.append(f"stderr:\n{_truncate(proc.stderr)}")
        return "\n".join(parts)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


async def run_sandboxed_code(
    code: str, language: str, timeout_seconds: int, max_memory_mb: int | None
) -> str:
    """Runs `code` in a fresh subprocess (see module docstring for exactly what is and isn't
    isolated) and returns exit code + captured stdout/stderr as one string. Off the event loop via
    asyncio.to_thread — subprocess.run itself blocks."""
    return await asyncio.to_thread(_run_sync, code, language, timeout_seconds, max_memory_mb)
