'use strict';

// Shared harness for the monitor.html browser test suite (tests/browser/monitor.test.js).
//
// Deliberately reuses real framework code instead of reimplementing it in JS: scaffolding a
// project shells out to the actual `inta new`, and seeding session history shells out to a
// one-line Python script using the actual `SQLiteCheckpointer` (runtime/memory.py) — so a seeded
// session is byte-for-byte what a real run would have produced, with no drift risk from
// hand-writing the on-disk JSON/SQL shape here.

const { spawn, execFileSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const REPO_ROOT = path.resolve(__dirname, '..', '..');

/**
 * Scaffolds a fresh IntaGrin project via the real `inta new` command in a temp directory.
 * Returns the absolute path to the scaffolded project.
 */
function scaffoldProject(projectName = 'demo') {
  const parentDir = fs.mkdtempSync(path.join(os.tmpdir(), 'intagrin-monitor-test-'));
  // `inta new` creates `<projectName>/` relative to cwd, so cwd must be the temp parent dir —
  // but `uv run` needs to be pointed at this repo's own project (pyproject.toml/.venv) to resolve
  // the `inta` console script, since the temp dir has neither. `--project` decouples the two.
  execFileSync('uv', ['run', '--project', REPO_ROOT, 'inta', 'new', projectName], {
    cwd: parentDir,
    stdio: 'pipe',
  });
  return path.join(parentDir, projectName);
}

/**
 * Seeds a session's checkpointed history directly into the project's SQLite memory db, via the
 * real SQLiteCheckpointer — not a hand-written INSERT. `sessionId` is the caller-visible id (no
 * tenant prefix); this namespaces it with `global_tenant:`, matching what verify_monitor_auth
 * (server/monitor.py) resolves for a project's default `server.auth.type: none`.
 */
function seedSession(projectDir, sessionId, messages, state = {}) {
  const namespacedId = `global_tenant:${sessionId}`;
  const script = [
    'import json',
    'from intagrin.runtime.memory import SQLiteCheckpointer',
    'cp = SQLiteCheckpointer(".ai/memory.db")',
    `cp.save_checkpoint(${JSON.stringify(namespacedId)}, json.loads(${JSON.stringify(
      JSON.stringify(messages)
    )}), json.loads(${JSON.stringify(JSON.stringify(state))}))`,
  ].join('\n');
  execFileSync('uv', ['run', '--project', REPO_ROOT, 'python', '-c', script], {
    cwd: projectDir,
    stdio: 'pipe',
  });
}

/**
 * Seeds a run-log row directly via the real `record_run_log` (runtime/run_logger.py) — not a
 * hand-written INSERT — matching seedSession's philosophy. `sessionId` is namespaced the same way.
 */
function seedRunLog(projectDir, sessionId, fields = {}) {
  const namespacedId = `global_tenant:${sessionId}`;
  const payload = { session_id: namespacedId, ...fields };
  const script = [
    'import json',
    'from pathlib import Path',
    'from intagrin.runtime.run_logger import record_run_log',
    'from unittest.mock import MagicMock',
    'mem_cfg = MagicMock()',
    'mem_cfg.type = "sqlite"',
    'mem_cfg.db_path = None',
    `record_run_log(mem_cfg, Path("."), **json.loads(${JSON.stringify(JSON.stringify(payload))}))`,
  ].join('\n');
  execFileSync('uv', ['run', '--project', REPO_ROOT, 'python', '-c', script], {
    cwd: projectDir,
    stdio: 'pipe',
  });
}

/** Polls a URL until it responds with HTTP 200, or throws after timeoutMs. */
async function waitForServer(url, timeoutMs = 25000) {
  const start = Date.now();
  let lastError = null;
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url);
      if (res.status === 200) return;
    } catch (e) {
      lastError = e;
    }
    await new Promise((r) => setTimeout(r, 300));
  }
  throw new Error(
    `Server at ${url} did not become ready within ${timeoutMs}ms` +
      (lastError ? ` (last error: ${lastError.message})` : '')
  );
}

/**
 * Spawns `inta monitor --port <port>` against projectDir and waits until it's serving. Returns
 * the child process — pass to stopMonitor() during teardown. Captures stdout/stderr so a startup
 * failure has a useful message instead of "server never came up".
 */
async function startMonitor(projectDir, port) {
  // `detached: true` puts this process in its own process group (pgid == its own pid), so
  // stopMonitor() can kill the whole tree via kill(-pgid) — `uv run` spawns `inta` as a real
  // child rather than exec'ing into it, so killing only proc.pid leaves an orphaned server
  // behind, still bound to the port, breaking every subsequent test run on the same port.
  const proc = spawn('uv', ['run', '--project', REPO_ROOT, 'inta', 'monitor', '--port', String(port)], {
    cwd: projectDir,
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: true,
  });
  let output = '';
  proc.stdout.on('data', (d) => {
    output += d.toString();
  });
  proc.stderr.on('data', (d) => {
    output += d.toString();
  });
  proc.getOutput = () => output;

  try {
    await waitForServer(`http://localhost:${port}/`);
  } catch (e) {
    stopMonitor(proc);
    throw new Error(`${e.message}\n--- monitor output ---\n${output}`);
  }
  return proc;
}

function stopMonitor(proc) {
  if (!proc || proc.killed) return;
  try {
    // Negative pid targets the whole process group (see the `detached: true` note above) —
    // kills `uv` and the `inta monitor` child it spawned, not just the immediate process.
    process.kill(-proc.pid, 'SIGKILL');
  } catch (e) {
    // already gone
  }
}

/** Best-effort recursive removal of a scaffolded project's parent temp dir. */
function cleanupProject(projectDir) {
  try {
    fs.rmSync(path.dirname(projectDir), { recursive: true, force: true });
  } catch (e) {
    // best-effort cleanup; a leftover temp dir isn't worth failing the test run over
  }
}

module.exports = {
  REPO_ROOT,
  scaffoldProject,
  seedSession,
  seedRunLog,
  waitForServer,
  startMonitor,
  stopMonitor,
  cleanupProject,
};
