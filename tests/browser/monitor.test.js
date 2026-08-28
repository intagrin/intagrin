'use strict';

// Browser regression suite for server/templates/monitor.html — the Monitor dashboard's React UI,
// loaded via CDN React + Babel-standalone with no build step and (until this suite) zero
// automated coverage. Covers, specifically: the Playground silently showing an empty chat on
// load instead of a session's real saved history, and the Architect chat's first-load session-id
// mismatch that silently swallowed every message sent in a browser's first-ever session. Neither
// test needs a real LLM call — both are pure UI-state bugs, reproduced by seeding state directly
// (a SQLite checkpoint row, or localStorage) and observing what renders.

const test = require('node:test');
const assert = require('node:assert/strict');
const puppeteer = require('puppeteer');

const {
  scaffoldProject,
  seedSession,
  seedRunLog,
  startMonitor,
  stopMonitor,
  cleanupProject,
} = require('./helpers');

const PORT = 8420;
const BASE_URL = `http://localhost:${PORT}`;
const SEEDED_SESSION_ID = 'playground';
const SEEDED_MESSAGES = [
  { role: 'user', content: 'Hello, is my history visible?' },
  { role: 'assistant', content: 'Yes — this was saved before the page ever loaded.' },
];
const SECOND_SESSION_ID = 'refund-review-42';
const SECOND_SESSION_MESSAGES = [
  { role: 'user', content: 'Please process a refund for order 42.' },
  { role: 'assistant', content: 'This refund exceeds the auto-approve limit and needs review.' },
];
const APPROVAL_SESSION_ID = 'approval-demo';
const APPROVAL_SESSION_MESSAGES = [
  { role: 'user', content: 'Book the hotel please.' },
  {
    role: 'assistant',
    tool_calls: [
      { id: 'call_1', type: 'function', function: { name: 'book_hotel', arguments: '{}' } },
    ],
  },
  {
    role: 'tool',
    tool_call_id: 'call_1',
    name: 'book_hotel',
    content: "Operation 'book_hotel' is paused awaiting human approval.",
  },
];

let projectDir;
let monitorProc;
let browser;

test.before(async () => {
  projectDir = scaffoldProject('demo');
  seedSession(projectDir, SEEDED_SESSION_ID, SEEDED_MESSAGES, {
    _metrics: { total_tokens: 42, total_cost: 0.0007 },
  });
  seedSession(projectDir, SECOND_SESSION_ID, SECOND_SESSION_MESSAGES, {
    _metrics: { total_tokens: 10, total_cost: 0.0002 },
  });
  seedRunLog(projectDir, SEEDED_SESSION_ID, {
    endpoint: '/chat',
    agent: 'triage',
    status: 'completed',
    error: null,
    tokens_delta: 42,
    cost_delta: 0.0007,
    total_tokens: 42,
    total_cost: 0.0007,
    message_count: 2,
    latency_ms: 812,
  });
  seedRunLog(projectDir, SECOND_SESSION_ID, {
    endpoint: '/resume',
    agent: 'billing',
    status: 'error',
    error: 'Refund gateway timed out after 30s',
    tokens_delta: 5,
    cost_delta: 0.0001,
    total_tokens: 15,
    total_cost: 0.0003,
    message_count: 3,
    latency_ms: 30210,
  });
  monitorProc = await startMonitor(projectDir, PORT);
  browser = await puppeteer.launch({ headless: true, args: ['--no-sandbox'] });
});

test.after(async () => {
  if (browser) await browser.close();
  stopMonitor(monitorProc);
  if (projectDir) cleanupProject(projectDir);
});

test('dashboard loads with no page errors', async () => {
  const page = await browser.newPage();
  const pageErrors = [];
  page.on('pageerror', (err) => pageErrors.push(err.toString()));

  try {
    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('body', { timeout: 10000 });
    // Give React + the initial /api/config, /api/memory fetches a moment to settle.
    await new Promise((r) => setTimeout(r, 1500));

    assert.deepEqual(pageErrors, [], `expected no uncaught page errors, got: ${pageErrors.join('; ')}`);

    const bodyText = await page.evaluate(() => document.body.innerText);
    assert.ok(bodyText.length > 0, 'expected the dashboard to render some content');
  } finally {
    await page.close();
  }
});

test('Playground shows a seeded session\'s history on load, with no message sent', async () => {
  const page = await browser.newPage();
  try {
    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
      () => document.body.innerText.includes('No messages in this session yet.') ||
        document.body.innerText.includes('Hello, is my history visible?'),
      { timeout: 15000 }
    );

    const bodyText = await page.evaluate(() => document.body.innerText);
    assert.ok(
      bodyText.includes('Hello, is my history visible?'),
      'expected the seeded user message to render in the Playground on load'
    );
    assert.ok(
      bodyText.includes('Yes — this was saved before the page ever loaded.'),
      'expected the seeded assistant reply to render in the Playground on load'
    );
    assert.ok(
      !bodyText.includes('No messages in this session yet.'),
      'the empty-state placeholder must not show once real history has loaded'
    );
  } finally {
    await page.close();
  }
});

test('Playground for a different, never-seeded session stays empty (history load is scoped correctly)', async () => {
  const page = await browser.newPage();
  try {
    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    // The "Active Session" label renders visually uppercase via CSS text-transform, so
    // innerText reports it as "ACTIVE SESSION" even though the JSX source is mixed-case.
    await page.waitForFunction(
      () => document.body.innerText.toUpperCase().includes('ACTIVE SESSION'),
      { timeout: 15000 }
    );

    // Switch to a fresh, never-seeded session via the real "+ New Session" button, which also
    // clears displayed messages (setMessages([])).
    const clicked = await page.evaluate(() => {
      const btn = [...document.querySelectorAll('button')].find((b) =>
        b.textContent.includes('New Session')
      );
      if (!btn) return false;
      btn.click();
      return true;
    });
    assert.ok(clicked, 'expected a "+ New Session" button in the Playground header');

    await page.waitForFunction(
      () => document.body.innerText.includes('No messages in this session yet.'),
      { timeout: 10000 }
    );

    const bodyText = await page.evaluate(() => document.body.innerText);
    assert.ok(
      !bodyText.includes('Hello, is my history visible?'),
      'a different session id must never show another session\'s history'
    );
  } finally {
    await page.close();
  }
});

test('Playground session selector finds and switches to a session via search', async () => {
  const page = await browser.newPage();
  try {
    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
      () => document.body.innerText.toUpperCase().includes('ACTIVE SESSION'),
      { timeout: 15000 }
    );

    // Open the session selector dropdown (the button showing the current session id).
    const opened = await page.evaluate((currentId) => {
      const btn = [...document.querySelectorAll('button')].find((b) =>
        b.textContent.includes(currentId)
      );
      if (!btn) return false;
      btn.click();
      return true;
    }, SEEDED_SESSION_ID);
    assert.ok(opened, 'expected a session-selector button showing the active session id');

    await page.waitForSelector('input[placeholder="Search sessions..."]', { timeout: 5000 });

    // Search for a substring that only matches the second seeded session.
    await page.type('input[placeholder="Search sessions..."]', 'refund-review');

    await page.waitForFunction(
      (id) => document.body.innerText.includes(id),
      { timeout: 5000 },
      SECOND_SESSION_ID
    );
    const bodyAfterSearch = await page.evaluate(() => document.body.innerText);
    assert.ok(
      bodyAfterSearch.includes(SECOND_SESSION_ID) &&
        !bodyAfterSearch.includes('No matching sessions.'),
      'expected the search to surface the matching session in the dropdown'
    );

    // Click the matching session row to select it.
    const selected = await page.evaluate((id) => {
      const row = [...document.querySelectorAll('button')].find((b) =>
        b.textContent.includes(id)
      );
      if (!row) return false;
      row.click();
      return true;
    }, SECOND_SESSION_ID);
    assert.ok(selected, 'expected a clickable row for the searched session');

    await page.waitForFunction(
      () => document.body.innerText.includes('Please process a refund for order 42.'),
      { timeout: 10000 }
    );
    const bodyAfterSelect = await page.evaluate(() => document.body.innerText);
    assert.ok(
      bodyAfterSelect.includes('Please process a refund for order 42.') &&
        bodyAfterSelect.includes('This refund exceeds the auto-approve limit and needs review.'),
      'expected the selected session\'s history to load into the Playground'
    );
    assert.ok(
      !bodyAfterSelect.includes('Hello, is my history visible?'),
      'switching sessions must not leave the previous session\'s messages on screen'
    );
  } finally {
    await page.close();
  }
});

test('Architect chat self-heals a mismatched first-load session id', async () => {
  const context = await browser.createBrowserContext();
  const page = await context.newPage();
  try {
    // Reproduces the exact bug condition: architectSessions and activeArchSessionId computed
    // from two independent Date.now() calls that didn't match. Seeded before any app script runs.
    const realSessionId = 'session-A-the-only-real-one';
    const mismatchedActiveId = 'session-B-does-not-match-anything';

    // architectSessions/activeArchSessionId are namespaced per-project (see monitor.html) —
    // seed under 'demo', the name scaffoldProject('demo') gives this suite's project.
    await page.evaluateOnNewDocument(
      (sessions, activeId) => {
        localStorage.setItem('architectSessions:demo', JSON.stringify(sessions));
        localStorage.setItem('activeArchSessionId:demo', activeId);
      },
      [{ id: realSessionId, name: 'Session 1', messages: [] }],
      mismatchedActiveId
    );

    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    // Proof the per-project load effect has actually run, not just that the page painted —
    // it only fires once /api/config resolves and the project name is known.
    await page.waitForFunction(
      () => localStorage.getItem('activeArchSessionId:demo') !== null,
      { timeout: 15000 }
    );
    // The self-heal effect runs right after; give it a beat to fire and persist back.
    await new Promise((r) => setTimeout(r, 500));

    const correctedId = await page.evaluate(() => localStorage.getItem('activeArchSessionId:demo'));
    assert.equal(
      correctedId,
      realSessionId,
      'a mismatched activeArchSessionId must self-heal to a real session id, not stay orphaned'
    );
  } finally {
    await page.close();
    await context.close();
  }
});

test('Architect chat history does not leak across different projects sharing one origin', async () => {
  const context = await browser.createBrowserContext();
  const page = await context.newPage();
  try {
    // `inta monitor` defaults to the same port (3000) for every project, so two differently-named
    // projects opened in the same browser land on the same origin — before architectSessions was
    // namespaced per-project (see monitor.html), they shared one global localStorage key and a
    // second project's Architect chat showed the first project's history. Seed a DIFFERENT
    // project's key here ('some-other-project', never 'demo') to prove this project's own load
    // never picks it up.
    await page.evaluateOnNewDocument(() => {
      localStorage.setItem(
        'architectSessions:some-other-project',
        JSON.stringify([
          {
            id: 'leaked-session',
            name: 'Session 1',
            messages: [{ role: 'user', content: 'Secret question about some-other-project.' }],
          },
        ])
      );
      localStorage.setItem('activeArchSessionId:some-other-project', 'leaked-session');
    });

    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    // Proof this project's ('demo') own per-project load/persist cycle has actually run.
    await page.waitForFunction(
      () => localStorage.getItem('architectSessions:demo') !== null,
      { timeout: 15000 }
    );

    const demoSessions = await page.evaluate(() => localStorage.getItem('architectSessions:demo'));
    assert.ok(
      !demoSessions.includes('Secret question about some-other-project'),
      "another project's Architect chat history must never leak into this project's own sessions"
    );
    assert.ok(
      !demoSessions.includes('leaked-session'),
      "another project's session id must never become this project's active/loaded session"
    );

    const otherProjectUntouched = await page.evaluate(() =>
      localStorage.getItem('architectSessions:some-other-project')
    );
    assert.ok(
      otherProjectUntouched && otherProjectUntouched.includes('leaked-session'),
      "loading this project must not overwrite another project's stored Architect history either"
    );
  } finally {
    await page.close();
    await context.close();
  }
});

test('Logs tab renders seeded run logs with created time, cost, and context size', async () => {
  const page = await browser.newPage();
  try {
    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
      () => document.body.innerText.toUpperCase().includes('ACTIVE SESSION'),
      { timeout: 15000 }
    );

    const clicked = await page.evaluate(() => {
      const btn = [...document.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Logs');
      if (!btn) return false;
      btn.click();
      return true;
    });
    assert.ok(clicked, 'expected a "Logs" nav button');

    await page.waitForFunction(
      // The heading renders visually uppercase via CSS text-transform, so innerText reports it
      // as "API RUN LOGS" even though the JSX source is mixed-case (same gotcha as "ACTIVE
      // SESSION" above).
      () => document.body.innerText.toUpperCase().includes('API RUN LOGS'),
      { timeout: 10000 }
    );
    await page.waitForFunction(
      () => document.body.innerText.includes('/chat') && document.body.innerText.includes('/resume'),
      { timeout: 10000 }
    );

    const bodyText = await page.evaluate(() => document.body.innerText);
    // Completed run: session id, cost, context size.
    assert.ok(bodyText.includes(SEEDED_SESSION_ID), 'expected the completed run\'s session id to render');
    assert.ok(bodyText.includes('0.0007') || bodyText.includes('.00070'), 'expected the completed run\'s cost to render');
    assert.ok(bodyText.includes('2 msgs'), 'expected the completed run\'s context size (message_count) to render');
    // Error run: distinct error message, second session id.
    assert.ok(bodyText.includes(SECOND_SESSION_ID), 'expected the error run\'s session id to render');
    assert.ok(
      bodyText.includes('Refund gateway timed out'),
      'expected the error run\'s error message to render'
    );
    // Every row has a created time.
    assert.ok(!/—\s*—/.test(bodyText), 'did not expect every timestamp to be missing');
  } finally {
    await page.close();
  }
});

test('Logs search filters by session id / status / error text', async () => {
  const page = await browser.newPage();
  try {
    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
      () => document.body.innerText.toUpperCase().includes('ACTIVE SESSION'),
      { timeout: 15000 }
    );

    await page.evaluate(() => {
      const btn = [...document.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Logs');
      btn.click();
    });
    await page.waitForFunction(
      () => document.body.innerText.toUpperCase().includes('API RUN LOGS'),
      { timeout: 10000 }
    );
    await page.waitForSelector('input[placeholder="Search logs..."]', { timeout: 5000 });

    await page.type('input[placeholder="Search logs..."]', 'gateway timed out');

    await page.waitForFunction(
      () => !document.body.innerText.includes('playground') || document.body.innerText.includes('refund-review-42'),
      { timeout: 5000 }
    );
    const bodyText = await page.evaluate(() => document.body.innerText);
    assert.ok(bodyText.includes(SECOND_SESSION_ID), 'expected the matching error row to remain visible');
    assert.ok(!bodyText.includes(SEEDED_SESSION_ID), 'expected the non-matching completed row to be filtered out');
  } finally {
    await page.close();
  }
});

test('Theme toggle switches to light mode and persists across a reload', async () => {
  // Isolated context + explicit media-feature emulation: the app only falls back to 'dark' when
  // prefers-color-scheme does NOT match light (monitor.html), which otherwise depends on the
  // runner's own ambient OS/browser default — not something this test should depend on. Without
  // pinning it, this passed locally (non-light ambient default) but failed in CI, whose headless
  // Chrome default apparently matches light. Same isolation reasoning as the dedicated
  // prefers-color-scheme test below, which already emulates explicitly for exactly this reason.
  const context = await browser.createBrowserContext();
  const page = await context.newPage();
  try {
    await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'dark' }]);
    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
      () => document.body.innerText.toUpperCase().includes('ACTIVE SESSION'),
      { timeout: 15000 }
    );

    const initialClass = await page.evaluate(() => document.documentElement.className);
    assert.equal(initialClass, 'dark', 'expected dark mode by default with nothing saved yet');

    const clicked = await page.evaluate(() => {
      const btn = [...document.querySelectorAll('button')].find(
        (b) => b.title && b.title.includes('light mode')
      );
      if (!btn) return false;
      btn.click();
      return true;
    });
    assert.ok(clicked, 'expected a theme toggle button offering to switch to light mode');

    await page.waitForFunction(
      () => document.documentElement.className === 'light',
      { timeout: 5000 }
    );

    const savedTheme = await page.evaluate(() => localStorage.getItem('theme'));
    assert.equal(savedTheme, 'light', 'expected the choice to be persisted to localStorage');

    // Reload — must come back in light mode, not reset to dark.
    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
      () => document.body.innerText.toUpperCase().includes('ACTIVE SESSION'),
      { timeout: 15000 }
    );
    const classAfterReload = await page.evaluate(() => document.documentElement.className);
    assert.equal(classAfterReload, 'light', 'expected the theme choice to survive a reload');
  } finally {
    await page.close();
    await context.close();
  }
});

test('Theme respects prefers-color-scheme on a first visit with nothing saved', async () => {
  const context = await browser.createBrowserContext();
  const page = await context.newPage();
  try {
    await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'light' }]);
    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
      () => document.body.innerText.toUpperCase().includes('ACTIVE SESSION'),
      { timeout: 15000 }
    );
    const themeClass = await page.evaluate(() => document.documentElement.className);
    assert.equal(
      themeClass,
      'light',
      'expected a first-ever visit with an OS light preference and nothing saved to default to light'
    );
  } finally {
    await page.close();
    await context.close();
  }
});

// Dynamic (spawn_agent) agents don't exist in ai.yaml, so the only way they ever become visible
// is the live SSE pipe (/api/stream/events) — these two tests fake that endpoint's response at
// the network layer (real end-to-end coverage would need a live spawn_agent-capable session and
// a real LLM call, which the rest of this suite deliberately avoids) rather than reimplementing
// EventStreamer server-side, so what's under test is purely the frontend's event handling.

test('Dragging a graph node persists its position across a reload (Studio layout localStorage)', async () => {
  // Isolated context: this test writes to the shared `studioLayout:demo` localStorage key and
  // shouldn't leak that into any other test in this file, the same reason the
  // prefers-color-scheme test above gets its own context.
  const context = await browser.createBrowserContext();
  const page = await context.newPage();
  try {
    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('.react-flow__node[data-id="triage"]', { timeout: 15000 });

    const getTriageTransform = () =>
      page.evaluate(
        () => document.querySelector('.react-flow__node[data-id="triage"]')?.style.transform
      );

    const beforeTransform = await getTriageTransform();
    assert.ok(beforeTransform, 'expected the triage node to render with a transform position');

    const box = await page.evaluate(() => {
      const rect = document
        .querySelector('.react-flow__node[data-id="triage"]')
        .getBoundingClientRect();
      return { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 };
    });

    await page.mouse.move(box.x, box.y);
    await page.mouse.down();
    await page.mouse.move(box.x + 160, box.y + 120, { steps: 10 });
    await page.mouse.up();
    // Give React a tick to apply the drag's position-change event before reading state back out.
    await new Promise((r) => setTimeout(r, 300));

    const afterDragTransform = await getTriageTransform();
    assert.notEqual(
      afterDragTransform,
      beforeTransform,
      'expected the drag to actually move the node'
    );

    const savedLayout = await page.evaluate(() => localStorage.getItem('studioLayout:demo'));
    assert.ok(savedLayout, 'expected onNodeDragStop to persist a Studio layout to localStorage');
    const parsed = JSON.parse(savedLayout);
    assert.ok(parsed.triage, "expected the dragged node's position to be saved under its id");

    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.waitForSelector('.react-flow__node[data-id="triage"]', { timeout: 15000 });
    // fetchConfig's first poll after mount needs a moment to apply the stored layout on top of
    // the freshly-computed formulaic positions.
    await new Promise((r) => setTimeout(r, 500));
    const afterReloadTransform = await getTriageTransform();
    assert.equal(
      afterReloadTransform,
      afterDragTransform,
      'expected the dragged position to survive a full page reload, not reset to the formulaic default'
    );
  } finally {
    await page.close();
    await context.close();
  }
});

test('Live "agent_spawned" event renders an ephemeral dashed node on the graph', async () => {
  const page = await browser.newPage();
  try {
    await page.setRequestInterception(true);
    page.on('request', (req) => {
      if (req.url().endsWith('/api/stream/events')) {
        const spawned = JSON.stringify({
          type: 'agent_spawned',
          data: { from: 'triage', to: 'triage_dyn_test1', role: 'Refund specialist' },
        });
        req.respond({
          status: 200,
          contentType: 'text/event-stream',
          body: `data: ${spawned}\n\n`,
        });
      } else {
        req.continue();
      }
    });

    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('body', { timeout: 10000 });
    await new Promise((r) => setTimeout(r, 1500));

    const bodyText = await page.evaluate(() => document.body.innerText);
    assert.ok(bodyText.includes('Refund specialist'), 'expected the ephemeral node label to render');
    assert.ok(bodyText.includes('Ephemeral'), 'expected the ephemeral badge text to render');
  } finally {
    await page.close();
  }
});

test('Live "router_decision" event with an error renders an amber misconfiguration toast', async () => {
  // Regression test for a real gap: Tracer.log_router_decision has always emitted this event over
  // SSE, but monitor.html had zero handling for it — a router/available_when condition that fails
  // open/closed silently by design (e.g. a typo'd state-key name) was invisible even to someone
  // watching the live dashboard at the exact moment it happened.
  const page = await browser.newPage();
  try {
    await page.setRequestInterception(true);
    page.on('request', (req) => {
      if (req.url().endsWith('/api/stream/events')) {
        const decision = JSON.stringify({
          type: 'router_decision',
          data: {
            kind: 'conditional',
            description: "triage -> billing if 'typo_balance < 0'",
            fired: false,
            target: 'billing',
            error: "Unknown variable: typo_balance",
          },
        });
        req.respond({
          status: 200,
          contentType: 'text/event-stream',
          body: `data: ${decision}\n\n`,
        });
      } else {
        req.continue();
      }
    });

    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('body', { timeout: 10000 });
    await new Promise((r) => setTimeout(r, 1500));

    const bodyText = await page.evaluate(() => document.body.innerText);
    // The heading renders as "Condition never fires" in the DOM, but a `uppercase` Tailwind
    // class visually transforms it — innerText reflects the CSS-computed text, not raw DOM text.
    assert.ok(bodyText.includes('CONDITION NEVER FIRES'), 'expected the misconfiguration toast heading to render');
    assert.ok(bodyText.includes('typo_balance'), 'expected the underlying error to render');
    assert.ok(bodyText.includes("triage -> billing"), 'expected the router description to render');
  } finally {
    await page.close();
  }
});

test('Live "router_decision" event with no error (a routine non-fire) does NOT render a toast', async () => {
  // The common case — a router simply evaluating to false — must stay quiet in the UI, same as it
  // stays at debug-level-only in the console log; only a genuinely broken condition should
  // interrupt the dashboard.
  const page = await browser.newPage();
  try {
    await page.setRequestInterception(true);
    page.on('request', (req) => {
      if (req.url().endsWith('/api/stream/events')) {
        const decision = JSON.stringify({
          type: 'router_decision',
          data: {
            kind: 'conditional',
            description: "triage -> billing if 'balance < 0'",
            fired: false,
            target: 'billing',
            error: null,
          },
        });
        req.respond({
          status: 200,
          contentType: 'text/event-stream',
          body: `data: ${decision}\n\n`,
        });
      } else {
        req.continue();
      }
    });

    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('body', { timeout: 10000 });
    await new Promise((r) => setTimeout(r, 1500));

    const bodyText = await page.evaluate(() => document.body.innerText);
    assert.ok(!bodyText.includes('CONDITION NEVER FIRES'), 'a routine non-fire must not surface a toast');
  } finally {
    await page.close();
  }
});

test('Approving a paused tool call shows what /resume actually returned, not a blind "Yes, proceed."', async () => {
  // Regression test for a real bug found live: the Approve button fired POST /api/resume, threw
  // away its response body (/resume's own turn loop can run the conversation arbitrarily far —
  // executing the approved tool, then continuing, e.g. into a freshly researched itinerary that
  // itself needs the human's review), and instead blindly sent a canned "Yes, proceed." message
  // via a completely separate /api/stream call against a chat transcript that never learned what
  // /resume had just done. On screen this looked like the itinerary was silently skipped — it was
  // sitting in the checkpoint the whole time, just never shown, before the auto-continuation
  // charged ahead past it.
  // Seeded fresh right here (not in test.before) — /api/memory only returns the 10 most-recently-
  // updated sessions, and by the time this test runs several earlier tests have already created
  // and touched their own real sessions, so a session seeded once at suite start would have aged
  // out of that list by now.
  seedSession(projectDir, APPROVAL_SESSION_ID, APPROVAL_SESSION_MESSAGES, {
    _metrics: { total_tokens: 5, total_cost: 0.0001 },
    _pending_approval: {
      tool: 'book_hotel',
      args: {},
      status: 'awaiting_approval',
      tool_call_id: 'call_1',
    },
  });

  const page = await browser.newPage();
  try {
    const ITINERARY_TEXT = 'Here is the Kuala Lumpur itinerary for your review before booking.';
    let streamCalled = false;

    await page.setRequestInterception(true);
    page.on('request', (req) => {
      if (req.url().endsWith('/api/resume')) {
        req.respond({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            response: ITINERARY_TEXT,
            active_agent: 'planner',
            status: 'completed',
            pending_action: null,
            queued_approvals: 0,
          }),
        });
      } else if (req.url().endsWith('/api/stream')) {
        streamCalled = true;
        req.continue();
      } else {
        req.continue();
      }
    });

    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
      () => document.body.innerText.toUpperCase().includes('ACTIVE SESSION'),
      { timeout: 15000 }
    );

    // Switch to the seeded session with a pending approval.
    const opened = await page.evaluate((currentId) => {
      const btn = [...document.querySelectorAll('button')].find((b) =>
        b.textContent.includes(currentId)
      );
      if (!btn) return false;
      btn.click();
      return true;
    }, SEEDED_SESSION_ID);
    assert.ok(opened, 'expected a session-selector button showing the active session id');

    await page.waitForSelector('input[placeholder="Search sessions..."]', { timeout: 5000 });
    await page.type('input[placeholder="Search sessions..."]', APPROVAL_SESSION_ID);
    await page.waitForFunction(
      (id) => document.body.innerText.includes(id),
      { timeout: 5000 },
      APPROVAL_SESSION_ID
    );
    const selected = await page.evaluate((id) => {
      const row = [...document.querySelectorAll('button')].find((b) => b.textContent.includes(id));
      if (!row) return false;
      row.click();
      return true;
    }, APPROVAL_SESSION_ID);
    assert.ok(selected, 'expected a clickable row for the approval-demo session');

    // The heading renders visually uppercase via CSS text-transform (same gotcha as "ACTIVE
    // SESSION" above), so innerText reports it as "APPROVAL REQUIRED" even though the JSX source
    // is mixed-case.
    await page.waitForFunction(() => document.body.innerText.toUpperCase().includes('APPROVAL REQUIRED'), {
      timeout: 10000,
    });

    // Click Approve.
    const approved = await page.evaluate(() => {
      const btn = [...document.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Approve');
      if (!btn) return false;
      btn.click();
      return true;
    });
    assert.ok(approved, 'expected an Approve button once a pending approval is showing');

    await page.waitForFunction(
      (text) => document.body.innerText.includes(text),
      { timeout: 10000 },
      ITINERARY_TEXT
    );

    const bodyText = await page.evaluate(() => document.body.innerText);
    assert.ok(
      bodyText.includes(ITINERARY_TEXT),
      "expected /resume's actual response to render in the chat transcript"
    );
    assert.ok(
      !streamCalled,
      'must not blindly fire a second /api/stream call ("Yes, proceed.") on top of what /resume already returned'
    );
  } finally {
    await page.close();
  }
});

test('A subsequent "agent_retired" event removes the ephemeral node again', async () => {
  const page = await browser.newPage();
  try {
    await page.setRequestInterception(true);
    page.on('request', (req) => {
      if (req.url().endsWith('/api/stream/events')) {
        const spawned = JSON.stringify({
          type: 'agent_spawned',
          data: { from: 'triage', to: 'triage_dyn_test2', role: 'Retired specialist' },
        });
        const retired = JSON.stringify({
          type: 'agent_retired',
          data: { agent: 'triage_dyn_test2', returned_to: 'triage' },
        });
        req.respond({
          status: 200,
          contentType: 'text/event-stream',
          body: `data: ${spawned}\n\ndata: ${retired}\n\n`,
        });
      } else {
        req.continue();
      }
    });

    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('body', { timeout: 10000 });
    await new Promise((r) => setTimeout(r, 1500));

    const bodyText = await page.evaluate(() => document.body.innerText);
    assert.ok(
      !bodyText.includes('Retired specialist'),
      'expected the ephemeral node to have been removed after agent_retired'
    );
  } finally {
    await page.close();
  }
});

test('Playground chat window does not show raw tool/system message content as a chat bubble', async () => {
  // Regression test for a real bug found live: the Playground rendered every non-user message
  // role identically to an assistant reply (marked.parse(m.content)) — including `role: 'tool'`
  // messages, whose `content` is the tool's raw JSON result string. Loading (or switching to) a
  // session with real tool calls in its history showed that raw JSON sitting in the chat window
  // as if the assistant had said it. The chat window must only ever show actual user/assistant
  // conversational turns — the raw tool trace has its own separate "Execution Traces" tab.
  const JSON_TOOL_SESSION_ID = 'json-leak-demo';
  const RAW_TOOL_JSON = '{"status": "confirmed", "booking_reference": "HT-XYZ-99999"}';
  seedSession(
    projectDir,
    JSON_TOOL_SESSION_ID,
    [
      { role: 'user', content: 'Book the hotel.' },
      {
        role: 'assistant',
        tool_calls: [
          { id: 'call_1', type: 'function', function: { name: 'book_hotel', arguments: '{}' } },
        ],
      },
      { role: 'tool', tool_call_id: 'call_1', name: 'book_hotel', content: RAW_TOOL_JSON },
      { role: 'assistant', content: 'Your hotel is booked!' },
    ],
    { _metrics: { total_tokens: 5, total_cost: 0.0001 } }
  );

  const page = await browser.newPage();
  try {
    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
      () => document.body.innerText.toUpperCase().includes('ACTIVE SESSION'),
      { timeout: 15000 }
    );

    const opened = await page.evaluate((currentId) => {
      const btn = [...document.querySelectorAll('button')].find((b) => b.textContent.includes(currentId));
      if (!btn) return false;
      btn.click();
      return true;
    }, SEEDED_SESSION_ID);
    assert.ok(opened, 'expected a session-selector button showing the active session id');

    await page.waitForSelector('input[placeholder="Search sessions..."]', { timeout: 5000 });
    await page.type('input[placeholder="Search sessions..."]', JSON_TOOL_SESSION_ID);
    await page.waitForFunction(
      (id) => document.body.innerText.includes(id),
      { timeout: 5000 },
      JSON_TOOL_SESSION_ID
    );
    const selected = await page.evaluate((id) => {
      const row = [...document.querySelectorAll('button')].find((b) => b.textContent.includes(id));
      if (!row) return false;
      row.click();
      return true;
    }, JSON_TOOL_SESSION_ID);
    assert.ok(selected, 'expected a clickable row for the json-leak-demo session');

    await page.waitForFunction(
      () => document.body.innerText.includes('Your hotel is booked!'),
      { timeout: 10000 }
    );

    const bodyText = await page.evaluate(() => document.body.innerText);
    assert.ok(bodyText.includes('Book the hotel.'), 'expected the user turn to still render');
    assert.ok(bodyText.includes('Your hotel is booked!'), 'expected the real assistant reply to still render');
    assert.ok(
      !bodyText.includes('booking_reference'),
      "the tool's raw JSON result must not appear in the chat window"
    );
  } finally {
    await page.close();
  }
});

test('Playground auto-scrolls to the bottom when a long session loads', async () => {
  // Regression test: the scrollable message container never programmatically scrolled — loading
  // (or switching to) a session whose history overflowed the visible area left the viewport
  // wherever the browser happened to leave it (usually the top), instead of showing the most
  // recent message the way a chat UI should.
  const LONG_SESSION_ID = 'long-history-demo';
  const longMessages = [];
  for (let i = 0; i < 30; i++) {
    longMessages.push({ role: 'user', content: `Message number ${i} to pad out the scroll area.` });
    longMessages.push({ role: 'assistant', content: `Reply number ${i}, also padding things out.` });
  }
  const LAST_MESSAGE_TEXT = 'Reply number 29, also padding things out.';
  seedSession(projectDir, LONG_SESSION_ID, longMessages, {
    _metrics: { total_tokens: 5, total_cost: 0.0001 },
  });

  const page = await browser.newPage();
  try {
    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
      () => document.body.innerText.toUpperCase().includes('ACTIVE SESSION'),
      { timeout: 15000 }
    );

    const opened = await page.evaluate((currentId) => {
      const btn = [...document.querySelectorAll('button')].find((b) => b.textContent.includes(currentId));
      if (!btn) return false;
      btn.click();
      return true;
    }, SEEDED_SESSION_ID);
    assert.ok(opened, 'expected a session-selector button showing the active session id');

    await page.waitForSelector('input[placeholder="Search sessions..."]', { timeout: 5000 });
    await page.type('input[placeholder="Search sessions..."]', LONG_SESSION_ID);
    await page.waitForFunction(
      (id) => document.body.innerText.includes(id),
      { timeout: 5000 },
      LONG_SESSION_ID
    );
    const selected = await page.evaluate((id) => {
      const row = [...document.querySelectorAll('button')].find((b) => b.textContent.includes(id));
      if (!row) return false;
      row.click();
      return true;
    }, LONG_SESSION_ID);
    assert.ok(selected, 'expected a clickable row for the long-history-demo session');

    await page.waitForFunction(
      (text) => document.body.innerText.includes(text),
      { timeout: 10000 },
      LAST_MESSAGE_TEXT
    );
    // Give the auto-scroll effect a moment to run after the DOM settles.
    await new Promise((r) => setTimeout(r, 300));

    const scrollState = await page.evaluate((text) => {
      const bubble = [...document.querySelectorAll('div')].find((d) => d.textContent.trim() === text);
      const container = bubble ? bubble.closest('.overflow-y-auto') : null;
      if (!container) return null;
      return {
        scrollTop: container.scrollTop,
        scrollHeight: container.scrollHeight,
        clientHeight: container.clientHeight,
      };
    }, LAST_MESSAGE_TEXT);

    assert.ok(scrollState, 'expected to find the scrollable message container');
    assert.ok(
      scrollState.scrollHeight > scrollState.clientHeight,
      'test setup check: the seeded history must actually overflow the container to prove anything'
    );
    const distanceFromBottom =
      scrollState.scrollHeight - scrollState.clientHeight - scrollState.scrollTop;
    assert.ok(
      distanceFromBottom < 40,
      `expected the container scrolled near the bottom (within 40px), got ${distanceFromBottom}px away`
    );
  } finally {
    await page.close();
  }
});

test('Architect chat auto-scrolls to the bottom when a long conversation is expanded', async () => {
  // Same auto-scroll regression as the Playground, for the separate IntaGrin Architect chat
  // panel — a different scrollable container with its own independent fix.
  const context = await browser.createBrowserContext();
  const page = await context.newPage();
  try {
    const archSessionId = 'arch-long-history';
    const longArchMessages = [];
    for (let i = 0; i < 30; i++) {
      longArchMessages.push({ role: 'user', content: `Architect question number ${i}.` });
      longArchMessages.push({ role: 'assistant', content: `Architect answer number ${i}.` });
    }
    const LAST_ARCH_TEXT = 'Architect answer number 29.';

    // architectSessions/activeArchSessionId are namespaced per-project (see monitor.html) —
    // seed under 'demo', the name scaffoldProject('demo') gives this suite's project.
    await page.evaluateOnNewDocument(
      (sessions, activeId) => {
        localStorage.setItem('architectSessions:demo', JSON.stringify(sessions));
        localStorage.setItem('activeArchSessionId:demo', activeId);
      },
      [{ id: archSessionId, name: 'Session 1', messages: longArchMessages }],
      archSessionId
    );

    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    // Proof the per-project load effect has actually run before we go looking for its content.
    await page.waitForFunction(
      () => localStorage.getItem('activeArchSessionId:demo') !== null,
      { timeout: 15000 }
    );

    // Expand the collapsed Architect chat panel (isChatExpanded defaults to false).
    const expanded = await page.evaluate(() => {
      const header = [...document.querySelectorAll('*')].find(
        (el) => el.children.length === 0 && el.textContent.trim() === 'IntaGrin Architect'
      );
      const clickTarget = header ? header.closest('.cursor-pointer') : null;
      if (!clickTarget) return false;
      clickTarget.click();
      return true;
    });
    assert.ok(expanded, 'expected a clickable "IntaGrin Architect" header to expand the panel');

    await page.waitForFunction(
      (text) => document.body.innerText.includes(text),
      { timeout: 10000 },
      LAST_ARCH_TEXT
    );
    await new Promise((r) => setTimeout(r, 300));

    const scrollState = await page.evaluate((text) => {
      const bubble = [...document.querySelectorAll('div')].find((d) => d.textContent.trim() === text);
      const container = bubble ? bubble.closest('.overflow-y-auto') : null;
      if (!container) return null;
      return {
        scrollTop: container.scrollTop,
        scrollHeight: container.scrollHeight,
        clientHeight: container.clientHeight,
      };
    }, LAST_ARCH_TEXT);

    assert.ok(scrollState, 'expected to find the Architect chat scrollable container');
    assert.ok(
      scrollState.scrollHeight > scrollState.clientHeight,
      'test setup check: the seeded conversation must actually overflow the container to prove anything'
    );
    const distanceFromBottom =
      scrollState.scrollHeight - scrollState.clientHeight - scrollState.scrollTop;
    assert.ok(
      distanceFromBottom < 40,
      `expected the Architect chat scrolled near the bottom (within 40px), got ${distanceFromBottom}px away`
    );
  } finally {
    await page.close();
    await context.close();
  }
});
