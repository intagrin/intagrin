import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="IntaGrin Human Approval Dashboard", version="1.0")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_DIR / ".ai" / "memory.db"
DEFIN_API_URL = os.environ.get("DEFIN_API_URL", "http://localhost:8000")


class ResumeActionRequest(BaseModel):
    session_id: str
    approved: bool
    edited_post_content: str | None = None
    reviewer_notes: str | None = None


class TriggerRunRequest(BaseModel):
    custom_prompt: str | None = None


def get_db_connection():
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/api/health")
async def health_check():
    api_online = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{DEFIN_API_URL}/docs")
            api_online = resp.status_code == 200
    except Exception:
        api_online = False

    return {
        "dashboard": "healthy",
        "defin_api_connected": api_online,
        "defin_api_url": DEFIN_API_URL,
        "db_exists": DB_PATH.exists()
    }


@app.get("/api/pending")
def get_pending_approvals():
    conn = get_db_connection()
    if not conn:
        return []

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT session_id, messages, state, updated_at FROM checkpoints ORDER BY updated_at DESC")
        rows = cursor.fetchall()
        
        pending_items = []
        for row in rows:
            try:
                state = json.loads(row["state"]) if row["state"] else {}
                pending = state.get("_pending_approval")
                if pending:
                    messages = json.loads(row["messages"]) if row["messages"] else []
                    raw_id = row["session_id"]
                    clean_id = raw_id.split(":", 1)[1] if ":" in raw_id else raw_id
                    pending_items.append({
                        "session_id": clean_id,
                        "raw_session_id": raw_id,
                        "updated_at": row["updated_at"],
                        "tool": pending.get("tool"),
                        "agent": pending.get("agent", "reviewer_agent"),
                        "args": pending.get("args", {}),
                        "message_count": len(messages)
                    })
            except Exception:
                continue
        return pending_items
    finally:
        conn.close()


@app.get("/api/history")
def get_session_history():
    conn = get_db_connection()
    if not conn:
        return []

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT session_id, messages, state, updated_at FROM checkpoints ORDER BY updated_at DESC LIMIT 30")
        rows = cursor.fetchall()
        
        history_items = []
        for row in rows:
            try:
                state = json.loads(row["state"]) if row["state"] else {}
                messages = json.loads(row["messages"]) if row["messages"] else []
                is_pending = bool(state.get("_pending_approval"))
                
                # Get last assistant message
                last_assistant_msg = ""
                for msg in reversed(messages):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        last_assistant_msg = msg["content"]
                        break
                        
                raw_id = row["session_id"]
                clean_id = raw_id.split(":", 1)[1] if ":" in raw_id else raw_id
                history_items.append({
                    "session_id": clean_id,
                    "raw_session_id": raw_id,
                    "updated_at": row["updated_at"],
                    "is_pending": is_pending,
                    "last_message": last_assistant_msg,
                    "message_count": len(messages)
                })
            except Exception:
                continue
        return history_items
    finally:
        conn.close()


@app.post("/api/resume")
async def resume_session(payload: ResumeActionRequest):
    request_body = {
        "session_id": payload.session_id,
        "approved": payload.approved,
        "reviewer_notes": payload.reviewer_notes or ("Approved by human reviewer" if payload.approved else "Rejected by human reviewer")
    }

    if payload.approved and payload.edited_post_content:
        # Pass updated args to IntaGrin engine
        request_body["edited_args"] = {
            "post_content": payload.edited_post_content
        }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{DEFIN_API_URL}/resume", json=request_body)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return resp.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Failed to connect to IntaGrin server at {DEFIN_API_URL}: {exc!s}")


@app.post("/api/trigger")
async def trigger_run(payload: TriggerRunRequest):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"daily_run_{timestamp}"
    
    prompt = payload.custom_prompt or (
        "Please research current trending breakthroughs in AI and technology, "
        "generate an engaging LinkedIn post, review it for accuracy and engagement, "
        "and submit it for human approval."
    )

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{DEFIN_API_URL}/chat",
                json={"message": prompt, "session_id": session_id}
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return {
                "status": "success",
                "session_id": session_id,
                "response": resp.json()
            }
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Failed to connect to IntaGrin server at {DEFIN_API_URL}: {exc!s}")


@app.get("/", response_class=HTMLResponse)
def index_page():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>IntaGrin - Human Approval Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    body { font-family: 'Inter', sans-serif; }
    .post-content-box { white-space: pre-wrap; font-family: inherit; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen">
  <!-- Top Navigation -->
  <header class="border-b border-slate-800 bg-slate-900/60 backdrop-blur sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-500 to-sky-400 flex items-center justify-center shadow-lg shadow-indigo-500/20">
          <i class="fa-solid fa-robot text-white text-lg"></i>
        </div>
        <div>
          <h1 class="text-base font-bold text-white tracking-tight">IntaGrin Social Media Gatekeeper</h1>
          <p class="text-xs text-slate-400">Human-in-the-Loop Review Dashboard</p>
        </div>
      </div>

      <div class="flex items-center gap-3">
        <div id="api-status-badge" class="flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium bg-slate-800 text-slate-400 border border-slate-700">
          <span class="w-2 h-2 rounded-full bg-slate-500 animate-pulse"></span>
          Checking Engine...
        </div>

        <button onclick="triggerNewRun()" id="trigger-btn" class="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-indigo-600 hover:bg-indigo-500 text-white transition shadow-sm hover:shadow-indigo-600/30">
          <i class="fa-solid fa-play text-xs"></i>
          <span>Trigger Daily Run</span>
        </button>

        <button onclick="loadDashboardData()" class="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition" title="Refresh">
          <i class="fa-solid fa-arrows-rotate"></i>
        </button>
      </div>
    </div>
  </header>

  <!-- Main Content -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
    
    <!-- Stats Row -->
    <div class="grid grid-cols-1 sm:grid-cols-3 gap-5">
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 flex items-center justify-between">
        <div>
          <p class="text-xs uppercase tracking-wider text-slate-400 font-semibold">Pending Approvals</p>
          <h3 id="stat-pending-count" class="text-3xl font-bold text-amber-400 mt-1">0</h3>
        </div>
        <div class="w-12 h-12 rounded-lg bg-amber-500/10 border border-amber-500/20 flex items-center justify-center text-amber-400 text-xl">
          <i class="fa-solid fa-clock"></i>
        </div>
      </div>

      <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 flex items-center justify-between">
        <div>
          <p class="text-xs uppercase tracking-wider text-slate-400 font-semibold">Processed Sessions</p>
          <h3 id="stat-total-count" class="text-3xl font-bold text-sky-400 mt-1">0</h3>
        </div>
        <div class="w-12 h-12 rounded-lg bg-sky-500/10 border border-sky-500/20 flex items-center justify-center text-sky-400 text-xl">
          <i class="fa-solid fa-list-check"></i>
        </div>
      </div>

      <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 flex items-center justify-between">
        <div>
          <p class="text-xs uppercase tracking-wider text-slate-400 font-semibold">Daily Cron Status</p>
          <h3 class="text-sm font-semibold text-emerald-400 mt-2 flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
            Active (09:00 AM)
          </h3>
        </div>
        <div class="w-12 h-12 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400 text-xl">
          <i class="fa-solid fa-calendar-day"></i>
        </div>
      </div>
    </div>

    <!-- Review Queue Section -->
    <section class="space-y-4">
      <div class="flex items-center justify-between">
        <div>
          <h2 class="text-xl font-bold text-white flex items-center gap-2">
            <i class="fa-solid fa-shield-halved text-amber-400"></i>
            Awaiting Human Approval
          </h2>
          <p class="text-sm text-slate-400">Review, modify, approve, or reject drafted LinkedIn posts staged by the reviewer agent.</p>
        </div>
      </div>

      <div id="pending-container" class="space-y-6">
        <div class="text-center py-12 bg-slate-900/50 border border-slate-800 rounded-2xl">
          <i class="fa-solid fa-circle-notch fa-spin text-slate-500 text-3xl mb-3"></i>
          <p class="text-slate-400 text-sm">Loading pending approval queue...</p>
        </div>
      </div>
    </section>

    <!-- History Archive Section -->
    <section class="space-y-4 pt-6 border-t border-slate-800">
      <div class="flex items-center justify-between">
        <div>
          <h2 class="text-lg font-bold text-white flex items-center gap-2">
            <i class="fa-solid fa-clock-rotate-left text-slate-400"></i>
            Recent Sessions & Activity
          </h2>
          <p class="text-xs text-slate-400">Audit log of recent agent interactions and completed executions.</p>
        </div>
      </div>

      <div id="history-container" class="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden divide-y divide-slate-800 text-sm">
        <div class="p-6 text-center text-slate-500 text-xs">Loading activity log...</div>
      </div>
    </section>
  </main>

  <script>
    async function checkHealth() {
      try {
        const res = await fetch('/api/health');
        const data = await res.json();
        const badge = document.getElementById('api-status-badge');
        if (data.defin_api_connected) {
          badge.className = 'flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium bg-emerald-950/60 text-emerald-300 border border-emerald-800/60';
          badge.innerHTML = '<span class="w-2 h-2 rounded-full bg-emerald-400"></span> API Server Online';
        } else {
          badge.className = 'flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium bg-rose-950/60 text-rose-300 border border-rose-800/60';
          badge.innerHTML = '<span class="w-2 h-2 rounded-full bg-rose-400"></span> API Offline (' + data.defin_api_url + ')';
        }
      } catch (e) {
        console.error(e);
      }
    }

    async function loadDashboardData() {
      await checkHealth();
      
      // Load pending
      try {
        const res = await fetch('/api/pending');
        const items = await res.json();
        document.getElementById('stat-pending-count').innerText = items.length;
        renderPending(items);
      } catch (e) {
        console.error("Failed to load pending:", e);
      }

      // Load history
      try {
        const res = await fetch('/api/history');
        const history = await res.json();
        document.getElementById('stat-total-count').innerText = history.length;
        renderHistory(history);
      } catch (e) {
        console.error("Failed to load history:", e);
      }
    }

    function renderPending(items) {
      const container = document.getElementById('pending-container');
      if (!items || items.length === 0) {
        container.innerHTML = `
          <div class="text-center py-12 bg-slate-900/40 border border-slate-800/80 rounded-2xl">
            <div class="w-12 h-12 rounded-full bg-emerald-500/10 text-emerald-400 mx-auto flex items-center justify-center text-xl mb-3">
              <i class="fa-solid fa-check"></i>
            </div>
            <h3 class="text-base font-semibold text-white">All Caught Up!</h3>
            <p class="text-xs text-slate-400 mt-1">No posts currently awaiting human review. The next scheduled daily run will appear here.</p>
          </div>
        `;
        return;
      }

      container.innerHTML = items.map((item, idx) => {
        const args = item.args || {};
        const postContent = args.post_content || 'No content provided';
        const reviewNotes = args.review_notes || 'No review critique supplied.';
        const suggestedStatus = args.suggested_status || 'approved_for_human';

        return `
          <div class="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl" id="card-${item.session_id}">
            <!-- Header Bar -->
            <div class="bg-slate-850 px-6 py-4 border-b border-slate-800 flex flex-wrap items-center justify-between gap-4">
              <div class="flex items-center gap-3">
                <span class="px-2.5 py-1 rounded-md text-xs font-semibold uppercase tracking-wider bg-amber-500/10 text-amber-300 border border-amber-500/30">
                  <i class="fa-solid fa-hourglass-half mr-1"></i> Awaiting Approval
                </span>
                <span class="text-sm font-mono text-slate-400">${item.session_id}</span>
              </div>
              <span class="text-xs text-slate-400">
                <i class="fa-regular fa-clock mr-1"></i> ${new Date(item.updated_at).toLocaleString()}
              </span>
            </div>

            <!-- Content Body -->
            <div class="p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
              <!-- Left: Reviewer Notes & Gatekeeper Info -->
              <div class="lg:col-span-4 space-y-4">
                <div class="bg-slate-950/70 border border-slate-800 rounded-xl p-4 space-y-3">
                  <h4 class="text-xs font-bold uppercase tracking-wider text-sky-400 flex items-center gap-2">
                    <i class="fa-solid fa-magnifying-glass-check"></i> Reviewer Assessment
                  </h4>
                  <div class="text-xs text-slate-300 leading-relaxed bg-slate-900 p-3 rounded-lg border border-slate-800">
                    ${reviewNotes}
                  </div>
                  <div class="flex items-center justify-between text-xs pt-1 border-t border-slate-800/80">
                    <span class="text-slate-400">Agent Recommendation:</span>
                    <span class="font-medium text-emerald-400 capitalize">${suggestedStatus.replace(/_/g, ' ')}</span>
                  </div>
                </div>

                <!-- Rejection Notes Form -->
                <div class="bg-slate-950/70 border border-slate-800 rounded-xl p-4 space-y-2">
                  <label class="text-xs font-semibold text-slate-300 block">Feedback / Revision Guidance</label>
                  <textarea id="feedback-${item.session_id}" rows="3" placeholder="If rejecting, specify revision notes for the agents..." class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 transition resize-none"></textarea>
                </div>
              </div>

              <!-- Right: LinkedIn Post Editor / Preview -->
              <div class="lg:col-span-8 flex flex-col space-y-3">
                <div class="flex items-center justify-between">
                  <label class="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                    <i class="fa-brands fa-linkedin text-sky-400"></i> LinkedIn Post Draft
                  </label>
                  <span class="text-xs text-slate-400">(Editable prior to approval)</span>
                </div>

                <div class="relative flex-1">
                  <textarea id="content-${item.session_id}" rows="12" class="w-full bg-slate-950 border border-slate-800 rounded-xl p-4 text-sm text-slate-100 font-sans leading-relaxed focus:outline-none focus:border-indigo-500 transition resize-y post-content-box">${postContent}</textarea>
                </div>

                <!-- Action Controls -->
                <div class="pt-3 flex items-center justify-end gap-3 border-t border-slate-800">
                  <button onclick="handleDecision('${item.session_id}', false)" class="px-4 py-2.5 rounded-lg text-xs font-semibold bg-rose-950/40 hover:bg-rose-900/60 text-rose-300 border border-rose-800/60 transition flex items-center gap-2">
                    <i class="fa-solid fa-xmark"></i> Reject & Request Revision
                  </button>
                  <button onclick="handleDecision('${item.session_id}', true)" class="px-5 py-2.5 rounded-lg text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-600/20 transition flex items-center gap-2">
                    <i class="fa-solid fa-check"></i> Approve & Resume
                  </button>
                </div>
              </div>
            </div>
          </div>
        `;
      }).join('');
    }

    function renderHistory(items) {
      const container = document.getElementById('history-container');
      if (!items || items.length === 0) {
        container.innerHTML = '<div class="p-6 text-center text-slate-500 text-xs">No session history recorded yet.</div>';
        return;
      }

      container.innerHTML = items.map(item => `
        <div class="p-4 flex items-center justify-between hover:bg-slate-850/50 transition">
          <div class="flex items-center gap-3">
            <div class="w-2.5 h-2.5 rounded-full ${item.is_pending ? 'bg-amber-400 animate-pulse' : 'bg-slate-600'}"></div>
            <div>
              <p class="font-mono text-xs text-white font-medium">${item.session_id}</p>
              <p class="text-xs text-slate-400 truncate max-w-xl mt-0.5">${item.last_message ? item.last_message.slice(0, 100) + '...' : 'Turn completed'}</p>
            </div>
          </div>
          <div class="text-right text-xs text-slate-400">
            <span>${new Date(item.updated_at).toLocaleTimeString()}</span>
            <span class="ml-2 text-slate-500">(${item.message_count} msgs)</span>
          </div>
        </div>
      `).join('');
    }

    async function handleDecision(sessionId, isApproved) {
      const content = document.getElementById(`content-${sessionId}`).value;
      const feedback = document.getElementById(`feedback-${sessionId}`).value;

      const confirmMsg = isApproved 
        ? "Approve this post and complete session?" 
        : "Reject this draft and pass feedback back to the agent?";
        
      if (!confirm(confirmMsg)) return;

      const card = document.getElementById(`card-${sessionId}`);
      if (card) card.style.opacity = '0.5';

      try {
        const res = await fetch('/api/resume', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: sessionId,
            approved: isApproved,
            edited_post_content: isApproved ? content : null,
            reviewer_notes: feedback || (isApproved ? "Approved by user." : "Rejected by user.")
          })
        });

        if (!res.ok) {
          const err = await res.json();
          alert(`Action failed: ${err.detail || 'Server error'}`);
          if (card) card.style.opacity = '1';
          return;
        }

        await loadDashboardData();
      } catch (e) {
        alert(`Error executing decision: ${e.message}`);
        if (card) card.style.opacity = '1';
      }
    }

    async function triggerNewRun() {
      const btn = document.getElementById('trigger-btn');
      btn.disabled = true;
      btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin text-xs"></i> <span>Running Swarm...</span>';

      try {
        const res = await fetch('/api/trigger', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({})
        });

        if (!res.ok) {
          const err = await res.json();
          alert(`Failed to trigger run: ${err.detail || 'Error'}`);
        } else {
          const data = await res.json();
          alert(`Swarm execution completed!\nSession ID: ${data.session_id}`);
          await loadDashboardData();
        }
      } catch (e) {
        alert(`Trigger error: ${e.message}`);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-play text-xs"></i> <span>Trigger Daily Run</span>';
      }
    }

    // Initial load and polling
    loadDashboardData();
    setInterval(loadDashboardData, 10000);
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
