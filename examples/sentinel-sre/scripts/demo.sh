#!/usr/bin/env bash
# Starts Sentinel's mock changelog API, then the Monitor dashboard (Agent Playground + live graph).
# Run from anywhere, with the repo's virtualenv active:  source ../../.venv/bin/activate
set -euo pipefail
cd "$(dirname "$0")/.."

# The Monitor's login and approver checks read SENTINEL_* straight from the environment.
set -a; . ./.env; set +a

# Fresh remediation ledger + incident clock (and optionally fresh sessions) for a clean take.
rm -f .ai/remediation_ledger.jsonl .ai/incident_anchor.txt
# The -wal/-shm sidecars must go with the db: deleting memory.db alone leaves SQLite pointing at
# sidecars that no longer match it, and the next open fails with "disk I/O error".
if [[ "${FRESH:-0}" == "1" ]]; then rm -f .ai/memory.db .ai/memory.db-wal .ai/memory.db-shm; fi

python mock_api/changelog_api.py &
MOCK_PID=$!
trap 'kill $MOCK_PID 2>/dev/null || true' EXIT

inta monitor --port "${PORT:-3000}"
