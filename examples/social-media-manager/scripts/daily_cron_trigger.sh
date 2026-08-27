#!/usr/bin/env bash
# ==============================================================================
# IntaGrin Daily Social Media Pipeline - Cron Trigger
# ==============================================================================
# Usage in Linux crontab:
# 0 9 * * * /home/anoop/Workspace/ai-platform/social-media-manager/scripts/daily_cron_trigger.sh >> /home/anoop/Workspace/ai-platform/social-media-manager/logs/cron.log 2>&1
# ==============================================================================

set -euo pipefail

API_URL="${DEFIN_API_URL:-http://localhost:8000}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SESSION_ID="daily_run_${TIMESTAMP}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../logs"
mkdir -p "${LOG_DIR}"

echo "[$(date -Iseconds)] [INFO] Starting daily pipeline run: ${SESSION_ID}"

PROMPT="Please research the latest trending breakthroughs in AI and tech, generate an engaging LinkedIn post with key insights and CTA, review it for high quality, and submit it for human approval."

# Send request to IntaGrin server
RESPONSE=$(curl -s -w "\nHTTP_STATUS:%{http_code}" -X POST "${API_URL}/chat" \
  -H "Content-Type: application/json" \
  -d "{
    \"message\": \"${PROMPT}\",
    \"session_id\": \"${SESSION_ID}\"
  }")

HTTP_STATUS=$(echo "$RESPONSE" | tr -d '\n' | sed -e 's/.*HTTP_STATUS://')
BODY=$(echo "$RESPONSE" | sed -e 's/HTTP_STATUS:.*//')

if [ "$HTTP_STATUS" -eq 200 ]; then
  echo "[$(date -Iseconds)] [SUCCESS] Triggered successfully. Session ID: ${SESSION_ID}"
  echo "[$(date -Iseconds)] [INFO] Response: ${BODY}"
else
  echo "[$(date -Iseconds)] [ERROR] Failed to trigger pipeline. HTTP Status: ${HTTP_STATUS}"
  echo "[$(date -Iseconds)] [ERROR] Body: ${BODY}"
  exit 1
fi
