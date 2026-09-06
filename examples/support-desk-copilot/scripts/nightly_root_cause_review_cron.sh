#!/usr/bin/env bash
# ==============================================================================
# IntaGrin Support Desk Copilot - Nightly Root-Cause Review
# ==============================================================================
# Runs the `nightly_root_cause_review` workflow declared in ai.yaml: recalls the day's
# recurring technical failure patterns from episodic memory, then debates the likely root
# cause across two independent hypotheses (infra vs. client-side) before a judge picks one.
#
# Named workflows run directly via `inta run`, not conversationally through /chat — see
# ai.yaml's `workflows:` section and `inta run --help`.
#
# Usage in Linux crontab (once nightly, e.g. 2am):
# 0 2 * * * /path/to/support-desk-copilot/scripts/nightly_root_cause_review_cron.sh >> /path/to/support-desk-copilot/logs/cron.log 2>&1
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "[$(date -Iseconds)] [INFO] Starting nightly_root_cause_review"

cd "${PROJECT_DIR}"
if inta run nightly_root_cause_review; then
  echo "[$(date -Iseconds)] [SUCCESS] nightly_root_cause_review completed"
else
  echo "[$(date -Iseconds)] [ERROR] nightly_root_cause_review failed"
  exit 1
fi
