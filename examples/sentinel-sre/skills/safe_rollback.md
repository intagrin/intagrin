# Runbook: safe rollback

1. **Target version**: the most recent version *before* the offending deploy whose canary passed.
   Never roll back more than one version without a reason.
2. **Approvals**: production rollbacks need two of on-call SRE, service owner, and incident
   commander. The tool pauses for this automatically.
3. **Reason**: one line naming the bad version and the evidence (e.g. "v2.14.0 shrank DB pool 50->5;
   pool timeouts began 2 min after deploy").
4. **After rollback**: post a "monitoring" status update, then watch error rate and p99 for 5
   minutes before calling it resolved.
5. **Follow-up**: config-only changes must still go through canary — add this as an action item
   whenever a skipped canary contributed.
