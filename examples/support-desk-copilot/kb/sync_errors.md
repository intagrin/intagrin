# Data Sync Errors

`sync-worker` processes account data asynchronously; a customer report of "my changes aren't
showing up" usually means one of:

- **Normal processing lag.** Sync can take up to 2 minutes under normal load. If the customer
  reports it's been longer than that, treat it as a real issue rather than lag.
- **Conflicting concurrent edits.** Two people editing the same record within a few seconds of
  each other can produce a last-write-wins conflict that looks like "my edit disappeared." Ask
  whether a teammate was editing the same record at the same time.
- **A stuck job.** If `sync-worker`'s own status check comes back anything other than
  "operational," sync delays are a known platform issue, not something specific to this account —
  reference the open incident rather than restarting troubleshooting from scratch.

There is no customer-facing "force resync" action today — if a sync appears permanently stuck
(not just delayed), this needs a technical escalation, not a repeated status check.
