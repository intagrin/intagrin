# Runbook: connection-pool exhaustion

## Client pool starved vs. database saturated

| Signal | Client pool starved | Database saturated |
|---|---|---|
| App logs | "Connection is not available, request timed out" with `active == total` | query timeouts, lock waits |
| DB active connections | far below `max_connections` | at or near `max_connections` |
| DB CPU / slow queries | normal | high |
| Pool `waiting` threads | high | normal-to-high |

If the database is healthy and the client pool is pinned at its maximum, the database is **not**
the problem — the pool is too small for the traffic.

## Confirming the cause
1. Compare the pool's configured max against its history. A recent deploy that changed it is the
   prime suspect.
2. Check that the timing lines up: pool waits climb within minutes of the deploy.
3. Rule out traffic spikes (requests/sec roughly flat means it's the pool, not the load).

## Fix
Roll back the deploy that changed the pool size (see the `safe_rollback` runbook). Don't raise
`max_connections` on the database — it isn't the constrained resource.
