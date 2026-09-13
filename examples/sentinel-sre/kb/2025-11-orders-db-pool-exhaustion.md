# Postmortem: orders-api 5xx storm (2025-11-18)

**Severity:** sev1 · **Duration:** 41 minutes · **Services:** orders-api, orders-db

## Summary
An environment-variable rename in orders-api v3.8.0 made the DB pool fall back to its library
default of 10 connections (previously 60). At peak traffic every request queued on the pool;
p99 latency hit 3.1s and 5xx reached 22%.

## What misled responders
The database was blamed first because the errors mentioned "connection". orders-db was healthy
(CPU 28%, 52 of 500 connections) — the constrained resource was the client-side pool.

## Resolution
Rolled back to v3.7.2. Recovery in about 3 minutes.

## Action items
- Alert on client pool `waiting` threads, not just DB connections.
- Config changes affecting pools or timeouts go through canary like code changes.
