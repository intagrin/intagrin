# Postmortem: storefront latency after full CDN purge (2026-03-02)

**Severity:** sev2 · **Duration:** 18 minutes · **Services:** storefront-web, image-service

## Summary
A full CDN cache purge (instead of a path-scoped one) sent all image traffic to origin.
image-service CPU hit 95% and storefront p99 rose to 1.4s. No errors, only latency.

## Resolution
Scaled image-service from 8 to 20 replicas until the cache warmed up (about 15 minutes), then
scaled back down.

## Action items
- The purge tooling requires a path prefix; full purges need approval.
