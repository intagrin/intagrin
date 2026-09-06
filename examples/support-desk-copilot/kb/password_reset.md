# Password Reset & Login Failures

Customers who can't log in almost always fall into one of three buckets:

1. **Expired session token.** Ask the customer to fully sign out (not just close the tab) and
   sign back in. This resolves the majority of "stuck on loading screen after login" reports.
2. **SSO provider mismatch.** If the account was originally created with email/password and the
   customer is now trying Google/Microsoft SSO (or vice versa), login will silently fail with no
   useful error. Confirm which method the account was created with before troubleshooting further.
3. **Auth service degradation.** If `auth-api` is reporting degraded/outage status, login failures
   are platform-wide, not account-specific — check service status before assuming it's a
   one-off account issue, and let the customer know it's a known incident rather than asking them
   to keep retrying.

Password reset emails can take up to 5 minutes to arrive and are commonly caught by corporate spam
filters for Enterprise-tier accounts — check the spam folder before escalating a "reset email never
arrived" report.
