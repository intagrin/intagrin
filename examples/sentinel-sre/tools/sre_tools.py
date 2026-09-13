"""Sentinel's local tools. Backed by tools.incident_store's fixture so the demo runs with no cloud
account. Every side-effecting action is DRY-RUN: recorded to the remediation ledger
(.ai/remediation_ledger.jsonl), never executed — swap record_action for real API calls to adapt."""

import json
import shlex

from intagrin.errors import AwaitingHumanInput

from . import incident_store

_READ_ONLY_KUBECTL_VERBS = {"get", "describe", "top", "logs"}
# scale_service's runtime AwaitingHumanInput threshold — small scale-ups are routine, big ones cost money.
_AUTO_APPROVE_MAX_REPLICAS = 10
_STATUSES = {"investigating", "identified", "monitoring", "resolved"}


def get_alert(alert_id: str) -> str:
    """Fetches a firing alert from Alertmanager.

    Args:
        alert_id: The alert id (e.g. 'ALRT-7731').
    """
    alert = incident_store.ALERTS.get(alert_id.strip().upper())
    if alert is None:
        return f"No alert found with id '{alert_id}'. Known alerts: {', '.join(incident_store.ALERTS)}."
    return json.dumps(alert, indent=2)


def execute_kubectl(command: str) -> str:
    """Runs a read-only kubectl command (get, describe, top, logs) against the production cluster.

    Args:
        command: The full kubectl command, e.g. 'kubectl get pods -l app=checkout-api'.
    """
    parts = shlex.split(command)
    if parts and parts[0] == "kubectl":
        parts = parts[1:]
    verb = parts[0] if parts else ""
    if verb not in _READ_ONLY_KUBECTL_VERBS:
        incident_store.record_action("kubectl_refused", command=command)
        return f"Refused: '{verb}' is not a read-only kubectl verb. This tool only allows {sorted(_READ_ONLY_KUBECTL_VERBS)}."
    if verb == "get" and "pods" in parts:
        return "\n".join(
            f"{svc}\tready={p['ready']}\trestarts={p['restarts_last_hour']}\timage={p['image']}"
            for svc, p in incident_store.PODS.items()
        )
    return f"[dry-run cluster] `kubectl {' '.join(parts)}` returned no anomalies."


def post_status_update(status: str, message: str) -> str:
    """Posts a public status-page update for the ongoing incident.

    Args:
        status: One of 'investigating', 'identified', 'monitoring', 'resolved'.
        message: Customer-facing update text (plain language, no internal details).
    """
    if status not in _STATUSES:
        return f"Invalid status '{status}'. Use one of {sorted(_STATUSES)}."
    incident_store.record_action("status_update", status=status, message=message)
    return f"Status page updated: [{status}] {message}"


def rollback_deploy(service: str, to_version: str, reason: str) -> str:
    """Rolls a service back to a previous deployed version (dry-run). Requires 2 human approvers.

    Args:
        service: The service to roll back (e.g. 'checkout-api').
        to_version: The known-good version to roll back to (e.g. 'v2.13.4').
        reason: One-line justification recorded in the audit ledger.
    """
    deploys = incident_store.DEPLOYS.get(service)
    if not deploys:
        return f"Unknown service '{service}' or no deploy history."
    versions = [d["version"] for d in deploys]
    if to_version not in versions:
        return f"'{to_version}' was never deployed for {service}. Known versions: {versions}."
    if to_version == versions[0]:
        return f"{to_version} is already the current version of {service}; nothing to roll back."
    incident_store.record_action(
        "rollback", service=service, from_version=versions[0], to_version=to_version, reason=reason
    )
    return f"[dry-run] {service} rolled back {versions[0]} -> {to_version}. Pods restarting; watch error rate for 5 minutes."


def scale_service(service: str, replicas: int) -> str:
    """Scales a service's replica count (dry-run). Large scale-ups pause for a human cost review.

    Args:
        service: The service to scale (e.g. 'checkout-api').
        replicas: Target replica count.
    """
    if service not in incident_store.PODS:
        return f"Unknown service '{service}'."
    # Decide *before* any side effect: resuming after approval re-invokes this function from the
    # start, so nothing non-idempotent may happen ahead of this check.
    if replicas > _AUTO_APPROVE_MAX_REPLICAS:
        raise AwaitingHumanInput(
            prompt=f"Scale {service} to {replicas} replicas? Above {_AUTO_APPROVE_MAX_REPLICAS} needs a cost review.",
            context={"service": service, "replicas": replicas},
        )
    incident_store.record_action("scale", service=service, replicas=replicas)
    return f"[dry-run] {service} scaled to {replicas} replicas."


def get_incident_timeline(alert_id: str) -> str:
    """Returns the known timeline for an incident, including remediation actions taken so far.

    Args:
        alert_id: The alert id the incident was opened for (e.g. 'ALRT-7731').
    """
    events = list(incident_store.TIMELINE.get(alert_id.strip().upper(), []))
    events += [
        {"at": e["at"], "event": f"{e['action']}: " + ", ".join(f"{k}={v}" for k, v in e.items() if k not in ("at", "action"))}
        for e in incident_store.read_ledger()
    ]
    if not events:
        return f"No timeline recorded for '{alert_id}'."
    return json.dumps(events, indent=2)
