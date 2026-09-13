"""Mock observability MCP server — Prometheus metrics, Loki logs, Kubernetes pod status — served
over stdio from tools.incident_store's fixture. Swap for real Prometheus/Loki/Kubernetes MCP
servers in ai.yaml's `observability` tool entry to point Sentinel at a real cluster."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from tools import incident_store  # noqa: E402

server = MCPServer("observability")


@server.tool(structured_output=False)
def query_metrics(service: str) -> str:
    """Current golden-signal metrics (latency, errors, saturation) for one service."""
    metrics = incident_store.METRICS.get(service)
    return json.dumps({"service": service, **metrics}, indent=2) if metrics else f"No metrics for '{service}'."


@server.tool(structured_output=False)
def get_pod_status(service: str) -> str:
    """Kubernetes pod readiness, restarts, and running image for one service."""
    pods = incident_store.PODS.get(service)
    return json.dumps({"service": service, **pods}, indent=2) if pods else f"No pods for '{service}'."


@server.tool(structured_output=False)
def search_logs(service: str, minutes: int = 30) -> str:
    """Raw application log lines for one service over the last N minutes."""
    return incident_store.build_logs(service, minutes)


if __name__ == "__main__":
    server.run()
