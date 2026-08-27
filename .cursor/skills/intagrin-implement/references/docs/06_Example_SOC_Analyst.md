# Example: Cybersecurity SOC Analyst

IntaGrin is built for high-stakes enterprise workflows. Here's a multi-agent SOC swarm:

1. **`Triage_Agent`:** Reads raw CrowdStrike alerts and handles simple false positives.
2. **`Forensics_Agent`:** Uses a Python tool to upload malware to a Cuckoo Sandbox. Because IntaGrin thread-pools synchronous tools, waiting 5 minutes for the sandbox analysis does not freeze the API server.
3. **`Containment_Agent`:** Has a `block_firewall_ip` tool. Because this tool is flagged with `requires_approval: true` in `ai.yaml`, execution halts until a human engineer clicks "Approve" in the dashboard.
4. **`Reporting_Agent`:** Compiles the final analysis into a PDF for compliance.
