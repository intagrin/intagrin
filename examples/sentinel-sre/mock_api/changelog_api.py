"""Mock deploy-changelog REST API, consumed by Sentinel through `type: "openapi"` — IntaGrin
generates the tool wrappers straight from this app's /openapi.json. Run: python mock_api/changelog_api.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402

from tools import incident_store  # noqa: E402

PORT = 8787

app = FastAPI(title="Changelog API", version="1.0", servers=[{"url": f"http://127.0.0.1:{PORT}"}])


@app.get(
    "/deploys/{service}",
    operation_id="list_deploys",
    summary="List recent deploys for a service, newest first, with summary and canary result.",
)
def list_deploys(service: str, limit: int = 5):
    if service not in incident_store.DEPLOYS:
        raise HTTPException(404, f"Unknown service '{service}'")
    return [{k: v for k, v in d.items() if k != "diff"} for d in incident_store.DEPLOYS[service][:limit]]


@app.get(
    "/deploys/{service}/{version}/diff",
    operation_id="get_deploy_diff",
    summary="Get the code and config diff shipped in one deploy version of a service.",
)
def get_deploy_diff(service: str, version: str):
    for d in incident_store.DEPLOYS.get(service, []):
        if d["version"] == version:
            return {"service": service, "version": version, "diff": d["diff"]}
    raise HTTPException(404, f"No deploy {version} for '{service}'")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
