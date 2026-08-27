
import pytest
import yaml
from fastapi.testclient import TestClient

from intagrin.server.monitor import app

client = TestClient(app)

@pytest.fixture
def mock_ai_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yaml_file = tmp_path / "ai.yaml"
    
    initial_data = {
        "version": "1.0",
        "name": "test_project",
        "memory": {
            "type": "sqlite",
            "db_path": ".ai/test_mem.db"
        },
        "model": {
            "primary": "test-model"
        },
        "default_agent": "triage",
        "agents": {
            "triage": {
                "system_prompt_file": "prompts/triage.jinja2",
                "handoffs": ["support"]
            },
            "support": {
                "system_prompt_file": "prompts/support.jinja2"
            },
            "billing": {
                "system_prompt_file": "prompts/billing.jinja2"
            }
        }
    }
    with open(yaml_file, "w") as f:
        yaml.dump(initial_data, f)

    # Needs a prompts directory to pass parse_project validation
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "triage.jinja2").write_text("Triage")
    (tmp_path / "prompts" / "support.jinja2").write_text("Support")
    (tmp_path / "prompts" / "billing.jinja2").write_text("Billing")

    return yaml_file

def test_sync_graph_handoff(mock_ai_yaml):
    response = client.post("/api/graph/sync", json={
        "agent_id": "support",
        "target_id": "triage",
        "edge_type": "handoff"
    })
    print(response.json())
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    
    with open(mock_ai_yaml, "r") as f:
        data = yaml.safe_load(f)
        
    assert "triage" in data["agents"]["support"]["handoffs"]

def test_sync_graph_delegation(mock_ai_yaml):
    response = client.post("/api/graph/sync", json={
        "agent_id": "triage",
        "target_id": "billing",
        "edge_type": "delegation"
    })
    
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    
    with open(mock_ai_yaml, "r") as f:
        data = yaml.safe_load(f)
        
    assert "billing" in data["agents"]["triage"]["delegations"]

def test_sync_graph_invalid_agent(mock_ai_yaml):
    response = client.post("/api/graph/sync", json={
        "agent_id": "nonexistent_agent",
        "target_id": "triage",
        "edge_type": "handoff"
    })

    assert response.status_code == 400
    assert "not found" in response.json()["detail"]


def test_sync_graph_invalid_target(mock_ai_yaml):
    """A drag-and-drop edge to a target agent that doesn't exist must be rejected, not silently
    written as a dangling reference nothing else validates."""
    response = client.post("/api/graph/sync", json={
        "agent_id": "triage",
        "target_id": "nonexistent_agent",
        "edge_type": "handoff",
    })

    assert response.status_code == 400

    with open(mock_ai_yaml, "r") as f:
        data = yaml.safe_load(f)
    assert "nonexistent_agent" not in data["agents"]["triage"].get("handoffs", [])
