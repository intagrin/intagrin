import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBasicCredentials
from fastapi.testclient import TestClient

from intagrin.errors import IntaGrinError
from intagrin.server import monitor
from intagrin.server.api import ChatRequest, ResumeRequest
from intagrin.server.monitor import (
    ApplyRequest,
    ArchitectRequest,
    SyncGraphRequest,
    apply_architect,
    get_config,
    get_docs,
    get_logs,
    get_memory,
    get_previewable_file,
    monitor_chat,
    monitor_resume,
    monitor_stream,
    run_architect,
    stream_events,
    sync_blueprint,
    sync_graph,
    verify_monitor_auth,
)

VALID_AI_YAML = """name: t
version: "1.0"
default_agent: triage
model:
  primary: mock/model
memory:
  type: sqlite
agents:
  triage:
    description: hi
  billing:
    description: money
"""

INVALID_AI_YAML = 'name: t\nversion: "1.0"\nagents: {}\n'  # missing model/memory/default_agent


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ai.yaml").write_text(VALID_AI_YAML)
    return tmp_path


def test_sync_graph_rejects_nonexistent_target_and_writes_nothing(project):
    before = (project / "ai.yaml").read_text()
    req = SyncGraphRequest(
        agent_id="triage", target_id="does_not_exist", edge_type="handoff", action="add"
    )
    with pytest.raises(IntaGrinError) as exc_info:
        sync_graph(req)
    assert exc_info.value.code == "IG-SRV-002"
    assert exc_info.value.http_status == 400
    assert (project / "ai.yaml").read_text() == before


def test_sync_graph_accepts_a_real_target(project):
    req = SyncGraphRequest(agent_id="triage", target_id="billing", edge_type="handoff", action="add")
    result = sync_graph(req)
    assert result == {"status": "success"}
    assert "billing" in (project / "ai.yaml").read_text()


def test_sync_graph_allows_removing_a_dangling_reference(project):
    """Removing a reference must stay allowed even if the target no longer exists — that's
    cleanup, not new corruption."""
    (project / "ai.yaml").write_text(
        VALID_AI_YAML.replace(
            "  triage:\n    description: hi\n",
            "  triage:\n    description: hi\n    handoffs: [\"already_deleted_agent\"]\n",
        )
    )
    req = SyncGraphRequest(
        agent_id="triage", target_id="already_deleted_agent", edge_type="handoff", action="remove"
    )
    result = sync_graph(req)
    assert result == {"status": "success"}
    assert "already_deleted_agent" not in (project / "ai.yaml").read_text()


def test_apply_architect_rejects_whole_batch_when_ai_yaml_is_invalid(project):
    prompt_path = project / "prompts" / "triage.jinja2"
    req = ApplyRequest(
        files_to_write=[
            {"filepath": "ai.yaml", "content": INVALID_AI_YAML},
            {"filepath": "prompts/triage.jinja2", "content": "You are triage."},
        ]
    )
    original_ai_yaml = (project / "ai.yaml").read_text()

    with pytest.raises(IntaGrinError) as exc_info:
        apply_architect(req)
    assert exc_info.value.code == "IG-SRV-003"
    assert exc_info.value.http_status == 400
    assert (project / "ai.yaml").read_text() == original_ai_yaml
    assert not prompt_path.exists()


def test_apply_architect_writes_the_batch_when_ai_yaml_is_valid(project):
    prompt_path = project / "prompts" / "triage.jinja2"
    req = ApplyRequest(
        files_to_write=[
            {"filepath": "ai.yaml", "content": VALID_AI_YAML},
            {"filepath": "prompts/triage.jinja2", "content": "You are triage."},
        ]
    )
    result = apply_architect(req)
    assert result == {"status": "success"}
    assert prompt_path.exists()
    assert prompt_path.read_text() == "You are triage."


def test_apply_architect_still_writes_non_yaml_files_with_no_ai_yaml_in_batch(project):
    prompt_path = project / "prompts" / "triage.jinja2"
    req = ApplyRequest(files_to_write=[{"filepath": "prompts/triage.jinja2", "content": "Hi."}])
    result = apply_architect(req)
    assert result == {"status": "success"}
    assert prompt_path.read_text() == "Hi."


def test_get_docs_falls_back_to_bundled_templates_copy_when_no_dev_docs_dir_exists(
    tmp_path, monkeypatch
):
    """Regression guard: a real (non-editable) install of intagrin has no repo-root docs/
    directory alongside the package — only src/intagrin/templates/copilot/docs/ ships with every
    install. Without this fallback, get_docs() returns "No Documentation Found" for any project
    not running from this repo's own dev checkout, which is exactly the bug report this guards."""
    fake_pkg_root = tmp_path / "fake_install" / "intagrin"
    fake_monitor_file = fake_pkg_root / "server" / "monitor.py"
    bundled_docs_dir = fake_pkg_root / "templates" / "copilot" / "docs"
    bundled_docs_dir.mkdir(parents=True)
    (bundled_docs_dir / "01_Getting_Started.md").write_text("# Getting Started\n")

    # No docs/ next to fake_pkg_root, and cwd has no docs/ either.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(monitor, "__file__", str(fake_monitor_file))

    result = get_docs()

    assert result != []
    assert result[0]["filename"] != "Error.md"
    assert any(d["filename"] == "01_Getting_Started.md" for d in result)


def test_get_docs_prefers_dev_repo_docs_dir_when_present(monkeypatch):
    """When a real docs/ directory exists (this repo's own dev checkout), it must still win over
    the bundled templates copy — the fallback added above must not change today's behavior here."""
    result = get_docs()
    assert result != []
    assert result[0]["filename"] != "Error.md"
    assert len(result) > 1


# --- Previously zero-coverage endpoints (server/monitor.py) --------------------------------


def test_serve_dashboard_returns_the_monitor_html(project):
    client = TestClient(monitor.app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_get_config_returns_the_parsed_ai_yaml_with_context_window(project):
    payload = get_config()
    assert payload["name"] == "t"
    assert payload["default_agent"] == "triage"
    assert "_context_window" in payload
    assert payload["_context_window"] > 0


def test_get_memory_returns_empty_list_for_a_fresh_project_with_no_checkpoints_yet(project):
    assert get_memory(user_context="global_tenant") == []


def test_get_logs_returns_empty_list_for_a_fresh_project_with_no_runs_yet(project):
    assert get_logs(user_context="global_tenant") == []


def test_stream_events_returns_a_streaming_response_without_blocking(project):
    """Constructing the StreamingResponse must not itself start consuming the (infinite)
    event_generator — this call must return immediately."""
    response = asyncio.run(stream_events(user_context="global_tenant"))
    assert isinstance(response, StreamingResponse)


def test_run_architect_returns_the_final_json_reply_with_no_tool_calls(project):
    mock_message = MagicMock(
        content='{"message": "Sure, here is my answer.", "files_to_write": []}',
        tool_calls=None,
    )
    mock_response = MagicMock(choices=[MagicMock(message=mock_message)])

    with patch(
        "litellm.acompletion", new_callable=AsyncMock
    ) as mock_acompletion:
        mock_acompletion.return_value = mock_response
        result = asyncio.run(
            run_architect(ArchitectRequest(messages=[{"role": "user", "content": "hi"}]))
        )

    assert result["status"] == "success"
    assert result["message"] == "Sure, here is my answer."
    assert result["files_to_write"] == []


def test_run_architect_dispatches_a_tool_call_then_returns_the_final_reply(project):
    """Exercises the async-converted tool-call loop end to end: a first turn requesting
    list_directory, a second turn with no more tool calls."""
    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function.name = "list_directory"
    tool_call.function.arguments = '{"dirpath": "."}'

    first_message = MagicMock(tool_calls=[tool_call])
    first_message.model_dump.return_value = {
        "role": "assistant",
        "tool_calls": [{"id": "call_1"}],
    }
    second_message = MagicMock(
        content='{"message": "Found your ai.yaml.", "files_to_write": []}', tool_calls=None
    )

    responses = [
        MagicMock(choices=[MagicMock(message=first_message)]),
        MagicMock(choices=[MagicMock(message=second_message)]),
    ]

    with patch(
        "litellm.acompletion", new_callable=AsyncMock
    ) as mock_acompletion:
        mock_acompletion.side_effect = responses
        result = asyncio.run(
            run_architect(
                ArchitectRequest(messages=[{"role": "user", "content": "what files are here?"}])
            )
        )

    assert result["status"] == "success"
    assert result["message"] == "Found your ai.yaml."


def test_run_architect_exhausting_max_turns_still_returns_a_success_response(project):
    """Real-world bug report: a genuinely complex, multi-file request made the Architect keep
    calling tools (list_directory/read_file) without ever reaching a final no-tool-calls answer,
    hitting max_turns and throwing away every turn of exploration as a dead-end 500 error. The
    fallback must force one final tools-less completion and still return status: "success" with
    whatever the LLM could summarize, not raise."""
    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function.name = "list_directory"
    tool_call.function.arguments = '{"dirpath": "."}'

    # Every regular turn keeps calling the same tool, forever — exactly what exhausts max_turns.
    looping_message = MagicMock(tool_calls=[tool_call])
    looping_message.model_dump.return_value = {
        "role": "assistant",
        "tool_calls": [{"id": "call_1"}],
    }
    looping_response = MagicMock(choices=[MagicMock(message=looping_message)])

    # The final, tools-omitted forced completion — no tool_calls attribute access needed since
    # run_architect's fallback path never checks msg.tool_calls at all.
    final_message = MagicMock(
        content='{"message": "I explored what I could but need more direction.", "files_to_write": []}'
    )
    final_response = MagicMock(choices=[MagicMock(message=final_message)])

    with patch(
        "litellm.acompletion", new_callable=AsyncMock
    ) as mock_acompletion:
        mock_acompletion.side_effect = [looping_response] * 10 + [final_response]
        result = asyncio.run(
            run_architect(
                ArchitectRequest(messages=[{"role": "user", "content": "a very complex ask"}])
            )
        )

    assert result["status"] == "success"
    assert result["message"] == "I explored what I could but need more direction."
    # 10 looping calls (max_turns) + 1 forced final call.
    assert mock_acompletion.call_count == 11
    # The forced final call must not offer tools — otherwise the LLM could just call one again.
    assert "tools" not in mock_acompletion.call_args_list[-1].kwargs


def test_run_architect_read_file_blocks_paths_outside_the_workspace(project, tmp_path):
    """Path-traversal guard, exercised through the now-async tool dispatch."""
    outside_file = tmp_path.parent / "secret.txt"
    outside_file.write_text("top secret")

    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function.name = "read_file"
    tool_call.function.arguments = f'{{"filepath": "{outside_file}"}}'

    first_message = MagicMock(tool_calls=[tool_call])
    first_message.model_dump.return_value = {
        "role": "assistant",
        "tool_calls": [{"id": "call_1"}],
    }
    second_message = MagicMock(content='{"message": "Denied."}', tool_calls=None)

    responses = [
        MagicMock(choices=[MagicMock(message=first_message)]),
        MagicMock(choices=[MagicMock(message=second_message)]),
    ]

    with patch(
        "litellm.acompletion", new_callable=AsyncMock
    ) as mock_acompletion:
        mock_acompletion.side_effect = responses
        asyncio.run(
            run_architect(ArchitectRequest(messages=[{"role": "user", "content": "read it"}]))
        )
    # The tool result fed back to the LLM (mock_acompletion's 2nd call) must show the denial,
    # not the secret file's contents.
    second_call_messages = mock_acompletion.call_args_list[1].kwargs["messages"]
    tool_result = next(m for m in second_call_messages if m.get("role") == "tool")
    assert "Access denied" in tool_result["content"]
    assert "top secret" not in tool_result["content"]


def test_sync_blueprint_writes_llm_generated_content(project):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="# Updated Blueprint\n\nNew content."))]

    with patch("litellm.completion", return_value=mock_response):
        result = sync_blueprint()

    assert result == {"status": "success"}
    assert "Updated Blueprint" in (project / "blueprint.md").read_text()


def test_monitor_chat_delegates_to_chat_endpoint(project):
    sentinel = MagicMock()
    with patch(
        "intagrin.server.monitor.chat_endpoint", new_callable=AsyncMock
    ) as mock_chat:
        mock_chat.return_value = sentinel
        result = asyncio.run(
            monitor_chat(ChatRequest(message="hi", session_id="s1"), user_context="global_tenant")
        )
    mock_chat.assert_awaited_once()
    assert mock_chat.call_args.kwargs["user_context"] == "global_tenant"
    assert result is sentinel


def test_monitor_stream_delegates_to_stream_endpoint(project):
    sentinel = MagicMock()
    with patch(
        "intagrin.server.monitor.stream_endpoint", new_callable=AsyncMock
    ) as mock_stream:
        mock_stream.return_value = sentinel
        result = asyncio.run(
            monitor_stream(
                ChatRequest(message="hi", session_id="s1"), user_context="global_tenant"
            )
        )
    mock_stream.assert_awaited_once()
    assert result is sentinel


def test_monitor_resume_delegates_to_resume_endpoint(project):
    sentinel = MagicMock()
    with patch(
        "intagrin.server.monitor.resume_endpoint", new_callable=AsyncMock
    ) as mock_resume:
        mock_resume.return_value = sentinel
        result = asyncio.run(
            monitor_resume(
                ResumeRequest(session_id="s1", approved=True),
                request=MagicMock(headers={}),
                user_context="global_tenant",
            )
        )
    mock_resume.assert_awaited_once()
    assert result is sentinel


API_KEY_AI_YAML = """name: t
version: "1.0"
default_agent: triage
model:
  primary: mock/model
memory:
  type: sqlite
server:
  auth:
    type: api_key
    env_var: MONITOR_TEST_KEY
agents:
  triage:
    description: hi
"""


@pytest.fixture
def project_with_api_key_auth(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ai.yaml").write_text(API_KEY_AI_YAML)
    monkeypatch.setenv("MONITOR_TEST_KEY", "s3cr3t")
    return tmp_path


def test_verify_monitor_auth_accepts_key_in_password_field(project_with_api_key_auth):
    creds = HTTPBasicCredentials(username="admin", password="s3cr3t")
    assert verify_monitor_auth(creds) == "global_tenant"


def test_verify_monitor_auth_ignores_username_value(project_with_api_key_auth):
    """Basic Auth requires *a* username, but there are no user accounts here — only the password
    field is checked, so any username (not just the documented 'admin' convention) works."""
    creds = HTTPBasicCredentials(username="whatever", password="s3cr3t")
    assert verify_monitor_auth(creds) == "global_tenant"


def test_verify_monitor_auth_rejects_key_in_username_field_only(project_with_api_key_auth):
    """Regression test: the key used to be accepted in *either* field. Putting it in the username
    field with a wrong/empty password must now be rejected — password is the only field checked."""
    creds = HTTPBasicCredentials(username="s3cr3t", password="wrong")
    with pytest.raises(HTTPException) as exc_info:
        verify_monitor_auth(creds)
    assert exc_info.value.status_code == 401


def test_verify_monitor_auth_rejects_wrong_password(project_with_api_key_auth):
    creds = HTTPBasicCredentials(username="admin", password="wrong")
    with pytest.raises(HTTPException) as exc_info:
        verify_monitor_auth(creds)
    assert exc_info.value.status_code == 401


# --- GET /api/files/{file_path} — inline media preview for tool-generated images/audio/video ---


def test_get_previewable_file_serves_an_allowed_image(project):
    (project / "generated_images").mkdir()
    (project / "generated_images" / "post.png").write_bytes(b"\x89PNG fake but real enough bytes")

    client = TestClient(monitor.app)
    response = client.get("/api/files/generated_images/post.png")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == b"\x89PNG fake but real enough bytes"
    # No forced download — an <img> tag needs this served inline, not as an attachment.
    assert "attachment" not in response.headers.get("content-disposition", "")


def test_get_previewable_file_rejects_path_traversal_outside_project(project):
    """`project` IS `tmp_path` (the fixture just returns it), so the sibling directory used here
    is genuinely outside the project root, not just a different-looking path inside it. Called
    directly rather than through the HTTP route: httpx/Starlette normalize `..` out of a URL
    before routing ever sees it, so a real `client.get("/api/files/../x")` never reaches this
    code at all — that's a second layer of defense, not a substitute for verifying the
    containment check itself actually rejects a literal `..` argument."""
    outside_dir = project.parent / "sibling_outside_project"
    outside_dir.mkdir()
    (outside_dir / "secret.png").write_bytes(b"should never be servable")

    with pytest.raises(IntaGrinError) as exc_info:
        get_previewable_file(f"../{outside_dir.name}/secret.png")
    assert exc_info.value.code == "IG-SRV-004"


def test_get_previewable_file_rejects_an_absolute_path_argument(project):
    """Path's own `/` operator discards the left operand entirely when the right operand is
    absolute (`Path("/a") / "/b" == Path("/b")`) — a naive `project_dir / file_path` join would
    silently serve ANY absolute path handed to it, bypassing containment entirely. Called
    directly (not through the HTTP route) since it's ambiguous whether a real URL can even
    encode a leading slash into this path param — this exercises the Python-level join gotcha
    itself, independent of that."""
    outside_dir = project.parent / "sibling_outside_project_2"
    outside_dir.mkdir()
    outside = outside_dir / "secret.png"
    outside.write_bytes(b"should never be servable")

    with pytest.raises(IntaGrinError) as exc_info:
        get_previewable_file(str(outside))
    assert exc_info.value.code == "IG-SRV-004"


def test_get_previewable_file_rejects_a_disallowed_extension(project):
    """Existing, inside the project, but not a previewable media type — e.g. this must never
    become a way to read ai.yaml/.env/source files inline, even though an authenticated
    dashboard user could already read them via the Architect chat's read_file tool."""
    (project / ".env").write_text("SECRET_KEY=doNotServeThis")

    client = TestClient(monitor.app)
    response = client.get("/api/files/.env")

    assert response.status_code == 404
    assert response.json()["code"] == "IG-SRV-004"


def test_get_previewable_file_rejects_a_missing_file(project):
    client = TestClient(monitor.app)
    response = client.get("/api/files/generated_images/does_not_exist.png")

    assert response.status_code == 404
    assert response.json()["code"] == "IG-SRV-004"


def test_get_previewable_file_rejects_a_file_over_the_size_limit(project, monkeypatch):
    monkeypatch.setattr(monitor, "_MAX_PREVIEW_FILE_BYTES", 10)
    (project / "big.png").write_bytes(b"x" * 11)

    client = TestClient(monitor.app)
    response = client.get("/api/files/big.png")

    assert response.status_code == 404
    assert response.json()["code"] == "IG-SRV-004"


def test_get_previewable_file_requires_auth_when_configured(project_with_api_key_auth):
    (project_with_api_key_auth / "post.png").write_bytes(b"fake png bytes")
    client = TestClient(monitor.app)

    unauthenticated = client.get("/api/files/post.png")
    assert unauthenticated.status_code == 401

    authenticated = client.get("/api/files/post.png", auth=("admin", "s3cr3t"))
    assert authenticated.status_code == 200
    assert authenticated.content == b"fake png bytes"
