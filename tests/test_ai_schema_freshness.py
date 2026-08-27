"""Guards against templates/ai.schema.json drifting out of sync with config/schema.py.

If someone edits src/intagrin/config/schema.py (adds a field, changes a description) but forgets
to re-run `uv run python scripts/generate_ai_schema.py`, this test fails — the Pydantic schema is
the single source of truth and the committed, bundled schema file must always match its rendered
output, the same freshness contract test_config_reference_freshness.py enforces for the human-
readable doc.
"""

from pathlib import Path

from intagrin.config.json_schema import render_json_schema

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_bundled_ai_schema_matches_schema_render():
    expected = render_json_schema()
    actual = (
        REPO_ROOT / "src" / "intagrin" / "templates" / "ai.schema.json"
    ).read_text(encoding="utf-8")
    assert actual == expected, (
        "templates/ai.schema.json is stale — run "
        "`uv run python scripts/generate_ai_schema.py` and commit the result."
    )
