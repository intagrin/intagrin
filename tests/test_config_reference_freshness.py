"""Guards against docs/13_Configuration_Reference.md drifting out of sync with config/schema.py.

If someone edits src/intagrin/config/schema.py (adds a field, changes a description) but forgets
to re-run `uv run python scripts/generate_config_reference.py`, this test fails — the Pydantic
schema is the single source of truth and the committed doc must always match its rendered output.
"""

from pathlib import Path

from intagrin.config.reference import render

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_docs_page_matches_schema_render():
    expected = render()
    actual = (REPO_ROOT / "docs" / "13_Configuration_Reference.md").read_text(encoding="utf-8")
    assert actual == expected, (
        "docs/13_Configuration_Reference.md is stale — run "
        "`uv run python scripts/generate_config_reference.py` and commit the result."
    )


def test_copilot_template_matches_schema_render():
    expected = render()
    actual = (
        REPO_ROOT / "src" / "intagrin" / "templates" / "copilot" / "reference_config.md"
    ).read_text(encoding="utf-8")
    assert actual == expected, (
        "templates/copilot/reference_config.md is stale — run "
        "`uv run python scripts/generate_config_reference.py` and commit the result."
    )
