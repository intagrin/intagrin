"""Guards against docs/13_Configuration_Reference.md drifting out of sync with config/schema.py.

If someone edits src/intagrin/config/schema.py (adds a field, changes a description) but forgets
to re-run `uv run python scripts/generate_config_reference.py`, this test fails — the Pydantic
schema is the single source of truth and the committed doc must always match its rendered output.
"""

import importlib.util
import sys
from pathlib import Path

from intagrin.config.reference import render

REPO_ROOT = Path(__file__).resolve().parent.parent


def _script_frontmatter() -> str:
    """The docs-site page carries SEO frontmatter the copilot template doesn't — read from the
    generator itself so the expected value can never drift from what it writes."""
    spec = importlib.util.spec_from_file_location(
        "generate_config_reference", REPO_ROOT / "scripts" / "generate_config_reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_config_reference"] = module
    spec.loader.exec_module(module)
    return module.FRONTMATTER


def test_docs_page_matches_schema_render():
    expected = _script_frontmatter() + render()
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
