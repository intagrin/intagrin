"""Guards against docs/12_Error_Reference.md drifting out of sync with the error registry.

If someone edits src/intagrin/errors.py but forgets to re-run
`uv run python scripts/generate_error_docs.py`, this test fails — the registry is the single
source of truth and the committed doc must always match its rendered output exactly.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_render():
    spec = importlib.util.spec_from_file_location(
        "generate_error_docs", REPO_ROOT / "scripts" / "generate_error_docs.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_error_docs"] = module
    spec.loader.exec_module(module)
    return module.render


def test_docs_page_matches_registry_render():
    render = _load_render()
    expected = render()
    actual = (REPO_ROOT / "docs" / "12_Error_Reference.md").read_text(encoding="utf-8")
    assert actual == expected, (
        "docs/12_Error_Reference.md is stale — run "
        "`uv run python scripts/generate_error_docs.py` and commit the result."
    )


def test_copilot_template_matches_registry_render():
    render = _load_render()
    expected = render()
    actual = (
        REPO_ROOT / "src" / "intagrin" / "templates" / "copilot" / "reference_error_codes.md"
    ).read_text(encoding="utf-8")
    assert actual == expected, (
        "templates/copilot/reference_error_codes.md is stale — run "
        "`uv run python scripts/generate_error_docs.py` and commit the result."
    )
