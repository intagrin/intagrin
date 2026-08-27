"""Guards against the bundled architecture reference drifting out of sync with docs/*.md.

If someone adds/edits/removes a docs/*.md page but forgets to re-run
`uv run python scripts/generate_architecture_reference.py`, this test fails — the real docs/ pages
are the single source of truth for both the bundled verbatim copies under
templates/copilot/docs/ and the auto-generated index in reference_architecture.md.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_architecture_reference", REPO_ROOT / "scripts" / "generate_architecture_reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_architecture_reference"] = module
    spec.loader.exec_module(module)
    return module


def test_every_source_doc_has_exactly_one_bundled_copy():
    gen = _load_generator()
    source_names = {p.name for p in gen._source_docs()}
    bundled_names = {p.name for p in gen.BUNDLE_DIR.glob("*.md")}
    assert bundled_names == source_names, (
        "templates/copilot/docs/ is stale relative to docs/*.md — run "
        "`uv run python scripts/generate_architecture_reference.py` and commit the result."
    )


def test_bundled_copies_match_their_source_byte_for_byte():
    gen = _load_generator()
    for doc_path in gen._source_docs():
        bundled = gen.BUNDLE_DIR / doc_path.name
        assert bundled.read_text(encoding="utf-8") == doc_path.read_text(encoding="utf-8"), (
            f"templates/copilot/docs/{doc_path.name} doesn't match docs/{doc_path.name} — run "
            "`uv run python scripts/generate_architecture_reference.py` and commit the result."
        )


def test_reference_architecture_matches_generator_output():
    gen = _load_generator()
    expected = gen.render_reference_architecture()
    actual = gen.REFERENCE_ARCHITECTURE_PATH.read_text(encoding="utf-8")
    assert actual == expected, (
        "templates/copilot/reference_architecture.md is stale — run "
        "`uv run python scripts/generate_architecture_reference.py` and commit the result."
    )


def test_excluded_docs_are_not_bundled():
    """The error-code and config-reference pages have their own, better delivery (dedicated
    Architect tools with narrow lookups) — bundling them again here would be pure redundancy."""
    gen = _load_generator()
    bundled_names = {p.name for p in gen.BUNDLE_DIR.glob("*.md")}
    assert bundled_names.isdisjoint(gen.EXCLUDED_DOCS)


def test_index_has_no_broken_tool_config_example():
    """Direct regression test for the confirmed bug this replaced: the old hand-written
    reference told the Architect to write a `function:` field on a LocalToolConfig tool block,
    which doesn't exist in the schema (only `name` + `module` do)."""
    gen = _load_generator()
    content = gen.REFERENCE_ARCHITECTURE_PATH.read_text(encoding="utf-8")
    assert "function:" not in content
