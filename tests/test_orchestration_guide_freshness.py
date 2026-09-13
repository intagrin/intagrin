"""Guards against docs/03_Choosing_an_Orchestration_Primitive.md drifting out of sync with
config/orchestration_guide.py.

If someone edits src/intagrin/config/orchestration_guide.py but forgets to re-run
`uv run python scripts/generate_orchestration_guide.py`, this test fails — GUIDE is the single
source of truth every consumer (inta compile, run_architect, the bundled IDE-skill docs) reads
from, and the committed doc must always match it exactly, the same freshness contract
config/reference.py and errors.py already have.
"""

import importlib.util
import sys
from pathlib import Path

from intagrin.config.orchestration_guide import GUIDE

REPO_ROOT = Path(__file__).resolve().parent.parent


def _script_frontmatter() -> str:
    """The docs-site page carries SEO frontmatter GUIDE itself doesn't (GUIDE is spliced into
    system prompts) — read from the generator so the expected value can't drift from what it
    writes."""
    spec = importlib.util.spec_from_file_location(
        "generate_orchestration_guide", REPO_ROOT / "scripts" / "generate_orchestration_guide.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_orchestration_guide"] = module
    spec.loader.exec_module(module)
    return module.FRONTMATTER


def test_docs_page_matches_guide():
    actual = (
        REPO_ROOT / "docs" / "03_Choosing_an_Orchestration_Primitive.md"
    ).read_text(encoding="utf-8")
    assert actual == _script_frontmatter() + GUIDE, (
        "docs/03_Choosing_an_Orchestration_Primitive.md is stale — run "
        "`uv run python scripts/generate_orchestration_guide.py` and commit the result."
    )


def test_guide_mentions_all_six_primitives():
    """A regression guard for the exact bug this module exists to fix: the guide previously
    lived as two independently hand-written paragraphs, neither of which mentioned auto_route."""
    for primitive in ["handoffs", "delegations", "routers", "auto_route", "spawns", "workflows"]:
        assert primitive in GUIDE, f"orchestration_guide.GUIDE never mentions {primitive!r}"
