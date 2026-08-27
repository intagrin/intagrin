"""Guards against docs/03_Choosing_an_Orchestration_Primitive.md drifting out of sync with
config/orchestration_guide.py.

If someone edits src/intagrin/config/orchestration_guide.py but forgets to re-run
`uv run python scripts/generate_orchestration_guide.py`, this test fails — GUIDE is the single
source of truth every consumer (inta compile, run_architect, the bundled IDE-skill docs) reads
from, and the committed doc must always match it exactly, the same freshness contract
config/reference.py and errors.py already have.
"""

from pathlib import Path

from intagrin.config.orchestration_guide import GUIDE

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_docs_page_matches_guide():
    actual = (
        REPO_ROOT / "docs" / "03_Choosing_an_Orchestration_Primitive.md"
    ).read_text(encoding="utf-8")
    assert actual == GUIDE, (
        "docs/03_Choosing_an_Orchestration_Primitive.md is stale — run "
        "`uv run python scripts/generate_orchestration_guide.py` and commit the result."
    )


def test_guide_mentions_all_six_primitives():
    """A regression guard for the exact bug this module exists to fix: the guide previously
    lived as two independently hand-written paragraphs, neither of which mentioned auto_route."""
    for primitive in ["handoffs", "delegations", "routers", "auto_route", "spawns", "workflows"]:
        assert primitive in GUIDE, f"orchestration_guide.GUIDE never mentions {primitive!r}"
