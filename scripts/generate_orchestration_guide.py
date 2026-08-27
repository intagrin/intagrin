"""Writes intagrin.config.orchestration_guide's GUIDE to
docs/03_Choosing_an_Orchestration_Primitive.md.

Run this after editing src/intagrin/config/orchestration_guide.py:

    uv run python scripts/generate_orchestration_guide.py

Then also run `uv run python scripts/generate_architecture_reference.py` — that script bundles
every docs/*.md page (this one included) into templates/copilot/docs/, which is how the guide
reaches any AI coding agent using `inta copilot`.

tests/test_orchestration_guide_freshness.py asserts the committed doc still matches GUIDE, so CI
catches an edit to orchestration_guide.py that forgot to regenerate the doc.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from intagrin.config.orchestration_guide import GUIDE

DOCS_PATH = REPO_ROOT / "docs" / "03_Choosing_an_Orchestration_Primitive.md"


def main() -> None:
    DOCS_PATH.write_text(GUIDE, encoding="utf-8")
    print(f"Wrote {DOCS_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
