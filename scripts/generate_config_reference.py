"""Writes intagrin.config.reference's rendered configuration doc to the two committed files.

Run this after editing src/intagrin/config/schema.py:

    uv run python scripts/generate_config_reference.py

tests/test_config_reference_freshness.py asserts the committed files still match render()'s
output, so CI catches a schema edit that forgot to regenerate the doc. The actual rendering logic
lives in src/intagrin/config/reference.py (not here) because server/monitor.py's
`lookup_config_reference` Architect tool needs it importable at runtime from an installed package.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from intagrin.config.reference import render

DOCS_PATH = REPO_ROOT / "docs" / "13_Configuration_Reference.md"
COPILOT_TEMPLATE_PATH = (
    REPO_ROOT / "src" / "intagrin" / "templates" / "copilot" / "reference_config.md"
)


def main() -> None:
    content = render()
    DOCS_PATH.write_text(content, encoding="utf-8")
    COPILOT_TEMPLATE_PATH.write_text(content, encoding="utf-8")
    print(f"Wrote {DOCS_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {COPILOT_TEMPLATE_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
