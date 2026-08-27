"""Writes intagrin.config.json_schema's rendered JSON Schema to the bundled template copy that
`inta new` scaffolds into every new project as `ai.schema.json`.

Run this after editing src/intagrin/config/schema.py:

    uv run python scripts/generate_ai_schema.py

tests/test_ai_schema_freshness.py asserts the committed file still matches render_json_schema()'s
output, so CI catches a schema edit that forgot to regenerate it. The actual rendering logic lives
in src/intagrin/config/json_schema.py (not here) because cli.py's `new` command needs it
importable at runtime from an installed package.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from intagrin.config.json_schema import render_json_schema

SCHEMA_PATH = REPO_ROOT / "src" / "intagrin" / "templates" / "ai.schema.json"


def main() -> None:
    SCHEMA_PATH.write_text(render_json_schema(), encoding="utf-8")
    print(f"Wrote {SCHEMA_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
