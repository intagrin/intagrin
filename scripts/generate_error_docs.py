"""Generates the error-code reference doc from the single source of truth: intagrin.errors.ERRORS.

Writes the SAME content to two places:
  - docs/12_Error_Reference.md            (the VitePress docs site)
  - src/intagrin/templates/copilot/reference_error_codes.md  (bundled into every project scaffolded
    via `inta copilot`, so IDE agents can look up a code without network access)

Run this after editing the registry in src/intagrin/errors.py:

    uv run python scripts/generate_error_docs.py

tests/test_error_docs_freshness.py asserts the committed docs/12_Error_Reference.md still matches
render()'s output, so CI catches a registry edit that forgot to regenerate the doc.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from intagrin.errors import ERRORS

DOCS_PATH = REPO_ROOT / "docs" / "12_Error_Reference.md"
COPILOT_TEMPLATE_PATH = (
    REPO_ROOT / "src" / "intagrin" / "templates" / "copilot" / "reference_error_codes.md"
)


def render() -> str:
    """Pure function: registry -> markdown. No I/O."""
    by_category: dict[str, list] = {}
    for spec in ERRORS.values():
        by_category.setdefault(spec.category, []).append(spec)
    for specs in by_category.values():
        specs.sort(key=lambda s: s.code)

    intro = (
        "Every codified IntaGrin error is listed below, grouped by category. Errors not yet "
        "migrated to a code keep today's plain-text messages — this list grows incrementally, "
        "it is not exhaustive."
    )
    generated_note = (
        "This file is generated from `src/intagrin/errors.py` by "
        "`scripts/generate_error_docs.py` — do not hand-edit it."
    )
    lines = [
        "# Error Code Reference",
        "",
        intro,
        "",
        generated_note,
        "",
    ]

    for category in sorted(by_category):
        lines.append(f"## {category}")
        lines.append("")
        lines.append("| Code | Title | Possible Causes |")
        lines.append("|---|---|---|")
        for spec in by_category[category]:
            causes = spec.causes.replace("\n", " ").replace("|", "\\|")
            title = spec.title.replace("|", "\\|")
            lines.append(f"| `{spec.code}` | {title} | {causes} |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    content = render()
    DOCS_PATH.write_text(content, encoding="utf-8")
    COPILOT_TEMPLATE_PATH.write_text(content, encoding="utf-8")
    print(f"Wrote {DOCS_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {COPILOT_TEMPLATE_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
