"""Bundles the real docs/*.md pages into the package and regenerates reference_architecture.md
from them, instead of hand-maintaining a separate condensed summary that can drift from what's
actually true (confirmed: it had — see CORE_PRINCIPLES's sibling, the deleted "Deep Implementation
Guide", which told the Architect to write a `function:` field on a tool block that doesn't exist
in the schema; docs/03_Tools_and_Actions.md, the real doc on the same topic, never had that bug).

Copies every docs/NN_*.md file EXCEPT the two already-generated reference pages (error codes,
config reference — they have their own better delivery: dedicated Architect tools with narrow
lookups, so flat-copying them here would be pure redundancy) into
src/intagrin/templates/copilot/docs/, verbatim. Also writes a new reference_architecture.md: a
short hand-written CORE_PRINCIPLES preface plus an auto-generated index — one row per bundled
file, with its title and opening sentence extracted directly from the file's own content, so a doc
added later appears in the index automatically with no separate description to hand-maintain.

Run this after adding/editing/removing a docs/*.md page:

    uv run python scripts/generate_architecture_reference.py

tests/test_architecture_reference_freshness.py asserts the bundled copies and the regenerated
reference_architecture.md still match this script's output, so CI catches a docs/ change that
forgot to regenerate.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DOCS_DIR = REPO_ROOT / "docs"
BUNDLE_DIR = REPO_ROOT / "src" / "intagrin" / "templates" / "copilot" / "docs"
REFERENCE_ARCHITECTURE_PATH = (
    REPO_ROOT / "src" / "intagrin" / "templates" / "copilot" / "reference_architecture.md"
)

# Already delivered better — as dedicated Architect tools with narrow, single-item lookups
# (lookup_error_code, lookup_config_reference) — rather than a page to read in full.
EXCLUDED_DOCS = {"12_Error_Reference.md", "13_Configuration_Reference.md"}

CORE_PRINCIPLES = """# IntaGrin Architecture Reference

## Core Principles
1. **Declarative YAML Orchestration (`ai.yaml`)**
   - Agents MUST define `system_prompt_file` and `handoffs`.
   - Workflows MUST define sequences of tasks.
2. **Python Vanilla Tools (`tools/*.py`)**
   - MUST use standard Python type hints (`str`, `int`, `bool`, `typing.Literal`).
   - MUST use docstrings for LLM tool binding and parameter descriptions.
3. **Jinja2 Prompts (`prompts/*.jinja2`)**
   - Variables are injected via standard jinja double braces (`{{ user_id }}`, `{{ state_var }}`).
"""


def _source_docs() -> list[Path]:
    return sorted(p for p in DOCS_DIR.glob("*.md") if p.name not in EXCLUDED_DOCS)


def _extract_title_and_blurb(doc_path: Path) -> tuple[str, str]:
    """Pulls the `# Title` and the first sentence of the following paragraph directly out of a
    doc's own content — no separate description to hand-write or let drift from the real title."""
    text = doc_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines()]

    title = doc_path.stem
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break

    # Find the first real prose paragraph after the title — skipping headings and bold
    # "**Label:** value" metadata lines (e.g. a report's Target/Auditor/Date block), which would
    # otherwise get concatenated into a non-sentence blurb.
    metadata_line = re.compile(r"^\*\*[^*]+:\*\*")
    paragraph = ""
    seen_title = False
    for line in lines:
        if not seen_title:
            if line.startswith("# "):
                seen_title = True
            continue
        if not line or line.startswith("#") or metadata_line.match(line):
            if paragraph:
                break
            continue
        paragraph += (" " if paragraph else "") + line

    first_sentence_match = re.search(r"^(.*?[.!?])(\s|$)", paragraph)
    blurb = first_sentence_match.group(1) if first_sentence_match else paragraph
    return title, blurb.strip()


def render_index() -> str:
    intro = (
        "Each topic below is a full page under `references/docs/` — read the specific page for "
        "your question rather than guessing from this one-line summary."
    )
    lines = [
        "## Deep Reference Index",
        "",
        intro,
        "",
        "| File | Title | What it covers |",
        "|---|---|---|",
    ]
    for doc_path in _source_docs():
        title, blurb = _extract_title_and_blurb(doc_path)
        blurb = blurb.replace("|", "\\|")
        lines.append(f"| `references/docs/{doc_path.name}` | {title} | {blurb} |")
    return "\n".join(lines)


def render_reference_architecture() -> str:
    return CORE_PRINCIPLES.rstrip() + "\n\n---\n\n" + render_index().rstrip() + "\n"


def main() -> None:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    # Remove any previously-bundled file that no longer has a source doc, so a renamed/deleted
    # docs/ page doesn't leave a stale orphan copy behind.
    current_names = {p.name for p in _source_docs()}
    for existing in BUNDLE_DIR.glob("*.md"):
        if existing.name not in current_names:
            existing.unlink()

    for doc_path in _source_docs():
        (BUNDLE_DIR / doc_path.name).write_text(
            doc_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    REFERENCE_ARCHITECTURE_PATH.write_text(render_reference_architecture(), encoding="utf-8")

    print(f"Bundled {len(current_names)} doc(s) into {BUNDLE_DIR.relative_to(REPO_ROOT)}")
    print(f"Wrote {REFERENCE_ARCHITECTURE_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
