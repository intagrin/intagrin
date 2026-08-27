"""Generates the raw JSON Schema for `ai.yaml`, for editor autocomplete/validation
(yaml-language-server, VS Code's redhat.vscode-yaml extension, and any other LSP client that
understands the `# yaml-language-server: $schema=...` modeline) — not for human reading, that's
`config/reference.py`. Both are derived from the same single source of truth,
`AppConfig.model_json_schema()`.

Lives in the package (not in scripts/) so a project scaffolded via an installed `pip install
intagrin` still gets a real, current schema file from `inta new` — `cli.py`'s `new` command
imports `render_json_schema()` at runtime. `scripts/generate_ai_schema.py` is the thin CLI wrapper
that writes this module's output to the bundled `templates/ai.schema.json` copy `new` reads from.
"""

import json

from .schema import AppConfig


def _forbid_unknown_properties(node: object) -> None:
    """Recursively sets `additionalProperties: false` on every object-with-`properties` fragment
    (every actual model schema, at any $defs nesting depth) that doesn't already declare it.

    Pydantic's runtime models default to `extra="ignore"` (a typo'd key is silently dropped, not
    an error), so model_json_schema() doesn't set this on its own — without it, an editor schema
    would validate YAML syntax but never catch a misspelled field name, which is the single most
    common thing this schema exists to catch. Skips `dict[str, X]`-shaped fields on purpose: those
    render as `additionalProperties: {$ref: ...}` (a real value schema, not absent) with no
    `properties` key of their own, so the `"properties" in node` guard below leaves them untouched.
    """
    if isinstance(node, dict):
        if "properties" in node and "additionalProperties" not in node:
            node["additionalProperties"] = False
        for value in node.values():
            _forbid_unknown_properties(value)
    elif isinstance(node, list):
        for item in node:
            _forbid_unknown_properties(item)


def render_json_schema() -> str:
    schema = AppConfig.model_json_schema()
    _forbid_unknown_properties(schema)
    schema["title"] = "IntaGrin ai.yaml"
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    ordered = {"$schema": schema.pop("$schema"), "title": schema.pop("title"), **schema}
    return json.dumps(ordered, indent=2, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    # `uv run python -m intagrin.config.json_schema > ai.schema.json` refreshes an existing
    # project's copy after an IntaGrin upgrade, without needing to re-run `inta new` — see
    # docs/01_Getting_Started.md.
    import sys

    sys.stdout.write(render_json_schema())
