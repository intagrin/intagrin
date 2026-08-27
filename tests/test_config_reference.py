from intagrin.config.reference import ROOT_SECTION_NAME, render, render_sections
from intagrin.config.schema import AppConfig


def test_render_sections_covers_every_defs_model_plus_root():
    schema = AppConfig.model_json_schema()
    expected_keys = set(schema["$defs"].keys()) | {ROOT_SECTION_NAME}

    assert set(render_sections().keys()) == expected_keys


def test_every_section_is_non_empty_and_has_a_table():
    for name, content in render_sections().items():
        assert content.strip(), f"section {name} is empty"
        assert "| Field | Type | Default | Description |" in content


def test_full_document_has_no_undocumented_fields():
    """Every field in the schema should have a real description — a regression here means a new
    field was added to schema.py without Field(description=...)."""
    assert "(undocumented)" not in render()


def test_full_document_contains_auth_config_details():
    """Direct regression test for the incident that prompted this feature: the Architect couldn't
    find how to configure monitor/server authentication."""
    doc = render()
    assert "AuthConfig" in doc
    assert "api_key" in doc
    assert "INTAGRIN_API_KEY" in doc


def test_render_is_deterministic():
    assert render() == render()
