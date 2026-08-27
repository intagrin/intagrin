import pytest

from intagrin.errors import ERRORS, ErrorSpec, IntaGrinError, get_error_spec


def test_every_error_has_non_empty_title_and_causes():
    for code, spec in ERRORS.items():
        assert isinstance(spec, ErrorSpec)
        assert spec.code == code
        assert spec.title.strip(), f"{code} has an empty title"
        assert spec.causes.strip(), f"{code} has empty causes"
        assert spec.category.strip(), f"{code} has an empty category"


def test_every_code_matches_its_category_prefix():
    prefix_by_category = {
        "Configuration": "IG-CFG-",
        "CLI Usage": "IG-CLI-",
        "Runtime": "IG-RT-",
        "MCP Integration": "IG-MCP-",
        "Server & API": "IG-SRV-",
    }
    for code, spec in ERRORS.items():
        expected_prefix = prefix_by_category[spec.category]
        assert code.startswith(expected_prefix), f"{code} doesn't match category {spec.category}"


def test_get_error_spec_returns_registered_spec():
    spec = get_error_spec("IG-CFG-001")
    assert spec.code == "IG-CFG-001"
    assert spec.title == "Missing ai.yaml configuration file"


def test_get_error_spec_raises_key_error_for_unregistered_code():
    with pytest.raises(KeyError):
        get_error_spec("IG-BOGUS-999")


def test_intagrin_error_stores_code_and_message():
    err = IntaGrinError("IG-CLI-001", "Directory 'foo' already exists.")
    assert err.code == "IG-CLI-001"
    assert err.message == "Directory 'foo' already exists."
    assert str(err) == "[IG-CLI-001] Directory 'foo' already exists."


def test_intagrin_error_falls_back_to_title_when_no_message_given():
    err = IntaGrinError("IG-MCP-001")
    assert err.message == "MCP tool not found"


def test_intagrin_error_http_status_defaults_from_registry():
    err = IntaGrinError("IG-SRV-001", "Server misconfiguration.")
    assert err.http_status == 500


def test_intagrin_error_http_status_can_be_overridden():
    err = IntaGrinError("IG-SRV-002", "Agent x not found.", http_status=404)
    assert err.http_status == 404


def test_intagrin_error_with_bogus_code_raises_key_error():
    with pytest.raises(KeyError):
        IntaGrinError("IG-BOGUS-999", "whatever")
