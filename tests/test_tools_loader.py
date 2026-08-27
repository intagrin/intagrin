from typing import Literal

import pytest

from intagrin.errors import IntaGrinError
from intagrin.runtime.tools_loader import get_tool_schema, load_local_tool


def sample_tool(user_id: int, status: Literal["active", "suspended"], balance: float = 100.0, is_verified: bool = True) -> str:
    """
    Fetch user record by ID.
    
    Args:
        user_id: Target user numeric identifier.
        status: Current user standing.
        balance: Account balance.
        is_verified: Verification status flag.
    """
    return f"User {user_id} ({status}) has balance ${balance}."

def test_get_tool_schema_types():
    schema = get_tool_schema(sample_tool)
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "sample_tool"
    
    params = schema["function"]["parameters"]["properties"]
    assert params["user_id"]["type"] == "integer"
    assert params["balance"]["type"] == "number"
    assert params["is_verified"]["type"] == "boolean"
    assert params["status"]["type"] == "string"
    
    required = schema["function"]["parameters"]["required"]
    assert "user_id" in required
    assert "status" in required
    assert "balance" not in required # has default
    assert "is_verified" not in required # has default

def test_load_local_tool_invalid():
    with pytest.raises(IntaGrinError) as exc_info:
        load_local_tool("non_existent_module", "fake_func")
    assert exc_info.value.code == "IG-RT-006"
