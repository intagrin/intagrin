import tempfile
from pathlib import Path

from intagrin.runtime.memory import SQLiteCheckpointer


def test_sqlite_checkpointer():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_mem.db"
        checkpointer = SQLiteCheckpointer(str(db_path))
        
        session_id = "test_session_1"
        messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}]
        state = {"user_tier": "enterprise", "counter": 42}
        
        checkpointer.save_checkpoint(session_id, messages, state)
        
        loaded_msgs, loaded_state = checkpointer.load_checkpoint(session_id)
        assert len(loaded_msgs) == 2
        assert loaded_msgs[0]["content"] == "Hello"
        assert loaded_state["user_tier"] == "enterprise"
        assert loaded_state["counter"] == 42
        
        # Test update checkpoint
        messages.append({"role": "user", "content": "Next turn"})
        state["counter"] = 43
        checkpointer.save_checkpoint(session_id, messages, state)
        
        loaded_msgs, loaded_state = checkpointer.load_checkpoint(session_id)
        assert len(loaded_msgs) == 3
        assert loaded_state["counter"] == 43

def test_sqlite_empty_session():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_mem.db"
        checkpointer = SQLiteCheckpointer(str(db_path))
        loaded_msgs, loaded_state = checkpointer.load_checkpoint("non_existent_session")
        assert loaded_msgs == []
        assert loaded_state == {}
