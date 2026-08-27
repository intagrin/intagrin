import tempfile
from pathlib import Path

import yaml

from intagrin.testing.synthesizer import SyntheticEvalSynthesizer


def test_synthesize_evals():
    with tempfile.TemporaryDirectory() as tmpdir:
        p_dir = Path(tmpdir)
        (p_dir / "tools").mkdir(parents=True, exist_ok=True)
        (p_dir / "tools" / "user_tools.py").write_text("""
def query_user(user_id: int, note: str = "") -> str:
    \"\"\"Query user info.\"\"\"
    return f"User {user_id}"
""")
        
        ai_yaml = """version: "1.0"
name: "synth-app"
default_agent: "support"
model:
  primary: "gemini/gemini-2.5-flash"
memory:
  type: "sqlite"
agents:
  support:
    handoffs: ["billing"]
    tools:
      - name: "query_user"
        module: "tools.user_tools"
  billing:
    handoffs: []
"""
        (p_dir / "ai.yaml").write_text(ai_yaml)
        
        synth = SyntheticEvalSynthesizer(project_dir=p_dir, count=10)
        synth.evolve()
        
        evals_path = p_dir / "tests" / "evals.yaml"
        assert evals_path.exists()
        
        data = yaml.safe_load(evals_path.read_text())
        cases = data.get("evaluations", [])
        assert len(cases) > 0
        test_types = [c.get("test_type") for c in cases]
        assert "happy_path" in test_types
        assert "numeric_boundary" in test_types
