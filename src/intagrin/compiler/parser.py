import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError
from ruamel.yaml import YAML

from ..config.schema import AppConfig
from ..errors import IntaGrinError


class ParserError(IntaGrinError):
    pass


class ExecutionGraph:
    """
    Immutable representation of the application's configuration and initial state.
    """

    def __init__(self, config: AppConfig, env_vars: dict[str, str]):
        self.config = config
        self.env_vars = env_vars


_graph_cache = {}
_cache_lock = threading.Lock()


def parse_project(project_dir: Path) -> ExecutionGraph:
    """
    Parses the ai.yaml, loads .env, and constructs the ExecutionGraph.
    Uses smart caching with modification-time checks to prevent massive disk I/O bottlenecks
    on API servers while fully supporting hot-reloading in local development.
    """
    with _cache_lock:
        if project_dir in _graph_cache:
            mtimes, graph = _graph_cache[project_dir]
            # Fast path: check if any of the tracked files have changed
            changed = False
            for path_str, old_mtime in mtimes.items():
                p = Path(path_str)
                if not p.exists() or p.stat().st_mtime > old_mtime:
                    changed = True
                    break
            if not changed:
                return graph

    # 1. Load Environment variables
    tracked_mtimes = {}
    env_file = project_dir / ".env"
    if env_file.exists():
        tracked_mtimes[str(env_file)] = env_file.stat().st_mtime
        load_dotenv(env_file)

    # Capture env vars to store in graph
    env_vars = dict(os.environ)

    # 2. Parse ai.yaml
    ai_yaml_path = project_dir / "ai.yaml"
    if not ai_yaml_path.exists():
        raise ParserError("IG-CFG-001", f"Missing configuration file: {ai_yaml_path}")

    tracked_mtimes[str(ai_yaml_path)] = ai_yaml_path.stat().st_mtime

    try:
        yaml_parser = YAML(typ="safe")
        with open(ai_yaml_path, "r", encoding="utf-8") as f:
            raw_config = yaml_parser.load(f)

        imports = raw_config.get("imports", [])
        for imp in imports:
            path = imp.get("path")
            ns = imp.get("namespace", "")
            if not path:
                continue
            imp_path = (project_dir / path).resolve()
            if imp_path.exists():
                tracked_mtimes[str(imp_path)] = imp_path.stat().st_mtime
                with open(imp_path, "r", encoding="utf-8") as inf:
                    sub_config = yaml_parser.load(inf)
                if sub_config:
                    if "agents" in sub_config:
                        for k, v in sub_config["agents"].items():
                            raw_config.setdefault("agents", {})[ns + k] = v
                    if "workflows" in sub_config:
                        for k, v in sub_config["workflows"].items():
                            raw_config.setdefault("workflows", {})[ns + k] = v
                    if "tools" in sub_config:
                        raw_config.setdefault("tools", []).extend(sub_config["tools"])
                    if "reducers" in sub_config:
                        raw_config.setdefault("reducers", []).extend(
                            sub_config["reducers"]
                        )
    except Exception as e:
        raise ParserError("IG-CFG-002", f"Failed to parse YAML syntax or process imports:\n{e}")

    # 3. Validate Schema
    try:
        config = AppConfig(**raw_config)
    except ValidationError as e:
        error_msgs = []
        for err in e.errors():
            loc = ".".join(str(part) for part in err["loc"])
            error_msgs.append(f"  - {loc}: {err['msg']}")
        raise ParserError(
            "IG-CFG-003", "Schema validation failed for ai.yaml:\n" + "\n".join(error_msgs)
        )

    graph = ExecutionGraph(config=config, env_vars=env_vars)

    # Save to cache
    with _cache_lock:
        _graph_cache[project_dir] = (tracked_mtimes, graph)

    return graph
