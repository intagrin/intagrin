import subprocess
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SRC = REPO_ROOT / "src" / "intagrin"


def test_wheel_includes_every_non_python_source_file(tmp_path):
    """Regression test for a real bug: db_migrations/alembic.ini (needed by run_auto_migrations
    the moment a project's memory.type is sqlite/postgres) existed in the source tree but wasn't
    covered by any [tool.setuptools.package-data] glob in pyproject.toml, so a real `pip install`
    (unlike this repo's own editable dev checkout, which reads straight from the source tree)
    shipped without it — "Auto-Migration Error: .../db_migrations/alembic.ini doesn't exist" the
    moment a real user's project actually hit that code path.

    Rather than hardcoding a list of files to check (which rots the moment a new asset is added
    without updating this test too), this builds the real wheel and diffs it against every
    non-.py file actually present in the source tree — so any future asset not covered by a
    package-data glob fails here, in CI, instead of in a user's install."""
    # setuptools reuses build/ and *.egg-info/ from any prior build in the repo root — without
    # clearing them first, a stale manifest from an earlier (correct) build silently papers over
    # a regression here, since the wheel gets built from the cached file list instead of a fresh
    # one reflecting the current pyproject.toml. Both are gitignored, safe to remove.
    import shutil

    shutil.rmtree(REPO_ROOT / "build", ignore_errors=True)
    for egg_info in REPO_ROOT.glob("src/*.egg-info"):
        shutil.rmtree(egg_info, ignore_errors=True)

    out_dir = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--wheel", "-o", str(out_dir)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    wheel_path = next(out_dir.glob("*.whl"))

    with zipfile.ZipFile(wheel_path) as z:
        wheel_names = set(z.namelist())

    missing = []
    for path in PACKAGE_SRC.rglob("*"):
        if path.is_dir() or path.suffix in (".py", ".pyc"):
            continue
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(PACKAGE_SRC.parent).as_posix()  # "intagrin/..."
        if rel not in wheel_names:
            missing.append(rel)

    assert not missing, (
        "these files exist in src/intagrin/ but are missing from the built wheel — add a "
        f"matching glob to [tool.setuptools.package-data] in pyproject.toml: {missing}"
    )
