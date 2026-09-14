"""The dependency rule is a test, not a convention."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_the_rings_only_point_inward() -> None:
    # A subprocess: run in-process, import-linter reconfigures logging and leaks into other tests.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from importlinter.cli import lint_imports; "
            "sys.exit(lint_imports(no_cache=True))",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
