"""Every script behind the documentation runs and passes its own checks."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "docs" / "scripts"


@pytest.mark.parametrize("script", sorted(p.name for p in SCRIPTS.glob("[!_]*.py")))
def test_docs_script(script, tmp_path):
    if script in ("rsm.py", "split_peaks.py"):
        pytest.importorskip("cristallina", reason="needs the [rsm] extra")
    env = {**os.environ, "DOCS_OUT": str(tmp_path), "MPLBACKEND": "Agg"}
    run = subprocess.run([sys.executable, script], cwd=SCRIPTS, env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-3000:]
