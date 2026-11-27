"""Run every docs script (regenerating figures and outputs), then zensical.

    python tools/build_docs.py          # build into site/
    python tools/build_docs.py serve    # live preview
"""
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).parents[1]
scripts = root / "docs" / "scripts"
env = {**os.environ, "DOCS_OUT": str(root / "docs" / "assets" / "generated"), "MPLBACKEND": "Agg"}
for script in sorted(scripts.glob("[!_]*.py")):
    print("running", script.name)
    subprocess.run([sys.executable, script.name], cwd=scripts, env=env, check=True, stdout=subprocess.DEVNULL)
zensical = Path(sys.executable).with_name("zensical")
subprocess.run([str(zensical), *(sys.argv[1:] or ["build"])], cwd=root, check=True)
