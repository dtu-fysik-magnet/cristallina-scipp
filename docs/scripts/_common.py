"""Shared setup for the documentation scripts: output folder and a quiet matplotlib."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

OUT = Path(os.environ.get("DOCS_OUT", Path(__file__).parents[1] / "assets" / "generated"))
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    """Save a matplotlib or plopp figure as SVG for the docs."""
    path = OUT / f"{name}.svg"
    fig.savefig(path, bbox_inches="tight") if hasattr(fig, "savefig") else fig.save(path)


def text(name, *lines):
    """Write printed results that a page includes with a snippet."""
    (OUT / f"{name}.txt").write_text("\n".join(str(line) for line in lines) + "\n")
