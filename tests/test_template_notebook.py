"""The experiment template runs end to end on the synthetic experiment."""

from pathlib import Path

import nbformat
import pytest

nbclient = pytest.importorskip("nbclient")

NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "experiment_template.ipynb"


def test_template_notebook_synthetic(tmp_path, monkeypatch):
    monkeypatch.setenv("CRISTALLINA_SYNTHETIC", "1")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code"), "commit the template without outputs"
    nbclient.NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}).execute()
    printed = "".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []))
    assert "SYNTHETIC DATA" in printed
    assert "not in any group: [78, 79]" in printed and "no scan.json" in printed
    assert "corrections: absorption" in printed and "saved" in printed
