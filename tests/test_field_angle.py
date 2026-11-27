"""Field direction relative to the sample, from the vector magnet's components."""

import numpy as np
import pytest
import scipp as sc

import cristallina_scipp as cs
from cristallina_scipp import testing


def series_with_fields(fields):
    b = np.asarray(fields, dtype=float)
    coords = {"run": sc.arange("run", len(b), unit=None)}
    coords |= {f"field_{c}": sc.array(dims=["run"], values=b[:, i], unit="T") for i, c in enumerate("xyz")}
    summary = sc.Dataset({"integral": sc.ones(dims=["run"], shape=[len(b)])}, coords=coords)
    return sc.DataGroup({"summary": summary, "settings": sc.DataGroup({})})


def test_angles_in_and_out_of_the_plane():
    s = 2 / np.sqrt(2)
    out = cs.add_field_angles(series_with_fields([[0, 0, 2], [0, s, s], [0, 2, 0], [0, -2, 0], [2, 0, 0], [0, 0, 0]]),
                              axis=(0, 0, 1), toward=(0, 1, 0))
    angle, tilt = (out["summary"].coords[c].values for c in ("field_angle", "field_tilt"))
    assert np.allclose(angle[:4], [0, 45, 90, -90]) and np.allclose(tilt[:4], 0)
    assert np.isclose(abs(tilt[4]), 90)
    assert np.isnan(angle[5]) and np.isnan(tilt[5])  # no field, no direction
    assert out["settings"]["field_axis"] == "0 0 1" and out["settings"]["field_toward"] == "0 1 0"


def test_toward_is_made_perpendicular_and_errors():
    out = cs.add_field_angles(series_with_fields([[0, 1, 1]]), axis=(0, 0, 1), toward=(0, 1, 1))
    assert np.isclose(out["summary"].coords["field_angle"].values[0], 45)
    with pytest.raises(ValueError, match="parallel"):
        cs.add_field_angles(series_with_fields([[0, 0, 1]]), axis=(0, 0, 1), toward=(0, 0, 2))
    with pytest.raises(ValueError, match="already has"):
        cs.add_field_angles(out, axis=(0, 0, 1), toward=(0, 1, 0))


ROTATION = """raw = "raw"
[rois]
"peak" = { left = 0, right = 64, bottom = 22, top = 38 }
[samples."S1"]
lattice = { a = 0.5, b = 0.5, c = 0.7, alp = 90, bet = 90, gam = 90 }
field_axis = [0, 0, 1]     # c along the magnet's z
field_toward = [0, 1, 0]   # b along the magnet's y
[groups."rotation"]
runs = "80-83"
roi = "peak"
sample = "S1"
"""


def test_catalogue_series_get_field_angles(tmp_path):
    for run, deg in zip(range(80, 84), (0, 30, 60, 90)):
        rad = np.radians(deg)
        testing.rocking_scan(tmp_path / "raw", run=run, seed=run, field=0.6, field_direction=(0, np.sin(rad), np.cos(rad)))
    (tmp_path / "experiment.toml").write_text(ROTATION)
    out = cs.Catalogue.load(tmp_path / "experiment.toml").series("rotation", cutoff=7.0, i0_cutoff=2.5, photon_energy=7.511)
    s = out["summary"]
    assert np.allclose(s.coords["field_angle"].values, [0, 30, 60, 90], atol=1e-6)
    assert np.allclose(s.coords["field_tilt"].values, 0, atol=1e-6)
    assert np.allclose(s.coords["field"].values, 0.6)
    assert out["settings"]["field_axis"] == "0 0 1"
