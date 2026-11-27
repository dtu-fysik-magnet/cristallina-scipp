"""Splitting against field, as a run series from catalogue groups, on the detector and in reciprocal space."""

import numpy as np
import pytest
import scipp as sc

import cristallina_scipp as cs
from cristallina_scipp import testing

KW = dict(cutoff=7.0, i0_cutoff=2.5, photon_energy=7.511)


@pytest.fixture(scope="module")
def cat(tmp_path_factory):
    pytest.importorskip("cristallina")
    return cs.Catalogue.load(testing.split_peak_experiment(tmp_path_factory.mktemp("split"), fields=(0.0, 3.0)))


def within(value, truth, sigmas=3):
    return abs(value.value - truth) < sigmas * np.sqrt(value.variance)


def test_catalogue_builds_sample_and_detector(cat):
    sample = cat.sample("S1")
    assert sample.sample_dict["a"] == pytest.approx(0.53484)
    det = cat.detector(90)
    assert det.gamma == pytest.approx(36.0)
    with pytest.raises(KeyError, match="not in"):
        cat.sample("S2")


def test_split_series_in_reciprocal_space(cat, tmp_path):
    out = cat.split_series("split vs field", reflection=(2, 0, 0), toward=(0, 0, 1),
                           bins={"strain": 16, "tilt_1": 16, "tilt_2": 16}, **KW)
    s = out["summary"]
    assert list(s.coords["field"].values) == [0.0, 3.0]
    assert list(s["split"].values) == [False, True]
    at3 = s["run", 1]
    assert within(at3["separation_strain"].data, 0.003) and within(at3["separation_tilt_1"].data, 0.15)
    assert within(at3["separation_tilt_2"].data, 0.0) and within(at3["ratio"].data, 0.5)
    assert out["settings"]["reflection"] == "2 0 0" and out["settings"]["space"] == "reciprocal"
    assert set(out["maps"]) == {"90", "91"} and "exposure" in out["maps"]["91"].coords
    out.save_hdf5(tmp_path / "split.h5")
    back = sc.io.load_hdf5(tmp_path / "split.h5")
    assert sc.identical(back["summary"], s)


def test_split_series_on_the_detector(cat):
    s = cat.split_series("split vs field", axis="x", **KW)["summary"]
    assert list(s["split"].values) == [False, True]
    assert s["separation_x"].unit is None or s["separation_x"].unit == sc.units.dimensionless


def test_split_series_needs_photon_counts(cat):
    with pytest.raises(ValueError, match="photon_energy"):
        cat.split_series("split vs field", cutoff=7.0)
