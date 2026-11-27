"""Voigt fits and one-versus-two-peak fits on synthetic split reflections."""

import numpy as np
import pytest
import scipp as sc
from scipy.special import voigt_profile
from sfdata import SFScanInfo

import cristallina_scipp as cs
from cristallina_scipp import channels, testing
from cristallina_scipp.peaks import fit_peaks
from cristallina_scipp.split import fit_split

CASES = {"single": {}, "2theta": dict(d_x=6.0), "unequal": dict(d_omega=0.06, d_x=3.0, ratio=0.7), "tilt": dict(d_y=6.0)}


@pytest.mark.parametrize("sigma, gamma", [(0.05, 0.02), (0.02, 0.05)])
def test_voigt_recovers_both_widths(sigma, gamma):
    rng = np.random.default_rng(0)
    x = np.linspace(-1, 1, 81)
    truth = 5 * voigt_profile(x - 0.1, sigma, gamma) + 2
    y = rng.poisson(truth * 20) / 20
    da = sc.DataArray(sc.array(dims=["x"], values=y, variances=truth / 20),
                      coords={"x": sc.array(dims=["x"], values=x, unit="deg")})
    p = fit_peaks(da, shape="voigt")["p0"]
    for name, true in (("sigma", sigma), ("gamma", gamma), ("center", 0.1), ("area", 5.0)):
        assert abs(p[name].value - true) < 3 * np.sqrt(p[name].variance), name


@pytest.fixture(scope="module")
def maps(tmp_path_factory):
    base = tmp_path_factory.mktemp("split")
    out = {}
    for run, (name, kw) in enumerate(CASES.items(), start=100):
        path = testing.split_rocking_scan(base, run=run, seed=run, **kw)
        stack = cs.reduce_scan(SFScanInfo(str(path)), channels.JF, channels.JF_I0, i0_cutoff=2.5, cutoff=7.0,
                               photon_energy=7.511)
        out[name] = stack / (stack.coords["i0"] / sc.scalar(1.0, unit=stack.coords["i0"].unit))
    return out


def within(value, truth, sigmas=3):
    return abs(value.value - truth) < sigmas * np.sqrt(value.variance)


def test_single_peak_is_not_split(maps):
    for dims in ("y", "x"):
        fit = fit_split(maps["single"].sum(dims), coords={"step": "readback"})
        assert not fit["split"] and fit["delta_bic"] < 0, (dims, fit["delta_bic"])


def test_splitting_in_two_theta_needs_the_detector_axis(maps):
    image = fit_split(maps["2theta"].sum("y"), coords={"step": "readback"})
    curve = fit_split(maps["2theta"].sum(["y", "x"]), coords={"step": "readback"})
    assert image["split"] and not curve["split"], (image["delta_bic"], curve["delta_bic"])
    assert within(image["two"]["separation"]["x"], 6.0)
    assert within(image["two"]["separation"]["readback"], 0.0)
    assert 0.8 < image["two"]["dispersion"] < 1.3  # Poisson scatter


def test_unequal_domains(maps):
    fit = fit_split(maps["unequal"].sum("y"), coords={"step": "readback"})
    two = fit["two"]
    assert fit["split"]
    assert within(two["separation"]["readback"], 0.06) and within(two["separation"]["x"], 3.0)
    assert within(two["ratio"], 0.7)
    assert two["separation"]["readback"].unit == sc.Unit("deg")


def test_tilt_out_of_the_scattering_plane_shows_in_rows(maps):
    rows = fit_split(maps["tilt"].sum("x"), coords={"step": "readback"})
    columns = fit_split(maps["tilt"].sum("y"), coords={"step": "readback"})
    assert rows["split"] and not columns["split"], (rows["delta_bic"], columns["delta_bic"])
    assert within(rows["two"]["separation"]["y"], 6.0)


def test_needs_variances(maps):
    with pytest.raises(ValueError, match="variances"):
        fit_split(sc.values(maps["single"].sum(["y", "x"])), coords={"step": "readback"})
