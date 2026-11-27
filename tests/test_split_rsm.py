"""Split peaks in reciprocal space: local coordinates and fits of binned maps."""

import numpy as np
import pytest
import scipp as sc

from cristallina_scipp import testing
from cristallina_scipp.rsm import _basis, histogram, local_coords
from cristallina_scipp.split import fit_split

LATTICE = {"a": 0.5, "b": 0.5, "c": 0.7, "alp": 90, "bet": 90, "gam": 90}


def hkl_stack(points):
    points = np.asarray(points, dtype=float)
    return sc.DataArray(sc.ones(dims=["p"], shape=[len(points)]),
                        coords={c: sc.array(dims=["p"], values=points[:, i]) for i, c in enumerate("hkl")})


def test_local_coords_measure_strain_and_tilt():
    g = np.array([2.0, 0, 0])
    tilt = np.radians(0.2)  # rotate G by 0.2 deg toward c*, and shrink it by 0.1 % (d 0.1 % longer)
    gc = _basis(LATTICE) @ g
    rotated = np.linalg.norm(gc) * np.array([np.cos(tilt), 0, np.sin(tilt)]) / 1.001
    hkl = np.linalg.solve(_basis(LATTICE), rotated)
    out = local_coords(hkl_stack([g, hkl]), g, LATTICE, toward=(0, 0, 1))
    assert np.allclose(out.coords["strain"].values, [0, 1e-3], atol=2e-5)
    assert np.allclose(out.coords["tilt_1"].values, [0, 0.2], atol=1e-3)
    assert np.allclose(out.coords["tilt_2"].values, 0, atol=1e-9)
    with pytest.raises(ValueError, match="parallel"):
        local_coords(hkl_stack([g]), g, LATTICE, toward=(1, 0, 0))


@pytest.fixture(scope="module")
def bins():
    pytest.importorskip("cristallina")
    n = 25  # voxels fitted at their mean pixel position, so coarse ones do not bias the fit
    return dict(strain=sc.linspace("strain", -0.012, 0.012, n), tilt_1=sc.linspace("tilt_1", -1.0, 1.0, n, unit="deg"),
                tilt_2=sc.linspace("tilt_2", -1.0, 1.0, n, unit="deg"))


def fit(bins, **split):
    return fit_split(histogram(testing.split_rsm_stack(**split), **bins), exposure="exposure", coupling=False)


def within(value, truth, sigmas=3):
    return abs(value.value - truth) < sigmas * np.sqrt(value.variance)


def test_histogram_keeps_exposure(bins):
    stack = testing.split_rsm_stack()
    rsm = histogram(stack, **bins)
    visited = rsm.coords["exposure"].values > 0
    assert 0 < visited.sum() < visited.size
    assert np.all(np.isnan(rsm.values[~visited]))


def test_one_domain_is_not_split(bins):
    out = fit(bins, seed=20)
    assert not out["split"] and out["delta_bic"] < 0


def test_strain_and_tilt_splits_are_measured(bins):
    out = fit(bins, seed=21, d_strain=0.006, d_tilt_1=0.3, ratio=0.7)
    two = out["two"]
    assert out["split"] and 0.8 < two["dispersion"] < 1.2
    assert within(two["separation"]["strain"], 0.006) and within(two["separation"]["tilt_1"], 0.3)
    assert within(two["separation"]["tilt_2"], 0.0) and within(two["ratio"], 0.7)
    assert two["separation"]["tilt_1"].unit == sc.Unit("deg")


def test_tilt_alone_is_measured(bins):
    out = fit(bins, seed=22, d_tilt_2=0.6)  # 4 sigma; at 3 sigma along one axis the evidence depends on the noise
    assert out["split"] and within(out["two"]["separation"]["tilt_2"], 0.6)
