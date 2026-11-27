"""Unit checks for peaks, angles and the scan reductions on synthetic scans."""

import numpy as np
import pytest
import scipp as sc
from sfdata import SFScanInfo

import cristallina_scipp as cs
from cristallina_scipp import testing
from cristallina_scipp.angles import detector_angles
from cristallina_scipp.peaks import fit_peaks, moments
from cristallina_scipp.rsm import histogram


def test_moments_of_a_gaussian():
    x = np.arange(100.0)
    da = sc.DataArray(sc.array(dims=["x"], values=np.exp(-0.5 * ((x - 40) / 5) ** 2)), coords={"x": sc.array(dims=["x"], values=x, unit="deg")})
    m = moments(da, "x")
    assert np.isclose(m["centroid"].value, 40) and np.isclose(m["fwhm"].value, 2.3548 * 5, rtol=1e-3)
    assert m["centroid"].unit == sc.Unit("deg")
    assert np.isnan(moments(da * 0, "x")["centroid"].value)


@pytest.mark.parametrize("shape", ["gaussian", "lorentzian"])
def test_fit_two_peaks(shape):
    from lmfit.lineshapes import gaussian, lorentzian

    f = {"gaussian": gaussian, "lorentzian": lorentzian}[shape]
    x = np.linspace(-1, 1, 201)
    y = f(x, amplitude=2.0, center=-0.3, sigma=0.05) + f(x, amplitude=1.0, center=0.4, sigma=0.08) + 0.5
    da = sc.DataArray(sc.array(dims=["theta"], values=y, unit="counts"), coords={"theta": sc.array(dims=["theta"], values=x, unit="deg")})
    fit = fit_peaks(da, n=2, shape=shape)
    found = sorted([fit["p0"], fit["p1"]], key=lambda p: p["center"].value)
    fwhm_per_sigma = 2.3548 if shape == "gaussian" else 2.0
    for p, (area, center, sigma) in zip(found, [(2.0, -0.3, 0.05), (1.0, 0.4, 0.08)]):
        assert np.isclose(p["center"].value, center, atol=1e-6)
        assert np.isclose(p["fwhm"].value, fwhm_per_sigma * sigma, rtol=1e-4)
        assert np.isclose(p["area"].value, area, rtol=1e-4)  # lmfit amplitude is the area
    assert np.isclose(fit["intercept"].value, 0.5, atol=1e-6)
    with pytest.raises(ValueError):
        fit_peaks(da, n=3, shape=shape, prominence=0.5)


def test_detector_angles_reference_pixel():
    da = sc.DataArray(sc.zeros(dims=["y", "x"], shape=[3, 3]),
                      coords={"y": sc.arange("y", 9, 12, unit=None), "x": sc.arange("x", 4, 7, unit=None)})
    a = detector_angles(da, distance=1.0, x0=5, y0=10, two_theta=30.0, pixel_size=1e-3)
    assert np.isclose(a.coords["two_theta"].values[1], -30.0)
    assert np.isclose(a.coords["nu"].values[1], 0.0)
    assert np.isclose(a.coords["nu"].values[2], np.degrees(np.arctan(1e-3)))


def test_reduce_scan_window_alignment_and_no_i0(tmp_path):
    scan = SFScanInfo(str(testing.fluorescence_energy_scan(tmp_path)))
    window = cs.reduce_scan(scan, testing.JF, testing.JF_I0, cutoff=5.5, upper_cutoff=6.8, i0_cutoff=2.5)
    assert window.coords["n_pulses"].values[10] == testing.PULSES - 1  # I0 missed a pulse; detector follows
    assert window.coords["readback"].unit == sc.Unit("eV") and window.coords["i0"].unit == sc.Unit("keV")
    plain = cs.reduce_scan(scan, testing.JF, cutoff=5.5, upper_cutoff=6.8)
    assert "i0" not in plain.coords and plain.coords["n_pulses"].values[10] == testing.PULSES
    # a plain list of steps works too
    single = cs.reduce_scan([scan[0]], testing.JF, cutoff=5.5, upper_cutoff=6.8, readbacks=[0.0])
    assert single.sizes["step"] == 1


def test_scan_means_errors(tmp_path):
    scan = SFScanInfo(str(testing.translation_scan(tmp_path)))
    sem = cs.scan_means(scan, [testing.GASMONITOR, "SARES31-GPS:TRX"])
    spread = cs.scan_means(scan, [testing.GASMONITOR], spread=True)
    assert np.allclose(spread[testing.GASMONITOR].variances, sem[testing.GASMONITOR].variances * testing.PULSES)
    assert np.allclose(sem["SARES31-GPS:TRX"].values, scan.readbacks)
    assert sem["SARES31-GPS:TRX"].coords["readback"].unit == sc.Unit("mm")


def test_histogram_normalises_and_leaves_gaps():
    # two steps hit the same voxel with different I0: result is total counts / total I0
    ones = np.ones((2, 1, 2))
    stack = sc.DataArray(
        sc.array(dims=["step", "y", "x"], values=[[[3.0, 0.0]], [[5.0, 0.0]]]),
        coords={
            "i0": sc.array(dims=["step"], values=[1.0, 3.0]),
            "h": sc.array(dims=["step", "y", "x"], values=ones * [0.1, 0.9]),
            "k": sc.array(dims=["step", "y", "x"], values=ones * 0.5),
            "l": sc.array(dims=["step", "y", "x"], values=ones * 0.5),
        },
    )
    edges = sc.array(dims=["h"], values=[0.0, 0.5, 1.0, 1.5])
    rsm = histogram(stack, h=edges, k=sc.array(dims=["k"], values=[0.0, 1.0]), l=sc.array(dims=["l"], values=[0.0, 1.0]))
    assert np.allclose(rsm.values[:2].ravel(), [8 / 4, 0 / 4])
    assert np.isnan(rsm.values[2]).all()  # never visited: NaN, not a neighbour's value


def test_roi_matches_cristallina_semantics():
    roi = cs.ROI(left=10, right=20, bottom=100, top=200, name="a")
    assert (roi.rows, roi.cols, roi.width, roi.height) == (slice(100, 200), slice(10, 20), 10, 100)
    assert roi.expand(5, 1) == cs.ROI(5, 25, 99, 201, name="b")  # name ignored, as in cristallina
    assert roi.shift(1, -1).name == "a" and roi.shift(1, -1).cols == slice(11, 21)
