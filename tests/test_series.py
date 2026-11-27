"""Run catalogue, background-subtracted integral and run series."""

import numpy as np
import pytest
import scipp as sc

import cristallina_scipp as cs
from cristallina_scipp import testing
from cristallina_scipp.catalogue import parse_runs
from cristallina_scipp.peaks import integrate

GEOMETRY = "x0 = 416.81\ny0 = 775.27\nd0 = 0.568\nz0 = 0.2\nstt0 = -0.09935076\n"


def curve(y, x, variances=None):
    return sc.DataArray(sc.array(dims=["step"], values=y, variances=variances, unit="counts"),
                        coords={"readback": sc.array(dims=["step"], values=x, unit="deg")})


def test_parse_runs_is_inclusive():
    assert parse_runs("501-505, 510") == [501, 502, 503, 504, 505, 510]
    assert parse_runs([7, "9-10"]) == [7, 9, 10]


def test_integrate_removes_a_sloped_background_exactly():
    x = np.linspace(-1, 1, 81)
    line = 3.0 + 2.0 * x
    assert abs(integrate(curve(line, x), n_edge=5, coord="readback")["area"].value) < 1e-12
    peak = 0.5 * np.exp(-0.5 * (x / 0.05) ** 2)  # area 0.5 * 0.05 * sqrt(2 pi)
    area = integrate(curve(line + peak, x), n_edge=5, coord="readback")["area"]
    assert area.unit == sc.Unit("counts*deg")
    assert np.isclose(area.value, 0.5 * 0.05 * np.sqrt(2 * np.pi), rtol=1e-3)


def test_integrate_propagates_variances():
    rng = np.random.default_rng(0)
    x = np.linspace(-1, 1, 41)
    truth = 10 + 50 * np.exp(-0.5 * (x / 0.1) ** 2)
    sigma = np.sqrt(truth)
    predicted = integrate(curve(truth, x, variances=sigma**2), n_edge=4, coord="readback")["area"]
    draws = [integrate(curve(truth + rng.normal(0, sigma), x), n_edge=4, coord="readback")["area"].value
             for _ in range(2000)]
    assert np.isclose(np.std(draws), np.sqrt(predicted.variance), rtol=0.08)


@pytest.fixture(scope="module")
def catalogue(tmp_path_factory):
    base = tmp_path_factory.mktemp("cat")
    for i, run in enumerate(range(60, 69)):  # energy series across the resonance
        e = 7495.0 + 4 * i
        testing.rocking_scan(base / "raw", run=run, seed=run, field=0.0, energy=e, scale=testing.resonance(e))
    for run, field in zip(range(70, 74), [0.0, 1.0, 2.0, 3.0]):  # field series, 0.02 deg/T
        testing.rocking_scan(base / "raw", run=run, seed=run, field=field, shift_per_tesla=0.02)
    (base / "p.toml").write_text(f'''
raw = "raw"
[rois]
peak = {{ left = 0, right = 64, bottom = 22, top = 38 }}
wide = {{ left = 0, right = 64, bottom = 18, top = 42 }}
[[geometry]]
runs = "1-100"
{GEOMETRY}
[groups."energy 0T"]
runs = "60-68"
roi = "peak"
reflection = "A"
[groups."field low"]
runs = "70-71"
roi = "peak"
reflection = "B"
field_angle = 0.0
[groups."field high"]
runs = "72-73"
roi = {{ "72" = "peak", "73" = "wide" }}
reflection = "B"
field_angle = 5.0
[groups."overlap"]
runs = "71-72"
roi = "peak"
''')
    return cs.Catalogue.load(base / "p.toml")


KW = dict(cutoff=7.0, i0_cutoff=2.5, photon_energy=7.511)


def test_catalogue_lookups(catalogue):
    assert catalogue.runs("energy 0T") == list(range(60, 69))
    assert catalogue.roi("field high", 73).name == "wide"
    assert catalogue.labels("field high") == {"reflection": "B", "field_angle": 5.0}
    assert catalogue.geometry(55)["d0"] == 0.568
    with pytest.raises(KeyError):
        catalogue.geometry(500)


def test_catalogue_rejects_unknown_roi(tmp_path):
    (tmp_path / "bad.toml").write_text('[groups.g]\nruns = "1-2"\nroi = "nope"\n')
    with pytest.raises(KeyError, match="nope"):
        cs.Catalogue.load(tmp_path / "bad.toml")


def test_energy_series_peaks_at_the_resonance(catalogue):
    summary = catalogue.series("energy 0T", **KW)["summary"]
    assert summary["integral"].unit == sc.Unit("counts*deg")
    best = summary["integral"].values.argmax()
    assert abs(summary.coords["mono_energy"].values[best] + 9 - testing.EDGE_EV) <= 4  # mono PV is offset by -9 eV
    assert set(summary.coords["reflection"].values) == {"A"}
    assert summary["integral"].variances is not None


def test_field_series_over_groups(catalogue):
    out = catalogue.series(["field low", "field high"], **KW)
    s = out["summary"]
    assert list(s.coords["run"].values) == [70, 71, 72, 73]
    assert list(s.coords["field_angle"].values) == [0.0, 0.0, 5.0, 5.0]
    slope = np.polyfit(s.coords["field"].values, s["center"].values, 1)[0]
    assert abs(slope - 0.02) < 0.006
    assert list(s.coords["roi"].values) == ["peak", "peak", "peak", "wide"]
    assert list(s.coords["roi_top"].values) == [38, 38, 38, 42]
    assert out["settings"]["groups"] == "field low, field high"
    assert len(out["settings"]["catalogue_sha256"]) == 64
    assert set(out["curves"]) == {"70", "71", "72", "73"}


def test_overlapping_groups_are_an_error(catalogue):
    with pytest.raises(ValueError, match="run 71"):
        catalogue.series(["field low", "overlap"], **KW)


def test_series_round_trips_through_hdf5(catalogue, tmp_path):
    out = catalogue.series("field low", upper_cutoff=None, **KW)
    path = tmp_path / "series.h5"
    out.save_hdf5(path)
    back = sc.io.load_hdf5(path)
    assert sc.identical(back["summary"], out["summary"])
    assert sc.identical(back["curves"]["70"], out["curves"]["70"])
    settings = back["settings"]
    assert sc.identical(settings["cutoff"], sc.scalar(7.0, unit="keV"))
    assert settings["upper_cutoff"] == "none"  # None is not storable
    assert settings["detector"] == "JF16T03V02" and settings["n_edge"] == 3
    assert settings["catalogue"].endswith("p.toml") and settings["created"]
    assert back["summary"].coords["scan"].values[0].endswith("run0070/meta/scan.json")
