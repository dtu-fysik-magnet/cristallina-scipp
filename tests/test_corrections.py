"""Transmission and absorption corrections of run series."""

import numpy as np
import pytest
import scipp as sc
from sfdata import SFScanInfo

from cristallina_scipp import channels, run_series, testing
from cristallina_scipp.corrections import correct_absorption, correct_transmission, fluorescence_yield

KW = dict(cutoff=7.0, photon_energy=7.511)
ATT = {80: (0.5, 0.1), 81: (0.25, 0.1), 82: (0.5, 0.04)}


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    base = tmp_path_factory.mktemp("corr")
    for run, att in ATT.items():  # same peak, three attenuations
        testing.rocking_scan(base, run=run, seed=run, attenuators=att)
    for i, run in enumerate(range(90, 99)):  # measured intensity suppressed by absorption above the edge
        e = 7495.0 + 4 * i
        testing.rocking_scan(base, run=run, seed=run, energy=e, scale=testing.resonance(e) / testing.edge(e))
    testing.fluorescence_energy_scan(base, run=45)
    return base


def scans(raw, runs):
    return {r: SFScanInfo(str(raw / f"run{r:04}" / "meta" / "scan.json")) for r in runs}


def spread(values):
    return np.std(values) / np.mean(values)


def test_gas_monitor_series_needs_both_attenuators(raw):
    series = run_series(scans(raw, ATT), channels.JF, channels.GASMONITOR, **KW)
    raw_integral = series["summary"]["integral"].values
    assert spread(raw_integral) > 0.3  # follows the attenuation
    corrected = correct_transmission(series)
    assert spread(corrected["summary"]["integral"].values) < 0.05
    assert corrected["settings"]["transmission_attenuators"] == "transmission_fe, transmission_exp"
    assert np.allclose(corrected["summary"].coords["transmission_factor"].values, [0.05, 0.025, 0.02])
    assert np.array_equal(corrected["summary"]["center"].values, series["summary"]["center"].values)


def test_pbps_series_needs_only_the_experiment_attenuator(raw):
    series = run_series(scans(raw, ATT), channels.JF, channels.PBPS149_INTENSITY, **KW)
    corrected = correct_transmission(series)
    assert corrected["settings"]["transmission_attenuators"] == "transmission_exp"
    assert spread(corrected["summary"]["integral"].values) < 0.05


def test_jf_i0_series_is_not_corrected_twice(raw):
    series = run_series(scans(raw, ATT), channels.JF, channels.JF_I0, i0_cutoff=2.5, **KW)
    assert spread(series["summary"]["integral"].values) < 0.05  # I0 after both attenuators
    with pytest.raises(ValueError, match="already corrects"):
        correct_transmission(series)


def exact_yield():
    e = np.arange(7440.0, 7601.0, 1.0)
    return sc.DataArray(sc.array(dims=["step"], values=testing.edge(e)),
                        coords={"readback": sc.array(dims=["step"], values=e, unit="eV"), "scan": sc.scalar("model")})


def test_fluorescence_yield_follows_the_edge(raw):
    yield_ = fluorescence_yield(scans(raw, [45])[45], cutoff=5.5, upper_cutoff=6.8, i0_cutoff=2.5, photon_energy=6.4)
    e = yield_.coords["readback"].values
    step = yield_.values[e > 7540].mean() / yield_.values[e < 7480].mean()
    model = testing.edge(e[e > 7540]).mean() / testing.edge(e[e < 7480]).mean()
    assert abs(step / model - 1) < 0.2
    assert yield_.coords["scan"].value.endswith("run0045/meta/scan.json")


def test_absorption_correction_recovers_the_resonance(raw):
    series = run_series(scans(raw, range(90, 99)), channels.JF, channels.JF_I0, i0_cutoff=2.5, **KW)
    corrected = correct_absorption(series, exact_yield(), coord="photon_energy")
    truth = testing.resonance(corrected["summary"].coords["photon_energy"].values * 1000)
    measured = series["summary"]["integral"].values / truth
    fixed = corrected["summary"]["integral"].values / truth
    sigma = np.sqrt(corrected["summary"]["integral"].variances) / truth
    mean = np.average(fixed, weights=sigma**-2)
    assert spread(measured) > 0.3  # absorption suppresses the peak above the edge
    assert np.all(np.abs(fixed - mean) < 4 * sigma), (fixed - mean) / sigma  # flat within the propagated errors
    assert corrected["settings"]["corrections"] == "absorption"
    assert corrected["settings"]["absorption_scan"] == "model"
    assert corrected["summary"]["integral"].variances is not None


def test_corrections_are_recorded_in_order_and_saved(raw, tmp_path):
    series = run_series(scans(raw, range(90, 93)), channels.JF, channels.GASMONITOR, **KW)
    yield_ = fluorescence_yield(scans(raw, [45])[45], cutoff=5.5, upper_cutoff=6.8, i0_cutoff=2.5, photon_energy=6.4)
    done = correct_transmission(correct_absorption(series, yield_, coord="photon_energy", exponent=-1))
    assert done["settings"]["corrections"] == "absorption; transmission"
    done.save_hdf5(tmp_path / "s.h5")
    back = sc.io.load_hdf5(tmp_path / "s.h5")
    assert back["settings"]["absorption_exponent"] == -1
    assert sc.identical(back["summary"], done["summary"])


def test_absorption_rejects_energies_outside_the_scan(raw):
    series = run_series(scans(raw, [80]), channels.JF, channels.JF_I0, i0_cutoff=2.5, **KW)
    yield_ = fluorescence_yield(scans(raw, [45])[45], cutoff=5.5, upper_cutoff=6.8, i0_cutoff=2.5, photon_energy=6.4)
    with pytest.raises(ValueError, match="outside"):
        correct_absorption(series, yield_, coord="mono_energy", reference=sc.scalar(8000.0, unit="eV"))
    shifted = yield_.copy()
    shifted.coords["readback"] = shifted.coords["readback"] + sc.scalar(500.0, unit="eV")
    with pytest.raises(ValueError, match=r"runs \[80\]"):
        correct_absorption(series, shifted, coord="photon_energy")
