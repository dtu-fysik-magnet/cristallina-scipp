"""Small synthetic scans in the SwissFEL file layout, for tests, docs and dry runs.

The files are what sfdata expects: per-channel groups with `data` and `pulse_id`,
plus a `meta/scan.json`, arranged as `<base>/run<NNNN>/...`. Channel names
match the Cristallina ones, so the code that reads them is the code you would run on real data. The
Jungfrau channels go in their own `acqNNNN.<detector>.h5` files, marked as
already converted (data in keV), as the March 2026 files were: sfdata reads
them through jungfrau_utils, which applies no calibration.

The three `*_scan` builders imitate typical runs (numbered 45, 51
and 55 by default): a fluorescence energy scan across an edge, a sample-translation scan,
and a rocking curve.
"""

import json
from pathlib import Path

import h5py
import numpy as np

from .channels import GASMONITOR, JF, JF_I0
from .channels import ATTENUATORS
from .channels import PBPS149_INTENSITY as PBPS149

ATT_FE, ATT_EXP = ATTENUATORS
SHAPE, I0_SHAPE, PULSES = (64, 64), (16, 16), 20
FLUORESCENCE_KEV = 6.4  # emission line inside the 5.5-6.8 keV window
EDGE_EV = 7511.0
BRAGG = {"center": -18.0, "fwhm": 0.08, "x": 40.0, "y": 30.0, "sigma_px": 2.0}

_PVS = {  # motor/beamline PVs read by the analysis, at their March values
    "SARES31-GPS:TD-PosAct": 0.0,
    "SARES30-MOBI2:MOT_Z.RBV": 0.0,
    "SARES31-GPS:ROT2THETA-PosAct": 36.0,
    "SARES31-MAG-IPS1-X:SIG_FLD": 0.0,
    "SARES31-MAG-IPS1-Y:SIG_FLD": 0.0,
    "SARES31-MAG-IPS1-Z:SIG_FLD": 1.5,
    "SARFE10-OATT053:UsrRec.TC1": 0.5,
    "SAROP31-OATA150:UsrRec.TC1": 0.1,
}


def write_scan(run_dir, readbacks, steps, *, name, units, pvs=None, first_pulse_id=26783989433):
    """Write a scan: one HDF5 file per step plus `meta/scan.json`.

    Args:
        run_dir: Directory to create, e.g. `<base>/run0045`.
        readbacks: One readback per step.
        steps: Per step, `{channel: (pulse_ids, values)}` for beam-synchronous
            channels (values have a leading pulse dimension).
        name: Scanned adjustable, stored in `scan_parameters`.
        units: Its unit.
        pvs: Per step, `{channel: value}` for PV channels (one entry each).

    Returns:
        Path to `scan.json`.
    """
    run_dir = Path(run_dir)
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    (run_dir / "meta").mkdir(exist_ok=True)
    files, pulse_ranges = [], []
    for i, chans in enumerate(steps):
        fname = run_dir / "data" / f"acq{i + 1:04}.SIMDATA.h5"
        lo = first_pulse_id + 1000 * i
        step_files = [str(fname)]
        for ch in [c for c in chans if c.startswith("JF")]:
            pids, values = chans[ch]
            jf_name = run_dir / "data" / f"acq{i + 1:04}.{ch}.h5"
            with h5py.File(jf_name, "w") as f:  # an already-converted Jungfrau file
                g = f.create_group(f"data/{ch}")
                g["pulse_id"] = np.asarray(pids, dtype=np.uint64).reshape(-1, 1)
                g["is_good_frame"] = np.ones(len(pids), dtype=bool)
                g.create_dataset("data", data=np.asarray(values, dtype=np.float32), compression="lzf")
                g["conversion_factor"] = np.ones(1)  # jungfrau_utils' marker for converted data
            step_files.append(str(jf_name))
        with h5py.File(fname, "w") as f:
            for ch, (pids, values) in chans.items():
                if ch.startswith("JF"):
                    continue
                g = f.create_group(ch)
                g["pulse_id"] = np.asarray(pids, dtype=np.int64)
                g.create_dataset("data", data=values, compression="lzf" if values.ndim > 1 else None)
            for ch, value in (pvs[i] if pvs else {}).items():
                g = f.create_group(ch)
                g["pulse_id"] = np.array([[lo]], dtype=np.int64)
                g["data"] = np.array([[value]], dtype=float)
        files.append(step_files)
        pulse_ranges.append([lo, lo + PULSES])
    info = {
        "scan_files": files,
        "pulseIds": pulse_ranges,
        "scan_parameters": {"name": [name], "Id": [name], "units": [units], "offset": [0], "conversion_factor": [1]},
        "scan_readbacks": [[float(r)] for r in readbacks],
        "scan_values": [[float(r)] for r in readbacks],
        "scan_readbacks_raw": [[float(r)] for r in readbacks],
        "scan_step_info": [[] for _ in readbacks],
    }
    path = run_dir / "meta" / "scan.json"
    path.write_text(json.dumps(info, indent=2))
    return path


def _detector(rng, rate, scale, energy_kev, noise_kev=0.3):
    """Pixel values in keV: Poisson photons of `energy_kev` plus electronic noise."""
    counts = rng.poisson(rate[None] * scale[:, None, None])
    return (counts * energy_kev + rng.normal(0, noise_kev, counts.shape)).astype(np.float32)


def _step(rng, i, fluo_rate, other_rate, other_kev, incident_kev, drop_i0_pulse=False, attenuators=None):
    pids = 26783989433 + 1000 * i + np.arange(PULSES)
    scale = rng.uniform(0.6, 1.4, PULSES)  # pulse-to-pulse intensity
    # beam after each attenuator, relative to the transmissions the PVs record by default
    fe, exp = attenuators or (_PVS[ATT_FE], _PVS[ATT_EXP])
    mid, down = fe / _PVS[ATT_FE], fe * exp / (_PVS[ATT_FE] * _PVS[ATT_EXP])
    image = (_detector(rng, fluo_rate * down, scale, FLUORESCENCE_KEV)
             + _detector(rng, other_rate * down, scale, other_kev, 0.0))
    i0 = _detector(rng, np.full(I0_SHAPE, 0.5 * down), scale, incident_kev)
    keep = np.arange(PULSES) != 7 if drop_i0_pulse else slice(None)
    return {
        JF: (pids, image),
        JF_I0: (pids[keep], i0[keep]),
        GASMONITOR: (pids, 2500 * scale + rng.normal(0, 50, PULSES)),
        PBPS149: (pids, 30 * mid * scale + rng.normal(0, 3, PULSES)),
    }


def _pvs(energy_ev, **extra):
    return {**_PVS, "SARUN:FELPHOTENE.VAL": energy_ev / 1000, "SAROP31-ODCC110:MOT_ENY.RBV": energy_ev - 9.0, **extra}


def edge(energy_ev):
    """Absorption (and so fluorescence yield) model across the synthetic edge."""
    return 0.2 + 1 / (1 + np.exp(-(np.asarray(energy_ev) - EDGE_EV) / 3.0))


def fluorescence_energy_scan(base, run=45, seed=1):
    """Photon-energy scan across the edge, fluorescence on the 1.5M detector.

    Elastic scattering at the incident energy (7.4-7.6 keV) falls above the
    5.5-6.8 keV window. In step 10, the I0 detector misses one pulse.
    """
    rng = np.random.default_rng(seed)
    energies = np.arange(7440.0, 7601.0, 4.0)
    readbacks = energies + rng.normal(0, 0.5, energies.size)
    steps = [
        _step(rng, i, np.full(SHAPE, 0.004 * edge(e)), np.full(SHAPE, 0.002), e / 1000, e / 1000, drop_i0_pulse=i == 10)
        for i, e in enumerate(energies)
    ]
    pvs = [_pvs(e) for e in energies]
    return write_scan(Path(base) / f"run{run:04}", readbacks, steps, name="Cristallina photon energy", units="eV", pvs=pvs)


def sample_profile(x_mm):
    """Fraction of the beam on the sample at translation `x_mm`."""
    from scipy.special import erf

    x = np.asarray(x_mm)
    return 0.5 * (erf((x + 0.9) / 0.07) - erf((x + 0.3) / 0.07))


def translation_scan(base, run=51, seed=2):
    """TRX translation scan across the sample, above the edge."""
    rng = np.random.default_rng(seed)
    positions = np.arange(-1.2, 0.21, 0.05)
    e = 7551.0
    steps = [
        _step(rng, i, np.full(SHAPE, 0.004 * edge(e) * sample_profile(x) + 1e-4), np.full(SHAPE, 0.002), e / 1000, e / 1000)
        for i, x in enumerate(positions)
    ]
    pvs = [_pvs(e, **{"SARES31-GPS:TRX": x}) for x in positions]
    return write_scan(Path(base) / f"run{run:04}", positions, steps, name="SARES31-GPS:TRX", units="mm", pvs=pvs)


def rocking_curve(theta):
    """Bragg-peak intensity (fraction of maximum) at sample angle `theta`."""
    half = BRAGG["fwhm"] / 2
    return half**2 / ((np.asarray(theta) - BRAGG["center"]) ** 2 + half**2)


def resonance(energy_ev):
    """Peak-intensity factor of the synthetic resonance at the edge (1 far below it)."""
    return 1 + 4 / (1 + ((np.asarray(energy_ev) - EDGE_EV) / 2.5) ** 2)


def rocking_scan(base, run=55, seed=3, field=1.5, shift_per_tesla=0.0, energy=7511.0, scale=1.0, attenuators=None):
    """Rocking curve through a Bragg peak; the spot drifts along x by 40 px/deg.

    `field` (T, along z) is written to the magnet PVs; `shift_per_tesla`
    (deg/T) moves the peak with it, for multi-run examples. `energy` (eV) is
    written to the energy PVs and `scale` multiplies the peak (not the
    background), e.g. `scale=resonance(energy)` for an energy series.
    `attenuators=(front_end, experiment)` sets both transmissions: the 1.5M
    and I0 detectors (downstream of both) and PBPS149 (between them) see the
    beam attenuated accordingly; the gas monitor (upstream) does not.
    """
    rng = np.random.default_rng(seed)
    center = BRAGG["center"] + shift_per_tesla * field
    thetas = np.arange(-18.5, -17.49, 0.025)
    e = energy
    yy, xx = np.mgrid[: SHAPE[0], : SHAPE[1]]
    steps = []
    for i, t in enumerate(thetas):
        x_spot = BRAGG["x"] + 40 * (t - center)
        spot = np.exp(-0.5 * (((xx - x_spot) ** 2 + (yy - BRAGG["y"]) ** 2) / BRAGG["sigma_px"] ** 2))
        peak = 0.6 * scale * rocking_curve(t - center + BRAGG["center"]) * spot + 5e-4
        steps.append(_step(rng, i, np.full(SHAPE, 1e-4), peak, e / 1000, e / 1000, attenuators=attenuators))
    att = dict(zip(ATTENUATORS, attenuators)) if attenuators else {}
    pvs = [_pvs(e, **{"SARES31-GPS:ROTTHETA": t, "SARES31-MAG-IPS1-Z:SIG_FLD": field, **att}) for t in thetas]
    return write_scan(Path(base) / f"run{run:04}", thetas, steps, name="SARES31-GPS:ROTTHETA", units="deg", pvs=pvs)


EXAMPLE_CATALOGUE = """\
# Synthetic experiment written by cristallina_scipp.testing.example_experiment.
raw = "raw"

[rois]
"peak A" = { left = 0, right = 64, bottom = 22, top = 38 }
"peak B" = { left = 0, right = 64, bottom = 22, top = 38 }

[[geometry]]
runs = "1-100"
x0 = 416.81
y0 = 775.27
d0 = 0.568
z0 = 0.2
stt0 = -0.09935076

[groups."fluorescence"]
runs = "45"
note = "energy scan across the edge, fluorescence window"

[groups."energy 0T"]
runs = "60-68"
roi = "peak A"
sample = "S1"
reflection = "A"
note = "rocking curves across the edge, 0 T"

[groups."field 0deg"]
runs = "70-73"
roi = "peak B"
sample = "S1"
reflection = "B"
field_angle = 0.0

[groups."field 15deg"]
runs = "74-77"
roi = "peak B"
sample = "S1"
reflection = "B"
field_angle = 15.0
"""


def example_experiment(base):
    """A small synthetic experiment with a catalogue, for the template notebook and dry runs.

    Writes under `base`: `raw/run0045` (fluorescence energy scan), runs 60-68
    (rocking curves across the edge, suppressed by absorption above it),
    70-73 and 74-77 (field series in two directions), run 78 (an alignment
    scan no group lists), run 79 without `scan.json` (still being written),
    and `experiment.toml`. Returns the catalogue path.
    """
    base = Path(base)
    raw = base / "raw"
    fluorescence_energy_scan(raw, run=45)
    for i, run in enumerate(range(60, 69)):
        e = 7495.0 + 4 * i
        rocking_scan(raw, run=run, seed=run, field=0.0, energy=e, scale=resonance(e) / edge(e))
    for first, shift in ((70, 0.02), (74, 0.012)):
        for run, field in zip(range(first, first + 4), [0.0, 1.0, 2.0, 3.0]):
            rocking_scan(raw, run=run, seed=run, field=field, shift_per_tesla=shift)
    rocking_scan(raw, run=78, seed=78)
    (raw / "run0079" / "data").mkdir(parents=True)
    path = base / "experiment.toml"
    path.write_text(EXAMPLE_CATALOGUE)
    return path


def march_2026(base):
    """Write all three synthetic March runs under `base`; returns `{run: scan.json}`."""
    return {45: fluorescence_energy_scan(base), 51: translation_scan(base), 55: rocking_scan(base)}


def split_rocking_scan(base, run=56, seed=4, d_omega=0.0, d_x=0.0, d_y=0.0, ratio=0.5, fwhm=BRAGG["fwhm"],
                       sigma_px=BRAGG["sigma_px"], drift=0.0, scale=1.0):
    """Rocking curve of a reflection split into two domains, with known splitting.

    The two peaks sit at ω = center ∓ `d_omega`/2 (deg), detector column
    x ∓ `d_x`/2 and row y ∓ `d_y`/2 (pixels), with intensity fractions `ratio`
    and 1 − `ratio`. Each is a Lorentzian of `fwhm` (deg) in ω times a
    Gaussian spot of `sigma_px`; `drift` (px/deg) moves the spots along x
    with ω. A tetragonal-to-orthorhombic distortion gives both: different
    d-spacings (`d_x`, through 2θ) and tilted twin domains (`d_omega`, `d_y`).
    """
    rng = np.random.default_rng(seed)
    thetas = np.arange(-18.5, -17.49, 0.025)
    yy, xx = np.mgrid[: SHAPE[0], : SHAPE[1]]
    domains = [(-0.5, ratio), (0.5, 1 - ratio)]
    steps = []
    for i, t in enumerate(thetas):
        peak = np.full(SHAPE, 5e-4)
        for sign, fraction in domains:
            c = BRAGG["center"] + sign * d_omega
            x0, y0 = BRAGG["x"] + sign * d_x + drift * (t - c), BRAGG["y"] + sign * d_y
            rock = (fwhm / 2) ** 2 / ((t - c) ** 2 + (fwhm / 2) ** 2)
            spot = np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / sigma_px**2)
            peak += 1.2 * scale * fraction * rock * spot
        steps.append(_step(rng, i, np.full(SHAPE, 1e-4), peak, 7.511, 7.511))
    pvs = [_pvs(7511.0, **{"SARES31-GPS:ROTTHETA": t}) for t in thetas]
    return write_scan(Path(base) / f"run{run:04}", thetas, steps, name="SARES31-GPS:ROTTHETA", units="deg", pvs=pvs)


TAS2 = {"a": 0.336, "b": 0.336, "c": 0.5853, "alp": 90, "bet": 90, "gam": 120}


def split_rsm_stack(d_strain=0.0, d_tilt_1=0.0, d_tilt_2=0.0, ratio=0.5, sigma_strain=0.002, sigma_tilt=0.15,
                    peak=20.0, background=0.5, seed=5):
    """A rocking scan through (0 0 2) with two domains, as a reduced stack with `h`, `k`, `l` and local coordinates.

    Uses `cristallina.rsm`'s geometry (the RSM tutorial's 1T-TaS2 at 9 keV;
    needs the `[rsm]` extra). The domains sit at strain ∓ `d_strain`/2 and
    tilts ∓ `d_tilt_1`/2, ∓ `d_tilt_2`/2 (deg) of the reflection
    ([`local_coords`][cristallina_scipp.rsm.local_coords] with `toward=(1, 0, 0)`),
    with fractions `ratio` and 1 − `ratio`; each is Gaussian with widths
    `sigma_strain` and `sigma_tilt` (deg). Counts are Poisson, scaled by a
    per-step I0.
    """
    import cristallina.rsm as crsm

    from .rsm import local_coords, pixel_hkl

    sample = crsm.Sample(dict(TAS2), "1T-TaS2")
    detector = crsm.Detector(R_m=0.15, px_size_um=75, center_pixel=(515, 807), detector_size_px=(1030, 1614))
    energy = 9000.0
    k = crsm.E_to_k(energy)
    detector.move_angles(gamma=2 * np.degrees(np.arcsin(2 * np.pi * 2 / TAS2["c"] / (2 * k))))
    alpha = np.linspace(10, 20, 41)
    import scipp as sc

    template = sc.DataArray(
        sc.zeros(dims=["step", "y", "x"], shape=[alpha.size, 120, 120], unit="counts"),
        coords={"readback": sc.array(dims=["step"], values=alpha, unit="deg"),
                "y": sc.arange("y", 747, 867, unit=None), "x": sc.arange("x", 455, 575, unit=None)},
    )
    stack = local_coords(pixel_hkl(template, sample, detector, energy), (0, 0, 2), TAS2, toward=(1, 0, 0))
    e, t1, t2 = (stack.coords[c].values for c in ("strain", "tilt_1", "tilt_2"))
    rate = np.full(e.shape, background)
    for sign, fraction in ((-0.5, ratio), (0.5, 1 - ratio)):
        rate += peak * fraction * np.exp(-0.5 * (((e - sign * d_strain) / sigma_strain) ** 2
                                                + ((t1 - sign * d_tilt_1) / sigma_tilt) ** 2
                                                + ((t2 - sign * d_tilt_2) / sigma_tilt) ** 2))
    rng = np.random.default_rng(seed)
    i0 = rng.uniform(0.5, 1.5, alpha.size)
    stack.values = rng.poisson(rate * i0[:, None, None]).astype(float)
    stack.variances = stack.values.copy()
    stack.coords["i0"] = sc.array(dims=["step"], values=i0)
    return stack
