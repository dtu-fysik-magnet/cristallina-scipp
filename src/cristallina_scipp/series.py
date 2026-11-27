"""Run series: one point per run, against energy, field, field angle or transmission.

Most physics runs are short rocking curves at one energy, field and
temperature. [`run_series`][cristallina_scipp.series.run_series] reduces each
run to its normalised ROI curve and measures it three ways (a peak fit, a
background-subtracted integral and moments) so that methods can be compared
on the same data. The runs' metadata become coordinates along `run`:

```python
cat = Catalogue.load("experiment.toml")
series = cat.series("energy 0T", cutoff=7.0, i0_cutoff=2.5, photon_energy=7.511)
pp.plot(series["summary"]["integral"], coords=["photon_energy"])
```
"""

import json
import warnings
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import scipp as sc

from . import channels
from .load import _open, reduce_scan
from .peaks import fit_peaks, integrate, moments


def run_metadata(scan) -> dict[str, sc.Variable]:
    """Energy, field and transmission of a run, read from its first step.

    These PVs are constant during a rocking curve. A channel missing from the
    files gives NaN, with a warning.

    Returns:
        Scalars: `photon_energy` (undulator, keV), `mono_energy` (eV),
            `field_x`, `field_y`, `field_z` and their magnitude `field` (T), the
            front-end and experiment attenuator transmissions `transmission_fe`
            and `transmission_exp` (`channels.ATTENUATORS`), and their product
            `transmission` (compare with the run table's "Transm FE" × "Transm EXP").
    """
    with _open(scan, 0) as data:
        def mean(name):
            try:
                return float(np.mean(data[name].data))
            except KeyError:
                warnings.warn(f"{name} not in the run's files; using NaN", stacklevel=3)
                return np.nan

        fel, mono = mean(channels.FEL_PHOTON_ENERGY), mean(channels.MONO_ENERGY)
        b = [mean(name) for name in channels.MAGNET]
        fe, exp = (mean(name) for name in channels.ATTENUATORS)
    return {
        "photon_energy": sc.scalar(fel, unit="keV"),
        "mono_energy": sc.scalar(mono, unit="eV"),
        **{f"field_{axis}": sc.scalar(v, unit="T") for axis, v in zip("xyz", b)},
        "field": sc.scalar(float(np.sqrt(np.sum(np.square(b)))), unit="T"),
        "transmission_fe": sc.scalar(fe),
        "transmission_exp": sc.scalar(exp),
        "transmission": sc.scalar(fe * exp),
    }


def run_overview(raw, runs=None):
    """One row per run in a raw-data folder: what was scanned, and at which energy, field and transmission.

    Reads each `run*/meta/scan.json` and the first step's PVs
    ([`run_metadata`][cristallina_scipp.series.run_metadata]). A run without a
    `scan.json` yet (still being written) or with unreadable files is listed
    with its `status` instead of being skipped, so gaps are visible.

    Args:
        raw: Folder that holds the `runNNNN` directories.
        runs: Run numbers to include; all by default.

    Returns:
        A `pandas.DataFrame` indexed by run, with columns `status`, `scanned`
            (adjustable name), `unit`, `start`, `stop`, `steps`, and the
            `run_metadata` values in their units (`photon_energy` keV,
            `mono_energy` eV, `field*` T, `transmission*`).
    """
    import pandas as pd
    from sfdata import SFScanInfo

    rows = {}
    for run_dir in sorted(Path(raw).glob("run[0-9][0-9][0-9][0-9]")):
        run = int(run_dir.name[3:])
        if runs is not None and run not in runs:
            continue
        path = run_dir / "meta" / "scan.json"
        if not path.exists():
            rows[run] = {"status": "no scan.json"}
            continue
        info = json.loads(path.read_text())
        readbacks = [r[0] for r in info.get("scan_readbacks", []) if r]
        row = {"status": "ok", "scanned": ", ".join(info["scan_parameters"]["name"]),
               "unit": ", ".join(info["scan_parameters"]["units"]),
               "start": readbacks[0] if readbacks else np.nan, "stop": readbacks[-1] if readbacks else np.nan,
               "steps": len(info.get("scan_files", []))}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                meta = run_metadata(SFScanInfo(str(path)))
            row |= {k: v.value for k, v in meta.items()}
            if caught:
                row["status"] = "missing PVs"
        except Exception as err:  # noqa: BLE001 -- an overview lists broken runs, it does not stop at them
            row["status"] = f"unreadable: {type(err).__name__}"
        rows[run] = row
    table = pd.DataFrame.from_dict(rows, orient="index").rename_axis("run")
    if "steps" in table:
        table["steps"] = table["steps"].astype("Int64")
    return table


def _peak(curve, shape):
    try:
        return fit_peaks(curve, shape=shape, coord="readback")["p0"]
    except ValueError:  # no peak found: keep the run, without fit values
        xu, yu = curve.coords["readback"].unit, curve.unit
        nan = lambda unit: sc.scalar(np.nan, variance=np.nan, unit=unit)  # noqa: E731
        return sc.DataGroup({"center": nan(xu), "fwhm": nan(xu), "height": nan(yu),
                             "area": nan(None if xu is None or yu is None else xu * yu)})


def run_series(scans, detector, i0=None, roi=None, *, labels=None, n_edge=3, shape="lorentzian",
               **reduce_kwargs) -> sc.DataGroup:
    """Reduce each run to one point: integrated intensity, position and width.

    For every run, the detector is summed over the ROI per step
    ([`reduce_scan`][cristallina_scipp.reduce_scan]) and divided by the step's I0,
    giving a curve against the scanned readback (per unit of I0, in I0's own
    unit, which `settings["i0_unit"]` records). That curve is measured three
    ways:

    - `integral`: area above a straight-line background
      ([`integrate`][cristallina_scipp.peaks.integrate]), no clipping,
      variances propagated;
    - `center`, `fwhm`, `height`, `area`: one-peak fit plus linear background
      ([`fit_peaks`][cristallina_scipp.peaks.fit_peaks]), NaN when no peak is found;
    - `centroid`, `moment_fwhm`: moments of the background-subtracted curve. For
      Lorentzian peaks `moment_fwhm` is much larger than the fitted FWHM,
      because the tails dominate the second moment.

    Args:
        scans: `{run: SFScanInfo}`.
        detector, i0, **reduce_kwargs: As for [`reduce_scan`][cristallina_scipp.reduce_scan];
            pass `photon_energy` for counts with Poisson variances.
        roi: One [`ROI`][cristallina_scipp.ROI] for all runs, or `{run: ROI}`.
        labels: `{run: {name: value}}`, e.g. sample, reflection or field angle
            from a [`Catalogue`][cristallina_scipp.catalogue.Catalogue]. Names
            given for every run become coordinates.
        n_edge: Points at each end used for the integral's background.
        shape: Peak shape for the fit, `"lorentzian"`, `"gaussian"` or `"voigt"`.

    Returns:
        A DataGroup with `summary`, a Dataset along `run` (coordinates: `run`,
            the [`run_metadata`][cristallina_scipp.series.run_metadata], the labels,
            and the provenance of each point: the `roi` name and the pixel bounds
            `roi_bottom`, `roi_top`, `roi_left`, `roi_right` actually summed, and
            the `scan` file); `curves`, the normalised curve of each run keyed by
            run number; and `settings`, the arguments used plus the package
            version and creation time. It saves and loads as one HDF5 file:
            `series.save_hdf5(path)`, `scipp.io.load_hdf5(path)`.
    """
    runs = list(scans)
    labels = labels or {}
    measures, meta, curves, where, i0_unit = [], [], {}, [], None
    for run in runs:
        r = roi.get(run) if isinstance(roi, dict) else roi
        stack = reduce_scan(scans[run], detector, i0, roi=r, **reduce_kwargs)
        curve = stack.sum(["y", "x"])
        if i0 is not None:  # per unit of I0, in I0's own unit (keV for the JF I0 detector)
            i0_unit = stack.coords["i0"].unit
            curve = curve / (stack.coords["i0"] / sc.scalar(1.0, unit=i0_unit))
        integral = integrate(curve, n_edge=n_edge, coord="readback")
        mom = moments(curve - integral["background"].data, "step", coord="readback")
        peak = _peak(curve, shape)
        measures.append({"integral": integral["area"], **{k: peak[k] for k in ("center", "fwhm", "height", "area")},
                         "centroid": mom["centroid"], "moment_fwhm": mom["fwhm"]})
        meta.append(run_metadata(scans[run]))
        curves[str(run)] = curve
        where.append(_where(r, stack, scans[run]))
    summary = _summary(runs, measures, meta, labels, where)
    settings = _settings(detector=detector, i0=i0, i0_unit=i0_unit, n_edge=n_edge, shape=shape, **reduce_kwargs)
    return sc.DataGroup({"summary": summary, "curves": sc.DataGroup(curves), "settings": settings})


def _where(roi, stack, scan):
    """Provenance of one run's point: the ROI name and pixel bounds summed, and the scan file."""
    y, x = stack.coords["y"].values, stack.coords["x"].values
    return {"roi": getattr(roi, "name", None) or "", "roi_bottom": int(y[0]), "roi_top": int(y[-1]) + 1,
            "roi_left": int(x[0]), "roi_right": int(x[-1]) + 1, "scan": str(getattr(scan, "fname", ""))}


def _summary(runs, measures, meta, labels, where):
    """A Dataset along `run` from per-run measures, with metadata, shared labels and provenance as coordinates."""
    coords = {"run": sc.array(dims=["run"], values=runs, unit=None)}
    for name in meta[0]:
        coords[name] = sc.concat([m[name] for m in meta], "run")
    shared = set.intersection(*(set(labels.get(run, {})) for run in runs)) if labels else set()
    for name in sorted(shared):
        coords[name] = sc.array(dims=["run"], values=[labels[run][name] for run in runs], unit=None)
    for name in where[0]:
        coords[name] = sc.array(dims=["run"], values=[w[name] for w in where], unit=None)
    return sc.Dataset({name: sc.concat([m[name] for m in measures], "run") for name in measures[0]}, coords=coords)


def _settings(**settings):
    """Arguments as storable values (None as "none", keV arguments with their unit), plus version and time."""
    settings = {k: "none" if v is None else (str(v) if isinstance(v, sc.Unit) else v) for k, v in settings.items()}
    settings |= {k: sc.scalar(float(v), unit="keV") for k, v in settings.items() if k in _KEV and v != "none"}
    settings |= {"cristallina_scipp": _version(), "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    return sc.DataGroup(settings)


_KEV = {"cutoff", "upper_cutoff", "i0_cutoff", "photon_energy"}  # reduce_scan arguments in keV


def _version():
    try:
        return version("cristallina-scipp")
    except PackageNotFoundError:
        return "unknown"


def split_series(scans, detector, i0=None, roi=None, *, labels=None, axis="x", to_q=None, bins=None,
                 shape="voigt", **reduce_kwargs) -> sc.DataGroup:
    """One point per run: is the reflection split, and by how much?

    For every run, [`fit_split`][cristallina_scipp.split.fit_split] compares
    one peak with two, either on the detector or in reciprocal space:

    - Detector (default): the ROI image normalised by I0 and summed over the
      other detector axis, `(step, x)` for splitting in 2θ (`axis="x"`) or
      `(step, y)` for tilts out of the scattering plane (`axis="y"`).
    - Reciprocal space: `to_q(run, stack)` adds `strain`, `tilt_1` and
      `tilt_2` coordinates (see
      [`Catalogue.split_series`][cristallina_scipp.catalogue.Catalogue.split_series],
      which builds it from the catalogue), and the
      [`histogram`][cristallina_scipp.rsm.histogram] in those, with `bins`,
      is fitted.

    Args:
        scans, detector, i0, roi, labels, **reduce_kwargs: As for
            [`run_series`][cristallina_scipp.series.run_series];
            `photon_energy` is required, since the fits use photon counts.
        axis: Detector axis kept with the scan axis, `"x"` or `"y"`.
        to_q: For reciprocal space, as above.
        bins: Bins for the reciprocal-space histogram; 24 per axis by default.
        shape: Peak shape, `"voigt"`, `"lorentzian"` or `"gaussian"`.

    Returns:
        A DataGroup with `summary`, a Dataset along `run` with `delta_bic`
            (BIC of one peak minus BIC of two, see
            [`fit_split`][cristallina_scipp.split.fit_split]), `split`, `dispersion`, the domain fraction `ratio`, and per axis
            the two-peak `separation_<axis>` and `fwhm_<axis>` and the
            one-peak `center_<axis>`, with the run metadata, labels and
            provenance as coordinates (as `run_series`); `maps`, the data
            fitted for each run; `best_fit`, the two-peak model of each run;
            and `settings`. Separations mean something only where `split`.
    """
    from .rsm import histogram
    from .split import fit_split

    if reduce_kwargs.get("photon_energy") is None:
        raise ValueError("split_series needs photon_energy: the fits use photon counts")
    runs = list(scans)
    labels = labels or {}
    measures, meta, where, maps, best = [], [], [], {}, {}
    for run in runs:
        r = roi.get(run) if isinstance(roi, dict) else roi
        stack = reduce_scan(scans[run], detector, i0, roi=r, **reduce_kwargs)
        if to_q is not None:
            if "i0" not in stack.coords:
                stack.coords["i0"] = sc.ones(dims=["step"], shape=[stack.sizes["step"]])
            data = histogram(to_q(run, stack), **(bins or {"strain": 24, "tilt_1": 24, "tilt_2": 24}))
            fit = fit_split(data, exposure="exposure", coupling=False, shape=shape)
        else:
            norm = stack if i0 is None else stack / (stack.coords["i0"] / sc.scalar(1.0, unit=stack.coords["i0"].unit))
            data = norm.sum({"x": "y", "y": "x"}[axis])
            fit = fit_split(data, coords={"step": "readback"}, shape=shape)
        two, one = fit["two"], fit["one"]
        m = {"delta_bic": sc.scalar(float(fit["delta_bic"])), "split": sc.scalar(fit["split"]),
             "dispersion": sc.scalar(float(two["dispersion"])), "ratio": two["ratio"]}
        for name in two["separation"]:
            m |= {f"separation_{name}": two["separation"][name], f"fwhm_{name}": two["fwhm"][name],
                  f"center_{name}": one["center"][name]}
        measures.append(m)
        meta.append(run_metadata(scans[run]))
        where.append(_where(r, stack, scans[run]))
        maps[str(run)], best[str(run)] = data, two["best_fit"]
    summary = _summary(runs, measures, meta, labels, where)
    settings = _settings(detector=detector, i0=i0, shape=shape, space="reciprocal" if to_q else f"detector {axis}",
                         **reduce_kwargs)
    return sc.DataGroup({"summary": summary, "maps": sc.DataGroup(maps), "best_fit": sc.DataGroup(best),
                         "settings": settings})


def add_field_angles(series, axis, toward, *, min_field=0.01) -> sc.DataGroup:
    """Field direction of every run, relative to the sample, from the measured field vector.

    With â the sample `axis` and t̂ the direction `toward`, made perpendicular
    to â (both in the vector magnet's x, y, z frame, e.g. `axis=(0, 0, 1)` if
    the sample's c axis lies along the magnet's z):

    - `field_angle` (deg) = atan2(B·t̂, B·â): 0 for B along â, 90 for B along
      t̂, signed, in the plane of â and t̂ (the plane a field rotation from
      B ∥ c to B ∥ b sweeps, with â = c and t̂ = b);
    - `field_tilt` (deg): the angle of B out of that plane.

    Both are NaN where |B| < `min_field` (T), where the direction means
    nothing. `axis` and `toward` may also be one vector per run, shape
    `(runs, 3)`, for series that combine samples mounted differently.

    Returns:
        A copy of the series with the two coordinates added to `summary` and
            the directions recorded in `settings`.
    """
    summary = series["summary"]
    if "field_angle" in summary.coords:
        raise ValueError("the series already has a field_angle coordinate (a catalogue label?); rename the label, "
                         "e.g. to nominal_field_angle")
    b = np.stack([summary.coords[f"field_{c}"].to(unit="T").values for c in "xyz"], axis=-1)
    a = np.broadcast_to(np.asarray(axis, dtype=float), b.shape)
    t = np.broadcast_to(np.asarray(toward, dtype=float), b.shape)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    t = t - np.sum(t * a, axis=-1, keepdims=True) * a
    norm = np.linalg.norm(t, axis=-1, keepdims=True)
    if np.any(norm < 1e-9):
        raise ValueError("toward is parallel to axis")
    t = t / norm
    n = np.cross(a, t)
    ba, bt, bn = (np.sum(b * v, axis=-1) for v in (a, t, n))
    weak = np.linalg.norm(b, axis=-1) < min_field
    out = series.copy()
    out["summary"] = summary.copy()
    for name, values in (("field_angle", np.arctan2(bt, ba)), ("field_tilt", np.arctan2(bn, np.hypot(ba, bt)))):
        out["summary"].coords[name] = sc.array(dims=["run"], values=np.where(weak, np.nan, np.degrees(values)),
                                               unit="deg")
    out["settings"] = series["settings"].copy()
    fmt = lambda v: "; ".join(" ".join(f"{x:g}" for x in row) for row in np.unique(np.atleast_2d(v), axis=0))  # noqa: E731
    out["settings"]["field_axis"], out["settings"]["field_toward"] = fmt(axis), fmt(toward)
    return out
