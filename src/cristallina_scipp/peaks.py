"""Peak position, width and fits for rocking curves and ROI profiles."""

import numpy as np
import scipp as sc
from lmfit.models import GaussianModel, LinearModel, LorentzianModel, VoigtModel
from scipy.signal import find_peaks, peak_widths

_FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))


def moments(da, dim, coord=None) -> sc.DataGroup:
    """Intensity-weighted centroid and Gaussian-equivalent FWHM along `dim`.

    Vectorised over every other dimension, so a `(step, x)` projection gives
    one centroid per step. Uses the `coord` coordinate (default: the one named
    `dim`), which can be pixel indices or angles (see [`detector_angles`][cristallina_scipp.angles.detector_angles]). Restrict
    `da` to the region of interest first; any background inside it biases
    both numbers.

    Returns:
        A DataGroup with `centroid` and `fwhm` (NaN where the weights sum to 0).
    """
    w = sc.values(da.data)
    x = da.coords[coord or dim].to(dtype="float64", copy=True)
    if x.unit is None:  # pixel indices
        x.unit = ""
    total = w.sum(dim)
    total = sc.where(total == sc.zeros_like(total), sc.scalar(np.nan, unit=total.unit), total)
    mu = (w * x).sum(dim) / total
    var = (w * (x - mu) ** 2).sum(dim) / total
    return sc.DataGroup({"centroid": mu, "fwhm": _FWHM_PER_SIGMA * sc.sqrt(var)})


def fit_peaks(da, n=1, shape="lorentzian", prominence=0.05, distance=5, coord=None) -> sc.DataGroup:
    """Fit `n` peaks plus a linear background to a 1-D curve.

    Peaks are seeded from `scipy.signal.find_peaks` (the `n` highest that
    pass the `prominence`, a fraction of the maximum, and `distance`, in points). The fit uses
    lmfit, weighted by `1/sigma` when `da` has variances.

    Args:
        da: 1-D DataArray, e.g. a rocking curve from
            [`roi_intensities`][cristallina_scipp.roi_intensities] divided by I0.
        n: Number of peaks.
        shape: `"lorentzian"`, `"gaussian"` or `"voigt"` (exact, via the Faddeeva
            function; Gaussian `sigma` and Lorentzian `gamma` both free, reported
            per peak).
        prominence: Minimum peak prominence as a fraction of `max(da)`.
        distance: Minimum separation between peaks, in points.
        coord: Name of the x coordinate, e.g. `"readback"`. Defaults to the
            coordinate named after the dimension.

    Returns:
        A DataGroup with, per peak (`p0`, `p1`, ...), `center`, `fwhm`,
            `height` and `area` as scalars whose variances are the fit's standard errors.
            It also holds the background `intercept` and `slope`, the `best_fit`
            curve, and the lmfit `result`.
    """
    (dim,) = da.dims
    coord = coord or dim
    x = da.coords[coord].values.astype(float)
    y = da.values.astype(float)
    weights = 1 / np.sqrt(da.variances) if da.variances is not None else None
    if weights is not None:
        weights = np.where(np.isfinite(weights), weights, 0.0)

    found, _ = find_peaks(y, prominence=prominence * np.nanmax(y), distance=distance)
    found = found[np.argsort(y[found])[::-1]][:n] if found.size else np.array([np.nanargmax(y)])
    if found.size < n:
        raise ValueError(f"found {found.size} peak(s), asked for {n}; lower prominence or distance")
    widths = peak_widths(y, found, rel_height=0.5)[0] * np.mean(np.diff(x))

    peak_model = {"lorentzian": LorentzianModel, "gaussian": GaussianModel, "voigt": VoigtModel}[shape]
    model = LinearModel(prefix="bg_")
    params = model.make_params(intercept=np.nanmin(y), slope=0.0)
    for i, (p, w) in enumerate(zip(found, np.abs(widths))):
        m = peak_model(prefix=f"p{i}_")
        model += m
        w = max(w, abs(x[1] - x[0]))
        sigma = w / {"lorentzian": 2, "gaussian": _FWHM_PER_SIGMA, "voigt": 3.6013}[shape]
        area = (y[p] - np.nanmin(y)) * w * {"lorentzian": np.pi / 2, "gaussian": 1.0645, "voigt": 1.3}[shape]
        params.update(m.make_params(
            center={"value": x[p], "min": x.min(), "max": x.max()},
            sigma={"value": sigma, "min": 0},
            amplitude={"value": area, "min": 0},
        ))
        if shape == "voigt":  # lmfit ties gamma to sigma by default
            params[f"p{i}_gamma"].set(value=sigma, vary=True, expr="", min=0)

    result = model.fit(y, params, x=x, weights=weights, nan_policy="omit")

    def scalar(name, unit):
        p = result.params[name]
        err = p.stderr if p.stderr is not None else np.nan
        return sc.scalar(p.value, variance=err**2, unit=unit)

    xu, yu = da.coords[coord].unit, da.unit
    out = {
        f"p{i}": sc.DataGroup({
            "center": scalar(f"p{i}_center", xu),
            "fwhm": scalar(f"p{i}_fwhm", xu),
            "height": scalar(f"p{i}_height", yu),
            "area": scalar(f"p{i}_amplitude", None if xu is None or yu is None else xu * yu),
            **({"sigma": scalar(f"p{i}_sigma", xu), "gamma": scalar(f"p{i}_gamma", xu)} if shape == "voigt" else {}),
        })
        for i in range(found.size)
    }
    out["intercept"] = scalar("bg_intercept", yu)
    out["slope"] = scalar("bg_slope", None if xu is None or yu is None else yu / xu)
    out["best_fit"] = sc.DataArray(sc.array(dims=[dim], values=result.best_fit, unit=yu), coords={coord: da.coords[coord]})
    out["result"] = result
    return sc.DataGroup(out)


def integrate(da, n_edge=3, coord=None) -> sc.DataGroup:
    """Area under a 1-D curve above a straight-line background.

    The background is a least-squares line through the first and last
    `n_edge` points; the area is the trapezoid integral of the curve minus that
    line. Nothing is clipped: on a curve with no peak the result scatters
    around zero, as it should.

    Every step is linear in the data, so the variances of `da` (if any) are
    propagated exactly, including the uncertainty of the background line.
    Treat the choice of `n_edge` as a systematic: compare a few values.

    Args:
        da: 1-D DataArray, e.g. a rocking curve normalised by I0.
        n_edge: Points used for the background at each end.
        coord: Name of the x coordinate; defaults to the dimension's.

    Returns:
        A DataGroup with the `area` (scalar, with variance if `da` has
            variances) and the `background` curve.
    """
    (dim,) = da.dims
    coord = coord or dim
    x = da.coords[coord].values.astype(float)
    n = x.size
    if n < 2 * n_edge + 1:
        raise ValueError(f"{n} points is too few for n_edge={n_edge}")
    edges = np.r_[:n_edge, n - n_edge : n]
    design = np.column_stack([np.ones(n), x])
    to_background = np.zeros((n, n))  # background = to_background @ y
    to_background[:, edges] = design @ np.linalg.pinv(design[edges])
    trapezoid = np.trapezoid(np.eye(n), x=x, axis=1)  # area = trapezoid @ y
    weights = trapezoid @ (np.eye(n) - to_background)
    y = da.values.astype(float)
    variance = None if da.variances is None else float(weights**2 @ da.variances)
    xu, yu = da.coords[coord].unit, da.unit
    area_unit = None if xu is None or yu is None else xu * yu
    return sc.DataGroup({
        "area": sc.scalar(float(weights @ y), variance=variance, unit=area_unit),
        "background": sc.DataArray(
            sc.array(dims=[dim], values=to_background @ y, unit=yu), coords={coord: da.coords[coord]}
        ),
    })
