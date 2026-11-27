"""One peak or two? Fits for split reflections, on a rocking curve or on the detector image per step.

A tetragonal-to-orthorhombic distortion splits a reflection two ways: twin
domains with different d-spacings scatter at different 2θ (detector column),
and domains tilted against each other reflect at different ω (rocking angle)
or, out of the scattering plane, at different ν (detector row). Summing the
ROI into a rocking curve keeps only the ω part, so a splitting mainly in 2θ
or ν shows up there as a broad or skewed peak. Fitting the `(step, x)` map,
or `(step, y)`, keeps it. In reciprocal space the two effects separate
physically: [`local_coords`][cristallina_scipp.rsm.local_coords] turns each
pixel's hkl into strain along the reciprocal vector and tilts across it, and
a [`histogram`][cristallina_scipp.rsm.histogram] in those coordinates is
fitted the same way.

[`fit_split`][cristallina_scipp.split.fit_split] fits one peak and two
peaks to the same data and compares them, so that a reported splitting is a
measured result with an uncertainty, not a choice of model:

```python
stack = cs.reduce_scan(scan, channels.JF, channels.JF_I0, roi=roi, cutoff=7.0, photon_energy=7.511)
image = (stack / stack.coords["i0"]).sum("y")        # (step, x): rocking angle and 2θ
fit = fit_split(image, coords={"step": "readback"})
fit["delta_bic"], fit["two"]["separation"]

# or in reciprocal space (needs the [rsm] extra for pixel_hkl)
stack = local_coords(pixel_hkl(stack, sample, detector, energy), (2, 0, 0), sample.sample_dict)
rsm = histogram(stack, strain=edges_e, tilt_1=edges_t, tilt_2=edges_t)
fit = fit_split(rsm, exposure="exposure", coupling=False)
```
"""

import itertools

import lmfit
import numpy as np
import scipp as sc
from scipy.special import voigt_profile, xlogy

_FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))
EXPLORE = 30  # evaluations per parameter for each exploratory start; 10 missed a 4-sigma split in tests
SPLIT_EVIDENCE = 10.0  # BIC difference above which two peaks are preferred ("very strong" on the Kass-Raftery scale)


def _fwhm(sigma, gamma):
    """Voigt FWHM (Olivero and Longbothum 1977, 0.02 %)."""
    fl, fg = 2 * gamma, _FWHM_PER_SIGMA * sigma
    return 0.5346 * fl + np.sqrt(0.2166 * fl**2 + fg**2)


class _Model:
    """Sum of `n` peaks, each a product of 1-D profiles along the axes, plus a planar background.

    Evaluated at points: `points[k]` holds every point's coordinate along axis
    k (a grid's coordinates broadcast, or each voxel's mean pixel position).
    Peak positions are written as a shared centre `c{k}` and separation `s{k}`
    per axis (peak 0 at c - s/2, peak 1 at c + s/2) with intensity fractions
    `ratio` and 1 - `ratio`, so the separation and its error come straight
    out of the fit. Widths are shared by the two peaks. With `coupling`, the
    position along each later axis may move linearly with the first
    (`k{j}`), as the beam's energy spread or divergence does on the detector.
    """

    def __init__(self, points, n, shape, coupling):
        self.points, self.n, self.shape = points, n, shape
        self.coupling = coupling and len(points) > 1
        self.mid = [float(np.mean(x)) for x in points]

    def profile(self, k, x, p):
        sigma = p[f"sig{k}"] if self.shape != "lorentzian" else 0.0
        gamma = p[f"gam{k}"] if self.shape != "gaussian" else 0.0
        return voigt_profile(x, sigma, gamma)

    def __call__(self, p, points=None):
        points = self.points if points is None else points
        fractions = (p["ratio"], 1 - p["ratio"]) if self.n == 2 else (1.0,)
        y = p["bg"] + sum(p[f"bg{k}"] * (x - m) for k, (x, m) in enumerate(zip(points, self.mid)))
        for i, fraction in enumerate(fractions):
            sign = (-0.5, 0.5)[i] if self.n == 2 else 0.0
            centre = [p[f"c{k}"] + sign * p.get(f"s{k}", 0.0) for k in range(len(points))]
            peak = self.profile(0, points[0] - centre[0], p)
            for k in range(1, len(points)):
                drift = p[f"k{k}"] * (points[0] - centre[0]) if self.coupling else 0.0
                peak = peak * self.profile(k, points[k] - centre[k] - drift, p)
            y = y + p["amp"] * fraction * peak
        return y


def _params(model, start):
    params = lmfit.Parameters()
    for name, value in start.items():
        params.add(name, value=value)
    for k, x in enumerate(model.points):
        span = float(np.ptp(x))
        params[f"c{k}"].set(min=float(x.min()), max=float(x.max()))
        for w, unused in (("sig", "lorentzian"), ("gam", "gaussian")):
            if model.shape == unused:
                params[f"{w}{k}"].set(value=0.0, vary=False)
            else:
                params[f"{w}{k}"].set(min=1e-6 * span)  # a Voigt with both widths 0 is undefined
        if model.n == 2:
            params[f"s{k}"].set(min=-span, max=span)  # the peaks are labelled after the fit
    params["amp"].set(min=0)
    if model.n == 2:
        params["ratio"].set(min=0, max=1)
    params.pop("k0", None)  # the first axis does not couple to itself
    if not model.coupling:
        for k in range(1, len(model.points)):
            params.pop(f"k{k}", None)
    return params


def _fit(model, counts, scale, start, max_nfev=None):
    """Maximum Poisson likelihood: least squares on deviance residuals of the counts."""

    def residual(p):
        mu = np.maximum(model(p.valuesdict()) / scale, 1e-12)
        d = 2 * (mu - counts + xlogy(counts, counts / mu))
        return np.sign(counts - mu) * np.sqrt(np.maximum(d, 0))

    params = start if isinstance(start, lmfit.Parameters) else _params(model, start)
    result = lmfit.minimize(residual, params, method="leastsq", scale_covar=False, max_nfev=max_nfev)
    if max_nfev is not None:  # a short exploratory run: no error analysis
        result.deviance = result.chisqr
        return result
    nvarys = result.nvarys  # the model's free parameters, for the BIC, even if some are fixed below
    for _ in range(3):  # a parameter on its bound (e.g. a Lorentzian width of 0) makes the covariance singular
        if result.errorbars:
            break
        params = result.params.copy()
        at_bound = [par for par in params.values()
                    if par.vary and np.isclose(par.value, [par.min, par.max], rtol=1e-3, atol=0).any()]
        if not at_bound:
            break
        for par in at_bound:
            par.vary = False  # report it without an error rather than lose every error
        result = lmfit.minimize(residual, params, method="leastsq", scale_covar=False)
    result.deviance = result.chisqr  # sum of squared deviance residuals
    result.k = nvarys
    result.dispersion = result.deviance / (result.ndata - nvarys)  # about 1 for Poisson scatter
    result.bic = result.deviance + np.log(result.ndata) * nvarys  # -2 ln L + k ln n, up to a constant
    return result


def _summary(result, model, names, units, data, all_points):
    p = result.params
    inflate = np.sqrt(max(1.0, result.dispersion))  # errors grow with scatter beyond Poisson, never shrink

    def scalar(name, unit):
        err = inflate * float(p[name].stderr) if p[name].vary and p[name].stderr is not None else np.nan
        return sc.scalar(float(p[name].value), variance=err**2, unit=unit)

    axes = range(len(names))
    out = {
        "fwhm": sc.DataGroup({names[k]: sc.scalar(_fwhm(p[f"sig{k}"].value, p[f"gam{k}"].value), unit=units[k]) for k in axes}),
        "sigma": sc.DataGroup({names[k]: scalar(f"sig{k}", units[k]) for k in axes}),
        "gamma": sc.DataGroup({names[k]: scalar(f"gam{k}", units[k]) for k in axes}),
        "amplitude": scalar("amp", None),
        "center": sc.DataGroup({names[k]: scalar(f"c{k}", units[k]) for k in axes}),
    }
    if model.n == 2:
        out["separation"] = sc.DataGroup({names[k]: scalar(f"s{k}", units[k]) for k in axes})
        out["ratio"] = scalar("ratio", None)
        out["peaks"] = sc.DataGroup({
            f"p{i}": sc.DataGroup({names[k]: sc.scalar(p[f"c{k}"].value + sgn * p[f"s{k}"].value, unit=units[k]) for k in axes})
            for i, sgn in enumerate((-0.5, 0.5))
        })
    if model.coupling:
        out["coupling"] = sc.DataGroup({names[k]: scalar(f"k{k}", None if units[k] is None or units[0] is None
                                                         else units[k] / units[0]) for k in range(1, len(names))})
    best = model(p.valuesdict(), [x.ravel() for x in all_points]).reshape(data.shape)
    out |= {"deviance": result.deviance, "dispersion": result.dispersion, "bic": result.bic,
            "best_fit": sc.DataArray(sc.array(dims=data.dims, values=best, unit=data.unit), coords=dict(data.coords)),
            "result": result}
    return sc.DataGroup(out)


def fit_split(da, coords=None, shape="voigt", coupling=True, exposure=None) -> sc.DataGroup:
    """Fit one peak and two peaks to 1-, 2- or 3-D data and compare them.

    Each peak is a product of profiles along the axes (Gaussian, Lorentzian
    or Voigt; the Voigt is exact, via the Faddeeva function), the two peaks
    share their widths, and a planar background is fitted with them. The
    two-peak model is parametrised by the mean centre, the separation along
    every axis, and the intensity fraction `ratio` of the first peak, so each
    comes with a standard error. Its fit is started from a grid of
    separations around the one-peak result and the best is kept.

    The fits maximise the Poisson likelihood of the photon counts, which stays
    unbiased where pixels hold few or no photons (a variance-weighted least
    squares does not). Counts and their scale come from the data's values and
    variances, so reduce with `photon_energy`; normalising by I0 is fine.

    The comparison uses the Bayesian information criterion (BIC) of each
    model, BIC = −2 ln L + k ln n, where L is the maximised likelihood, k the
    number of free parameters and n the number of data points. The first
    term rewards a better fit, the second penalises every extra parameter, so
    a lower BIC is the better model once its complexity is paid for.
    `delta_bic` = BIC(one peak) − BIC(two peaks): positive values favour two
    peaks. Differences of 2-6 are positive, 6-10 strong and above 10 very
    strong evidence (Kass and Raftery, 1995); `split` is True above
    `SPLIT_EVIDENCE` (10). Near or below zero the data do not need a second
    peak, whatever the two-peak fit returns. The likelihood ratio is divided
    by the two-peak fit's `dispersion` (deviance per degree of freedom) when
    that exceeds 1, so scatter beyond Poisson does not fake evidence; standard
    errors are inflated by its square root.

    The standard errors of the two-peak parameters mean something only when
    `split` is True: two peaks closer than about twice their width look like
    one broader peak, the fit can trade width for separation, and the errors
    from the curvature of the likelihood are then far too small.

    On a [`histogram`][cristallina_scipp.rsm.histogram] the model is
    evaluated at each voxel's `mean_<dim>` coordinates, the exposure-weighted
    mean position of the pixels in it, rather than at voxel centres, so the
    voxel size does not bias the fit; empty voxels are left out.

    The peaks are labelled so that the most significant separation is
    positive: peak `p1` lies at the larger coordinate along that axis, and
    `ratio` is the fraction in `p0`.

    Args:
        da: Data with the scan axis first: a rocking curve `(step,)`, or the
            ROI image summed over rows, `(step, x)`, for splitting in 2θ, or
            over columns, `(step, y)`, for tilts out of the scattering plane.
            Fit both projections: `(step, y, x)` also works but takes minutes
            and, with more parameters, gives weaker evidence.
        coords: `{dim: coordinate name}` for the axes, e.g.
            `{"step": "readback", "x": "two_theta"}`. Dimensions not listed use
            `mean_<dim>` if present (from `histogram`), else the coordinate
            of the same name. Results are keyed by the names given here, else
            by dimension.
        shape: `"voigt"`, `"lorentzian"` or `"gaussian"`.
        coupling: For 2-D and 3-D data, let the peak position along each
            detector axis move linearly with the scan axis. Not needed in
            reciprocal-space coordinates.
        exposure: Name of a coordinate, or a Variable, with the summed I0 per
            point, such that `values = counts / exposure`: the `exposure` of a
            [`histogram`][cristallina_scipp.rsm.histogram]. Points with zero
            exposure (no pixel landed there) are left out. Without it, the
            scale is taken from the values and variances.

    Returns:
        A DataGroup with `one` and `two` (the fitted parameters with standard
            errors as variances: `center`, `separation`, `ratio`, `fwhm`,
            `sigma`, `gamma` and `coupling` per axis, the `peaks` positions,
            `deviance`, `dispersion`, `bic`, `best_fit` and the lmfit `result`),
            `delta_bic`, and `split` (True if `delta_bic > SPLIT_EVIDENCE`).
    """
    dims = list(da.dims)
    given = coords or {}
    coords = {d: given.get(d, f"mean_{d}" if f"mean_{d}" in da.coords else d) for d in dims}
    names = [given.get(d, d) for d in dims]  # results are keyed by the coordinate asked for, else the dimension
    points = []
    for d in dims:
        c = da.coords[coords[d]]
        c = sc.midpoints(c, d) if da.coords.is_edges(coords[d], d) else c
        points.append(sc.broadcast(c, sizes=da.sizes).transpose(dims).values.astype(float))
    units = [da.coords[coords[d]].unit for d in dims]
    y = da.values.astype(float)
    if da.variances is None:
        raise ValueError("fit_split needs variances (reduce with photon_energy) to weight and compare the fits")
    var = da.variances.astype(float)
    if exposure is not None:
        e = (da.coords[exposure] if isinstance(exposure, str) else exposure).values.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(e > 0, 1 / e, np.nan)
    else:
        # values = scale * counts and variances = scale**2 * counts, so scale = variance / value where
        # there are counts; empty points take their scan step's scale (normalisation is per step)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where((y > 0) & (var > 0), var / y, np.nan)
        rest = tuple(range(1, y.ndim))
        per_step = np.nanmedian(scale, axis=rest, keepdims=True) if rest else np.full_like(scale, np.nanmedian(scale))
        scale = np.where(np.isfinite(scale), scale, np.broadcast_to(per_step, scale.shape))
        scale = np.where(np.isfinite(scale), scale, np.nanmedian(scale))
    mask = np.isfinite(y) & np.isfinite(scale) & np.all([np.isfinite(x) for x in points], axis=0)
    fit = _compare([x[mask] for x in points], y[mask] / scale[mask], scale[mask], shape, coupling)
    return sc.DataGroup({
        "one": _summary(fit["one"], fit["one_model"], names, units, da, points),
        "two": _summary(fit["two"], fit["two_model"], names, units, da, points),
        "delta_bic": fit["delta_bic"],
        "split": bool(fit["delta_bic"] > SPLIT_EVIDENCE),
    })


def _compare(points, counts, scale, shape, coupling):
    """One-peak fit, the best two-peak fit from a grid of starts, and their BIC difference."""
    y = counts * scale
    ndim = len(points)
    one = _Model(points, 1, shape, coupling)
    # start: centroid and RMS width of the background-subtracted data, per axis
    bg = float(np.percentile(y, 10))
    w = np.clip(y - bg, 0, None)
    start = {"bg": bg}
    norm = 1.0
    for k, x in enumerate(points):
        c = float(np.average(x, weights=w)) if w.sum() > 0 else float(x.mean())
        rms = float(np.sqrt(np.average((x - c) ** 2, weights=w))) if w.sum() > 0 else float(np.ptp(x)) / 4
        width = max(rms, 1e-3 * float(np.ptp(x))) / 2
        start |= {f"c{k}": c, f"sig{k}": width, f"gam{k}": width, f"bg{k}": 0.0, f"k{k}": 0.0}
        norm *= np.sqrt(2 * np.pi) * width
    start["amp"] = float(w.max() * norm)  # height times the volume of a unit-height Gaussian
    fit1 = _fit(one, counts, scale, start)

    two = _Model(points, 2, shape, coupling)
    p1 = fit1.params.valuesdict()
    fwhm = [_fwhm(p1[f"sig{k}"], p1[f"gam{k}"]) for k in range(ndim)]
    # every start on a grid of separations (in units of the one-peak FWHM) runs briefly; the two
    # best are then converged, which finds the same optimum as converging all at a fraction of the cost
    grid = [(0.0, 0.6)] + [(-0.6, 0.0, 0.6)] * (ndim - 1)  # first axis >= 0: the other sign is the same model
    trials = []
    for steps in itertools.product(*grid):
        if not any(steps):
            continue
        seed = dict(p1) | {f"s{k}": steps[k] * fwhm[k] for k in range(ndim)} | {"ratio": 0.5}
        seed |= {f"{w}{k}": 0.7 * p1[f"{w}{k}"] for k in range(ndim) for w in ("sig", "gam")}
        trials.append(_fit(two, counts, scale, seed, max_nfev=EXPLORE * (len(seed) + 1)))
    trials.sort(key=lambda r: r.deviance)
    best = min((_fit(two, counts, scale, r.params) for r in trials[:2]), key=lambda r: r.deviance)

    # Label the peaks so the most significant separation is positive (peak 1 at the larger
    # coordinate along that axis): the same model, with the peaks swapped
    signif = [abs(best.params[f"s{k}"].value) / (best.params[f"s{k}"].stderr or np.inf) for k in range(ndim)]
    if best.params[f"s{int(np.argmax(signif))}"].value < 0:
        for k in range(ndim):
            best.params[f"s{k}"].value *= -1
        best.params["ratio"].value = 1 - best.params["ratio"].value
    phi = max(1.0, best.dispersion)
    delta = (fit1.deviance - best.deviance) / phi - np.log(best.ndata) * (best.k - fit1.k)
    return {"one": fit1, "one_model": one, "two": best, "two_model": two, "delta_bic": delta}
