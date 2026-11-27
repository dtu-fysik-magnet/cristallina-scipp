"""One peak or two? Fits for split reflections, on a rocking curve or on the detector image per step.

A tetragonal-to-orthorhombic distortion splits a reflection two ways: twin
domains with different d-spacings scatter at different 2θ (detector column),
and domains tilted against each other reflect at different ω (rocking angle)
or, out of the scattering plane, at different ν (detector row). Summing the
ROI into a rocking curve keeps only the ω part, so a splitting mainly in 2θ
or ν shows up there as a broad or skewed peak. Fitting the `(step, x)` map,
or `(step, y)`, keeps it.

[`fit_split`][cristallina_scipp.split.fit_split] fits one peak and two
peaks to the same data and compares them, so that a reported splitting is a
measured result with an uncertainty, not a choice of model:

```python
stack = cs.reduce_scan(scan, channels.JF, channels.JF_I0, roi=roi, cutoff=7.0, photon_energy=7.511)
image = (stack / stack.coords["i0"]).sum("y")        # (step, x): rocking angle and 2θ
fit = fit_split(image, coords={"step": "readback"})
fit["delta_bic"], fit["two"]["separation"]
```
"""

import itertools

import lmfit
import numpy as np
import scipp as sc
from scipy.special import voigt_profile, xlogy

_FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))
SPLIT_EVIDENCE = 10.0  # BIC difference above which two peaks are preferred ("very strong" on the Kass-Raftery scale)


def _fwhm(sigma, gamma):
    """Voigt FWHM (Olivero and Longbothum 1977, 0.02 %)."""
    fl, fg = 2 * gamma, _FWHM_PER_SIGMA * sigma
    return 0.5346 * fl + np.sqrt(0.2166 * fl**2 + fg**2)


class _Model:
    """Sum of `n` peaks, each a product of 1-D profiles along the axes, plus a planar background.

    Peak positions are written as a shared centre `c{k}` and separation `s{k}`
    per axis (peak 0 at c - s/2, peak 1 at c + s/2) with intensity fractions
    `ratio` and 1 - `ratio`, so the separation and its error come straight
    out of the fit. Widths are shared by the two peaks. With `coupling`, the
    position along each detector axis may move linearly with the scan axis
    (`k{j}`), as from the beam's energy spread or divergence.
    """

    def __init__(self, axes, n, shape, coupling):
        self.axes, self.n, self.shape, self.coupling = axes, n, shape, coupling and len(axes) > 1
        self.grids = np.meshgrid(*axes, indexing="ij")
        self.mid = [a.mean() for a in axes]

    def profile(self, k, x, p):
        sigma = p[f"sig{k}"] if self.shape != "lorentzian" else 0.0
        gamma = p[f"gam{k}"] if self.shape != "gaussian" else 0.0
        return voigt_profile(x, sigma, gamma)

    def peak(self, p, i):
        sign = (-0.5, 0.5)[i] if self.n == 2 else 0.0
        centre = [p[f"c{k}"] + sign * p.get(f"s{k}", 0.0) for k in range(len(self.axes))]
        shape = [len(a) for a in self.axes]
        out = self.profile(0, self.axes[0] - centre[0], p).reshape([-1] + [1] * (len(shape) - 1))
        for k in range(1, len(self.axes)):
            if self.coupling:  # position along axis k drifts with the scan axis
                arg = self.grids[k] - centre[k] - p[f"k{k}"] * (self.grids[0] - centre[0])
                index = tuple(slice(None) if j in (0, k) else 0 for j in range(len(shape)))
                out = out * self.profile(k, arg[index], p).reshape([shape[j] if j in (0, k) else 1 for j in range(len(shape))])
            else:
                out = out * self.profile(k, self.axes[k] - centre[k], p).reshape([shape[k] if j == k else 1 for j in range(len(shape))])
        return out

    def __call__(self, p):
        fractions = (p["ratio"], 1 - p["ratio"]) if self.n == 2 else (1.0,)
        y = sum(p["amp"] * f * self.peak(p, i) for i, f in enumerate(fractions))
        background = p["bg"] + sum(p[f"bg{k}"] * (g - m) for k, (g, m) in enumerate(zip(self.grids, self.mid)))
        return y + background


def _params(model, start):
    params = lmfit.Parameters()
    for name, value in start.items():
        params.add(name, value=value)
    for k, axis in enumerate(model.axes):
        params[f"c{k}"].set(min=axis.min(), max=axis.max())
        floor = 1e-3 * np.min(np.abs(np.diff(axis)))  # a Voigt with both widths 0 is undefined
        for w, unused in (("sig", "lorentzian"), ("gam", "gaussian")):
            if model.shape == unused:
                params[f"{w}{k}"].set(value=0.0, vary=False)
            else:
                params[f"{w}{k}"].set(min=floor)
        if model.n == 2:
            span = np.ptp(axis)
            params[f"s{k}"].set(min=0 if k == 0 else -span, max=span)  # s0 >= 0 labels the peaks
    params["amp"].set(min=0)
    if model.n == 2:
        params["ratio"].set(min=0, max=1)
    params.pop("k0", None)  # the scan axis does not couple to itself
    if not model.coupling:
        for k in range(1, len(model.axes)):
            params.pop(f"k{k}", None)
    return params


def _fit(model, counts, scale, start):
    """Maximum Poisson likelihood: least squares on deviance residuals of the counts."""
    mask = np.isfinite(counts) & np.isfinite(scale)
    n = counts[mask]

    def residual(p):
        mu = np.maximum(model(p.valuesdict())[mask] / scale[mask], 1e-12)
        d = 2 * (mu - n + xlogy(n, n / mu))
        return np.sign(n - mu) * np.sqrt(np.maximum(d, 0))

    result = lmfit.minimize(residual, _params(model, start), method="leastsq", scale_covar=False)
    nvarys = result.nvarys  # the model's free parameters, for the BIC, even if some are fixed below
    if not result.errorbars:  # a parameter on its bound (e.g. a Lorentzian width of 0) makes the covariance singular
        params = result.params.copy()
        for par in params.values():
            if par.vary and np.isclose(par.value, [par.min, par.max], rtol=1e-6, atol=1e-12).any():
                par.vary = False  # report it without an error rather than lose every error
        result = lmfit.minimize(residual, params, method="leastsq", scale_covar=False)
    result.deviance = result.chisqr  # sum of squared deviance residuals
    result.k = nvarys
    result.dispersion = result.deviance / (result.ndata - nvarys)  # about 1 for Poisson scatter
    result.bic = result.deviance + np.log(result.ndata) * nvarys  # -2 ln L + k ln n, up to a constant
    return result


def _summary(result, model, units, dims, coords, data):
    p = result.params
    names = [coords[d] for d in dims]

    inflate = np.sqrt(max(1.0, result.dispersion))  # errors grow with scatter beyond Poisson, never shrink

    def scalar(name, unit):
        err = inflate * float(p[name].stderr) if p[name].vary and p[name].stderr is not None else np.nan
        return sc.scalar(float(p[name].value), variance=err**2, unit=unit)

    axes = range(len(dims))
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
                                                         else units[k] / units[0]) for k in range(1, len(dims))})
    out |= {"deviance": result.deviance, "dispersion": result.dispersion, "bic": result.bic,
            "best_fit": sc.DataArray(sc.array(dims=dims, values=model(p.valuesdict()), unit=data.unit),
                                     coords={c: data.coords[c] for c in coords.values()}),
            "result": result}
    return sc.DataGroup(out)


def fit_split(da, coords=None, shape="voigt", coupling=True) -> sc.DataGroup:
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

    The comparison is the Bayesian information criterion: `delta_bic` =
    BIC(one) − BIC(two), with the likelihood ratio divided by the two-peak
    fit's `dispersion` (deviance per degree of freedom) when that exceeds 1,
    so scatter beyond Poisson does not fake evidence; standard errors are
    inflated by its square root. Values above `SPLIT_EVIDENCE` (10) are very
    strong evidence for two peaks; near or below zero the data do not need a
    second peak, whatever the two-peak fit returns.

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
            the coordinate of the same name.
        shape: `"voigt"`, `"lorentzian"` or `"gaussian"`.
        coupling: For 2-D and 3-D data, let the peak position along each
            detector axis move linearly with the scan axis.

    Returns:
        A DataGroup with `one` and `two` (the fitted parameters with standard
            errors as variances: `center`, `separation`, `ratio`, `fwhm`,
            `sigma`, `gamma` and `coupling` keyed by coordinate name, the `peaks` positions,
            `deviance`, `dispersion`, `bic`, `best_fit` and the lmfit `result`),
            `delta_bic`, and `split` (True if `delta_bic > SPLIT_EVIDENCE`).
    """
    dims = list(da.dims)
    coords = {d: (coords or {}).get(d, d) for d in dims}
    axes = [da.coords[coords[d]].values.astype(float) for d in dims]
    units = [da.coords[coords[d]].unit for d in dims]
    y = da.values.astype(float)
    if da.variances is None:
        raise ValueError("fit_split needs variances (reduce with photon_energy) to weight and compare the fits")
    var = da.variances.astype(float)
    # values = scale * counts and variances = scale**2 * counts, so scale = variance / value where
    # there are counts; empty points take their scan step's scale (normalisation is per step)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where((y > 0) & (var > 0), var / y, np.nan)
    rest = tuple(range(1, y.ndim))
    per_step = np.nanmedian(scale, axis=rest, keepdims=True) if rest else np.full_like(scale, np.nanmedian(scale))
    scale = np.where(np.isfinite(scale), scale, np.broadcast_to(per_step, scale.shape))
    scale = np.where(np.isfinite(scale), scale, np.nanmedian(scale))
    counts = y / scale

    ndim = len(dims)
    one = _Model(axes, 1, shape, coupling)
    # start: centroid and RMS width of the background-subtracted data, per axis
    w = np.clip(y - np.nanpercentile(y, 10), 0, None)
    start = {"bg": float(np.nanpercentile(y, 10)), "amp": 0.0}
    for k in range(ndim):
        marginal = w.sum(axis=tuple(j for j in range(ndim) if j != k))
        c = float(np.average(axes[k], weights=marginal)) if marginal.sum() > 0 else float(axes[k].mean())
        rms = np.sqrt(np.average((axes[k] - c) ** 2, weights=marginal)) if marginal.sum() > 0 else np.ptp(axes[k]) / 4
        width = max(rms, np.min(np.abs(np.diff(axes[k])))) / 2
        start |= {f"c{k}": c, f"sig{k}": width, f"gam{k}": width, f"bg{k}": 0.0, f"k{k}": 0.0}
    volume = np.prod([np.abs(np.mean(np.diff(a))) for a in axes])
    start["amp"] = float(w.sum() * volume)
    fit1 = _fit(one, counts, scale, start)

    two = _Model(axes, 2, shape, coupling)
    p1 = fit1.params.valuesdict()
    fwhm = [_fwhm(p1[f"sig{k}"], p1[f"gam{k}"]) for k in range(ndim)]
    best = None
    grid = [(0.0, 0.6)] + [(-0.6, 0.0, 0.6)] * (ndim - 1)  # separations in units of the one-peak FWHM
    for steps in itertools.product(*grid):
        if not any(steps):
            continue
        seed = dict(p1) | {f"s{k}": steps[k] * fwhm[k] for k in range(ndim)} | {"ratio": 0.5}
        seed |= {f"{w}{k}": 0.7 * p1[f"{w}{k}"] for k in range(ndim) for w in ("sig", "gam")}
        fit2 = _fit(two, counts, scale, seed)
        if best is None or fit2.deviance < best.deviance:
            best = fit2

    # Label the peaks so the most significant separation is positive (peak 1 at the larger
    # coordinate along that axis): the same model, with the peaks swapped
    signif = [abs(best.params[f"s{k}"].value) / (best.params[f"s{k}"].stderr or np.inf) for k in range(ndim)]
    if best.params[f"s{int(np.argmax(signif))}"].value < 0:
        for k in range(ndim):
            best.params[f"s{k}"].value *= -1
        best.params["ratio"].value = 1 - best.params["ratio"].value
    phi = max(1.0, best.dispersion)
    delta = (fit1.deviance - best.deviance) / phi - np.log(best.ndata) * (best.k - fit1.k)
    return sc.DataGroup({
        "one": _summary(fit1, one, units, dims, coords, da),
        "two": _summary(best, two, units, dims, coords, da),
        "delta_bic": delta,
        "split": bool(delta > SPLIT_EVIDENCE),
    })
