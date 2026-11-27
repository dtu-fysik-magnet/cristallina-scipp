# Rocking curve

This follows run 55 of the March 2026 beamtime: the sample angle is stepped through a Bragg reflection, and the
reflected intensity is measured on the Jungfrau 1.5M. Here the data are a synthetic copy. The spot drifts across the
detector as the sample rocks, so there is something for the per-step statistics to find.

## Reduce, then define ROIs

```python
--8<-- "docs/scripts/rocking_curve.py:reduce"
```

```python
--8<-- "docs/scripts/rocking_curve.py:image"
```

![Summed image of the step closest to the peak](../assets/generated/rocking_image.svg)

`rocking` is the ROI intensity, normalised and
background-subtracted, as a function of the readback.

## Fit the peak

[`fit_peaks`][cristallina_scipp.peaks.fit_peaks] seeds peaks with `scipy.signal.find_peaks`, fits Lorentzians or
Gaussians plus a linear background with lmfit, and weights the fit by the propagated errors. Each result is a scalar
with a variance, so it can go straight into further arithmetic.

```python
--8<-- "docs/scripts/rocking_curve.py:fit"
```

```text
--8<-- "docs/assets/generated/rocking_fit.txt"
```

![Rocking curve and fit](../assets/generated/rocking_fit.svg)

The synthetic peak is a Lorentzian centred at −18.000° with a FWHM of 0.080°.

## Spot position and width per step

[`moments`][cristallina_scipp.peaks.moments] gives an intensity-weighted centroid and a Gaussian-equivalent FWHM, for
every step at once. Projections are taken **inside the ROI** in both directions:

```python
--8<-- "docs/scripts/rocking_curve.py:moments"
```

![Centroid ± FWHM/2 per step](../assets/generated/rocking_moments.svg)

The synthetic spot moves 40 px/deg along x and not at all along y, and that is what comes back.

## Angles instead of pixels

See [scattering angles](../how-to/pixel-angles.md).
