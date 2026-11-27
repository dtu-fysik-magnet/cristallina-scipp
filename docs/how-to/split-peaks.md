# Decide whether a reflection is split

A tetragonal-to-orthorhombic distortion splits a reflection into twin domains in two ways: domains with different
d-spacings scatter at different 2θ, and domains tilted against each other reflect at different ω or, out of the
scattering plane, at different ν. How large either is depends on the sample, temperature and field, so the question
for every run is the same: one peak, or two, and how far apart?

[`fit_split`][cristallina_scipp.split.fit_split] answers it by fitting one peak and two peaks to the same data and
comparing them with the Bayesian information criterion. `delta_bic` above 10 is very strong evidence for two peaks;
near or below zero, the data do not need a second one, whatever the two-peak fit returns. The fits maximise the
Poisson likelihood of the photon counts, so reduce with `photon_energy`.

## On the detector

Summing the ROI into a rocking curve keeps only ω. Fit the image summed over rows, `(step, x)`, for a splitting in
2θ, and over columns, `(step, y)`, for a tilt out of the scattering plane:

```python
--8<-- "docs/scripts/split_peaks.py:detector"
```

```text
--8<-- "docs/assets/generated/split_detector.txt"
```

Here the two domains differ only in d-spacing. The rocking curve looks like one peak, and its fit is fine; the
`(step, x)` map shows two spots and measures their separation.

## In reciprocal space

On the detector, strain and tilt mix: a tilt moves a spot in ω and ν, a change of d-spacing moves it in 2θ, and the
beam's energy spread couples ω and 2θ. In reciprocal space they separate.
[`local_coords`][cristallina_scipp.rsm.local_coords] turns each pixel's h, k, l (from
[`pixel_hkl`][cristallina_scipp.rsm.pixel_hkl]) into `strain`, the relative change of d-spacing along the reciprocal
vector G, and two tilts across it, in degrees. These do not depend on the crystal's orientation, so they mean the
same for every run and reflection. A [`histogram`][cristallina_scipp.rsm.histogram] in those coordinates keeps the
summed I0 of each voxel as `exposure` and the mean position of its pixels as `mean_strain`, `mean_tilt_1` and
`mean_tilt_2`. The fit uses both: an exact Poisson model of the binned counts, evaluated where the pixels actually
were:

```python
--8<-- "docs/scripts/split_peaks.py:rsm"
```

```text
--8<-- "docs/assets/generated/split_rsm.txt"
```

![A split in 2θ on the detector, the rocking curve that hides it, and two domains in reciprocal space](../assets/generated/split_peaks.svg)

## What to trust

- **`delta_bic` first.** The separations and their errors mean something only when `split` is True. Two peaks closer
  than about twice their width look like one broader peak; the two-peak fit then trades width for separation, and its
  errors come out far too small.
- **Resolution.** In synthetic tests with known splitting, a separation of 3σ (peak widths, as Gaussian σ) along a
  single axis is borderline (`delta_bic` from about 4 to 40 depending on the noise), 4σ is detected reliably, and
  smaller separations along several axes at once add up (2σ in strain plus 2σ in tilt gave `delta_bic` near 190).
  Measure the limit for your own peak widths and counts with
  [`split_rocking_scan`][cristallina_scipp.testing.split_rocking_scan] or
  [`split_rsm_stack`][cristallina_scipp.testing.split_rsm_stack].
- **Voxel size.** Because the model is evaluated at each voxel's mean pixel position, the voxel size does not bias
  the fit: voxels of half the peak's σ gave the same, unbiased separations as finer ones, at a fraction of the time
  (evaluating at voxel centres instead biased the strain by 1.6 standard errors). Strain is sampled in discrete
  steps, one per detector column, so voxels finer than that leave empty rows; they are simply left out.
- **Errors.** Where `split` is True, the errors are calibrated: over repeated synthetic runs the separations scatter
  around the truth by their quoted errors. Scatter beyond Poisson (`dispersion` above 1) inflates them and weakens
  `delta_bic` accordingly.
- **The full detector volume**, `(step, y, x)`, also works, but takes minutes and, with more parameters, gives weaker
  evidence than the two projections. In reciprocal space a 3-D fit of a 24 × 24 × 24 map takes about half a minute.
