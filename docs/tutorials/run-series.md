# Run series

Most physics runs of a resonant diffraction experiment are short rocking curves at one energy, field and temperature. A result is a
series of them: integrated intensity against photon energy (a resonance), or peak position against field, for one or
more field directions. This tutorial builds both from a [run catalogue](../how-to/catalogue.md), on synthetic runs
of two reflections, A and B.

## Load the catalogue

The catalogue holds what the data files cannot tell you: which runs belong together and why, the ROIs, and the
detector calibration.

```toml
--8<-- "docs/scripts/example_catalogue.toml"
```

```python
--8<-- "docs/scripts/run_series.py:load"
```

```text
--8<-- "docs/assets/generated/series_load.txt"
```

## An energy series

[`series`][cristallina_scipp.catalogue.Catalogue.series] reduces every run of a group to its ROI curve, normalised
by I0 per step, and measures that curve three ways: a background-subtracted `integral`, a one-peak fit (`center`,
`fwhm`, `height`, `area`) and moments (`centroid`, `moment_fwhm`). The run metadata (undulator and monochromator
energy, field components and magnitude, attenuator transmission) and the group's labels become coordinates along
`run`.

```python
--8<-- "docs/scripts/run_series.py:energy"
```

```text
--8<-- "docs/assets/generated/series_energy.txt"
```

Every number carries a variance: Poisson counting statistics from `photon_energy`, propagated through the
normalisation, the background line and the integral, or the fit's standard error.

```python
--8<-- "docs/scripts/run_series.py:energy-plot"
```

![Run curves and the resonance of reflection A](../assets/generated/series_energy.svg)

## Several groups: field and field direction

Pass a list of groups to combine them. Each run keeps its group's labels, so `field_angle` here separates the two
field directions. A run listed in two of the groups raises an error, which catches ranges that overlap by one.

```python
--8<-- "docs/scripts/run_series.py:field"
```

```text
--8<-- "docs/assets/generated/series_field.txt"
```

![Peak position of reflection B against field for two field directions](../assets/generated/series_field.svg)

## Which integrated intensity?

The integral and the fitted area answer slightly different questions. The integral is model-free but covers only
the scanned range; the Lorentzian area includes the tails beyond it.

```python
--8<-- "docs/scripts/run_series.py:methods"
```

```text
--8<-- "docs/assets/generated/series_methods.txt"
```

Both are in every summary, so a plot can say which it shows, and two series can only be compared like for like. To
check how much the background choice matters, rerun with another `n_edge`, e.g.
`cat.series(group, n_edge=5, **reduce)`.

## Save and reload

A series saves as one HDF5 file with everything needed to reproduce or question it: the summary, every run's curve,
and the settings. Each point records the ROI name and the pixel bounds actually summed, and its `scan.json`;
`settings` record the cutoffs, I0 normalisation and integration choices, the package version, when the series was
made, and the catalogue file with a SHA-256 of its contents. Share the file instead of a CSV: anyone can reload it
and see how every number was obtained.

```python
--8<-- "docs/scripts/run_series.py:save"
```

```text
--8<-- "docs/assets/generated/series_save.txt"
```

If the catalogue changes later, its hash no longer matches the one in the file, so you can tell whether a saved
series was made with the current definitions.

## Clipping

The background-subtracted integral is never clipped at zero. Clipping turns negative noise into positive area, so
weak peaks would come out too large; an integral that scatters around zero is the correct result for no signal.
