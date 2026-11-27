# Check the beam and the photon energy

Before trusting a scan, check what the beam did during it: whether the monitors were steady and agree, whether every
step has all its pulses, which photon energy actually reached the sample, and which pixel energies to count as
fluorescence. Each check takes one scan; the example is a fluorescence energy scan across an absorption edge.

## Beam monitors

[`scan_means`][cristallina_scipp.scan_means] gives the mean of any beam-synchronous or PV channel per step. With
`spread=True` the variances are the pulse-to-pulse variance within each step, the beam's stability; without it they
are the squared standard error of the mean, which is what to use when dividing by a monitor.

```python
--8<-- "docs/scripts/beam_checks.py:monitors"
```

```text
--8<-- "docs/assets/generated/beam_monitors.txt"
```

The gas monitor sits upstream of both attenuators and PBPS149 between them (see
[corrections](corrections.md#transmission)), so their ratio changes with the front-end attenuator, but within a scan
their step means should follow each other. A poor correlation points at a monitor problem, not at the sample.

## Dropped pulses

[`reduce_scan`][cristallina_scipp.reduce_scan] sums the detector and the I0 over the pulses both recorded, and keeps
the count as `n_pulses`. A step with fewer pulses than the others lost frames on one of the two:

```python
--8<-- "docs/scripts/beam_checks.py:pulses"
```

```text
--8<-- "docs/assets/generated/beam_pulses.txt"
```

Normalising by I0 summed over the same pulses keeps such a step correct, but it has fewer counts and larger errors.

## Photon energy

Three numbers describe the energy: the undulator setting (`SARUN:FELPHOTENE.VAL`, keV), the monochromator readback
(`SAROP31-ODCC110:MOT_ENY.RBV`, eV) and the value the scan records as its readback. Compare them before choosing which
one an energy series is plotted against, or matched to a fluorescence scan with (`coord` in
[`correct_absorption`][cristallina_scipp.corrections.correct_absorption]):

```python
--8<-- "docs/scripts/beam_checks.py:energy"
```

```text
--8<-- "docs/assets/generated/beam_energy.txt"
```

A constant offset is a calibration question for the beamline; an offset that drifts during the scan is not.

## The fluorescence window

[`pixel_energies`][cristallina_scipp.pixel_energies] histograms the energies of the pixels hit in a few steps. The
fluorescence line, the elastic line at the incident energy and the noise near the threshold are separate peaks.
Compare steps below and above the edge: the fluorescence peak grows across the edge and the elastic one does not.
Set `cutoff` and `upper_cutoff` of [`reduce_scan`][cristallina_scipp.reduce_scan] around the fluorescence peak,
clear of the elastic one:

```python
--8<-- "docs/scripts/beam_checks.py:window"
```

```text
--8<-- "docs/assets/generated/beam_window.txt"
```

![Monitors per step, photon-energy offsets, and pixel-energy histograms below and above the edge](../assets/generated/beam_checks.svg)

The same histogram, taken on a rocking curve, shows where to put the Bragg `cutoff`: below the elastic line, above
the fluorescence.
