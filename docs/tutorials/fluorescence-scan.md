# Fluorescence energy scan

This tutorial reproduces the photon-energy scan from the March 2026 beamtime (run 45): the incident energy is stepped
across an absorption edge, and the fluorescence is counted on the Jungfrau 1.5M in a 5.5–6.8 keV window, normalised by
the JF-I0 detector.

It runs on a synthetic scan written in the SwissFEL file layout, with the real channel names. On real data only the
line that opens the scan changes.

## Open the scan

```python
--8<-- "docs/scripts/fluorescence_scan.py:open"
```

```text
--8<-- "docs/assets/generated/fluorescence_scan_info.txt"
```

## Beam monitors and photon-energy offsets

[`scan_means`][cristallina_scipp.scan_means] replaces the per-step `np.mean` / `np.std` loops. With `spread=True`
the error bars show the pulse-to-pulse spread within each step, which is what you want when judging beam stability.
The default is the standard error of the mean, which is what you want before dividing by it.

```python
--8<-- "docs/scripts/fluorescence_scan.py:beam"
```

![Monitors per step](../assets/generated/fluorescence_beam.svg)

```text
--8<-- "docs/assets/generated/fluorescence_offsets.txt"
```

## Reduce the scan in an energy window

`cutoff` and `upper_cutoff` are applied to every pixel of every pulse before summing. The window keeps the
fluorescence line and rejects both electronic noise and elastic scattering at the incident energy.

`photon_energy` is the energy of the **detected** photons, here the emission line, not the scanned incident energy.
With it, the result is in photon counts with Poisson variances.

```python
--8<-- "docs/scripts/fluorescence_scan.py:reduce"
```

```text
--8<-- "docs/assets/generated/fluorescence_stack.txt"
```

```python
--8<-- "docs/scripts/fluorescence_scan.py:plot"
```

![Fluorescence scan](../assets/generated/fluorescence_scan.svg)

## Check pulse alignment

In the synthetic data the I0 detector misses one pulse in step 10. `reduce_scan` drops that pulse from the 1.5M sum as
well, and records the pulse count per step:

```python
--8<-- "docs/scripts/fluorescence_scan.py:alignment"
```

```text
--8<-- "docs/assets/generated/fluorescence_alignment.txt"
```

The beamline loop and `calculate_JF_stacks` sum the two detectors in separate calls, so a missed frame leaves
numerator and denominator covering different pulses. See [normalisation](../explanation/normalisation.md).
