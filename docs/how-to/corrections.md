# Correct for transmission and absorption

Corrections are separate, explicit steps on a [run series](../tutorials/run-series.md). Each returns a corrected
copy, changes only the intensity measures (`integral`, `area`, `height`), stores the factors it applied as a
coordinate, and appends its name and parameters to `settings`, so a [saved series](../tutorials/run-series.md#save-and-reload)
says what was done to it.

## Transmission

Normalising by an I0 monitor removes the effect of every attenuator upstream of that monitor, but not of those
between it and the sample. SwissFEL device names give the position along the beamline in metres, which fixes the
order:

| I0 monitor | Position | Attenuators after it | Correction |
|---|---|---|---|
| gas monitor `SARFE10-PBIG050` | 50 m | front end `OATT053` (53 m), experiment `OATA150` (150 m) | divide by both |
| `SAROP31-PBPS149` | 149 m | experiment `OATA150` | divide by the experiment attenuator |
| JF I0 detector `JF20T01V01` | experiment hutch | none | none |
| no I0 normalisation | | both | divide by both |

[`correct_transmission`][cristallina_scipp.corrections.correct_transmission] reads the monitor from the series'
settings and divides by the right product. The attenuator transmissions of each run are in the summary as
`transmission_fe` and `transmission_exp`.

```python
--8<-- "docs/scripts/corrections.py:transmission"
```

```text
--8<-- "docs/assets/generated/corr_transmission.txt"
```

A series normalised by the JF I0 detector needs no correction, and asking for one is an error rather than a silent
second division:

```python
--8<-- "docs/scripts/corrections.py:jf-i0"
```

```text
--8<-- "docs/assets/generated/corr_jf_i0.txt"
```

## Absorption, from a fluorescence scan

For a thick crystal in reflection, the kinematic integrated intensity is proportional to |F|²/μ(E). Above the
absorption edge the sample absorbs more and every reflection weakens, whatever its resonance. The fluorescence yield
of an energy scan across the edge follows μ(E), so the correction multiplies by
y(E)/y(E<sub>ref</sub>), with the reference below the edge by default.

```python
--8<-- "docs/scripts/corrections.py:absorption"
```

```text
--8<-- "docs/assets/generated/corr_absorption.txt"
```

![Fluorescence yield and the energy series before and after correction](../assets/generated/corr_absorption.svg)

`coord` says which of the series' energies to match with the scan's readback. Check which one the fluorescence scan
followed; in the synthetic data the monochromator readback is offset by 9 eV from the undulator energy, and the scan
readback matches the undulator. Runs outside the scan's energy range are rejected rather than extrapolated.

!!! warning "Direction of the correction"
    Dividing an energy series by the fluorescence yield is the opposite direction for absorption: it deepens the
    dip above the edge instead of removing it. `exponent=-1` does that division, to compare with analyses that used
    it. Removing something else, for example a fluorescence background inside the ROI, is a different correction.

Both the proportionality to μ(E) and the 1/μ dependence are approximations. Fluorescence from a concentrated, thick
sample saturates, so it follows μ(E) less than proportionally near the edge, and the geometric factor
depends on incidence and exit angles, which differ between reflections and from the fluorescence scan. The yield's
own counting noise propagates into the corrected intensities; a noisy yield scan below the edge limits the result.
