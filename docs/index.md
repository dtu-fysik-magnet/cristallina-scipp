# cristallina-scipp

Tools for looking at SwissFEL/Cristallina data with [scipp](https://scipp.github.io) and
[plopp](https://scipp.github.io/plopp), written while preparing for a Cristallina-Q beamtime.

The beamline stack still does the I/O and the geometry. This package takes over once the data are reduced:

| Step | Done by |
|---|---|
| files, pulse-id matching, scan steps | `sfdata` |
| Jungfrau gain, pedestal and geometry | `jungfrau_utils` |
| pixel to hkl | `cristallina.rsm` |
| labelled arrays, units, errors, fitting, binning | **cristallina-scipp** |

What you gain over the loops in the beamline notebooks:

- **The same pulses in numerator and denominator.** [`reduce_scan`][cristallina_scipp.reduce_scan] drops pulses
  missing from either the detector or I0. See [normalisation](explanation/normalisation.md).
- **Readbacks, units and errors travel with the data.** Plots label themselves, and Poisson errors propagate through
  normalisation and background subtraction into fits.
- **Fits and per-step peak statistics** that carry uncertainties ([`fit_peaks`][cristallina_scipp.peaks.fit_peaks],
  [`moments`][cristallina_scipp.peaks.moments]).
- **Reciprocal-space maps by histogramming** instead of nearest-neighbour interpolation
  ([why](explanation/rsm-binning.md)).

## Quick look

```python
from sfdata import SFScanInfo
import cristallina_scipp as cs
from cristallina_scipp import channels

scan = SFScanInfo("/sf/cristallina/data/p2xxxx/raw/run0055/meta/scan.json")
stack = cs.reduce_scan(scan, channels.JF, channels.JF_I0, cutoff=7.0, i0_cutoff=2.5)
curve = cs.roi_intensities(stack, [peak_roi])["roi", 0] / stack.coords["i0"]
```

## Where to go

- **Tutorials** walk through complete analyses on synthetic scans that imitate the March 2026 runs. Start with the
  [fluorescence energy scan](tutorials/fluorescence-scan.md).
- **How-to guides** cover single tasks: [finding the sample](how-to/sample-search.md), [describing the experiment in a run catalogue](how-to/catalogue.md), and
  [translating existing cristallina code](how-to/from-cristallina.md).
- **Reference**: the [API](reference/api.md), generated from the docstrings.

Every code block in the tutorials and how-tos comes from a script in `docs/scripts/`. The test suite runs those
scripts, so the code shown is the code that ran.
