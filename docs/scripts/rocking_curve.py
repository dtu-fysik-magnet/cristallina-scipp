"""Tutorial: a rocking curve, like run 55 of the March 2026 beamtime.

Covers the ROI rocking curve, a peak fit, the spot's position and width per step, and angle coordinates.
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plopp as pp
import scipp as sc
from _common import save, text

from sfdata import SFScanInfo

from cristallina_scipp import testing

scan = SFScanInfo(str(testing.rocking_scan(Path(tempfile.mkdtemp()))))
# real data: scan = SFScanInfo("/sf/cristallina/data/p2xxxx/raw/run0055/meta/scan.json")

# --8<-- [start:reduce]
import cristallina_scipp as cs
from cristallina_scipp import channels
from cristallina_scipp import ROI

stack = cs.reduce_scan(scan, channels.JF, channels.JF_I0, cutoff=7.0, i0_cutoff=2.5, photon_energy=7.511)
normalised = stack / stack.coords["i0"]

# real data: e.g. ROI(left=367, right=467, bottom=664, top=764, name="peak_01")
peak = ROI(left=20, right=62, bottom=22, top=38, name="peak_01")
background = ROI(left=20, right=62, bottom=2, top=18, name="background")
curves = cs.roi_intensities(normalised, [peak, background])
rocking = curves["roi", 0] - curves["roi", 1]
full_image = normalised.sum(["y", "x"])
# --8<-- [end:reduce]

# --8<-- [start:image]
on_peak = int(np.argmax(rocking.values))
fig = pp.plot(stack["step", on_peak], title=f"step {on_peak}")
for r in (peak, background):
    fig.ax.add_patch(plt.Rectangle((r.left, r.bottom), r.right - r.left, r.top - r.bottom, fill=False, color="w"))
# --8<-- [end:image]
save(fig, "rocking_image")

# --8<-- [start:fit]
from cristallina_scipp.peaks import fit_peaks

fit = fit_peaks(rocking, shape="lorentzian", coord="readback")
p0 = fit["p0"]
print(f"centre {p0['center'].value:.4f} ± {p0['center'].variance**0.5:.4f} deg")
print(f"FWHM   {p0['fwhm'].value:.4f} ± {p0['fwhm'].variance**0.5:.4f} deg")
print(f"area   {p0['area'].value:.3g} ± {p0['area'].variance**0.5:.2g} (counts/keV of I0) deg")
fig = pp.plot(rocking, coords=["readback"], title="ROI - background, Lorentzian fit")
fig.ax.plot(fit["best_fit"].coords["readback"].values, fit["best_fit"].values, "-", color="C1")
# --8<-- [end:fit]
save(fig, "rocking_fit")
text(
    "rocking_fit",
    f"centre {p0['center'].value:.4f} ± {p0['center'].variance**0.5:.4f} deg",
    f"FWHM   {p0['fwhm'].value:.4f} ± {p0['fwhm'].variance**0.5:.4f} deg",
    f"area   {p0['area'].value:.3g} ± {p0['area'].variance**0.5:.2g} (counts/keV of I0) deg",
    f"reduced chi^2 {fit['result'].redchi:.2f}",
)

# --8<-- [start:moments]
from cristallina_scipp.peaks import moments

in_roi = cs.crop(stack, peak)
horizontal = moments(in_roi.sum("y"), "x")  # centroid and FWHM along x, per step
vertical = moments(in_roi.sum("x"), "y")
strong = rocking.values > 0.3 * rocking.values.max()  # only steps with a clear spot
# --8<-- [end:moments]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
for ax, (name, m) in zip(axs, [("x", horizontal), ("y", vertical)]):
    ax.errorbar(stack.coords["readback"].values[strong], m["centroid"].values[strong],
                yerr=m["fwhm"].values[strong] / 2, fmt="o")
    ax.set_xlabel("readback [deg]")
    ax.set_ylabel(f"{name} centroid ± FWHM/2 [px]")
save(fig, "rocking_moments")

# --8<-- [start:angles]
from cristallina_scipp.angles import MARCH_2026, detector_angles, geometry_from_step

with scan[0] as first:
    geometry = geometry_from_step(first, **MARCH_2026)
angular = detector_angles(stack["step", on_peak], **geometry)
profile = cs.crop(angular, peak).sum("y")
fig = pp.plot(profile, coords=["two_theta"], title="horizontal profile through the peak")
# --8<-- [end:angles]
save(fig, "rocking_angles")
text("rocking_geometry", {k: round(v, 6) for k, v in geometry.items()})

# checks
assert abs(p0["center"].value - testing.BRAGG["center"]) < 5 * p0["center"].variance ** 0.5 + 0.003
assert abs(p0["fwhm"].value - testing.BRAGG["fwhm"]) < 0.25 * testing.BRAGG["fwhm"]
drift = np.polyfit(stack.coords["readback"].values[strong], horizontal["centroid"].values[strong], 1)[0]
assert 30 < drift < 50, drift  # synthetic spot drifts 40 px/deg
assert angular.coords["two_theta"].unit == sc.Unit("deg")
