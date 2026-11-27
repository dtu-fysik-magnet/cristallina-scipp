"""How-to: decide whether a reflection is split, on the detector and in reciprocal space.

Synthetic reflections split into two twin domains with known separations.
"""

import tempfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipp as sc
from _common import save, text

from sfdata import SFScanInfo

import cristallina_scipp as cs
from cristallina_scipp import channels, testing

warnings.filterwarnings("ignore", message="Could not")  # cristallina's cache-directory notices
base = Path(tempfile.mkdtemp())
path = testing.split_rocking_scan(base, run=56, d_x=6.0)  # two d-spacings: 6 px apart in 2theta, same omega

# --8<-- [start:detector]
from cristallina_scipp.split import fit_split

stack = cs.reduce_scan(SFScanInfo(str(path)), channels.JF, channels.JF_I0, cutoff=7.0, i0_cutoff=2.5,
                       photon_energy=7.511)
norm = stack / (stack.coords["i0"] / sc.scalar(1.0, unit=stack.coords["i0"].unit))
fits = {
    "rocking curve (step)": fit_split(norm.sum(["y", "x"]), coords={"step": "readback"}),
    "2theta map (step, x)": fit_split(norm.sum("y"), coords={"step": "readback"}),
    "tilt map (step, y)": fit_split(norm.sum("x"), coords={"step": "readback"}),
}
for name, fit in fits.items():
    print(f"{name:22s} delta_bic {fit['delta_bic']:7.1f}  split: {fit['split']}")
sep = fits["2theta map (step, x)"]["two"]["separation"]["x"]
print(f"separation in x: {sep.value:.2f} ± {np.sqrt(sep.variance):.2f} px")
# --8<-- [end:detector]
sep = fits["2theta map (step, x)"]["two"]["separation"]["x"]
text("split_detector", *(f"{n:22s} delta_bic {f['delta_bic']:7.1f}  split: {f['split']}" for n, f in fits.items()),
     f"separation in x: {sep.value:.2f} ± {np.sqrt(sep.variance):.2f} px")

# --8<-- [start:rsm]
from cristallina_scipp.rsm import histogram

# a rocking scan with h, k, l per pixel (from rsm.pixel_hkl) and local coordinates around (0 0 2)
# (rsm.local_coords); here two domains 0.6 % apart in d-spacing and tilted 0.3 deg against each other
q_stack = testing.split_rsm_stack(d_strain=0.006, d_tilt_1=0.3, ratio=0.7, seed=21)
edges = dict(strain=sc.linspace("strain", -0.012, 0.012, 25),
             tilt_1=sc.linspace("tilt_1", -1.0, 1.0, 25, unit="deg"),
             tilt_2=sc.linspace("tilt_2", -1.0, 1.0, 25, unit="deg"))
rsm = histogram(q_stack, **edges)
fit = fit_split(rsm, exposure="exposure", coupling=False)
print(f"delta_bic {fit['delta_bic']:.0f}, split: {fit['split']}")
for name, value in fit["two"]["separation"].items():
    print(f"separation in {name}: {value.value:.4f} ± {np.sqrt(value.variance):.4f} {value.unit}")
print(f"fraction in the first domain: {fit['two']['ratio'].value:.2f} ± {np.sqrt(fit['two']['ratio'].variance):.2f}")
# --8<-- [end:rsm]
two = fit["two"]
text("split_rsm", f"delta_bic {fit['delta_bic']:.0f}, split: {fit['split']}",
     *(f"separation in {n}: {v.value:.4f} ± {np.sqrt(v.variance):.4f} {v.unit}" for n, v in two["separation"].items()),
     f"fraction in the first domain: {two['ratio'].value:.2f} ± {np.sqrt(two['ratio'].variance):.2f}")

fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), layout="constrained")
image = norm.sum("y")
axs[0].pcolormesh(image.coords["x"].values, image.coords["readback"].values, image.values, shading="nearest")
axs[0].set(xlabel="detector column x (2θ)", ylabel="ω [deg]", title="(step, x): two spots")
curve = norm.sum(["y", "x"])
axs[1].errorbar(curve.coords["readback"].values, curve.values, yerr=np.sqrt(curve.variances), fmt=".", label="data")
one = fits["rocking curve (step)"]["one"]["best_fit"]
axs[1].plot(one.coords["readback"].values, one.values, label="one peak")
axs[1].set(xlabel="ω [deg]", ylabel="counts per I0 keV", title="rocking curve: looks like one peak")
axs[1].legend()
exposure = rsm.coords["exposure"].values
proj = np.nansum(np.nan_to_num(rsm.values) * exposure, axis=2) / exposure.sum(axis=2)  # counts / I0, over tilt_2
e_mid, t_mid = (sc.midpoints(rsm.coords[c]).values for c in ("strain", "tilt_1"))
axs[2].pcolormesh(t_mid, e_mid * 100, proj, shading="nearest")
for label, peak in two["peaks"].items():
    axs[2].plot(peak["tilt_1"].value, peak["strain"].value * 100, "w+", ms=12, mew=2)
axs[2].set(xlabel="tilt₁ [deg]", ylabel="strain [%]", title="reciprocal space: fitted domains (+)")
save(fig, "split_peaks")

# checks
assert not fits["rocking curve (step)"]["split"] and fits["2theta map (step, x)"]["split"]
assert not fits["tilt map (step, y)"]["split"]
assert abs(sep.value - 6.0) < 3 * np.sqrt(sep.variance)
assert fit["split"] and abs(two["separation"]["strain"].value - 0.006) < 3 * np.sqrt(two["separation"]["strain"].variance)
