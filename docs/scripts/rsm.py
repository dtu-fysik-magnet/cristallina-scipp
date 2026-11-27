"""Tutorial: reciprocal-space map by histogramming, compared with nearest-neighbour griddata.

Synthetic rocking scan through 1T-TaS2 (0 0 2) at 9 keV, using cristallina.rsm's geometry.
"""

import matplotlib.pyplot as plt
import numpy as np
import plopp as pp
import scipp as sc
from _common import save, text

# --8<-- [start:geometry]
import cristallina.rsm as crsm

from cristallina_scipp.rsm import histogram, pixel_hkl

sample = crsm.Sample({"a": 0.336, "b": 0.336, "c": 0.5853, "alp": 90, "bet": 90, "gam": 120}, "1T-TaS2")
detector = crsm.Detector(R_m=0.15, px_size_um=75, center_pixel=(515, 807), detector_size_px=(1030, 1614))
energy = 9000.0  # eV
k = crsm.E_to_k(energy)  # 1/nm
detector.move_angles(gamma=2 * np.degrees(np.arcsin(2 * np.pi * 2 / sample.sample_dict["c"] / (2 * k))))

alpha = np.linspace(10, 20, 41)
template = sc.DataArray(
    sc.zeros(dims=["step", "y", "x"], shape=[alpha.size, 120, 120], unit="counts"),
    coords={
        "readback": sc.array(dims=["step"], values=alpha, unit="deg"),
        "y": sc.arange("y", 747, 867, unit=None),
        "x": sc.arange("x", 455, 575, unit=None),
    },
)
geometry = pixel_hkl(template, sample, detector, energy)  # adds h, k, l coords (step, y, x)
# --8<-- [end:geometry]

# --8<-- [start:synthetic]
rng = np.random.default_rng(2026)
h, k_, l = (geometry.coords[c].values for c in "hkl")
sigma = {"h": 0.004, "k": 0.008, "l": 0.006}


def model(h, k, l):
    return 0.5 + 20 * np.exp(-0.5 * ((h / sigma["h"]) ** 2 + (k / sigma["k"]) ** 2 + ((l - 2) / sigma["l"]) ** 2))


i0 = rng.uniform(0.5, 1.5, alpha.size)
stack = geometry.copy()
stack.values = rng.poisson(model(h, k_, l) * i0[:, None, None]).astype(float)
stack.variances = stack.values.copy()
stack.coords["i0"] = sc.array(dims=["step"], values=i0)
# --8<-- [end:synthetic]

# --8<-- [start:maps]
gs = 0.004
h_grid, k_grid = np.arange(-0.06, 0.06 + gs, gs), np.arange(-0.1, 0.1 + gs, gs)
l_grid = np.arange(l.min(), l.max() + gs, gs)


def edges(dim, grid):
    return sc.array(dims=[dim], values=np.append(grid - gs / 2, grid[-1] + gs / 2))


binned = histogram(stack, h=edges("h", h_grid), k=edges("k", k_grid), l=edges("l", l_grid))

# the beamline method, for comparison (cristallina.rsm.RSM -> RSM_interpolate)
nearest = crsm.RSM_interpolate(
    h.ravel(), k_.ravel(), l.ravel(), (stack / stack.coords["i0"]).values.ravel(), (-0.06, 0.06), (-0.1, 0.1), gs
)
nearest = sc.DataArray(
    sc.array(dims=["l", "h", "k"], values=nearest, unit="counts"),
    coords={"h": sc.array(dims=["h"], values=h_grid), "k": sc.array(dims=["k"], values=k_grid),
            "l": sc.array(dims=["l"], values=l_grid)},
).transpose(["h", "k", "l"])
# --8<-- [end:maps]

h0, l0 = int(np.argmin(np.abs(h_grid))), int(np.argmin(np.abs(l_grid - 2)))
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, layout="constrained")
for ax, (name, m) in zip(axs, [("griddata nearest", nearest), ("histogram / monitor", binned)]):
    pp.plot(m["h", h0].transpose(["l", "k"]), ax=ax, vmin=0, vmax=10, title=name)
axs[1].set_ylabel("")
save(fig, "rsm_slices")

fig = pp.plot({"griddata nearest": nearest["h", h0]["l", l0], "histogram / monitor": binned["h", h0]["l", l0]},
              title="k cut at h=0, l=2")
kk = np.linspace(-0.1, 0.1, 400)
fig.ax.plot(kk, model(h_grid[h0], kk, l_grid[l0]), "k--", lw=1, label="model")
fig.ax.legend()
save(fig, "rsm_cut")

visited = np.isfinite(binned.values)
text("rsm_coverage", f"voxels: {binned.data.size}, visited: {visited.sum()} ({visited.mean():.0%}); "
     f"griddata filled {np.isfinite(nearest.values).sum()}")

# checks: binned peak voxel close to the model; unvisited voxels stay NaN
cut = binned["h", h0]["l", l0]
assert abs(k_grid[np.nanargmax(cut.values)]) <= gs and sc.nanmax(cut).value > 0.6 * model(h_grid[h0], 0, l_grid[l0])
assert 0.2 < visited.mean() < 0.95 and np.isfinite(nearest.values).all()
