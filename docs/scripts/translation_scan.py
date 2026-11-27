"""How-to: find the sample with a fluorescence translation scan (run 51 in March 2026)."""

import tempfile
from pathlib import Path

import numpy as np
import plopp as pp
from _common import save, text

from sfdata import SFScanInfo

from cristallina_scipp import testing

scan = SFScanInfo(str(testing.translation_scan(Path(tempfile.mkdtemp()))))
# real data: scan = SFScanInfo("/sf/cristallina/data/p2xxxx/raw/run0051/meta/scan.json")

# --8<-- [start:reduce]
import cristallina_scipp as cs
from cristallina_scipp import channels

stack = cs.reduce_scan(scan, channels.JF, channels.JF_I0, cutoff=5.5, upper_cutoff=6.8, i0_cutoff=2.5, photon_energy=6.4)
profile = stack.sum(["y", "x"]) / stack.coords["i0"]
# --8<-- [end:reduce]

# --8<-- [start:edges]
def half_maximum_crossings(da, coord="readback"):
    """Coordinate values where ``da`` crosses half its maximum (linear interpolation)."""
    x, y = da.coords[coord].values, da.values
    half = 0.5 * (np.nanmax(y) + np.nanmin(y))
    i = np.flatnonzero(np.diff(np.sign(y - half)))
    return x[i] + (half - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])


edges = half_maximum_crossings(profile)
print(f"sample from {edges[0]:.3f} to {edges[-1]:.3f} mm, centre {edges.mean():.3f} mm")
# --8<-- [end:edges]
text("translation_edges", f"sample from {edges[0]:.3f} to {edges[-1]:.3f} mm, centre {edges.mean():.3f} mm")

fig = pp.plot(profile, coords=["readback"], title="5.5-6.8 keV / JF-I0 vs TRX")
for e in edges:
    fig.ax.axvline(e, color="C1", lw=1)
save(fig, "translation_scan")

# checks: synthetic sample spans -0.9 to -0.3 mm
assert len(edges) == 2 and abs(edges[0] + 0.9) < 0.03 and abs(edges[1] + 0.3) < 0.03, edges
