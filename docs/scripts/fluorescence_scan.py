"""Tutorial: a fluorescence energy scan, like run 45 of the March 2026 beamtime.

Runs on a synthetic scan written in the SwissFEL layout; on real data only the
lines marked "real data" change.
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plopp as pp
import scipp as sc
from _common import save, text

# --8<-- [start:open]
from sfdata import SFScanInfo

from cristallina_scipp import testing

scan = SFScanInfo(str(testing.fluorescence_energy_scan(Path(tempfile.mkdtemp()))))
# real data: scan = SFScanInfo("/sf/cristallina/data/p2xxxx/raw/run0045/meta/scan.json")
# --8<-- [end:open]
text("fluorescence_scan_info", scan, scan.parameters)

# --8<-- [start:beam]
import cristallina_scipp as cs

from cristallina_scipp import channels

GASMONITOR, PBPS149 = channels.GASMONITOR, channels.PBPS149_INTENSITY
UNDULATOR = "SARUN:FELPHOTENE.VAL"  # keV
MONO = "SAROP31-ODCC110:MOT_ENY.RBV"  # eV

beam = cs.scan_means(scan, [GASMONITOR, PBPS149], units={GASMONITOR: "uJ", PBPS149: "uJ"}, spread=True)
energies = cs.scan_means(scan, [UNDULATOR, MONO], units={UNDULATOR: "keV", MONO: "eV"})

readback = energies[MONO].coords["readback"]
undulator_offset = energies[UNDULATOR].to(unit="eV") - readback
mono_offset = energies[MONO] - readback
# --8<-- [end:beam]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
pp.plot(beam[GASMONITOR], coords=["readback"], ax=axs[0], title="gas monitor: mean ± spread per step")
pp.plot(beam[PBPS149], coords=["readback"], ax=axs[1], title="PBPS149: mean ± spread per step")
save(fig, "fluorescence_beam")
text(
    "fluorescence_offsets",
    f"undulator offset = {np.mean(undulator_offset.values):6.1f} ± {np.std(undulator_offset.values):.1f} eV",
    f"mono offset      = {np.mean(mono_offset.values):6.1f} ± {np.std(mono_offset.values):.1f} eV",
)

# --8<-- [start:reduce]
stack = cs.reduce_scan(
    scan,
    channels.JF,
    channels.JF_I0,
    cutoff=5.5,  # keV: fluorescence window ...
    upper_cutoff=6.8,  # ... excludes elastic scattering at the incident energy
    i0_cutoff=2.5,
    photon_energy=6.4,  # emission line, so the result is in photon counts
)
raw = stack.sum(["y", "x"])
normalised = raw / stack.coords["i0"]
# --8<-- [end:reduce]
text("fluorescence_stack", stack)

# --8<-- [start:plot]
fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True, layout="constrained")
pp.plot(raw, coords=["readback"], ax=axs[0], title="5.5-6.8 keV, raw")
pp.plot(normalised, coords=["readback"], ax=axs[1], title="5.5-6.8 keV / JF-I0")
# --8<-- [end:plot]
save(fig, "fluorescence_scan")

# --8<-- [start:alignment]
dropped = stack.coords["n_pulses"].values < stack.coords["n_pulses"].values.max()
print("steps with dropped pulses:", np.flatnonzero(dropped), stack.coords["n_pulses"].values[dropped])
# --8<-- [end:alignment]
text("fluorescence_alignment", f"steps with dropped pulses: {np.flatnonzero(dropped)} {stack.coords['n_pulses'].values[dropped]}")

# checks (run by the test suite)
edge = readback.values[np.argmax(np.gradient(normalised.values))]
assert abs(edge - testing.EDGE_EV) < 6, edge
assert list(np.flatnonzero(dropped)) == [10]
assert abs(np.mean(mono_offset.values) + 9) < 1
assert stack.coords["i0"].unit == sc.Unit("keV") and stack.unit == sc.Unit("counts")
