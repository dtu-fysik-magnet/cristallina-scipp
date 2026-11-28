"""How-to: check the beam monitors, the photon energy and the fluorescence window before trusting a scan.

Synthetic stand-in for a fluorescence energy scan across an edge (run 45 by default).
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipp as sc
from _common import save, text

from sfdata import SFScanInfo

import cristallina_scipp as cs
from cristallina_scipp import channels, testing

base = Path(tempfile.mkdtemp())
scan = SFScanInfo(str(testing.fluorescence_energy_scan(base, run=45)))
# real data: scan = SFScanInfo("/sf/cristallina/data/p2xxxx/raw/run0045/meta/scan.json")

# --8<-- [start:monitors]
GAS, PBPS = channels.GASMONITOR, channels.PBPS149_INTENSITY
beam = cs.scan_means(scan, [GAS, PBPS], spread=True)  # variances: the pulse-to-pulse spread in each step
for name, da in beam.items():
    mean, spread = da.values, np.sqrt(da.variances)
    print(f"{name}: mean {mean.mean():.1f}, step-to-step {mean.std() / mean.mean():.1%}, "
          f"pulse-to-pulse {np.mean(spread / mean):.1%}, pulses per step {da.coords['n'].values.min()}")
print(f"correlation of the step means: {np.corrcoef(beam[GAS].values, beam[PBPS].values)[0, 1]:.2f}")
# --8<-- [end:monitors]
lines = []
for name, da in beam.items():
    mean, spread = da.values, np.sqrt(da.variances)
    lines.append(f"{name}: mean {mean.mean():.1f}, step-to-step {mean.std() / mean.mean():.1%}, "
                 f"pulse-to-pulse {np.mean(spread / mean):.1%}, pulses per step {da.coords['n'].values.min()}")
lines.append(f"correlation of the step means: {np.corrcoef(beam[GAS].values, beam[PBPS].values)[0, 1]:.2f}")
text("beam_monitors", *lines)

# --8<-- [start:pulses]
stack = cs.reduce_scan(scan, channels.JF, channels.JF_I0, cutoff=5.5, upper_cutoff=6.8, i0_cutoff=2.5)
n = stack.coords["n_pulses"].values
print("pulses per step:", n.min(), "-", n.max(), "| steps with dropped pulses:", np.flatnonzero(n < n.max()).tolist())
# --8<-- [end:pulses]
text("beam_pulses", f"pulses per step: {n.min()} - {n.max()} | steps with dropped pulses: "
     f"{np.flatnonzero(n < n.max()).tolist()}")

# --8<-- [start:energy]
UNDULATOR, MONO = channels.FEL_PHOTON_ENERGY, channels.MONO_ENERGY
energies = cs.scan_means(scan, [UNDULATOR, MONO], units={UNDULATOR: "keV", MONO: "eV"})
readback = energies[MONO].coords["readback"]  # what the scan says it set
for name, e in (("undulator", energies[UNDULATOR].to(unit="eV")), ("monochromator", energies[MONO])):
    d = (e.data - readback).values
    print(f"{name:13s} - scan readback = {d.mean():+6.1f} ± {d.std():.1f} eV")
# --8<-- [end:energy]
lines = []
for name, e in (("undulator", energies[UNDULATOR].to(unit="eV")), ("monochromator", energies[MONO])):
    d = (e.data - readback).values
    lines.append(f"{name:13s} - scan readback = {d.mean():+6.1f} ± {d.std():.1f} eV")
text("beam_energy", *lines)

# --8<-- [start:window]
window = (5.5, 6.8)  # keV: the fluorescence line in, the elastic line (the incident energy) out
below = cs.pixel_energies(scan, range(5))  # first five steps, below the edge
above = cs.pixel_energies(scan, range(len(scan) - 5, len(scan)))  # last five, above it
for label, h in (("below the edge", below), ("above the edge", above)):
    inside = h["energy", sc.scalar(window[0], unit="keV"):sc.scalar(window[1], unit="keV")].sum().value
    print(f"{label}: {inside:.0f} of {h.sum().value:.0f} hits in the window")
# --8<-- [end:window]
lines = []
for label, h in (("below the edge", below), ("above the edge", above)):
    inside = h["energy", sc.scalar(window[0], unit="keV"):sc.scalar(window[1], unit="keV")].sum().value
    lines.append(f"{label}: {inside:.0f} of {h.sum().value:.0f} hits in the window")
text("beam_window", *lines)

fig, axs = plt.subplots(1, 3, figsize=(14, 3.5), layout="constrained")
for name, marker in ((GAS, "o"), (PBPS, "s")):
    da = beam[name]
    axs[0].errorbar(da.coords["readback"].values, da.values / da.values.mean(), yerr=np.sqrt(da.variances) / da.values.mean(),
                    fmt=marker, ms=3, label=name.split(":")[0])
axs[0].set(xlabel="scan readback [eV]", ylabel="relative to the mean", title="monitors per step (± pulse spread)")
axs[0].legend(fontsize="small")
for name, e in (("undulator", energies[UNDULATOR].to(unit="eV")), ("monochromator", energies[MONO])):
    axs[1].plot(readback.values, (e.data - readback).values, ".", label=name)
axs[1].set(xlabel="scan readback [eV]", ylabel="minus scan readback [eV]", title="photon energy")
axs[1].legend()
for h, label in ((below, "first 5 steps (below the edge)"), (above, "last 5 steps (above the edge)")):
    centres = sc.midpoints(h.coords["energy"]).values
    axs[2].step(centres, h.values, where="mid", label=label)
for e in window:
    axs[2].axvline(e, color="red", lw=1)
axs[2].set(yscale="log", xlabel="pixel energy [keV]", ylabel="hits", title="choosing the fluorescence window")
axs[2].legend(fontsize="small")
save(fig, "beam_checks")

# checks
assert np.flatnonzero(n < n.max()).tolist() == [10]
mono = (energies[MONO].data - readback).values.mean()
assert abs(mono + 9) < 1, mono  # the synthetic monochromator reads 9 eV low
in_below = below["energy", sc.scalar(5.5, unit="keV"):sc.scalar(6.8, unit="keV")].sum().value
in_above = above["energy", sc.scalar(5.5, unit="keV"):sc.scalar(6.8, unit="keV")].sum().value
assert in_above > 2 * in_below  # fluorescence switches on above the edge
