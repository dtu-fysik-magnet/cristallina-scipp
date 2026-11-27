"""How-to: correct a run series for transmission and for absorption.

Synthetic runs: a gas-monitor-normalised peak at three attenuations, and an
energy series whose measured intensity is suppressed by absorption above the
edge, plus the fluorescence energy scan (run 45) that measures that edge.
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipp as sc
from _common import save, text

from sfdata import SFScanInfo

from cristallina_scipp import testing

base = Path(tempfile.mkdtemp())
attenuations = {80: (0.5, 0.1), 81: (0.25, 0.1), 82: (0.5, 0.04)}
for run, att in attenuations.items():
    testing.rocking_scan(base, run=run, seed=run, attenuators=att)
for i, run in enumerate(range(90, 99)):
    e = 7495.0 + 4 * i
    testing.rocking_scan(base, run=run, seed=run, energy=e, scale=testing.resonance(e) / testing.edge(e))
testing.fluorescence_energy_scan(base, run=45)


def open_scan(run):
    return SFScanInfo(str(base / f"run{run:04}" / "meta" / "scan.json"))
    # real data: cat.scan(run), or SFScanInfo(".../raw/run0045/meta/scan.json")


# --8<-- [start:transmission]
from cristallina_scipp import channels, run_series
from cristallina_scipp.corrections import correct_transmission

reduce = dict(cutoff=7.0, photon_energy=7.511)
by_gas = run_series({r: open_scan(r) for r in (80, 81, 82)}, channels.JF, channels.GASMONITOR, **reduce)
fixed = correct_transmission(by_gas)
print(by_gas["summary"]["integral"].values.round(5))
print(fixed["summary"]["integral"].values.round(5))
print(fixed["settings"]["corrections"], "|", fixed["settings"]["transmission_attenuators"])
# --8<-- [end:transmission]
text("corr_transmission", by_gas["summary"]["integral"].values.round(5), fixed["summary"]["integral"].values.round(5),
     f'{fixed["settings"]["corrections"]} | {fixed["settings"]["transmission_attenuators"]}')

# --8<-- [start:jf-i0]
by_jf_i0 = run_series({r: open_scan(r) for r in (80, 81, 82)}, channels.JF, channels.JF_I0, i0_cutoff=2.5, **reduce)
print(by_jf_i0["summary"]["integral"].values.round(5))
try:
    correct_transmission(by_jf_i0)
except ValueError as err:
    print(err)
# --8<-- [end:jf-i0]
try:
    correct_transmission(by_jf_i0)
except ValueError as err:
    message = str(err)
text("corr_jf_i0", by_jf_i0["summary"]["integral"].values.round(5), message)

# --8<-- [start:absorption]
from cristallina_scipp.corrections import correct_absorption, fluorescence_yield

energy = run_series({r: open_scan(r) for r in range(90, 99)}, channels.JF, channels.JF_I0, i0_cutoff=2.5, **reduce)
fluorescence = fluorescence_yield(open_scan(45), cutoff=5.5, upper_cutoff=6.8, i0_cutoff=2.5, photon_energy=6.4)
corrected = correct_absorption(energy, fluorescence, coord="photon_energy")
print(corrected["summary"].coords["absorption_factor"].values.round(2))
# --8<-- [end:absorption]
text("corr_absorption", corrected["summary"].coords["absorption_factor"].values.round(2))

fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
x = fluorescence.coords["readback"].values / 1000
axs[0].errorbar(x, fluorescence.values, yerr=np.sqrt(fluorescence.variances), fmt=".", color="C2")
axs[0].set(xlabel="photon energy [keV]", ylabel="fluorescence per I0 keV", title="fluorescence yield, run 45")
e = energy["summary"].coords["photon_energy"].values
for series, label, marker in ((energy, "measured", "o"), (corrected, "absorption corrected", "s")):
    s = series["summary"]["integral"]
    axs[1].errorbar(e, s.values, yerr=np.sqrt(s.variances), fmt=marker + "-", label=label)
axs[1].plot(e, testing.resonance(e * 1000) * corrected["summary"]["integral"].values[0] / testing.resonance(e[0] * 1000),
            "k:", label="true shape")
axs[1].set(xlabel="photon energy [keV]", ylabel="integrated intensity [counts·deg]", title="energy series")
axs[1].legend()
save(fig, "corr_absorption")

# checks
assert np.std(fixed["summary"]["integral"].values) / np.mean(fixed["summary"]["integral"].values) < 0.05
assert "already corrects" in message
truth = testing.resonance(e * 1000)
ratio = corrected["summary"]["integral"].values / truth
assert np.std(ratio) / np.mean(ratio) < 0.25  # the run-45 yield is noisy (about 65 photons per point below the edge)
measured = energy["summary"]["integral"].values / truth
assert np.std(measured) / np.mean(measured) > 0.5
