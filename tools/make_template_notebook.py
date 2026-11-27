"""Regenerate notebooks/experiment_template.ipynb (outputs cleared: it is meant to be copied and run on real data)."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))  # noqa: E731

md(r"""
# Experiment analysis template

One notebook per analysis, all starting from the experiment's shared run catalogue:

1. **Overview**: every run in the raw folder, what it scanned, and which catalogue group it belongs to.
2. **Energy series**: integrated intensity per run across the edge, with propagated errors.
3. **Corrections**: absorption (from a fluorescence energy scan) and, if needed, transmission, each recorded.
4. **Field series**: peak position against field, for one or more field directions.
5. **Save**: one HDF5 file per series, with the runs, ROIs, cutoffs and corrections that produced it.

Copy this notebook to your own folder and edit only the configuration cell. If a step needs different code, change it
in cristallina-scipp (or open an issue) rather than in the copy, so everyone gets the fix.

Dry run without data: `CRISTALLINA_SYNTHETIC=1 jupyter lab experiment_template.ipynb` writes a small synthetic
experiment to a temporary folder and uses it instead.
""")

md("## Configuration")

code(r"""
import os
from pathlib import Path

PGROUP = "p2xxxx"
DATA_ROOT = Path(os.environ.get("CRISTALLINA_DATA", "/sf/cristallina/data"))
CATALOGUE = DATA_ROOT / PGROUP / "work" / "catalogue"           # shared checkout (cristallina-catalogue setup)
RESULTS = DATA_ROOT / PGROUP / "work" / "results" / os.environ.get("USER", "me")

# One set of reduction settings for every series that ends up on the same plot.
REDUCE = dict(
    cutoff=7.0,           # keV: pixel values below are not Bragg photons
    i0_cutoff=2.5,        # keV: the same for the JF I0 detector
    photon_energy=7.511,  # keV: converts to photon counts with Poisson errors
)

ENERGY_GROUP = "energy 0T"                     # catalogue group(s) of the energy series
ENERGY_COORD = "photon_energy"                 # which energy the fluorescence scan's readback follows
FIELD_GROUPS = ["field 0deg", "field 15deg"]   # groups combined in the field series
FLUORESCENCE = dict(
    run=45,               # energy scan across the edge, same sample and geometry if possible
    cutoff=5.5, upper_cutoff=6.8,  # keV: the fluorescence window
    i0_cutoff=2.5,
    photon_energy=6.4,    # keV: the emission line, for counts with errors
)

SYNTHETIC = os.environ.get("CRISTALLINA_SYNTHETIC") == "1"
""")

code(r"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipp as sc

import cristallina_scipp as cs
from cristallina_scipp import channels
from cristallina_scipp.corrections import correct_absorption, correct_transmission, fluorescence_yield

if SYNTHETIC:
    import tempfile

    from cristallina_scipp import testing

    work = Path(tempfile.mkdtemp())
    CATALOGUE = testing.example_experiment(work)
    RESULTS = work / "results"
    print("SYNTHETIC DATA in", work)

pd.set_option("display.max_rows", 500)
cat = cs.Catalogue.load(CATALOGUE)
print(f"{cat.path}\nraw data: {cat.raw}\ngroups: {', '.join(cat.groups)}")
""")

md(r"""
## 1. Run overview

One row per run folder. `group` is empty for runs no catalogue group lists: during the beamtime, that is the list of
runs still to describe. `status` flags runs still being written (`no scan.json`), runs a group lists that are not in
the raw folder, and runs whose files lack expected PVs.
""")

code(r"""
overview = cat.overview()
overview[["status", "scanned", "unit", "start", "stop", "steps", "photon_energy", "field", "transmission", "group"]]
""")

code(r"""
print("not in any group:", overview.index[overview["group"] == ""].tolist())
print("in two groups:   ", overview.index[overview["group"].str.contains(";")].tolist())
print("not ok:          ", overview.loc[overview["status"] != "ok", "status"].to_dict())
""")

md(r"""
## 2. Energy series

Each run is reduced to its ROI curve, normalised by I0 per step, and measured three ways: a background-subtracted
`integral` (no clipping), a one-peak fit (`center`, `fwhm`, `height`, `area`) and moments. Look at the run curves
before trusting the numbers.
""")

code(r"""
energy = cat.series(ENERGY_GROUP, **REDUCE)
summary = energy["summary"]

fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
for run, curve in energy["curves"].items():
    axs[0].plot(curve.coords["readback"].values, curve.values, ".-", lw=0.8, label=run)
axs[0].set(xlabel=f"readback [{curve.coords['readback'].unit}]", ylabel=f"counts per I0 {energy['settings']['i0_unit']}",
           title="run curves")
axs[0].legend(fontsize="small", ncols=2, title="run")
e = summary.coords[ENERGY_COORD]
for name, marker in (("integral", "o"), ("area", "v")):
    axs[1].errorbar(e.values, summary[name].values, yerr=np.sqrt(summary[name].variances), fmt=marker,
                    label={"integral": "integral", "area": "fit area"}[name])
axs[1].set(xlabel=f"{ENERGY_COORD} [{e.unit}]", ylabel="integrated intensity", title=ENERGY_GROUP)
axs[1].legend();
""")

md(r"""
## 3. Corrections

**Absorption.** For a thick crystal in reflection the intensity goes as |F|²/μ(E), so every reflection weakens above
the edge. `correct_absorption` multiplies by the fluorescence yield y(E)/y(E_ref), with the reference at the low end of
the fluorescence scan. Check the yield curve first: its noise goes into every corrected point.

**Transmission.** Normalising by the JF I0 detector already includes both attenuators; a gas-monitor or PBPS149
normalisation does not, and `correct_transmission` divides by the attenuators after that monitor.
""")

code(r"""
fluo_settings = {k: v for k, v in FLUORESCENCE.items() if k != "run"}
fluorescence = fluorescence_yield(cat.scan(FLUORESCENCE["run"]), **fluo_settings)
corrected = correct_absorption(energy, fluorescence, coord=ENERGY_COORD)
if energy["settings"]["i0"] != channels.JF_I0:
    corrected = correct_transmission(corrected)
print("corrections:", corrected["settings"]["corrections"])

fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
x = fluorescence.coords["readback"]
axs[0].errorbar(x.values, fluorescence.values, yerr=np.sqrt(fluorescence.variances), fmt=".", color="C2")
axs[0].set(xlabel=f"readback [{x.unit}]", ylabel="fluorescence per I0", title=f"fluorescence yield, run {FLUORESCENCE['run']}")
for series, label, marker in ((energy, "measured", "o"), (corrected, "corrected", "s")):
    s = series["summary"]["integral"]
    axs[1].errorbar(e.values, s.values, yerr=np.sqrt(s.variances), fmt=marker + "-", label=label)
axs[1].set(xlabel=f"{ENERGY_COORD} [{e.unit}]", ylabel="integrated intensity", title=ENERGY_GROUP)
axs[1].legend();
""")

md(r"""
## 4. Field series

Several groups combine into one series; each run keeps its group's labels (here `field_angle`), and a run listed in
two of the groups is an error.
""")

code(r"""
field = cat.series(FIELD_GROUPS, **REDUCE)
fs = field["summary"]

fig, ax = plt.subplots(figsize=(6, 3.8), layout="constrained")
for group in FIELD_GROUPS:
    one = fs[fs.coords["group"] == sc.scalar(group, unit=None)]
    ax.errorbar(one.coords["field"].values, one["center"].values, yerr=np.sqrt(one["center"].variances),
                fmt="o-", label=group)
ax.set(xlabel="field [T]", ylabel=f"fitted centre [{fs['center'].unit}]", title="peak position")
ax.legend();
""")

md(r"""
## 5. Save

One HDF5 file per series holds the summary, every run's curve, and the settings: cutoffs, I0, corrections, the package
version, and the catalogue with a SHA-256 of its contents. Share these files rather than CSVs; anyone can reload one
with `scipp.io.load_hdf5` and see how each number was obtained.
""")

code(r"""
RESULTS.mkdir(parents=True, exist_ok=True)
saved = {}
for name, series in ((ENERGY_GROUP + " corrected", corrected), (" + ".join(FIELD_GROUPS), field)):
    path = RESULTS / (name.replace(" ", "_").replace("+", "and") + ".h5")
    series.save_hdf5(path)
    saved[name] = path
    print("saved", path)

back = sc.io.load_hdf5(saved[ENERGY_GROUP + " corrected"])
for key in ("groups", "catalogue_commit", "catalogue_sha256", "corrections", "i0", "cutoff", "cristallina_scipp", "created"):
    value = back["settings"][key]
    print(f"{key}: {value.value} {value.unit}" if isinstance(value, sc.Variable) else f"{key}: {value}")
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, "notebooks/experiment_template.ipynb")
