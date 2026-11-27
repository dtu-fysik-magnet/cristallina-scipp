"""Tutorial: energy and field series from a run catalogue.

Synthetic runs: rocking curves of one reflection across an absorption edge,
and of another at four fields for two field directions. The
catalogue is docs/scripts/example_catalogue.toml.
"""

import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipp as sc
from _common import save, text

from cristallina_scipp import testing

base = Path(tempfile.mkdtemp())
for i, run in enumerate(range(60, 69)):
    e = 7495.0 + 4 * i
    testing.rocking_scan(base / "raw", run=run, seed=run, field=0.0, energy=e, scale=testing.resonance(e))
for run, field in zip(range(70, 74), [0.0, 1.0, 2.0, 3.0]):
    testing.rocking_scan(base / "raw", run=run, seed=run, field=field, shift_per_tesla=0.02)
for run, field in zip(range(74, 78), [0.0, 1.0, 2.0, 3.0]):
    testing.rocking_scan(base / "raw", run=run, seed=run, field=field, shift_per_tesla=0.012)
shutil.copy(Path(__file__).with_name("example_catalogue.toml"), base / "experiment.toml")

# --8<-- [start:load]
import cristallina_scipp as cs

cat = cs.Catalogue.load(base / "experiment.toml")  # real data: the experiment's catalogue file
print(cat.runs("energy 0T"))
print(cat.labels("field 15deg"))
# --8<-- [end:load]
text("series_load", cat.runs("energy 0T"), cat.labels("field 15deg"))

# --8<-- [start:energy]
reduce = dict(cutoff=7.0, i0_cutoff=2.5, photon_energy=7.511)  # Bragg cutoff, I0 cutoff (keV), photons
energy = cat.series("energy 0T", **reduce)
summary = energy["summary"]
keep = ["run", "mono_energy", "photon_energy", "field", "transmission"]
print(sc.Dataset({k: summary[k] for k in ("integral", "area", "center", "fwhm")})
      .drop_coords([c for c in summary.coords if c not in keep]))
# --8<-- [end:energy]
text("series_energy", sc.Dataset({k: summary[k] for k in ("integral", "area", "center", "fwhm")})
     .drop_coords([c for c in summary.coords if c not in keep]))

# --8<-- [start:energy-plot]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
for run, curve in energy["curves"].items():
    e = summary["run", list(energy["curves"]).index(run)].coords["photon_energy"].to(unit="eV").value
    axs[0].plot(curve.coords["readback"].values, curve.values, ".-", lw=0.8, label=f"{e:.0f} eV")
axs[0].set(xlabel="θ [deg]", ylabel="counts per I0 keV", title="run curves")
axs[0].legend(fontsize="small", ncols=2)
e = summary.coords["photon_energy"].values
for name, marker in (("integral", "o"), ("area", "v")):
    axs[1].errorbar(e, summary[name].values, yerr=np.sqrt(summary[name].variances), fmt=marker,
                    label={"integral": "integral", "area": "fit area"}[name])
axs[1].set(xlabel="photon energy [keV]", ylabel="integrated intensity [counts·deg]", title="reflection A, 0 T")
axs[1].legend()
# --8<-- [end:energy-plot]
save(fig, "series_energy")

# --8<-- [start:field]
field = cat.series(["field 0deg", "field 15deg"], **reduce)["summary"]
for angle in (0.0, 15.0):
    one = field[field.coords["field_angle"] == sc.scalar(angle, unit=None)]
    slope = np.polyfit(one.coords["field"].values, one["center"].values, 1)[0]
    print(f"field angle {angle:4.1f} deg: peak moves {slope:.4f} deg/T")
# --8<-- [end:field]
slopes = {}
for angle in (0.0, 15.0):
    one = field[field.coords["field_angle"] == sc.scalar(angle, unit=None)]
    slopes[angle] = np.polyfit(one.coords["field"].values, one["center"].values, 1)[0]
text("series_field", *(f"field angle {a:4.1f} deg: peak moves {s:.4f} deg/T" for a, s in slopes.items()))

fig, ax = plt.subplots(figsize=(5, 3.4), layout="constrained")
for angle, marker in ((0.0, "o"), (15.0, "s")):
    one = field[field.coords["field_angle"] == sc.scalar(angle, unit=None)]
    ax.errorbar(one.coords["field"].values, one["center"].values, yerr=np.sqrt(one["center"].variances),
                fmt=marker + "-", label=f"field angle {angle:g}°")
ax.set(xlabel="field [T]", ylabel="fitted centre [deg]", title="reflection B peak position")
ax.legend()
save(fig, "series_field")

# --8<-- [start:methods]
ratio = summary["integral"] / summary["area"]
print(ratio.values.round(3))
# --8<-- [end:methods]
text("series_methods", ratio.values.round(3))

# --8<-- [start:save]
energy.save_hdf5(base / "energy_0T.h5")

saved = sc.io.load_hdf5(base / "energy_0T.h5")
for key, value in saved["settings"].items():
    print(f"{key}: {value.value} {value.unit}" if isinstance(value, sc.Variable) else f"{key}: {value}")
first = saved["summary"]["run", 0]
print(first.coords["run"].value, first.coords["roi"].value, first.coords["scan"].value)
# --8<-- [end:save]
lines = [f"{k}: {v.value} {v.unit}" if isinstance(v, sc.Variable) else f"{k}: {v}" for k, v in saved["settings"].items()]
lines = [line if not line.startswith(("catalogue:", "created:")) else line.split(":")[0] + ": ..." for line in lines]
text("series_save", *lines, f"{first.coords['run'].value} {first.coords['roi'].value} .../raw/run0060/meta/scan.json")

# checks
assert sc.identical(saved["summary"], summary)
best = summary.coords["mono_energy"].values[summary["integral"].values.argmax()] + 9  # mono PV offset -9 eV
assert abs(best - testing.EDGE_EV) <= 4, best
assert abs(slopes[0.0] - 0.02) < 0.006 and abs(slopes[15.0] - 0.012) < 0.006, slopes
assert np.all((ratio.values > 0.75) & (ratio.values < 1.0)), ratio.values  # the fit adds the tails outside the scan
