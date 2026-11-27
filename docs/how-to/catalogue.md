# Describe the experiment in a catalogue

Without one, every notebook retypes its own run lists (`np.arange(598, 605)`), ROIs and geometry. A run
catalogue is one TOML file per experiment, next to the data, that everyone loads instead. Results built from it can
be compared because they start from the same definitions.

## What goes in it

Only what the data files do not record:

- **`raw`**: the folder with the `runNNNN` directories, absolute or relative to the catalogue file.
- **`[rois]`**: named ROIs, `{ left, right, bottom, top }` in pixels (rows `bottom:top`, columns `left:right`, as
  [`ROI`][cristallina_scipp.ROI]).
- **`[[geometry]]`**: detector calibration by run range, with the keywords of
  [`geometry_from_step`][cristallina_scipp.angles.geometry_from_step]. Add an entry whenever the detector is
  recalibrated.
- **`[groups."name"]`**: runs that belong together, with `runs`, an optional `roi`, and any labels you want as
  coordinates later (`sample`, `reflection`, `field_angle`, `note`, ...).

Energy, field and transmission are read from each run's files by
[`run_metadata`][cristallina_scipp.series.run_metadata], so they are not typed in.

```toml
--8<-- "docs/scripts/example_catalogue.toml"
```

## Run ranges

`runs` takes inclusive ranges: `"467-486, 525-533"`. Python's `range(501, 507)` stops at 506, so neighbouring
`range` calls that share an end point put that run in two series. Write the first and last run of the group.

## A peak that moves

When the ROI depends on the run, map run ranges to ROI names:

```toml
roi = { "781-786" = "peak 0T", "787-789" = "peak 0.6T", "791-796" = "peak 1T" }
```

[`Catalogue.load`][cristallina_scipp.catalogue.Catalogue.load] checks that every run of every group has a defined
ROI, so a typo fails when the file is loaded, not halfway through a series.

## Use it

```python
import cristallina_scipp as cs

cat = cs.Catalogue.load("/sf/cristallina/data/p2xxxx/work/catalogue.toml")
cat.runs("energy 0T")        # [467, ..., 486, 525, ..., 533]
cat.roi("energy 0T", 470)    # the ROI for that run
cat.geometry(470)                # calibration for angles.geometry_from_step
scan = cat.scan(470)             # SFScanInfo
series = cat.series("energy 0T", cutoff=7.0, i0_cutoff=2.5, photon_energy=7.511)
```
