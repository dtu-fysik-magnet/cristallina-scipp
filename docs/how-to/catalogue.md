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

## Keep it up to date

[`cat.overview()`][cristallina_scipp.catalogue.Catalogue.overview] lists every run in `raw` with what it scanned, its
energy, field and transmission, and the group(s) it belongs to. Runs with an empty `group` are the ones still to
describe; `status` flags runs still being written and runs a group lists that are not in `raw`. The
[experiment template](https://github.com/dtu-fysik-magnet/cristallina-scipp/blob/main/notebooks/experiment_template.ipynb)
starts with this table.

## Samples, for reciprocal space

To map runs to hkl, the catalogue also needs the sample: its lattice and how it was mounted. Each `[samples]` entry
holds the arguments of `cristallina.rsm.Sample`, and groups refer to it by their `sample` label:

```toml
[samples."S1"]
lattice = { a = 0.694, b = 0.694, c = 0.6068, alp = 90, bet = 90, gam = 90 }   # nm, degrees
surface_normal = [1, 0, 0]
alpha0 = 45.7      # mounting offsets, degrees
chi0 = -0.53
offset = 0.0       # added to the rocking readback to give alpha

[groups."energy 0T"]
runs = "467-486"
sample = "S1"
```

A remounted or realigned sample gets a new entry. The detector for each run comes from `[[geometry]]` and the run's
motor positions ([`cat.detector(run)`][cristallina_scipp.catalogue.Catalogue.detector]), and
[`cat.to_q(run, stack, reflection)`][cristallina_scipp.catalogue.Catalogue.to_q] adds h, k, l and the strain and
tilts around a reflection to a reduced run.

## Share it during the experiment

Several people edit the catalogue during a beamtime, often at once. A plain shared file loses edits (the last save
wins) and can be left broken by a typo. Keep it in git instead, with a check that refuses a broken version:

```sh
cristallina-catalogue setup /sf/cristallina/data/p2xxxx/work --from my_catalogue.toml
```

This creates two things in the work folder:

- `catalogue.git`, a repository the whole group can push to. A push whose catalogue does not load (invalid TOML,
  an undefined ROI, a run range in two `[[geometry]]` entries, a name defined twice) is refused, with the reason.
- `catalogue/`, a read-only checkout that is updated after every accepted push. Notebooks load this folder, and
  every saved series records its git commit as `catalogue_commit` (with `+modified` if it was changed by hand).

To change it, work in your own clone:

```sh
git clone /sf/cristallina/data/p2xxxx/work/catalogue.git ~/catalogue
cd ~/catalogue
# edit, then check before pushing
cristallina-catalogue check .
git commit -am "Add the 30 degree field series"
git pull --rebase && git push
```

If someone else pushed first, `git pull --rebase` merges their change with yours; edits to different groups merge
without conflicts. `check` also warns about runs listed in two groups and runs that are not in the raw folder (yet).

To find the catalogue a saved series was made with: `git show <catalogue_commit>:<file>` in any clone.

### Split it into files

A catalogue can be a folder of TOML files, read as one: for example `00-setup.toml` with `raw`, the ROIs and the
geometry, and one file per series or per person. People then rarely edit the same file. A name defined in two files
is an error, as is a `raw` that differs between files.
