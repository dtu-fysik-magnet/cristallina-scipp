# Install

## Anywhere else

`sfdata` and `jungfrau_utils` are not on PyPI (the `sfdata` there is an unrelated package), so `pyproject.toml`
names their public repositories and pip or uv fetch them from git.

!!! warning "Python 3.13 and later: get bitshuffle from conda-forge"
    `sfdata` needs `bitshuffle.h5`. bitshuffle publishes PyPI wheels only up to Python 3.12; on newer Pythons pip
    builds it from source, and without HDF5 headers it silently leaves out `bitshuffle.h5`
    (`ModuleNotFoundError: No module named 'bitshuffle.h5'` on `import sfdata`). In a conda environment,
    `conda install -c conda-forge bitshuffle` first.

```sh
pip install /path/to/cristallina-scipp            # or git+<this repository's URL>
pip install "/path/to/cristallina-scipp[rsm]"     # adds cristallina, for cristallina_scipp.rsm.pixel_hkl
```

`cristallina` is optional because its repository carries about 200 MB of test data, which pip and uv download in full
(uv shows no progress while it does). Only the RSM needs it: [`ROI`][cristallina_scipp.ROI] and the channel names
(`cristallina_scipp.channels`) are copied into this package.

For development, `uv.lock` pins the upstream commits:

```sh
uv sync --extra notebook --extra test     # no cristallina; its two tests are skipped
uv run pytest
uv sync --all-extras                      # with cristallina, needed to build the docs (the RSM tutorial)
```

To work on one of the upstream packages at the same time, install your checkout over the pinned one, e.g.
`uv pip install -e ../sf_datafiles`.

## On the Ra cluster or the beamline consoles

The pgroup kernel already has `cristallina`, `sfdata` and `jungfrau_utils`. Add this package without letting pip
replace them with the git versions:

```sh
pip install --no-deps -e /path/to/cristallina-scipp
pip install scipp plopp lmfit
```

Check from the notebook kernel:

```python
import cristallina_scipp, scipp, plopp
print(cristallina_scipp.__file__, scipp.__version__)
```

!!! warning "Importing cristallina patches jungfrau_utils"
    `import cristallina` replaces `jungfrau_utils.file_adapter.locate_gain_file` and `locate_pedestal_file`. If you
    mock those, for example to read data offline, do it after importing cristallina.
