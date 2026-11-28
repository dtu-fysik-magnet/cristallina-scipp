# Install

## Anywhere else

`sfdata` and `jungfrau_utils` are not on PyPI (the `sfdata` there is an unrelated package), so `pyproject.toml`
names their public repositories, pinned to tested commits, and pip or uv fetch them from git.

!!! warning "Python 3.13 and later: get bitshuffle from conda-forge"
    `sfdata` needs `bitshuffle.h5`. bitshuffle publishes PyPI wheels only up to Python 3.12; on newer Pythons pip
    builds it from source, and without HDF5 headers it silently leaves out `bitshuffle.h5`
    (`ModuleNotFoundError: No module named 'bitshuffle.h5'` on `import sfdata`). In a conda environment,
    `conda install -c conda-forge bitshuffle` first.

Install a release (see [versions](#versions-and-releases)):

```sh
pip install "cristallina-scipp @ git+https://github.com/dtu-fysik-magnet/cristallina-scipp.git@v0.1.0"
pip install "cristallina-scipp[rsm] @ git+https://github.com/dtu-fysik-magnet/cristallina-scipp.git@v0.1.0"  # adds cristallina, for rsm.pixel_hkl
```

or a local checkout, `pip install /path/to/cristallina-scipp`.

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

## Versions and releases

Releases are tagged `vX.Y.Z` on GitHub, with the wheel and source tarball attached and the documentation published
for each. Before 1.0, a minor release (0.2 → 0.3) may change or remove things notebooks and saved files rely on; a
patch release (0.2.0 → 0.2.1) does not. [`CHANGELOG.md`](https://github.com/dtu-fysik-magnet/cristallina-scipp/blob/main/CHANGELOG.md)
says what changed and, for anything that breaks, what to do. Install one release for a whole beamtime, so everyone
on shift runs the same code.

The version comes from the git tag. A checkout between releases reports a version such as `0.2.1.dev3+gabc1234`
(three commits after 0.2.0, at commit `abc1234`), and saved series record it in `settings["cristallina_scipp"]`, so
every result names the code that made it.

To make a release, on an up-to-date `main`:

1. In `CHANGELOG.md`, rename "Unreleased" to the version and date, e.g. `## [0.2.0] - 2027-02-15`, and start a new
   empty "Unreleased" section above it; merge that as a pull request.
2. Create the release from the merged `main`, with that section as its notes:

    ```sh
    awk '/^## \[0.2.0\]/{take=1; next} take && /^## \[/{exit} take' CHANGELOG.md > notes.md
    gh release create v0.2.0 --target main --title v0.2.0 --notes-file notes.md
    ```

The release workflow attaches the wheel and source tarball, after checking they carry the tag's version; the docs
workflow publishes the documentation. To move the pinned `sfdata` or `jungfrau_utils` commits, change them in `pyproject.toml`, run `uv lock`, and note it in the changelog.
