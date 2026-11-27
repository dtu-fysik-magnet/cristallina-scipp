# cristallina-scipp

[![tests](https://github.com/dtu-fysik-magnet/cristallina-scipp/actions/workflows/tests.yml/badge.svg)](https://github.com/dtu-fysik-magnet/cristallina-scipp/actions/workflows/tests.yml)
[![docs](https://github.com/dtu-fysik-magnet/cristallina-scipp/actions/workflows/docs.yml/badge.svg)](https://dtu-fysik-magnet.github.io/cristallina-scipp/)

Analyse SwissFEL/Cristallina data with [scipp](https://scipp.github.io) and [plopp](https://scipp.github.io/plopp).
The beamline packages (`sfdata`, `jungfrau_utils`, `cristallina`) still read the files; this package takes over from
there, with photon-counting errors carried through every step.

- **Reduce scans**: per-step Jungfrau images and I0 summed over the same pulses, with energy cutoffs.
- **Measure peaks**: fits, moments and background-subtracted integrals.
- **Run series**: one point per run against energy, field or angle, defined in a shared run catalogue and saved to
  HDF5 with its provenance.
- **Correct**: transmission and absorption corrections, recorded in the result.
- **Map**: scattering angles and binned reciprocal-space maps.

Start an analysis by copying `notebooks/experiment_template.ipynb`: run overview, energy and field series,
corrections, and saved results, from one shared run catalogue.

Documentation: <https://dtu-fysik-magnet.github.io/cristallina-scipp/>

## Install

```sh
pip install "cristallina-scipp @ git+https://github.com/dtu-fysik-magnet/cristallina-scipp.git"
```

This fetches `sfdata` and `jungfrau_utils` from their public repositories. Add the `[rsm]` extra for
reciprocal-space maps (it downloads `cristallina`, about 200 MB). Python 3.11 or 3.12; see the
[install guide](https://dtu-fysik-magnet.github.io/cristallina-scipp/how-to/install/) for newer Pythons and the Ra cluster.

## Develop

```sh
uv sync --all-extras
uv run pytest
uv run python tools/build_docs.py    # or: ... serve
```

Every code block in the documentation comes from a script in `docs/scripts/` that the tests run.

## Licence

BSD 3-Clause; see [LICENSE](LICENSE).
