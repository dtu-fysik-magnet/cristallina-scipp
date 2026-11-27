# Reciprocal-space map

`cristallina.rsm` converts every pixel of every step to (h, k, l). The beamline code then fills a grid with
`scipy.interpolate.griddata(method="nearest")`. Here the conversion is reused unchanged, and the grid is filled by
histogramming counts and monitor separately. The [explanation](../explanation/rsm-binning.md) covers why.

## Geometry

1T-TaS₂ at 9 keV, the detector at 2θ for (0 0 2), and the sample rocked through 10°:

```python
--8<-- "docs/scripts/rsm.py:geometry"
```

## A synthetic rocking scan

A Gaussian Bragg peak on a flat background, with I0 varying from step to step, and Poisson counts:

```python
--8<-- "docs/scripts/rsm.py:synthetic"
```

## Both maps on the same grid

```python
--8<-- "docs/scripts/rsm.py:maps"
```

```text
--8<-- "docs/assets/generated/rsm_coverage.txt"
```

![k-l slices](../assets/generated/rsm_slices.svg)

Outside the parallelogram the scan swept, the nearest-neighbour map is filled with copies of distant pixels. The
histogram leaves those voxels empty (NaN). Inside, each voxel is the average of every pixel that landed in it, so the
background is smoother and every value has an error bar:

![k cut](../assets/generated/rsm_cut.svg)

The executed notebook `notebooks/rsm.ipynb` contains the same comparison, plus the bundled `cristallina` test run.
