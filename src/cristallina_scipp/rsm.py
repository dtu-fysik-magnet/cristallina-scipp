"""Reciprocal-space maps by histogramming instead of interpolation.

Pixel -> hkl uses `cristallina.rsm` (Sample/Detector/Experiment) unchanged,
so the geometry is the beamline's. What changes is the last step:
`cristallina.rsm.RSM_interpolate` fills an hkl grid with
`griddata(method='nearest')`, which copies values into voxels no pixel
visited and never normalises by how many pixels landed in a voxel. Here
counts and monitor are histogrammed separately and divided, so each voxel is
counts per unit I0 and unvisited voxels are NaN.

`pixel_hkl` needs `cristallina`: `pip install cristallina-scipp[rsm]`.
"""

from copy import copy

import numpy as np
import scipp as sc


def pixel_hkl(stack, sample, detector, energy, offset=0.0) -> sc.DataArray:
    """Add per-pixel `h`, `k`, `l` coords (dims `step, y, x`) to a stack.

    `stack` is from [`reduce_scan`][cristallina_scipp.reduce_scan]; its `readback`
    is taken as the sample angle `alpha` (plus `offset`), as in
    `cristallina.rsm.RSM`. `energy` in eV; `detector` already moved to
    its 2theta (`detector.move_angles(gamma=tth)`).
    """
    from cristallina import rsm as crsm  # optional dependency, the [rsm] extra

    stack = stack.transpose(["step", "y", "x"])
    y, x = stack.coords["y"].values, stack.coords["x"].values
    roi = [int(x[0]), int(x[-1]) + 1, int(y[0]), int(y[-1]) + 1]  # [xmin, xmax, ymin, ymax]
    sample = copy(sample)
    hkl = []
    for alpha in stack.coords["readback"].values:
        sample.rotate_sample(alpha=alpha + offset)
        hkl.append(crsm.Experiment(sample, detector, energy).ROI_hkl(roi))
    hkl = np.stack(hkl, axis=1)  # (3, step, y, x)
    out = stack.copy(deep=False)
    for name, values in zip("hkl", hkl):
        out.coords[name] = sc.array(dims=stack.dims, values=values)
    return out


def histogram(stack, **bins) -> sc.DataArray:
    """Normalised RSM: sum(counts) / sum(I0) per voxel.

    `bins` are passed to `DataArray.hist`, e.g. `h=50, k=50, l=80` or
    bin-edge Variables. Variances (if any) propagate from the counts.
    """
    coords = {c: stack.coords[c] for c in "hkl"}
    signal = sc.DataArray(stack.data, coords=coords).flatten(to="pixel")
    monitor = sc.DataArray(
        sc.broadcast(stack.coords["i0"], sizes=stack.sizes).copy(), coords=coords
    ).flatten(to="pixel")
    signal = signal.hist(**bins)
    edges = {c: signal.coords[c] for c in bins}
    return signal / monitor.hist(**edges).data
