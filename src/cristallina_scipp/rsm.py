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

    `bins` are passed to `DataArray.hist`, e.g. `h=50, k=50, l=80`, or
    `strain=60, tilt_1=40, tilt_2=40` after
    [`local_coords`][cristallina_scipp.rsm.local_coords], or bin-edge
    Variables. Variances (if any) propagate from the counts. The summed I0 of
    each voxel is kept as the `exposure` coordinate (0 where no pixel landed),
    and the exposure-weighted mean position of its pixels as `mean_<coord>`
    (NaN where empty). [`fit_split`][cristallina_scipp.split.fit_split] uses
    both: an exact Poisson model of the binned counts, evaluated where the
    pixels actually were, so the voxel size does not bias the fit.
    """
    coords = {c: stack.coords[c] for c in bins}
    signal = sc.DataArray(stack.data, coords=coords).flatten(to="pixel")
    monitor = sc.DataArray(
        sc.broadcast(stack.coords["i0"], sizes=stack.sizes).copy(), coords=coords
    ).flatten(to="pixel")
    signal = signal.hist(**bins)
    edges = {c: signal.coords[c] for c in bins}
    exposure = monitor.hist(**edges).data
    out = signal / exposure
    out.coords["exposure"] = exposure
    for c in bins:  # exposure-weighted mean position of the pixels in each voxel
        weighted = monitor.copy()
        weighted.data = weighted.data * weighted.coords[c].to(dtype="float64")
        out.coords[f"mean_{c}"] = weighted.hist(**edges).data / exposure
    return out


def _basis(lattice):
    """Reciprocal basis (columns a*, b*, c*, in 1/length) of `lattice` = {a, b, c, alp, bet, gam} (degrees)."""
    a, b, c = lattice["a"], lattice["b"], lattice["c"]
    al, be, ga = np.radians([lattice["alp"], lattice["bet"], lattice["gam"]])
    c2 = (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    real = np.column_stack([[a, 0, 0], [b * np.cos(ga), b * np.sin(ga), 0],
                            [c * np.cos(be), c * c2, c * np.sqrt(1 - np.cos(be) ** 2 - c2**2)]])
    return 2 * np.pi * np.linalg.inv(real).T


def local_coords(stack, hkl, lattice, toward=(0, 0, 1)) -> sc.DataArray:
    """Add `strain`, `tilt_1` and `tilt_2` coordinates around the reflection `hkl`.

    With G the reciprocal vector of `hkl` and Q that of each pixel (from its
    `h`, `k`, `l` coordinates, e.g. [`pixel_hkl`][cristallina_scipp.rsm.pixel_hkl]):

    - `strain` = −(Q − G)·Ĝ / |G|, the relative change of d-spacing to first
      order: positive for a longer d. Twin domains of different d-spacing
      separate along it.
    - `tilt_1`, `tilt_2` (deg) = (Q − G)·ê / |G|, rotations of the lattice
      planes about two axes perpendicular to G: `tilt_1` toward the reflection
      `toward`, `tilt_2` about the axis Ĝ × ê₁. Twin domains tilted against
      each other separate along these.

    These do not depend on the crystal's orientation, so they mean the same
    for every run and reflection.

    Args:
        stack: Data with `h`, `k`, `l` coordinates.
        hkl: The reflection, e.g. `(2, 0, 0)`.
        lattice: `{"a", "b", "c", "alp", "bet", "gam"}` (lengths in any unit,
            angles in degrees), e.g. a `cristallina.rsm.Sample`'s `sample_dict`.
        toward: A reflection not parallel to `hkl` that fixes the direction of
            `tilt_1`.
    """
    basis = _basis(lattice)
    g = basis @ np.asarray(hkl, dtype=float)
    ghat = g / np.linalg.norm(g)
    e1 = basis @ np.asarray(toward, dtype=float)
    e1 = e1 - (e1 @ ghat) * ghat
    if np.linalg.norm(e1) < 1e-9 * np.linalg.norm(g):
        raise ValueError(f"toward={toward} is parallel to hkl={hkl}")
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ghat, e1)
    hkl_values = np.stack([stack.coords[c].values for c in "hkl"], axis=-1)
    dq = hkl_values @ basis.T - g
    out = stack.copy(deep=False)
    dims = stack.coords["h"].dims
    gnorm = np.linalg.norm(g)
    out.coords["strain"] = sc.array(dims=dims, values=-(dq @ ghat) / gnorm)
    for name, e in (("tilt_1", e1), ("tilt_2", e2)):
        out.coords[name] = sc.array(dims=dims, values=np.degrees((dq @ e) / gnorm), unit="deg")
    return out
