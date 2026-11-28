"""sfdata -> scipp adapters.

sfdata (with jungfrau_utils) still does the file handling, pulse-id matching and
Jungfrau gain/pedestal/geometry corrections; these functions only wrap the
results as labelled scipp objects.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field, replace

import jungfrau_utils
import numpy as np
import scipp as sc

from . import channels


@dataclass
class ROI:
    """Rectangular region of interest, as `cristallina.utils.ROI`.

    Rows are `bottom:top` and columns `left:right`; the name is ignored when
    comparing ROIs. Any object with `.rows`/`.cols` slices works where a ROI
    is expected, so cristallina's own ROIs can be passed too.
    """

    left: int
    right: int
    bottom: int
    top: int
    name: str | None = field(default=None, compare=False)

    @property
    def rows(self):
        return slice(self.bottom, self.top)

    @property
    def cols(self):
        return slice(self.left, self.right)

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.top - self.bottom

    def shift(self, x=0, y=0) -> "ROI":
        """A copy moved `x` pixels right and `y` pixels up."""
        return replace(self, left=self.left + x, right=self.right + x, bottom=self.bottom + y, top=self.top + y)

    def expand(self, x=0, y=0) -> "ROI":
        """A copy grown by `x` pixels left and right and `y` pixels up and down."""
        return replace(self, left=self.left - x, right=self.right + x, bottom=self.bottom - y, top=self.top + y)

_DIMS = {1: ("pulse",), 2: ("pulse", "x"), 3: ("pulse", "y", "x")}


def channel(ch, unit=None, dims=None) -> sc.DataArray:
    """One sfdata channel as a DataArray with a `pulse_id` coordinate.

    Reads the whole channel; for Jungfrau images prefer [`reduce_scan`][cristallina_scipp.reduce_scan].

    Args:
        ch: An `sfdata` channel, e.g. `data["SARFE10-PBIG050-EVR0:CALCI"]`.
        unit: Unit of the values (sfdata files carry none).
        dims: Dimension labels; by default `pulse`, then `x` (spectra) or
            `y, x` (images).

    Returns:
        A DataArray with an int64 `pulse_id` coordinate (Jungfrau files store
            unsigned, beam-synchronous files signed pulse ids; casting lets scipp
            check that they line up).
    """
    data = np.asarray(ch.data)
    dims = dims or _DIMS[data.ndim]
    return sc.DataArray(
        sc.array(dims=dims, values=data, unit=unit),
        coords={"pulse_id": sc.array(dims=[dims[0]], values=np.asarray(ch.pids, dtype=np.int64), unit=None)},
    )


def step(data, names, units=None) -> sc.DataGroup:
    """Pulse-aligned channels of one acquisition/step as a DataGroup.

    Pulses missing from any of `names` are dropped, so all entries share the
    same `pulse_id`. Don't include PV channels here: they have one entry per
    change, and aligning on them drops nearly every pulse. Read them with
    [`channel`][cristallina_scipp.channel] instead.

    Args:
        data: An `SFDataFiles` or a scan step (`scan[i]`).
        names: Channel names.
        units: Optional `{name: unit}`.
    """
    units = units or {}
    sub = data[tuple(names)]
    sub.drop_missing(show_stats=False)
    return sc.DataGroup({n: channel(sub[n], units.get(n)) for n in names})


@contextmanager
def _open(scan, i):
    """`with scan[i]`, minus jungfrau_utils' two-line notice for every converted file it opens."""
    verbose, jungfrau_utils.verbose = jungfrau_utils.verbose, False
    try:
        data = scan[i]
    finally:
        jungfrau_utils.verbose = verbose
    with data:
        yield data


def _ndim(ch):
    # ponytail: sfdata's SFChannelJF.shape calls juf.handler, which is None for
    # converted files; jungfrau_utils' File.get_shape_out handles both cases.
    # Drop once sfdata uses juf.get_shape_out().
    juf = getattr(ch, "juf", None)
    return 1 + len(juf.get_shape_out()) if juf is not None else ch.ndim


def _batched_sum(ch, lower, upper, rows, cols, batch_size):
    """Sum over pulses, with per-pulse cutoffs, read in batches. Scalars for 1-D channels."""
    if _ndim(ch) == 1:
        return np.sum(ch.data, dtype=np.float64)
    total = 0.0
    for _, batch in ch.in_batches(batch_size):
        im = batch[:, rows, cols]
        if lower is not None:
            im = np.where(im < lower, 0, im)
        if upper is not None:
            im = np.where(im > upper, 0, im)
        total = total + im.sum(axis=0, dtype=np.float64)
    return total


def _unit(name):
    try:
        return sc.Unit(name) if name else None
    except Exception:  # scan.json units are free text
        return None


def reduce_scan(
    scan,
    detector,
    i0=None,
    roi=None,
    cutoff=None,
    upper_cutoff=None,
    i0_cutoff=None,
    photon_energy=None,
    readbacks=None,
    readback_unit=None,
    i0_unit=None,
    batch_size=100,
) -> sc.DataArray:
    """Pulse-summed detector image per scan step, dims `(step, y, x)`.

    Mirrors `cristallina.analysis.perform_image_stack_sum` (per-pulse cutoffs
    in keV, batched reading). Unlike `calculate_JF_stacks`, the detector and
    `i0` are summed over the *same* pulses: pulses missing from either are
    dropped from both.

    Args:
        scan: An `SFScanInfo`, or any sequence of `SFDataFiles` with
            `readbacks` given explicitly.
        detector: Image channel, e.g. `channels.JF`.
        i0: Optional normalisation channel. Either a beam-synchronous scalar,
            like the gas monitor, or an image detector, like
            `channels.JF_I0`, which is summed over all pixels
            after `i0_cutoff`.
        roi: A [`ROI`][cristallina_scipp.ROI] (anything with `.rows`/`.cols`).
        cutoff: Pixel values below this (keV) are set to 0 in each pulse.
        upper_cutoff: Pixel values above this (keV) are set to 0. Together with
            `cutoff` this is an energy window, e.g. 5.5-6.8 keV for
            fluorescence.
        i0_cutoff: Lower cutoff (keV) for an image `i0`.
        photon_energy: Energy (keV) of the *detected* photons. For Bragg peaks
            this is the incident energy; for a fluorescence window, the emission
            line. If given, the result is in photon counts with Poisson
            variances.
        readbacks: Override `scan.readbacks`.
        readback_unit: Override the unit from `scan.parameters["units"]`.
        i0_unit: Unit of `i0`. Defaults to keV for an image `i0` and
            dimensionless for a scalar channel.
        batch_size: Pulses per read.

    Returns:
        A DataArray with coords `readback` and `n_pulses` (step), `i0`
            (step, only if given), `y`/`x` pixel indices, and a scalar
            `scanned` holding the scanned adjustable's name. `da / da.coords["i0"]`
            is the normalised signal.
    """
    params = getattr(scan, "parameters", {}) or {}
    readbacks = np.asarray(scan.readbacks if readbacks is None else readbacks, dtype=float)
    if readbacks.ndim == 2:  # multi-adjustable scans: take the first adjustable
        readbacks = readbacks[:, 0]
    if readback_unit is None:
        readback_unit = _unit((params.get("units") or [None])[0])

    rows, cols = (roi.rows, roi.cols) if roi is not None else (slice(None), slice(None))
    names = (detector,) if i0 is None else (detector, i0)
    images, i0_sums, n_pulses = [], [], []
    for i in range(len(scan)):
        with _open(scan, i) as data:
            sub = data[names]
            sub.drop_missing(show_stats=False)
            images.append(_batched_sum(sub[detector], cutoff, upper_cutoff, rows, cols, batch_size))
            if i0 is not None:
                i0_sums.append(np.sum(_batched_sum(sub[i0], i0_cutoff, None, slice(None), slice(None), batch_size)))
                i0_image = _ndim(sub[i0]) > 1
            n_pulses.append(sub[detector].nvalid)

    values = np.stack(images)
    _, ny, nx = values.shape
    y0, x0 = rows.start or 0, cols.start or 0
    coords = {
        "readback": sc.array(dims=["step"], values=readbacks, unit=readback_unit),
        "n_pulses": sc.array(dims=["step"], values=n_pulses, unit=None),
        "y": sc.arange("y", y0, y0 + ny, unit=None),
        "x": sc.arange("x", x0, x0 + nx, unit=None),
        "scanned": sc.scalar((params.get("name") or [""])[0]),
    }
    if i0 is not None:
        if i0_unit is None:
            i0_unit = "keV" if i0_image else "dimensionless"
        coords["i0"] = sc.array(dims=["step"], values=i0_sums, unit=i0_unit)
    da = sc.DataArray(sc.array(dims=["step", "y", "x"], values=values, unit="keV"), coords=coords)
    return da if photon_energy is None else to_photons(da, photon_energy)


def to_photons(da, photon_energy) -> sc.DataArray:
    """Convert summed keV to photon counts with Poisson variances.

    Use this after [`crop`][cristallina_scipp.crop] when a full-detector stack is too large to carry
    variances for every pixel.

    Args:
        da: Data in keV, e.g. from [`reduce_scan`][cristallina_scipp.reduce_scan].
        photon_energy: Energy of the detected photons in keV.
    """
    out = da / photon_energy
    out.unit = "counts"
    # ponytail: Poisson on thresholded, gain-corrected counts is approximate;
    # I0 is treated as exact.
    out.variances = np.clip(out.values, 0, None)
    return out


def crop(da, roi) -> sc.DataArray:
    """The part of `da` inside `roi`, matched against the `y`/`x` pixel coords.

    ROIs follow [`ROI`][cristallina_scipp.ROI]: rows `bottom:top`, columns
    `left:right`, in detector pixel indices.
    """
    def label(v):
        return sc.scalar(v, unit=None, dtype=da.coords["y"].dtype)

    return da["y", label(roi.bottom):label(roi.top)]["x", label(roi.left):label(roi.right)]


def scan_means(scan, names, units=None, spread=False, readbacks=None) -> sc.DataGroup:
    """Per-step mean of beam-synchronous or PV channels.

    Replaces the `for step in scan: means.append(np.mean(...))` loops.

    Args:
        scan: An `SFScanInfo` (or sequence of steps with `readbacks`).
        names: Channel names; each may be BS (one value per pulse) or PV (one
            value per change).
        units: Optional `{name: unit}`.
        spread: If False (default) the variances are the squared standard error
            of the mean. If True they are the per-step variance of the channel
            itself, i.e. the beam stability that `mean ± std` plots show.
        readbacks: Override `scan.readbacks`.

    Returns:
        A DataGroup of `(step)` DataArrays with `readback` and `n` coords.
    """
    units = units or {}
    params = getattr(scan, "parameters", {}) or {}
    readbacks = np.asarray(scan.readbacks if readbacks is None else readbacks, dtype=float)
    if readbacks.ndim == 2:
        readbacks = readbacks[:, 0]
    stats = {n: ([], [], []) for n in names}
    for i in range(len(scan)):
        with _open(scan, i) as data:
            for n in names:
                v = np.asarray(data[n].data, dtype=float).ravel()
                mean, var, count = stats[n]
                mean.append(v.mean())
                var.append(v.var(ddof=1) if v.size > 1 else 0.0)
                count.append(v.size)
    rb = sc.array(dims=["step"], values=readbacks, unit=_unit((params.get("units") or [None])[0]))
    out = {}
    for n, (mean, var, count) in stats.items():
        var, count = np.asarray(var), np.asarray(count)
        out[n] = sc.DataArray(
            sc.array(dims=["step"], values=mean, variances=var if spread else var / count, unit=units.get(n)),
            coords={"readback": rb, "n": sc.array(dims=["step"], values=count, unit=None)},
        )
    return sc.DataGroup(out)


def roi_intensities(da, rois) -> sc.DataArray:
    """Sum `da` over each ROI, stacked along a new `roi` dimension.

    See [`crop`][cristallina_scipp.crop] for the ROI convention; works on full images or on an
    ROI-cropped [`reduce_scan`][cristallina_scipp.reduce_scan] result alike.
    """
    out = sc.concat([crop(da, r).sum(["y", "x"]) for r in rois], "roi")
    out.coords["roi"] = sc.array(dims=["roi"], values=[r.name for r in rois])
    return out


def pixel_energies(scan, step, detector=channels.JF, threshold=2.5, bins=None) -> sc.DataArray:
    """Histogram of the energies of the pixels hit in one or more steps, over every pulse.

    The fluorescence line, the elastic line (at the incident energy) and the
    noise peak show up as separate peaks, so this is how to choose the
    `cutoff` and `upper_cutoff` windows of
    [`reduce_scan`][cristallina_scipp.reduce_scan]: compare a step below an
    absorption edge with one above it. It reads whole steps, so pick a few.

    Args:
        scan: An `SFScanInfo`.
        step: Step index, or several (e.g. `range(5)`), histogrammed together.
        detector: Jungfrau channel (values in keV).
        threshold: Pixel values at or below this (keV) are not counted as hits.
        bins: Bin edges (a Variable in keV along `energy`) or a number of bins
            from `threshold` to 10 keV; 150 by default.

    Returns:
        Hits per energy bin, with an `energy` bin-edge coordinate.
    """
    values = []
    for i in [step] if isinstance(step, (int, np.integer)) else step:
        with _open(scan, i) as data:
            v = np.asarray(data[detector].data, dtype=float).ravel()
        values.append(v[v > threshold])
    hits = np.concatenate(values)
    if bins is None or isinstance(bins, int):
        bins = sc.linspace("energy", threshold, 10.0, (bins or 150) + 1, unit="keV")
    counts = sc.DataArray(sc.ones(dims=["hit"], shape=[hits.size], unit="counts"),
                          coords={"energy": sc.array(dims=["hit"], values=hits, unit="keV")})
    return counts.hist(energy=bins)
