"""Corrections to a run series, each applied explicitly and recorded.

Every function takes a [`run_series`][cristallina_scipp.series.run_series]
result and returns a corrected copy. Only the intensity measures (`integral`,
`area`, `height`) change; positions and widths do not. The factors applied
become coordinates of the summary, and `settings["corrections"]` lists the
corrections in the order applied, so a saved series says what was done to it.
"""

import numpy as np
import scipp as sc

from . import channels
from .load import reduce_scan

INTENSITIES = ("integral", "area", "height")

# Attenuators between each I0 monitor and the sample. SwissFEL device names carry
# the position along the beamline in metres: the gas monitor SARFE10-PBIG050
# (50 m) is upstream of the front-end attenuator OATT053 (53 m); PBPS149 (149 m)
# is upstream of the experiment attenuator OATA150 (150 m); the JF I0 detector
# is in the experiment hutch, after both.
ATTENUATORS_AFTER_MONITOR = {
    "none": ("transmission_fe", "transmission_exp"),  # no I0 normalisation
    channels.GASMONITOR: ("transmission_fe", "transmission_exp"),
    channels.PBPS149_INTENSITY: ("transmission_exp",),
    channels.JF_I0: (),
}


def _record(series, name, **params):
    out = series.copy()
    out["summary"] = series["summary"].copy()
    out["settings"] = series["settings"].copy()
    done = out["settings"].get("corrections", "")
    out["settings"]["corrections"] = f"{done}; {name}" if done else name
    out["settings"].update({f"{name}_{k}": v for k, v in params.items()})
    return out


def correct_transmission(series, attenuators=None) -> sc.DataGroup:
    """Divide intensities by the transmission of the attenuators after the I0 monitor.

    Normalising by an I0 monitor removes the effect of any attenuator upstream
    of it, but not of those between it and the sample. Which those are follows
    from the monitor recorded in `settings["i0"]` (see
    `ATTENUATORS_AFTER_MONITOR`): both for the gas monitor or no normalisation,
    the experiment attenuator for PBPS149, none for the JF I0 detector.

    Args:
        series: A run series.
        attenuators: Override, as summary coordinate names, e.g.
            `("transmission_exp",)`; needed for monitors not in the table.

    Raises:
        ValueError: If no attenuator lies after the monitor. Correcting such a
            series again would divide by the transmission twice.
    """
    i0 = series["settings"]["i0"]
    if attenuators is None:
        if i0 not in ATTENUATORS_AFTER_MONITOR:
            raise ValueError(f"unknown position of I0 monitor {i0!r}; pass attenuators=")
        attenuators = ATTENUATORS_AFTER_MONITOR[i0]
    if not attenuators:
        raise ValueError(f"{i0} is downstream of both attenuators: normalising by it already corrects for transmission")
    summary = series["summary"]
    factor = summary.coords[attenuators[0]].copy()
    for name in attenuators[1:]:
        factor = factor * summary.coords[name]
    out = _record(series, "transmission", attenuators=", ".join(attenuators))
    for name in INTENSITIES:
        out["summary"][name] = summary[name] / factor
    out["summary"].coords["transmission_factor"] = factor
    return out


def fluorescence_yield(scan, detector=channels.JF, i0=channels.JF_I0, roi=None, **reduce_kwargs) -> sc.DataArray:
    """Fluorescence per unit I0 against energy, from an energy scan across the edge.

    Pass the fluorescence window as `cutoff`/`upper_cutoff` (keV), as in the
    [fluorescence tutorial](../tutorials/fluorescence-scan.md). The result is
    normalised like a run series (per unit of I0, in I0's own unit), with the
    scan's readback as `readback` and the scan file as a scalar `scan` coordinate.
    """
    stack = reduce_scan(scan, detector, i0, roi=roi, **reduce_kwargs)
    out = stack.sum(["y", "x"])
    if i0 is not None:
        out = out / (stack.coords["i0"] / sc.scalar(1.0, unit=stack.coords["i0"].unit))
    out.coords["scan"] = sc.scalar(str(getattr(scan, "fname", "")))
    return out


def _interpolate(curve, x):
    """Linear interpolation of a 1-D curve (values and variances) at `x`."""
    xs = curve.coords["readback"].values.astype(float)
    order = np.argsort(xs)
    xs, ys = xs[order], curve.values[order]
    i = np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2)
    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    values = (1 - t) * ys[i] + t * ys[i + 1]
    if curve.variances is None:
        return values, None
    vs = curve.variances[order]
    return values, (1 - t) ** 2 * vs[i] + t**2 * vs[i + 1]


def correct_absorption(series, fluorescence, *, coord, reference=None, exponent=1) -> sc.DataGroup:
    """Correct intensities for absorption in the sample, using fluorescence as a proxy for μ(E).

    For a thick crystal in reflection, the kinematic integrated intensity is
    proportional to |F|²/μ(E): above the edge the sample absorbs more and the
    measured intensity drops. The correction multiplies by μ(E)/μ(E_ref),
    taking the fluorescence yield `y(E)` as proportional to μ(E):

        corrected = measured × (y(E) / y(E_ref)) ** exponent

    with `exponent=1`. `exponent=-1` divides by the yield instead; use it only to
    compare with analyses that did that.

    Both steps are approximations. Fluorescence from a concentrated, thick
    sample saturates, so it follows μ(E) less than proportionally near the
    edge, and the geometric factor depends on the incidence and exit angles,
    which differ between reflections and from the fluorescence scan.

    Args:
        series: A run series.
        fluorescence: Yield against energy, from
            [`fluorescence_yield`][cristallina_scipp.corrections.fluorescence_yield].
        coord: The series' energy coordinate to match with the scan's readback,
            e.g. `"mono_energy"` or `"photon_energy"`; units are converted.
            Check which one the fluorescence scan's readback follows.
        reference: Energy (scalar with unit) where the factor is 1. Defaults to
            the lowest energy of the fluorescence scan, below the edge, so
            intensities there are unchanged. The yield there is taken as exact;
            its noise rescales the whole series but does not change its shape.
        exponent: `1` (absorption correction) or `-1` (division by the yield).

    Raises:
        ValueError: If a run's energy is outside the fluorescence scan.
    """
    if exponent not in (1, -1):
        raise ValueError("exponent must be 1 or -1")
    summary = series["summary"]
    xunit = fluorescence.coords["readback"].unit
    energies = summary.coords[coord].to(unit=xunit, dtype="float64").values
    xs = fluorescence.coords["readback"].values
    outside = summary.coords["run"].values[(energies < xs.min()) | (energies > xs.max())]
    if outside.size:
        raise ValueError(f"runs {outside.tolist()} are outside the fluorescence scan ({xs.min()}-{xs.max()} {xunit})")
    ref = xs.min() if reference is None else reference.to(unit=xunit).value
    if not xs.min() <= ref <= xs.max():
        raise ValueError(f"reference {ref} {xunit} is outside the fluorescence scan ({xs.min()}-{xs.max()} {xunit})")
    values, variances = _interpolate(fluorescence, energies)
    y_ref = _interpolate(fluorescence, np.array([ref]))[0][0]  # a normalisation constant: no variance
    ratio = sc.array(dims=["run"], values=values / y_ref, variances=None if variances is None else variances / y_ref**2)
    factor = ratio if exponent == 1 else sc.reciprocal(ratio)
    out = _record(series, "absorption", coord=coord, exponent=exponent,
                  reference=sc.scalar(float(ref), unit=xunit), scan=fluorescence.coords["scan"].value)
    for name in INTENSITIES:
        out["summary"][name] = summary[name] * factor
    out["summary"].coords["absorption_factor"] = sc.values(factor)
    out["fluorescence"] = fluorescence
    return out
