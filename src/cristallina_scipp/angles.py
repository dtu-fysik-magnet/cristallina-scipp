"""Pixel -> scattering angle coordinates for the Jungfrau 1.5M.

A port of `convert_angle` from the March 2026 beamtime scripts, with the
calibration constants turned into arguments. It treats the detector as flat and
perpendicular to the beam and separates the axes: `two_theta` depends only on
the column `x`, and `nu` only on the row `y`. That is fine close to the
detector centre, not for wide-angle maps (use `cristallina_scipp.rsm` for those).
"""

import numpy as np
import scipp as sc

# Calibration used at the March 2026 beamtime, from convert_angle.
MARCH_2026 = {"x0": 416.81, "y0": 775.27, "d0": 0.568, "z0": 0.2, "stt0": -0.09935076}


def detector_angles(da, *, distance, x0, y0, two_theta=0.0, pixel_size=75e-6) -> sc.DataArray:
    """Add `two_theta` (along `x`) and `nu` (along `y`) coordinates, in degrees.

    Args:
        da: Anything with pixel-index `x` and `y` coords, e.g. from
            [`reduce_scan`][cristallina_scipp.reduce_scan].
        distance: Sample-detector distance in metres.
        x0: Pixel column of the direct beam at `two_theta = 0`.
        y0: Pixel row of the direct beam.
        two_theta: Detector arm angle in degrees.
        pixel_size: Pixel pitch in metres.

    The sign convention, `atan(x/distance) - two_theta`, is copied from the
    beamtime script and has not been checked against a known reflection.
    """
    x = (da.coords["x"].values - x0) * pixel_size
    y = (da.coords["y"].values - y0) * pixel_size
    out = da.copy(deep=False)
    out.coords["two_theta"] = sc.array(dims=["x"], values=np.degrees(np.arctan2(x, distance)) - two_theta, unit="deg")
    out.coords["nu"] = sc.array(dims=["y"], values=np.degrees(np.arctan2(y, distance)), unit="deg")
    return out


def geometry_from_step(step, x0, y0, d0, z0, stt0, pixel_size=75e-6) -> dict:
    """Arguments for [`detector_angles`][cristallina_scipp.angles.detector_angles] from the motor positions in a step.

    Reads the detector translation (`SARES31-GPS:TD-PosAct`, mm), the vertical
    offset (`SARES30-MOBI2:MOT_Z.RBV`, mm) and the arm angle
    (`SARES31-GPS:ROT2THETA-PosAct`, deg), as `convert_angle` did. Use
    `geometry_from_step(scan[0], **MARCH_2026)` for that beamtime's
    calibration.
    """
    def first(name):
        return float(np.asarray(step[name].data).ravel()[0])

    vertical = first("SARES30-MOBI2:MOT_Z.RBV") / 1000
    return {
        "distance": d0 + (z0 - first("SARES31-GPS:TD-PosAct") / 1000),
        "x0": x0,
        "y0": y0 - vertical / pixel_size,
        "two_theta": first("SARES31-GPS:ROT2THETA-PosAct") - stt0,
        "pixel_size": pixel_size,
    }
