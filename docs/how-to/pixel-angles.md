# Use scattering angles instead of pixels

[`detector_angles`][cristallina_scipp.angles.detector_angles] adds `two_theta` (along x) and `nu` (along y) coordinates
in degrees. Plots and [`moments`][cristallina_scipp.peaks.moments] then use angles directly.
[`geometry_from_step`][cristallina_scipp.angles.geometry_from_step] reads the detector distance, vertical offset and
arm angle from the motor PVs in a step:

```python
--8<-- "docs/scripts/rocking_curve.py:angles"
```

```text
--8<-- "docs/assets/generated/rocking_geometry.txt"
```

![Profile in two-theta](../assets/generated/rocking_angles.svg)

!!! warning "Unverified convention"
    The formula, `atan(x/distance) - arm angle`, and the `MARCH_2026` calibration come from `convert_angle` in the
    March scripts. With the arm at about 36°, it puts the peak at about −38°. Check the sign against a reflection of
    known 2θ before quoting angles.

The conversion treats the detector as flat and its axes as separable. For full maps in (h, k, l), use
[`cristallina_scipp.rsm`](../tutorials/rsm.md).
