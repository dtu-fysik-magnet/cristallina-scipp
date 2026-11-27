# Normalisation and errors

## Same pulses, top and bottom

A SwissFEL step holds many pulses, and each detector can lose frames independently. Normalising a sum over the 1.5M
by a sum over JF-I0 is only right if both sums cover the same pulses.

- `perform_image_calculations` and `perform_image_stack_sum` drop missing pulses only among the channels in their
  call. The beamline notebooks call them once for the 1.5M and once for JF-I0.
- `calculate_JF_stacks` matches the pulses of both detectors, but only to log a warning. The two sums are still made
  separately.
- [`reduce_scan`][cristallina_scipp.reduce_scan] selects both channels together, calls `drop_missing()` once, then
  sums both over the surviving pulses. The `n_pulses` coordinate records how many survived in each step.

## Cutoffs

Cutoffs are applied to each pixel of each pulse before summing, as in cristallina:

- `cutoff` (lower) removes electronic noise. For Bragg peaks it sits just below the photon energy (7.0 keV at
  7.5 keV in March).
- `upper_cutoff` with `cutoff` forms an energy window: fluorescence at 5.5–6.8 keV, excluding elastic scattering at
  the incident energy.
- `i0_cutoff` does the same for an image I0. March used 2.5 keV in the notebook and 2 keV inside
  `calculate_JF_stacks`. Pick one and use it for every run you compare.

## Photon counts and variances

With `photon_energy` (the detected photons' energy, in keV), summed keV are divided by it and given Poisson variances.
From there scipp propagates errors through `/ i0`, background subtraction and fits. Two approximations:

- Gain-corrected, thresholded Jungfrau values are not exact photon counts. Pixels with charge sharing or pile-up make
  Poisson only an approximation.
- I0 is treated as exact. A JF-I0 sum over many pixels and pulses has a small relative error, but it is not zero.

For full-detector stacks, convert only the region you need with [`to_photons`][cristallina_scipp.to_photons] after
[`crop`][cristallina_scipp.crop]. Variances double the memory.
