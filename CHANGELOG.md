# Changelog

All notable changes to cristallina-scipp. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [semantic versioning](https://semver.org/): before 1.0, a minor release (0.2 → 0.3) may change or
remove things notebooks and saved files rely on, and a patch release (0.2.0 → 0.2.1) does not.

Every pull request adds its user-facing changes under "Unreleased". **Changed** and **Removed** entries say what to do
in an existing notebook or catalogue.

## [Unreleased]

## [0.1.0] - 2026-11-28

The first release. Everything below is new.

### Added

- Reduction: `reduce_scan` (per-step Jungfrau sums with energy cutoffs, detector and I0 over the same pulses, optional
  photon counts with Poisson variances), `scan_means`, `roi_intensities`, `crop`, `to_photons`, `channel`, `step`.
- Peaks: `peaks.fit_peaks` (Lorentzian, Gaussian or exact Voigt), `peaks.moments`, `peaks.integrate` (area above a
  straight-line background, variances propagated, no clipping).
- Beam checks: `pixel_energies` histograms the energies of the pixels hit in one or several steps, for choosing the
  fluorescence and Bragg windows (replaces cristallina's `calculate_image_histograms`); a how-to covers beam monitors,
  dropped pulses, photon-energy offsets and the fluorescence window.
- Run catalogue (`Catalogue`): one TOML file, or a folder of them, per experiment with run groups, ROIs, detector
  calibration by run range and samples; strict loading that names the file at fault.
- `cristallina-catalogue` command: `check` a catalogue, and `setup` a shared git repository with hooks that refuse a
  catalogue that does not load and keep a read-only checkout up to date.
- Run overview: `run_overview` and `Catalogue.overview` list every run in a raw folder with its scan and metadata and
  the groups it belongs to.
- Run series: `run_series` and `Catalogue.series`, one point per run (integral, fit, moments) with energy, field,
  transmission and labels as coordinates, saved to HDF5 with provenance (settings, package version, catalogue path,
  SHA-256 and git commit).
- Field direction: `add_field_angles` computes `field_angle` and `field_tilt` from each run's measured field vector;
  a catalogue sample's `field_angle_zero` and `field_angle_90` (field directions in the magnet's frame) add them to
  every series of its runs.
- Corrections: `correct_transmission` (attenuators after the I0 monitor) and `correct_absorption` (fluorescence
  yield as a proxy for μ(E)), each recorded in the series.
- Split peaks: `split.fit_split` compares one and two peaks on a rocking curve, a detector projection or a
  reciprocal-space map, by maximum Poisson likelihood and the Bayesian information criterion; `split_series` and
  `Catalogue.split_series` do it per run.
- Reciprocal space: `rsm.pixel_hkl`, `rsm.histogram` (normalised, with per-voxel exposure and mean pixel position),
  `rsm.local_coords` (strain and tilts around a reflection); `Catalogue.sample`, `.detector` and `.to_q` build them
  from the catalogue.
- Angles: `angles.detector_angles`, `angles.geometry_from_step`.
- Synthetic data in the SwissFEL layout (`testing`) for tests, docs and dry runs.
- Notebooks: `experiment_template.ipynb` and `rsm.ipynb`; zensical documentation built from tested scripts.
