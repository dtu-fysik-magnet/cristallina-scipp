# Map existing cristallina code

| Beamline code | cristallina-scipp |
|---|---|
| `for step in scan: np.mean(step[ch].data)` / `np.std` | [`scan_means(scan, names, spread=True)`][cristallina_scipp.scan_means] |
| `perform_image_calculations(..., lower_cutoff_threshold, upper_cutoff_threshold)` per step, plus a separate JF-I0 call | [`reduce_scan(scan, JF, JF_I0, cutoff=, upper_cutoff=, i0_cutoff=)`][cristallina_scipp.reduce_scan], with the same pulses for both |
| `calculate_JF_stacks(scan, ...)` → `stacks, I0_norms` | `stack = reduce_scan(...)`; `stack.coords["i0"]` |
| `calculate_JF_stacks_no_I0` | `reduce_scan(scan, JF)` (no `i0`) |
| `calculate_image_histograms(scan[i].fnames, ...)` | [`pixel_energies(scan, i)`][cristallina_scipp.pixel_energies], also over several steps |
| `stacks[:, roi.rows, roi.cols].sum(axis=(1, 2)) / I0` | `roi_intensities(stack, [roi]) / stack.coords["i0"]` |
| `perform_image_stack_sum(..., roi=roi)` | `reduce_scan(..., roi=roi)` or [`crop(stack, roi)`][cristallina_scipp.crop] |
| `convert_angle(step)`, `convert_roi` | [`detector_angles(da, **geometry_from_step(step, **MARCH_2026))`][cristallina_scipp.angles.detector_angles] |
| `roi_intensity_FWHM` (centroids, FWHM) | [`moments(crop(stack, roi).sum("y"), "x")`][cristallina_scipp.peaks.moments] |
| `new_fitter` / `build_model` / `extract_info` | [`fit_peaks(curve, n=, shape=, coord="readback")`][cristallina_scipp.peaks.fit_peaks] |
| `batch_fitter`, `data_sorter`, `plot_overlap`, `load_and_plot_intensities_series`, `extraxt_sum_intensity`, `integrate_omega_roi_trapz_bgsub` | [run series](../tutorials/run-series.md): [`Catalogue.series`][cristallina_scipp.catalogue.Catalogue.series] |
| `I / f(E)` with `interp1d` of the run-45 fluorescence, `get_run_transmission` | [corrections](corrections.md): `correct_absorption` (multiplies; `exponent=-1` reproduces the division), `correct_transmission` |
| run lists retyped per notebook, `setup/roi_peaks.py`, `setup/rsmconfig.py`, `roi_map` | a [run catalogue](catalogue.md) |
| `interactive_scan_plot` | `pp.slicer(stack / stack.coords["i0"], keep=["y", "x"])` |
| `collapsed_axis_plot` | `pp.plot(crop(stack, roi).sum("y"))` (readback against x or 2θ) |
| `RSM` + `RSM_interpolate` | [`pixel_hkl`][cristallina_scipp.rsm.pixel_hkl] + [`histogram`][cristallina_scipp.rsm.histogram] |

Kept as they are: `calculate_image_histograms` (choosing the energy window), `cristallina.rsm`'s geometry and
the pgroup bookkeeping module. `cristallina.utils.ROI` and the channel names used here are copied as
[`ROI`][cristallina_scipp.ROI] and `cristallina_scipp.channels`, so only the RSM needs `cristallina` installed;
cristallina's own ROIs work wherever a `ROI` is expected.
