# Super-resolution comparison figures (40-44)

Compact deconvolved model vs the independent extended model, NGC 3110 CO(2-1).

| script | figure |
|---|---|
| fig40_overview.py | 40_superres_overview.png: compact * compact beam, compact * ext beam, extended * ext beam; spectrum, contours, residual |
| fig41_zooms.py | 41_superres_zooms.png: four zoom boxes + box spectra |
| fig42_channels.py | 42_superres_channels.png: channel maps |
| fig43_profiles.py | 43_superres_profiles.png: brightness cuts |
| fig44_dirty_raw_restored.py | 44_dirty_raw_restored.png: dirty / raw model / model * ext beam |

Configured by environment variables (defaults in brackets):

- `COMPACT_MODEL`  compact model cube, Jy/pixel, (32, 800, 800) at 0.02" [data/ngc3110/compact/pd/model.npy]
- `EXTENDED_MODEL` extended model cube [data/ngc3110/extended/pd/model.npy]
- `OUT_DIR`        output folder [data/ngc3110/compact/debug]
- `COMPACT_LABEL`  label in "Compact (<label>)" panel headings [deconvolved]

Run with the CASA-free env, e.g.

    COMPACT_MODEL=data/ngc3110/compact/fista/model_k2_20k.npy OUT_DIR=figures/fista COMPACT_LABEL=FISTA \
        /scratch/alahiry/conda/envs/deconvolver2d1d/bin/python scripts/superres_figures/fig40_overview.py

## Target resolution vs none, and vs the extended CLEAN (53-60)

Written to `figures/experiments/11_target_resolution/`, numbers to `data/experiments/11_target_resolution/`.
Compact pdtau k = 5 with no target (`data/experiments/04_pd_tau_reweighting/compact/pdtau_k5.npy`) and with a
target resolution of lambda/B_max = 0.1134" (`data/experiments/11_target_resolution/pdtau_k5_t0.1134.npy`),
both convolved with the extended clean beam. Need the GPU (residuals via `simple/gpu_pd.Op`).

| script | figure |
|---|---|
| fig53_target_vs_none_extbeam.py | 53_mom0_extbeam: moment 0 of both, and target - no target |
| | 54_residuals: residual on the compact dirty cube; rms and mean per channel |
| | 55_spectra_extbeam: integrated and one-beam spectra |
| fig56_compact_vs_extended_clean.py | 56_vs_clean_mom0: both compact models vs extended CLEAN (components * beam, image) |
| | 57_vs_clean_diff: moment-0 differences against CLEAN's components |
| | 58_vs_clean_residuals: residual of all three models on the EXTENDED dirty cube (one operator) |
| | 59_vs_clean_spectra: integrated and one-beam spectra |
| fig61_dendro_mask_extended.py | 61_dendro_mask_extended_ms_clean: astrodendro mask of the extended CLEAN image (per-channel sigma, peak bar set by the negated image); saves data/ngc3110/extended/ms_clean_masks/dendro_<CLEAN>.npy |
| fig60_metrics_summary.py | 60_metrics_summary: the numbers of 53-59 side by side, one panel per metric (reads their JSONs; no GPU) |

`TAG` picks the target model (`pdtau_k5_<TAG>.npy`, default t0.1134); `CLEAN` picks the extended CLEAN run
(`data/ngc3110/extended/<CLEAN>/`, default ms_clean; other runs get a `_<CLEAN>` suffix).

`MASK=data/ngc3110/extended/ms_clean_masks/dendro_ms_clean.npy` (fig53, fig56, fig60) multiplies every model cube
(both compact models, CLEAN's components and its image) by fig61's dendrogram mask before the convolution with the
extended beam and before the residuals; moment-0 comparisons are then taken over the mask's footprint. Outputs get
the suffix `_dendro_ms_clean`.
