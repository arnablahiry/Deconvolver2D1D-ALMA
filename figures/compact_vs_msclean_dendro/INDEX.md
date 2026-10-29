# Compact deconvolved vs extended MS-CLEAN, inside the astrodendro mask

Script: `scripts/compact_vs_msclean_dendro/make_figures.py`. Compact: `data/experiments/04_pd_tau_reweighting/compact/pdtau_k5.npy` (pdtau k = 5, raw, Jy/pixel). CLEAN: `data/ngc3110/extended/ms_clean/`. Mask: `data/ngc3110/extended/ms_clean_masks/dendro_ms_clean.npy` (scripts/superres_figures/fig61_dendro_mask_extended.py). Both models are multiplied by the 3D mask and convolved with the extended clean beam (0.221″×0.172″, PA -74.7°); the CLEAN image (components * beam + residual) is shown as a reference. Numbers in `stats.json`.

- `50_msclean_complete.png`: the extended MS-CLEAN before and after finishing it inside the growing dendrogram mask (scripts/msclean_complete.py): emission-only moment 0, residual, rms per channel, spectrum. (scripts/compact_vs_msclean_dendro/fig50_msclean_complete.py)
- `40_overview.png`: Moment 0 of the compact model, CLEAN components and CLEAN image (all × mask, extended beam); compact − CLEAN; compact / CLEAN; integrated spectra inside the mask.
- `41_zooms.png`: Four boxes (nucleus/bar, north end of the bar, east arm, south arm): moment 0 of compact and CLEAN on a common scale per box, and the box spectra (CLEAN image dashed).
- `42_channels.png`: Eight line channels: compact, CLEAN (common scale per channel) and compact − CLEAN (±50% of it), mJy/beam.
- `43_profiles.png`: Moment-0 cuts north-south along the bar, north-south across the east arm and east-west across the south arm: compact, CLEAN components, CLEAN image.
- `44_moment1.png`: Intensity-weighted velocity of compact and CLEAN (where moment 0 > 5% of the peak) and their difference.
- `45_pixel_scatter.png`: Compact vs CLEAN per moment-0 pixel and per voxel (log density; dashed 1:1), and the median compact/CLEAN ratio as a function of CLEAN's brightness.
- `46_residuals_extended.png`: Residual of the masked compact model and the masked CLEAN components on the extended dirty cube (moment 0 and rms per channel, whole field and inside the footprint).
- `47_spectra_peaks.png`: One-beam spectra at the eight brightest local maxima of CLEAN's moment 0: compact, CLEAN components, CLEAN image.
- `48_radial.png`: Moment-0 flux per annulus, cumulative flux and the compact/CLEAN ratio per annulus.
- `49_summary.png`: The numbers: flux, moment-0 peak, extended-data residual rms (whole field, footprint, 3D mask) and mean, compact-vs-CLEAN correlations and moment-0 slope, moment-1 difference.
