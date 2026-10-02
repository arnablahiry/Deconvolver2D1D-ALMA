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
