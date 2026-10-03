#!/usr/bin/env python
"""
Grids a dirty PSF with CASA for a given baseline (uvdist) selection, so the
"compact" and "extended" configurations in
`notebooks/fire_cii_simulated_obs.ipynb` are both *real* CASA PSFs rather
than beams re-derived with this repo's simple `histogram2d` gridder.

Why this exists: `psf.dirty_beam_from_uv` bins each visibility into a single
uv cell and inverse-FFTs. CASA instead convolves each visibility onto ~6x6
cells with a prolate spheroidal, applies the per-pointing primary beams of
the 3-pointing mosaic, and grid-corrects. Measured against the real PSF, the
simple gridder correlates at only 0.48; adding a Gaussian gridding kernel
gets to ~0.58, and the rest is the mosaic primary beam, which a single-field
gridder cannot reproduce at all. The only way to get the real thing is to let
CASA do it.

The imaging setup is taken unchanged from `legacy/simple/grid_with_casa.py`
(same cell, image size, channels, phase centre, briggs robust=0.5, mosaic
ftmachine); the ONLY difference between configurations is the `uvdist`
selection passed to `selectdata`.

The all-baselines case is already on disk as `simple_results/data/psf.npy`,
so by default this only grids the compact one.

Usage:
    python3 scripts/grid_fire_configs_with_casa.py
    python3 scripts/grid_fire_configs_with_casa.py "0~793m" psf_compact
"""

import os
import shutil
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "legacy", "simple"))
import grid_with_casa as cfg   # MS path + imaging config, unchanged

OUT_DIR = os.path.join(REPO, "simple_results", "data")


def grid_psf(uvdist, out_name, work_stem=None, imsize=None, cell_arcsec=None):
    """Grid one PSF cube under the given uvdist selection and save it as .npy."""
    from casatools import synthesisimager, synthesisnormalizer, image as image_tool

    imsize = imsize or cfg.IMSIZE
    cell_arcsec = cell_arcsec or cfg.CELL_ARCSEC
    work_stem = work_stem or os.path.join(OUT_DIR, f"casa_{out_name}")
    for suffix in (".psf", ".residual", ".model", ".sumwt", ".weight", ".gridwt", ".pb"):
        path = work_stem + suffix
        if os.path.exists(path):
            shutil.rmtree(path)

    imager = synthesisimager()
    imager.selectdata({
        "msname": os.path.join(REPO, cfg.MS), "spw": cfg.SPW, "field": "",
        "uvdist": uvdist,                      # <-- the only thing that varies
        "usescratch": True, "readonly": False, "datacolumn": "data",
    })
    impars = {
        "imagename": work_stem, "specmode": "cube", "nchan": cfg.NCHAN,
        "imsize": [imsize, imsize], "cell": [f"{cell_arcsec}arcsec"] * 2,
        "stokes": "I", "phasecenter": cfg.PHASECENTER, "projection": "SIN",
        "start": f"{cfg.START_KMS}km/s", "width": f"{cfg.WIDTH_KMS}km/s",
        "restfreq": f"{cfg.RESTFREQ_GHZ}GHz", "outframe": "LSRK",
        "veltype": "radio", "deconvolver": "hogbom",
    }
    gridpars = {
        "imagename": work_stem, "ftmachine": "mosaic", "wprojplanes": 1,
        "padding": 1.2, "pblimit": 0.2, "normtype": "flatnoise",
        "useautocorr": False, "usedoubleprec": True,
        "interpolation": "linear", "conjbeams": False,
    }
    imager.defineimage(impars, gridpars)
    imager.setweighting(type="briggs", rmode="norm", robust=0.5)

    normalizer = synthesisnormalizer()
    normalizer.setupnormalizer({
        "imagename": work_stem, "normtype": "flatnoise", "workdir": OUT_DIR,
        "deconvolver": "hogbom", "nterms": 1, "imindex": 0, "psfcutoff": 0.35,
    })
    imager.normalizerinfo({
        "imagename": work_stem, "normtype": "flatnoise", "workdir": OUT_DIR,
        "deconvolver": "hogbom", "nterms": 1, "imindex": 0, "psfcutoff": 0.35,
    })

    print(f"[casa] gridding PSF for uvdist='{uvdist or 'all'}' ...")
    imager.makepsf()

    ia = image_tool()
    ia.open(work_stem + ".psf")
    arr = ia.getchunk()                       # CASA axes: (x, y, stokes, chan)
    beam = ia.restoringbeam()
    ia.close()
    psf = np.transpose(arr[:, :, 0, :], (2, 1, 0))     # -> (chan, y, x)

    if "beams" in beam:
        per_chan = [b["*0"] for b in beam["beams"].values()]
        bmaj = float(np.median([b["major"]["value"] for b in per_chan]))
        bmin = float(np.median([b["minor"]["value"] for b in per_chan]))
        bpa = float(np.median([b["positionangle"]["value"] for b in per_chan]))
    else:
        bmaj, bmin, bpa = (beam["major"]["value"], beam["minor"]["value"],
                           beam["positionangle"]["value"])

    imager.done()
    normalizer.done()

    out_path = os.path.join(OUT_DIR, f"{out_name}.npy")
    np.save(out_path, psf)
    np.save(os.path.join(OUT_DIR, f"{out_name}_beam_arcsec_deg.npy"),
            np.array([bmaj, bmin, bpa]))
    print(f"[casa] wrote {out_path}, shape {psf.shape}")
    print(f"[casa] PSF peak per channel: {psf.max(axis=(1, 2)).min():.4f} .. "
          f"{psf.max(axis=(1, 2)).max():.4f}  (should be ~1.0)")
    print(f"[casa] clean beam {bmaj:.4f}\" x {bmin:.4f}\" @ {bpa:.1f} deg "
          f"= {bmaj / cfg.CELL_ARCSEC:.2f} x {bmin / cfg.CELL_ARCSEC:.2f} px")
    return psf


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        imsize = int(sys.argv[3]) if len(sys.argv) >= 4 else None
        grid_psf(sys.argv[1], sys.argv[2], imsize=imsize)
    else:
        # The FIRE deconvolution notebooks want both configs gridded directly
        # onto the FIRE crop's own 204px/0.04" grid -- no dead space around the
        # source, unlike the padded 320px CASA-imaging grid gridded above.
        # This is real gridding from the visibilities at that field of view,
        # not interpolation of the 320px PSF, but it is not identical to it:
        # a smaller FOV means a coarser uv cell for Briggs weighting, and (more
        # importantly, measured against a plain crop of the 320px version)
        # some of the compact beam's far sidelobe power aliases back into the
        # smaller frame -- corr 0.983, up to ~3.5% of peak. Accepted trade for
        # no dead space and ~2.5x fewer pixels (faster solves).
        #
        # extended uses ALL baselines (the full array's native resolution,
        # 4.07 x 3.15 px); compact keeps its own upper uvdist cut.
        grid_psf("", "fire204_psf_extended", imsize=204)
        grid_psf("0~450m", "fire204_psf_compact", imsize=204)
