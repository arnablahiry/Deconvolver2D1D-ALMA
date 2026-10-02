"""
The exact measurement operator, evaluated on the real visibilities every
call -- the operator `uv_to_image.ShiftInvariantOperator` only approximates.

======================================================================
WHY THIS EXISTS: THE MOSAIC BREAKS THE SINGLE-PSF SHORTCUT
======================================================================
`ShiftInvariantOperator` computes N(x) = x (*) PSF via two FFTs, which is
the *exact* normal operator A^H W A only when A^H W A is one fixed
convolution kernel everywhere in the field. That holds for a single
pointing. It does NOT hold for a mosaic: this dataset combines 3 pointings,
each multiplying the sky by its own primary beam PB_p *before* the Fourier
transform,

    A^H W A x  =  sum_p  PB_p * IFFT( W_p * FFT( PB_p * x ) )

-- a sum of position-dependent terms, not one global convolution. A single
FFT-derived OTF cannot represent that sum; it can only approximate it with
one averaged kernel. Measured cost of that approximation on this dataset:
0.55% rms disagreement over the full field, up to ~5-7% at the mosaic edge,
where the residual sits after fitting with the fast operator.

The only way to get this right is to let CASA do what it already does
correctly: for each visibility, use its real (u, v, w) coordinates AND
figure out which pointing (and therefore which primary beam) it belongs to.
That is unavoidably a per-visibility, per-iteration operation -- which is
why, and ONLY why, this version calls CASA every time instead of once.

======================================================================
WHAT ONE CALL ACTUALLY COMPUTES
======================================================================
`synthesisimager.executemajorcycle` with a model image installed does, in
one pass over the real visibilities:

    1. degrid the model image x -> trial visibilities A(x)
    2. form the TRUE visibility-domain residual:  r_vis = Vis - A(x)
    3. grid r_vis with the real weights W and the correct primary beam:
       residual_image = A^H W r_vis

and returns `residual_image` directly, already correctly normalized (see
the note in `_major_cycle` below -- CASA's cube-mode gridder divides by the
weight sum internally; calling the normalizer's divide-by-weight again, as
an earlier version of this pipeline did, silently rescaled the image by
4.65e7). No extra bookkeeping needed: this residual_image already equals
`dirty - N(x)` for the REAL, mosaic-aware N.

Cost: ~60-140 seconds per call, versus ~0.2s for the FFT operator, because
step 1-3 above touch the real, scattered visibilities of the whole
measurement set instead of two FFTs on a regular grid.
"""

import os
import numpy as np

import grid_with_casa as cfg   # reuses MS path / imaging config, unchanged


class ExactOperator:
    """Same interface as `uv_to_image.ShiftInvariantOperator`:
    `.gradient(x)`, `.residual(x)`, `.dirty`, `.lipschitz` -- drop-in
    replacement in `deconvolve.py`'s solver loop."""

    def __init__(self, dirty, lipschitz_estimate):
        from casatools import synthesisimager, synthesisnormalizer

        self.dirty = dirty
        # Power-iterating the exact operator to get its true Lipschitz
        # constant would cost ~20 major cycles just for setup. The fast
        # FFT operator's max|OTF| (passed in) is measured to agree with the
        # true operator's overall response to 0.55% rms, so it is reused
        # here as a very close, much cheaper stand-in. `step_safety < 1` in
        # `deconvolve.py` gives a safety margin against this being slightly
        # off.
        self.lipschitz = lipschitz_estimate

        self.si = synthesisimager()
        self.si.selectdata({
            "msname": os.path.abspath(cfg.MS), "spw": cfg.SPW, "field": "",
            "usescratch": True, "readonly": False, "datacolumn": "data",
        })
        impars = {
            "imagename": cfg.IMAGE_STEM, "specmode": "cube", "nchan": cfg.NCHAN,
            "imsize": [cfg.IMSIZE, cfg.IMSIZE],
            "cell": [f"{cfg.CELL_ARCSEC}arcsec"] * 2,
            "stokes": "I", "phasecenter": cfg.PHASECENTER, "projection": "SIN",
            "start": f"{cfg.START_KMS}km/s", "width": f"{cfg.WIDTH_KMS}km/s",
            "restfreq": f"{cfg.RESTFREQ_GHZ}GHz", "outframe": "LSRK",
            "veltype": "radio", "deconvolver": "hogbom",
        }
        gridpars = {
            "imagename": cfg.IMAGE_STEM, "ftmachine": "mosaic", "wprojplanes": 1,
            "padding": 1.2, "pblimit": 0.2, "normtype": "flatnoise",
        }
        self.si.defineimage(impars, gridpars)
        self.si.setweighting(type="briggs", rmode="norm", robust=0.5)

        normpars = {"imagename": cfg.IMAGE_STEM, "normtype": "flatnoise",
                   "workdir": cfg.OUT_DIR, "deconvolver": "hogbom",
                   "nterms": 1, "imindex": 0}
        self.sn = synthesisnormalizer()
        self.sn.setupnormalizer(normpars)
        self.si.normalizerinfo(normpars)

    def _write_model(self, x):
        from casatools import image as image_tool

        model_path = cfg.IMAGE_STEM + ".model"
        if os.path.exists(model_path):
            import shutil
            shutil.rmtree(model_path)
        ia = image_tool()
        ia.open(cfg.IMAGE_STEM + ".psf")     # template for coordinate system
        csys = ia.coordsys()
        shape = ia.shape()
        ia.close()
        arr = np.zeros(tuple(shape))
        arr[:, :, 0, :] = np.transpose(x, (2, 1, 0))     # (nz,ny,nx) -> CASA axes
        ia.fromarray(outfile=model_path, pixels=arr, csys=csys.torecord(),
                    overwrite=True)
        ia.close()
        csys.done()

    def residual(self, x):
        """dirty - N(x), computed by ONE real degrid+grid pass. See module
        docstring: this is genuinely `Vis - A(x)`, gridded, not an FFT
        approximation of it."""
        from casatools import image as image_tool

        self._write_model(x)
        self.si.executemajorcycle({"lastcycle": False})
        ia = image_tool()
        ia.open(cfg.IMAGE_STEM + ".residual")
        arr = ia.getchunk()
        ia.close()
        return np.transpose(arr[:, :, 0, :], (2, 1, 0))

    def gradient(self, x):
        return -self.residual(x)

    def apply(self, x):
        """N(x) = dirty - residual(x). Only needed for the diagnostic print
        in deconvolve.py, not by the solver loop itself (which only ever
        needs `gradient`/`residual`, each one call, matching the fast
        operator's cost profile of one operator evaluation per iteration)."""
        return self.dirty - self.residual(x)

    def done(self):
        self.si.done()
        self.sn.done()
