"""
Export simple_results/data/*.npy (from legacy/simple/deconvolve.py) as FITS,
for viewing in CARTA/DS9.

legacy/simple/deconvolve.py only ever saves raw numpy arrays -- no WCS, no
BUNIT, no beam, so nothing but numpy can open them directly. This reuses the
same `write_casa_image`/`export_fits` helpers src/alma_fourier.py already
uses for results/, templated off the CASA image grid_with_casa.py wrote
(simple_results/data/casa.psf), so the coordinate system here is identical
to results/'s -- these FITS files line up pixel-for-pixel with the ones in
results/ if you load both in CARTA.

Run from the repo root:
    python simple_results/export_fits.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "legacy", "simple"))

from alma_fourier import write_casa_image, export_fits
from uv_to_image import ShiftInvariantOperator
from fista_2d1d import restore, gaussian_beam


def main():
    psf = np.load(os.path.join(DATA_DIR, "psf.npy"))
    dirty = np.load(os.path.join(DATA_DIR, "dirty.npy"))
    model = np.load(os.path.join(DATA_DIR, "model.npy"))
    bmaj, bmin, bpa = np.load(os.path.join(DATA_DIR, "beam_arcsec_deg.npy"))
    cell_arcsec = 0.04   # matches grid_with_casa.py's CELL_ARCSEC

    # residual wasn't saved by deconvolve.py -- recompute it the same way:
    # dirty - N(model), via the exact operator the solver itself used.
    op = ShiftInvariantOperator(psf, dirty)
    residual = op.residual(model)

    # `model` (Jy/pixel, un-diluted point brightness) is NOT on the same
    # scale as `dirty` (Jy/beam, flux integrated over ~15 pixels/beam) --
    # comparing their raw colorbars is comparing different units, not a
    # bug. `restored` puts the model back on that same Jy/beam footing by
    # convolving it with the clean (Gaussian) beam and adding the leftover
    # residual back -- the standard CLEAN-style product, directly
    # comparable to `dirty` in CARTA.
    cb = gaussian_beam(model.shape[1:], bmaj / cell_arcsec,
                       bmin / cell_arcsec, bpa)
    restored = restore(model, residual, cb)

    template = os.path.join(DATA_DIR, "casa.psf")
    outputs = {
        "model":    (model, "Jy/pixel", None),              # sparse model, no beam
        "dirty":    (dirty, "Jy/beam", (bmaj, bmin, bpa)),
        "residual": (residual, "Jy/beam", (bmaj, bmin, bpa)),
        "restored": (restored, "Jy/beam", (bmaj, bmin, bpa)),  # comparable to dirty
    }
    for name, (cube, bunit, beam) in outputs.items():
        img = os.path.join(DATA_DIR, f"{name}.image")
        fits = os.path.join(DATA_DIR, f"ngc7469_simple_{name}.fits")
        write_casa_image(img, cube, template=template, bunit=bunit, beam=beam)
        export_fits(img, fits)
        print(f"wrote {fits}")


if __name__ == "__main__":
    main()
