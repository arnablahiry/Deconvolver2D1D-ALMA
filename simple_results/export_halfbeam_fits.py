"""
Export simple_results/data/model.npy convolved with a beam half the FWHM of
the full clean beam (same position angle), for viewing in CARTA/DS9.

This is not a CLEAN-style restored image (no residual added back) -- it is
just the sparse model smoothed to a sharper beam than the standard restoring
beam, to show finer structure than ngc7469_simple_restored.fits while still
being in Jy/beam units.

Run from the repo root:
    python simple_results/export_halfbeam_fits.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "legacy", "simple"))

from alma_fourier import write_casa_image, export_fits
from fista_2d1d import gaussian_beam
from scipy.signal import fftconvolve


def main():
    model = np.load(os.path.join(DATA_DIR, "model.npy"))
    bmaj, bmin, bpa = np.load(os.path.join(DATA_DIR, "beam_arcsec_deg.npy"))
    cell_arcsec = 0.04   # matches grid_with_casa.py's CELL_ARCSEC

    half_bmaj, half_bmin = bmaj / 2.0, bmin / 2.0
    beam_pix = gaussian_beam(model.shape[1:], half_bmaj / cell_arcsec,
                             half_bmin / cell_arcsec, bpa)

    smoothed = np.empty_like(model)
    for k in range(model.shape[0]):
        smoothed[k] = fftconvolve(model[k], beam_pix, mode="same")

    template = os.path.join(DATA_DIR, "casa.psf")
    img = os.path.join(DATA_DIR, "halfbeam.image")
    fits = os.path.join(DATA_DIR, "ngc7469_simple_model_halfbeam.fits")
    write_casa_image(img, smoothed, template=template, bunit="Jy/beam",
                     beam=(half_bmaj, half_bmin, bpa))
    export_fits(img, fits)
    print(f"clean beam:      {bmaj:.4f}\" x {bmin:.4f}\" @ {bpa:.1f} deg")
    print(f"half-width beam: {half_bmaj:.4f}\" x {half_bmin:.4f}\" @ {bpa:.1f} deg")
    print(f"wrote {fits}")


if __name__ == "__main__":
    main()
