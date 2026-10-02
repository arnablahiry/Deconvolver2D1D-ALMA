"""
The one step that genuinely has to touch real visibilities: turning the
measurement set into a PSF cube and a dirty-image cube.

This is CASA's `synthesisimager` doing exactly the gridding described in
`operator.py`'s docstring -- steps 1-3 (weight, grid, IFFT) -- for real,
scattered (u, v) points, real Briggs weighting, and (for a mosaic like this
one) each pointing's primary beam. It is run ONCE, offline; nothing in the
deconvolution loop calls CASA again.

Run this first:
    python simple/grid_with_casa.py

Everything else in `simple/` only ever reads the two .npy files it writes.
"""

import os
import numpy as np


MS = "data/ngc7469_co21.ms"          # from ../scripts/split_line_ms.py
OUT_DIR = "simple/data"
IMAGE_STEM = "simple/data/casa"

# Same imaging choices as the full pipeline (src/alma_fourier.ImagingConfig),
# picked from a measured survey of this dataset -- see the project README.
IMSIZE = 320
CELL_ARCSEC = 0.04
NCHAN = 90
START_KMS = 4610.0
WIDTH_KMS = 5.0
RESTFREQ_GHZ = 230.538
PHASECENTER = "J2000 23:03:15.61 +08.52.25.8"   # measured source position
SPW = "0,1,2,3"


def grid_psf_and_dirty():
    from casatools import synthesisimager, synthesisnormalizer, image as image_tool

    os.makedirs(OUT_DIR, exist_ok=True)

    # --- 1. tell the imager which visibilities and which image grid -----
    imager = synthesisimager()
    imager.selectdata({
        "msname": os.path.abspath(MS), "spw": SPW, "field": "",
        "usescratch": True, "readonly": False, "datacolumn": "data",
    })
    impars = {
        "imagename": IMAGE_STEM, "specmode": "cube", "nchan": NCHAN,
        "imsize": [IMSIZE, IMSIZE], "cell": [f"{CELL_ARCSEC}arcsec"] * 2,
        "stokes": "I", "phasecenter": PHASECENTER, "projection": "SIN",
        "start": f"{START_KMS}km/s", "width": f"{WIDTH_KMS}km/s",
        "restfreq": f"{RESTFREQ_GHZ}GHz", "outframe": "LSRK",
        "veltype": "radio", "deconvolver": "hogbom",
    }
    gridpars = {
        "imagename": IMAGE_STEM, "ftmachine": "mosaic", "wprojplanes": 1,
        "padding": 1.2, "pblimit": 0.2, "normtype": "flatnoise",
    }
    imager.defineimage(impars, gridpars)
    imager.setweighting(type="briggs", rmode="norm", robust=0.5)

    # Cube gridding needs the normalizer's image names too, or it aborts.
    normalizer = synthesisnormalizer()
    normalizer.setupnormalizer({
        "imagename": IMAGE_STEM, "normtype": "flatnoise", "workdir": OUT_DIR,
        "deconvolver": "hogbom", "nterms": 1, "imindex": 0,
    })
    imager.normalizerinfo({
        "imagename": IMAGE_STEM, "normtype": "flatnoise", "workdir": OUT_DIR,
        "deconvolver": "hogbom", "nterms": 1, "imindex": 0,
    })

    # --- 2. grid the PSF: response of the (u,v) coverage + weights to a
    #        point source. This IS "IFFT2(W(u,v))" from operator.py.
    print("[casa] gridding the PSF ...")
    imager.makepsf()

    # --- 3. grid the dirty image with a zero sky model: IFFT2(W * V_obs). -
    print("[casa] gridding the dirty image ...")
    imager.executemajorcycle({"lastcycle": False})

    def read_cube(path):
        ia = image_tool()
        ia.open(path)
        arr = ia.getchunk()          # CASA axis order: (x, y, stokes, chan)
        ia.close()
        return np.transpose(arr[:, :, 0, :], (2, 1, 0))   # -> (chan, y, x)

    psf = read_cube(IMAGE_STEM + ".psf")
    dirty = read_cube(IMAGE_STEM + ".residual")

    ia = image_tool()
    ia.open(IMAGE_STEM + ".psf")
    beam = ia.restoringbeam()
    ia.close()
    if "beams" in beam:
        beams = [b["*0"] for b in beam["beams"].values()]
        bmaj = float(np.median([b["major"]["value"] for b in beams]))
        bmin = float(np.median([b["minor"]["value"] for b in beams]))
        bpa = float(np.median([b["positionangle"]["value"] for b in beams]))
    else:
        bmaj, bmin, bpa = (beam["major"]["value"], beam["minor"]["value"],
                          beam["positionangle"]["value"])

    imager.done()
    normalizer.done()

    np.save(os.path.join(OUT_DIR, "psf.npy"), psf)
    np.save(os.path.join(OUT_DIR, "dirty.npy"), dirty)
    np.save(os.path.join(OUT_DIR, "beam_arcsec_deg.npy"), np.array([bmaj, bmin, bpa]))
    print(f"[casa] wrote {OUT_DIR}/psf.npy and dirty.npy, shape {psf.shape}")
    print(f"[casa] PSF peak per channel: min={psf.max(axis=(1,2)).min():.4f} "
          f"max={psf.max(axis=(1,2)).max():.4f}  (should both be ~1.0)")
    print(f"[casa] clean beam: {bmaj:.4f}\" x {bmin:.4f}\" @ {bpa:.1f} deg "
          f"(cell={CELL_ARCSEC}\")")


if __name__ == "__main__":
    grid_psf_and_dirty()
