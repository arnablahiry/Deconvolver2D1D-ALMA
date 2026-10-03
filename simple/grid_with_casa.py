"""
The one step that genuinely has to touch real visibilities: turning the
measurement set into a PSF cube and a dirty-image cube.

This is CASA's `synthesisimager` doing exactly the gridding described in
`uv_to_image.py`'s docstring -- steps 1-3 (weight, grid, IFFT) -- for real,
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

# Imaging choices for the NGC 7469 CO(2-1) cube,
# picked from a measured survey of this dataset.
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
        "useautocorr": False, "usedoubleprec": True,
        "interpolation": "linear", "conjbeams": False,
    }
    imager.defineimage(impars, gridpars)
    imager.setweighting(type="briggs", rmode="norm", robust=0.5)

    # Cube gridding needs the normalizer's image names too, or it aborts.
    normalizer = synthesisnormalizer()
    normalizer.setupnormalizer({
        "imagename": IMAGE_STEM, "normtype": "flatnoise", "workdir": OUT_DIR,
        "deconvolver": "hogbom", "nterms": 1, "imindex": 0, "psfcutoff": 0.35,
    })
    imager.normalizerinfo({
        "imagename": IMAGE_STEM, "normtype": "flatnoise", "workdir": OUT_DIR,
        "deconvolver": "hogbom", "nterms": 1, "imindex": 0, "psfcutoff": 0.35,
    })

    # --- 2. grid the PSF: response of the (u,v) coverage + weights to a
    #        point source. This IS "IFFT2(W(u,v))" from uv_to_image.py.
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
    bmaj, bmin, bpa = median_beam(ia.restoringbeam())
    ia.close()

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


def median_beam(record):
    """(bmaj", bmin", bpa deg) from an `image.restoringbeam()` record: the
    single beam, or the per-channel median for a cube."""
    if "beams" in record:
        beams = [b["*0"] for b in record["beams"].values()]
    else:
        beams = [record]
    for b in beams:
        assert b["major"]["unit"] == b["minor"]["unit"] == "arcsec", b
        assert b["positionangle"]["unit"] == "deg", b
    return tuple(float(np.median([b[k]["value"] for b in beams]))
                 for k in ("major", "minor", "positionangle"))


def tclean_dirty_cube(ms_list, spw_list, field, phasecenter, out_dir, imsize,
                      cell_arcsec, start_kms, width_kms, nchan, restfreq_ghz,
                      weighting="natural", psf_factor=2, log_name="tclean_dirty_cube.log"):
    """
    Same step as `grid_psf_and_dirty` -- MS -> PSF cube + dirty cube, once,
    no deconvolution -- for single-pointing data, via `tclean(niter=0)` so
    several MSs (e.g. the execution blocks of one array configuration) grid
    onto one cube. Standard gridder, LSRK radio-velocity channels.

    Writes `out_dir`/dirty_cube.fits (Jy/beam, `imsize` x `imsize`) and
    dirty_beam_cube.fits (peak 1 per channel, `psf_factor` x larger, same
    centre -- see uv_to_image.ShiftInvariantOperator for why the PSF must be
    twice the image: an image-sized PSF makes the normal operator
    indefinite). Both come from one tclean run at the larger size, the dirty
    cube cropped to its central `imsize` (a dirty pixel does not depend on
    the image size). Returns their paths plus the clean beam CASA fitted to
    the PSF main lobe (per-channel median, see `median_beam`).
    Module-level and file-in/file-out so it can run in a worker process;
    `log_name` is its CASA log in casa_logs/.
    """
    import glob
    import shutil
    import casatools
    from casatasks import tclean
    from casa_logs import redirect_casa_logs

    os.makedirs(out_dir, exist_ok=True)
    redirect_casa_logs(log_name)
    base = os.path.join(out_dir, "_tclean_cube")
    for p in glob.glob(base + ".*"):
        shutil.rmtree(p)
    tclean(vis=list(ms_list), field=field, spw=list(spw_list), datacolumn="data",
           imagename=base, imsize=psf_factor * imsize, cell=f"{cell_arcsec}arcsec",
           phasecenter=phasecenter, specmode="cube", start=f"{start_kms}km/s",
           width=f"{width_kms}km/s", nchan=nchan, restfreq=f"{restfreq_ghz}GHz",
           outframe="LSRK", veltype="radio", gridder="standard",
           weighting=weighting, niter=0, restoration=False, pbcor=False)

    ia = casatools.image()
    ia.open(base + ".psf")
    beam = median_beam(ia.restoringbeam())
    ia.close()

    out = dict(dirty=os.path.join(out_dir, "dirty_cube.fits"),
               psf=os.path.join(out_dir, "dirty_beam_cube.fits"), beam=beam)
    # niter=0: the residual IS the dirty cube; crop it to the central imsize
    # (refpix psf_factor*imsize/2 -> imsize/2: the same grid as imaging at imsize)
    lo = (psf_factor - 1) * imsize // 2
    ia.open(base + ".residual")
    dirty = ia.subimage(region=casatools.regionmanager().box(blc=[lo, lo], trc=[lo + imsize - 1] * 2))
    dirty.setbrightnessunit("Jy/beam")
    dirty.tofits(out["dirty"], overwrite=True)
    dirty.done()
    ia.close()
    ia.open(base + ".psf")
    ia.tofits(out["psf"], overwrite=True)
    ia.close()
    for p in glob.glob(base + ".*"):
        shutil.rmtree(p)
    return out


if __name__ == "__main__":
    grid_psf_and_dirty()
