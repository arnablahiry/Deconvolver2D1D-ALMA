#!/usr/bin/env python
"""
The tapered extended dirty cube of NGC 3110: the extended data, natural weighting, the uv taper of
msclean_extended_taper.py (data/ngc3110/extended/msclean_taper/taper.json, matched to the compact clean beam), no
CLEAN. With its PSF this is the measured data at the compact resolution, to be compared with a model * that PSF
(forward model) rather than with a CLEAN image.

Writes data/ngc3110/extended/dirty_taper/{dirty,psf}.fits (Jy per dirty beam; PSF peak 1). CASA env:

    python scripts/experiments/dirty_extended_taper.py
"""
import glob
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("scripts", "simple"):
    sys.path.insert(0, os.path.join(REPO, p))
from msclean_cube import GALAXIES, IMSIZE, CELL_ARCSEC, RESTFREQ_GHZ, NCHAN, WIDTH_KMS, phasecenter_string, line_setup  # noqa: E402
from casa_logs import redirect_casa_logs  # noqa: E402

redirect_casa_logs("dirty_ngc3110_extended_taper.log")
from casatasks import tclean, exportfits  # noqa: E402

TAPER = json.load(open(os.path.join(REPO, "data/ngc3110/extended/msclean_taper/taper.json")))["uvtaper"]
OUT = os.path.join(REPO, "data/ngc3110/extended/dirty_taper"); os.makedirs(OUT, exist_ok=True)
base = os.path.join(OUT, "_tclean")
for p in glob.glob(base + ".*"):
    shutil.rmtree(p)
g = GALAXIES["ngc3110"]; paths = g["configs"]["extended"]; ref = paths[0]
_, vsys = line_setup(ref, g["field"])
tclean(vis=list(paths), field=g["field"], spw=[str(line_setup(p, g["field"])[0]) for p in paths], datacolumn="data",
       imagename=base, imsize=IMSIZE, cell=f"{CELL_ARCSEC}arcsec", phasecenter=phasecenter_string(ref, g["field"]),
       specmode="cube", start=f"{vsys - (NCHAN - 1) / 2 * WIDTH_KMS}km/s", width=f"{WIDTH_KMS}km/s", nchan=NCHAN,
       restfreq=f"{RESTFREQ_GHZ}GHz", outframe="LSRK", veltype="radio", gridder="standard", weighting="natural",
       uvtaper=TAPER, niter=0, pbcor=False, restoration=False)
exportfits(imagename=base + ".residual", fitsimage=os.path.join(OUT, "dirty.fits"), velocity=True, overwrite=True)
exportfits(imagename=base + ".psf", fitsimage=os.path.join(OUT, "psf.fits"), velocity=True, overwrite=True)
print("uvtaper", TAPER, "-> FITS in", OUT, flush=True)
