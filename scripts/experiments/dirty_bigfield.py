#!/usr/bin/env python
"""
NGC 3110 dirty cube and PSF of one configuration on a field much larger than the 12 m primary beam (default 2048 px
at 0.04" = 82"; FWHM of the primary beam ~26" at 227 GHz), natural weighting, the channels and phase centre of
msclean_cube.py, no CLEAN, no taper. For the model-free cross-convolution test (cross_convolution.py): emission
outside the old 16" field and its sidelobes are then inside the image.

    python scripts/experiments/dirty_bigfield.py compact|extended [--imsize 2048] [--cell 0.04] [--taper]
    -> data/ngc3110/<config>/dirty_big[_taper]/{dirty,psf}.fits   (Jy per dirty beam; PSF peak 1, image-sized)
--taper (extended): the uv taper of msclean_taper/taper.json (the compact-beam match), on the same grid.
"""
import argparse
import glob
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("scripts", "simple"):
    sys.path.insert(0, os.path.join(REPO, p))
from msclean_cube import GALAXIES, RESTFREQ_GHZ, NCHAN, WIDTH_KMS, phasecenter_string, line_setup  # noqa: E402
from casa_logs import redirect_casa_logs  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("config", choices=("compact", "extended"))
ap.add_argument("--imsize", type=int, default=2048); ap.add_argument("--cell", type=float, default=0.04)
ap.add_argument("--taper", action="store_true")
a = ap.parse_args()
name = "dirty_big_taper" if a.taper else "dirty_big"
redirect_casa_logs(f"{name}_ngc3110_{a.config}.log")
from casatasks import tclean, exportfits  # noqa: E402

OUT = os.path.join(REPO, "data/ngc3110", a.config, name); os.makedirs(OUT, exist_ok=True)
extra = {}
if a.taper:
    import json
    extra["uvtaper"] = json.load(open(os.path.join(REPO, "data/ngc3110/extended/msclean_taper/taper.json")))["uvtaper"]
base = os.path.join(OUT, "_tclean")
for p in glob.glob(base + ".*"):
    shutil.rmtree(p)
g = GALAXIES["ngc3110"]; paths = g["configs"][a.config]; ref = g["configs"]["extended"][0]
_, vsys = line_setup(ref, g["field"])
tclean(vis=list(paths), field=g["field"], spw=[str(line_setup(p, g["field"])[0]) for p in paths], datacolumn="data",
       imagename=base, imsize=a.imsize, cell=f"{a.cell}arcsec", phasecenter=phasecenter_string(ref, g["field"]),
       specmode="cube", start=f"{vsys - (NCHAN - 1) / 2 * WIDTH_KMS}km/s", width=f"{WIDTH_KMS}km/s", nchan=NCHAN,
       restfreq=f"{RESTFREQ_GHZ}GHz", outframe="LSRK", veltype="radio", gridder="standard", weighting="natural",
       niter=0, pbcor=False, restoration=False, pblimit=-0.001, **extra)
exportfits(imagename=base + ".residual", fitsimage=os.path.join(OUT, "dirty.fits"), velocity=True, overwrite=True)
exportfits(imagename=base + ".psf", fitsimage=os.path.join(OUT, "psf.fits"), velocity=True, overwrite=True)
for p in glob.glob(base + ".*"):
    shutil.rmtree(p)
print(f"{a.config}: {a.imsize} px x {a.cell}\" -> {OUT}", flush=True)
