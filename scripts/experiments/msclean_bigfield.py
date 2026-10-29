#!/usr/bin/env python
"""
NGC 3110 multiscale CLEAN on a field that holds all the emission (the 16" field of msclean_cube.py misses ~40% of the
compact line emission, whose sidelobes then stall CLEAN), natural weighting, the channels and phase centre of
msclean_cube.py, no primary-beam cut-off on the image (pblimit < 0); tclean with auto-multithresh (ALMA pipeline
values: short-baseline for compact and tapered runs, long-baseline for the full-resolution extended), then
msclean_complete.finish inside the growing astrodendro mask to 1.5 sigma.

    python scripts/experiments/msclean_bigfield.py compact                 # 0.04" x 1280 px, scales 0 30 90
    python scripts/experiments/msclean_bigfield.py extended --taper        # same grid, the compact-beam taper
                                                                           #   (msclean_taper/taper.json), restored
                                                                           #   with exactly the compact clean beam
    python scripts/experiments/msclean_bigfield.py extended                # 0.02" x 2560 px, scales 0 10 30 60 120
    ... [--finish-only]                                                    # only the finishing, from <name>/_tclean.*

Writes data/ngc3110/<config>/big_<name>{,_finished}/ (FITS as msclean_cube / msclean_complete). CASA env.
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("scripts", "simple"):
    sys.path.insert(0, os.path.join(REPO, p))
from msclean_cube import GALAXIES, RESTFREQ_GHZ, NCHAN, WIDTH_KMS, phasecenter_string, line_setup  # noqa: E402
from casa_logs import redirect_casa_logs  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("config", choices=("compact", "extended"))
ap.add_argument("--taper", action="store_true", help="extended only: taper to the compact beam")
ap.add_argument("--imsize", type=int); ap.add_argument("--cell", type=float)
ap.add_argument("--scales", type=int, nargs="+")
ap.add_argument("--finish-only", action="store_true", help="skip tclean: rerun only the finishing from <name>/")
a = ap.parse_args()
lowres = a.config == "compact" or a.taper
cell = a.cell or (0.04 if lowres else 0.02); imsize = a.imsize or (1280 if lowres else 2560)
scales = a.scales or ([0, 30, 90] if lowres else [0, 10, 30, 60, 120])
name = "big_taper" if a.taper else "big"
redirect_casa_logs(f"msclean_ngc3110_{a.config}_{name}.log")
from casatasks import tclean, exportfits  # noqa: E402
from astropy.io import fits  # noqa: E402

OUT = os.path.join(REPO, "data/ngc3110", a.config, name); os.makedirs(OUT, exist_ok=True)
base = os.path.join(OUT, "_tclean")
for p in ([] if a.finish_only else glob.glob(base + ".*")):
    shutil.rmtree(p)
g = GALAXIES["ngc3110"]; paths = g["configs"][a.config]; ref = g["configs"]["extended"][0]
_, vsys = line_setup(ref, g["field"])
common = dict(vis=list(paths), field=g["field"], spw=[str(line_setup(p, g["field"])[0]) for p in paths],
              datacolumn="data", imsize=imsize, cell=f"{cell}arcsec", phasecenter=phasecenter_string(ref, g["field"]),
              specmode="cube", start=f"{vsys - (NCHAN - 1) / 2 * WIDTH_KMS}km/s", width=f"{WIDTH_KMS}km/s", nchan=NCHAN,
              restfreq=f"{RESTFREQ_GHZ}GHz", outframe="LSRK", veltype="radio", gridder="standard", weighting="natural",
              deconvolver="multiscale", scales=scales, pblimit=-0.05, pbcor=False, fullsummary=True,
              restoringbeam="common")
if a.taper:
    common["uvtaper"] = json.load(open(os.path.join(REPO, "data/ngc3110/extended/msclean_taper/taper.json")))["uvtaper"]
    h = fits.getheader(os.path.join(REPO, "results/NGC3110/compact/msclean/image.fits"))
    common["restoringbeam"] = [f"{h['BMAJ'] * 3600}arcsec", f"{h['BMIN'] * 3600}arcsec", f"{h['BPA']}deg"]
automask = (dict(sidelobethreshold=2.0, noisethreshold=4.25, lownoisethreshold=1.5, negativethreshold=15.0) if lowres else
            dict(sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=7.0))
print(f"{a.config} {name}: {imsize} px x {cell}\" = {imsize * cell:.0f}\", scales {scales}, taper {common.get('uvtaper')}, "
      f"restoring beam {common['restoringbeam']}", flush=True)
json.dump(dict(imsize=imsize, cell=cell, scales=scales, uvtaper=common.get("uvtaper"), restoringbeam=common["restoringbeam"],
               automask=automask), open(os.path.join(OUT, "setup.json"), "w"), indent=1)
if not a.finish_only:
    rec = tclean(imagename=base, niter=1_000_000, nsigma=3.0, usemask="auto-multithresh", minbeamfrac=0.3, growiterations=75,
                 **automask, **common)
    print(f"tclean: stopcode {rec.get('stopcode')}, {rec.get('iterdone')} iterations, {rec.get('nmajordone')} major cycles", flush=True)
    for ext in ("image", "model", "residual", "psf", "mask"):
        exportfits(imagename=f"{base}.{ext}", fitsimage=os.path.join(OUT, f"{ext}.fits"), velocity=True, overwrite=True)
from msclean_complete import finish  # noqa: E402
vel = vsys + WIDTH_KMS * (np.arange(NCHAN) - (NCHAN - 1) / 2)
line = [i for i, v in enumerate(vel) if 4750 <= v <= 5280]
finish(OUT, OUT + "_finished", common, line, cell, nsigma=1.5, label=f"ngc3110 {a.config} {name}")
print("DONE", flush=True)
