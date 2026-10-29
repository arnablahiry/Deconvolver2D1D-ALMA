#!/usr/bin/env python
"""
Finish the tapered extended MS-CLEAN (data/ngc3110/extended/big_taper_finished, which stalls at ~9x its noise because its
own dendrogram mask cannot grow past the tapered data's sidelobes) inside the COMPACT MS-CLEAN's mask
(compact/big_finished/mask_final.npy, same 51" / 0.04" grid). The mask only says where CLEAN may place components;
every flux comes from the extended visibilities. Same tclean setup as msclean_bigfield.py --taper (uvtaper to the
compact beam, multiscale 0/30/90 px, natural, restored with the compact clean beam), continued from the stalled run,
nsigma 1.5, then exported with emission = image - residual.

    /scratch/alahiry/conda/envs/deconvolver2d1d/bin/python scripts/experiments/msclean_taper_compactmask.py
    -> data/ngc3110/extended/big_taper_cmask/{image,model,residual,psf,mask,emission}.fits, run.json
"""
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("scripts", "simple"):
    sys.path.insert(0, os.path.join(REPO, p))
from msclean_cube import GALAXIES, RESTFREQ_GHZ, NCHAN, WIDTH_KMS, phasecenter_string, line_setup  # noqa: E402
from casa_logs import redirect_casa_logs  # noqa: E402

redirect_casa_logs("msclean_ngc3110_extended_big_taper_cmask.log")
import casatools  # noqa: E402
from casatasks import tclean, exportfits  # noqa: E402
from astropy.io import fits  # noqa: E402

D_ = os.path.join(REPO, "data/ngc3110")
# THRESH_MJY: an absolute stopping threshold (tclean's nsigma uses each channel's residual rms, which the uncleaned
# emission's sidelobes inflate, so it stops at ~6x the real noise); SRC/OUT to continue a previous run
THRESH_MJY = float(os.environ.get("THRESH_MJY", 0))
SRC = os.path.join(D_, os.environ.get("SRC", "extended/big_taper_finished"))
OUT = os.path.join(D_, os.environ.get("OUT", "extended/big_taper_cmask"))
os.makedirs(OUT, exist_ok=True)
base = os.path.join(OUT, "_tclean")
for p in glob.glob(base + ".*"):
    shutil.rmtree(p)
for p in glob.glob(os.path.join(SRC, "_tclean.*")):
    if not p.endswith((".mask", ".usermask")) and os.path.isdir(p):
        shutil.copytree(p, os.path.join(OUT, os.path.basename(p)))
setup = json.load(open(os.path.join(D_, "extended/big_taper/setup.json")))
g = GALAXIES["ngc3110"]; paths = g["configs"]["extended"]; ref = paths[0]
_, vsys = line_setup(ref, g["field"])
common = dict(vis=list(paths), field=g["field"], spw=[str(line_setup(p, g["field"])[0]) for p in paths], datacolumn="data",
              imsize=setup["imsize"], cell=f"{setup['cell']}arcsec", phasecenter=phasecenter_string(ref, g["field"]),
              specmode="cube", start=f"{vsys - (NCHAN - 1) / 2 * WIDTH_KMS}km/s", width=f"{WIDTH_KMS}km/s", nchan=NCHAN,
              restfreq=f"{RESTFREQ_GHZ}GHz", outframe="LSRK", veltype="radio", gridder="standard", weighting="natural",
              deconvolver="multiscale", scales=setup["scales"], pblimit=-0.05, pbcor=False, fullsummary=True,
              restoringbeam=setup["restoringbeam"], uvtaper=setup["uvtaper"])
ia = casatools.image()
ia.open(base + ".image"); cs = ia.coordsys().torecord(); ia.close()
m = np.load(os.path.join(D_, "compact/big_finished/mask_final.npy")).astype(np.float32)        # (chan, y, x)
ia.fromarray(outfile=base + ".usermask", pixels=m.transpose(2, 1, 0)[:, :, None, :], csys=cs, overwrite=True); ia.close()
print(f"compact mask: {int(m.sum())} voxels; continuing {SRC} with nsigma 1.5", flush=True)
t0 = time.time()
stop = dict(threshold=f"{THRESH_MJY}mJy", nsigma=0.0) if THRESH_MJY else dict(nsigma=1.5)
print("stopping:", stop, flush=True)
rec = tclean(imagename=base, niter=1_000_000, **stop, usemask="user", mask=base + ".usermask", calcpsf=False,
             calcres=False, **common)
print(f"tclean: stopcode {rec.get('stopcode')}, {rec.get('iterdone')} iterations, {rec.get('nmajordone')} major cycles, "
      f"{time.time() - t0:.0f} s", flush=True)
for ext in ("image", "model", "residual", "psf", "mask"):
    exportfits(imagename=f"{base}.{ext}", fitsimage=os.path.join(OUT, f"{ext}.fits"), velocity=True, overwrite=True)
with fits.open(os.path.join(OUT, "image.fits")) as hi, fits.open(os.path.join(OUT, "residual.fits")) as hr:
    fits.writeto(os.path.join(OUT, "emission.fits"), (hi[0].data - hr[0].data).astype(np.float32), hi[0].header, overwrite=True)
json.dump(dict(stopcode=rec.get("stopcode"), iterations=rec.get("iterdone"), major_cycles=rec.get("nmajordone"),
               mask="compact/big_finished/mask_final.npy", stopping=stop, source=SRC, seconds=round(time.time() - t0)),
          open(os.path.join(OUT, "run.json"), "w"), indent=1)
print("DONE", flush=True)
