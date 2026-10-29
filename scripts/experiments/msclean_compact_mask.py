#!/usr/bin/env python
"""
Continue NGC 3110's finished extended MS-CLEAN (data/ngc3110/extended/msclean_complete, = results/NGC3110/extended/
msclean_finished) inside its own mask OR the compact model's footprint (compact_only_residual.compact_footprint): the
mask only allows components, so CLEAN still models only what the extended residual holds above nsigma.

Null test: each line-free channel gets the footprint of a line channel (cycling through them), so whatever CLEAN puts
there is noise it would also pick up inside the footprint.

Writes results/NGC3110/extended/<out>/{image,model,residual,psf,mask,emission}.fits, mask.npy, run.json. CASA env:

    python scripts/experiments/msclean_compact_mask.py --nsigma 3 --out msclean_cmask_n3
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("scripts", "simple", "scripts/experiments"):
    sys.path.insert(0, os.path.join(REPO, p))
from msclean_cube import GALAXIES, IMSIZE, CELL_ARCSEC, RESTFREQ_GHZ, NCHAN, WIDTH_KMS, WEIGHTING, phasecenter_string, line_setup  # noqa: E402
from casa_logs import redirect_casa_logs  # noqa: E402
from compact_only_residual import compact_footprint  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--nsigma", type=float, default=3.0); ap.add_argument("--out", required=True)
ap.add_argument("--scales", type=int, nargs="+", default=[0, 10, 30, 60])
ap.add_argument("--niter", type=int, default=1_000_000)
a = ap.parse_args()

import casatools  # noqa: E402
from casatasks import tclean, exportfits  # noqa: E402

GAL = os.path.join(REPO, "results", "NGC3110")
SRC = os.path.join(REPO, "data", "ngc3110", "extended", "msclean_complete")
out = os.path.join(GAL, "extended", a.out); base = os.path.join(out, "_tclean")
redirect_casa_logs(f"msclean_ngc3110_extended_{a.out}.log")
os.makedirs(out, exist_ok=True)
for p in glob.glob(base + ".*"):
    shutil.rmtree(p)
for p in glob.glob(os.path.join(SRC, "_tclean.*")):
    shutil.copytree(p, os.path.join(out, os.path.basename(p)))

# the tclean keywords of msclean_complete.main (same phase centre string and spw selection as the run continued)
g = GALAXIES["ngc3110"]; paths = g["configs"]["extended"]; ref = paths[0]
_, vsys = line_setup(ref, g["field"])
common = dict(vis=list(paths), field=g["field"], spw=[str(line_setup(p, g["field"])[0]) for p in paths],
              datacolumn="data", imagename=base, imsize=IMSIZE, cell=f"{CELL_ARCSEC}arcsec",
              phasecenter=phasecenter_string(ref, g["field"]), specmode="cube",
              start=f"{vsys - (NCHAN - 1) / 2 * WIDTH_KMS}km/s", width=f"{WIDTH_KMS}km/s", nchan=NCHAN,
              restfreq=f"{RESTFREQ_GHZ}GHz", outframe="LSRK", veltype="radio", gridder="standard", weighting=WEIGHTING,
              deconvolver="multiscale", scales=list(a.scales), restoringbeam="common", pbcor=False, fullsummary=True)

L = json.load(open(os.path.join(GAL, "line_channels.json")))["line"]
free = [c for c in range(NCHAN) if c not in L]
E0 = np.load(os.path.join(GAL, "extended", "msclean_finished", "mask_final.npy")) > 0
C = compact_footprint(GAL)
mask = E0.copy()
mask[L] |= C[L]
for i, c in enumerate(free):                   # null: a line channel's footprint in each line-free channel
    mask[c] |= C[L[i % len(L)]]
np.save(os.path.join(out, "mask.npy"), mask.astype(np.uint8))
print(f"mask: {mask[L].mean():.2%} of line-channel voxels (original {E0[L].mean():.2%}), {mask[free].mean():.2%} of "
      f"line-free; nsigma {a.nsigma}, scales {a.scales} px -> {out}", flush=True)

ia = casatools.image()
ia.open(f"{base}.image"); cs = ia.coordsys(); ia.close()
for p in (f"{base}.usermask", f"{base}.mask"):
    if os.path.exists(p):
        shutil.rmtree(p)
ia.fromarray(outfile=f"{base}.usermask", pixels=mask.astype(np.float32).transpose(2, 1, 0)[:, :, None, :],
             csys=cs.torecord(), overwrite=True); ia.close()
t0 = time.time()
rec = tclean(niter=a.niter, nsigma=a.nsigma, usemask="user", mask=f"{base}.usermask", calcpsf=False, calcres=False,
             **common)
run = dict(nsigma=a.nsigma, scales_px=a.scales, src=os.path.relpath(SRC, REPO), stopcode=rec.get("stopcode"),
           iterations=rec.get("iterdone"), major_cycles=rec.get("nmajordone"), seconds=round(time.time() - t0),
           mask_line=float(mask[L].mean()), mask_free=float(mask[free].mean()))
print(run, flush=True)
json.dump(run, open(os.path.join(out, "run.json"), "w"), indent=1)
for ext in ("image", "model", "residual", "psf", "mask"):
    exportfits(imagename=f"{base}.{ext}", fitsimage=os.path.join(out, f"{ext}.fits"), velocity=True, overwrite=True)
from astropy.io import fits  # noqa: E402
with fits.open(os.path.join(out, "image.fits")) as hi, fits.open(os.path.join(out, "residual.fits")) as hr:
    hdr = hi[0].header.copy(); hdr["HISTORY"] = "emission only: image - residual = CLEAN components * clean beam"
    fits.writeto(os.path.join(out, "emission.fits"), (hi[0].data - hr[0].data).astype(np.float32), hdr, overwrite=True)
print("DONE", flush=True)
