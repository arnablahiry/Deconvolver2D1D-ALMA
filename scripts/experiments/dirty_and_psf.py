#!/usr/bin/env python
"""
Dirty cube + dirty beam for deconvolution experiments (the 2D-1D solver needs the PSF at 2x the field): one tclean
niter=0 at 2 x IMSIZE on the pre-averaged MSs (regrid_ms.py), natural weighting, the line cube's channels; the dirty cube
cropped to the central IMSIZE, the PSF kept at 2 x IMSIZE (peak 1 per channel). Optionally the uv-tapered extended data.

    CONFIG=compact|extended [TAPER_JSON=...] CELL=0.16 IMSIZE=320 REGRID=/scratch/alahiry/regrid/NGC3110_64x20kms \
        python scripts/experiments/dirty_and_psf.py NGC3110
    -> results/<T>/deconv_inputs/<config>[_taper]_<cell>/{dirty_cube.fits, dirty_beam_cube.fits, info.json}
"""
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
from casa_logs import redirect_casa_logs  # noqa: E402

T = sys.argv[1]; CFG = os.environ.get("CONFIG", "compact")
CELL, IMSIZE = float(os.environ["CELL"]), int(os.environ["IMSIZE"])
RG = json.load(open(os.path.join(os.environ["REGRID"], "ms.json")))
TAP = json.load(open(os.environ["TAPER_JSON"]))["uvtaper"] if os.environ.get("TAPER_JSON") else None
name = f"{CFG}{'_taper' if TAP else ''}_{CELL:g}"
redirect_casa_logs(f"dirty_and_psf_{T}_{name}.log")
import casatools  # noqa: E402
from casatasks import tclean  # noqa: E402

GC = json.load(open(os.path.join(REPO, "results", T, "config.json")))
OUT = os.path.join(REPO, "results", T, "deconv_inputs", name); os.makedirs(OUT, exist_ok=True)
base = os.path.join(OUT, "_tclean")
for p in glob.glob(base + ".*"):
    shutil.rmtree(p)
t0 = time.time()
kw = dict(vis=RG["ms"][CFG], spw=RG["spw"], datacolumn="data", imagename=base, imsize=2 * IMSIZE, cell=f"{CELL}arcsec",
          phasecenter=GC["phasecenter"], specmode="cube", start=f"{RG['start_kms']}km/s", width=f"{RG['width_kms']}km/s",
          nchan=RG["nchan"], restfreq=f"{GC['restfreq_ghz']}GHz", outframe="LSRK", veltype="radio", gridder="standard",
          weighting="natural", niter=0, restoration=False, pbcor=False, pblimit=-0.05)
if TAP:
    kw["uvtaper"] = TAP
tclean(**kw)
ia, rg = casatools.image(), casatools.regionmanager()
lo = IMSIZE // 2
ia.open(base + ".residual")
sub = ia.subimage(region=rg.box(blc=[lo, lo], trc=[lo + IMSIZE - 1, lo + IMSIZE - 1]))
sub.setbrightnessunit("Jy/beam"); sub.tofits(os.path.join(OUT, "dirty_cube.fits"), overwrite=True, velocity=True); sub.done(); ia.close()
ia.open(base + ".psf"); b = ia.restoringbeam(); ia.tofits(os.path.join(OUT, "dirty_beam_cube.fits"), overwrite=True, velocity=True); ia.close()
if "beams" in b:
    bb = b["beams"]; ch = [bb[k]["*0"] for k in sorted(bb, key=lambda s: int(s.strip("*")))]
    b = ch[len(ch) // 2]
qa = casatools.quanta()
beam = [qa.convert(b["major"], "arcsec")["value"], qa.convert(b["minor"], "arcsec")["value"], b["positionangle"]["value"]]
json.dump(dict(target=T, config=CFG, uvtaper=TAP, cell_arcsec=CELL, imsize=IMSIZE, psf_imsize=2 * IMSIZE, nchan=RG["nchan"],
               width_kms=RG["width_kms"], start_kms=RG["start_kms"], phasecenter=GC["phasecenter"], weighting="natural",
               ms=RG["ms"][CFG], fitted_psf_beam_arcsec_deg=beam, seconds=round(time.time() - t0)),
          open(os.path.join(OUT, "info.json"), "w"), indent=1)
for p in glob.glob(base + ".*"):
    shutil.rmtree(p)
print(f"{name}: dirty {IMSIZE} px, PSF {2 * IMSIZE} px, beam {beam[0]:.3f}\" x {beam[1]:.3f}\" @ {beam[2]:.1f}, {time.time() - t0:.0f} s",
      flush=True)
