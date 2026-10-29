#!/usr/bin/env python
"""
Joint compact + extended MS-CLEAN of NGC 3110 without a uv taper (natural weighting), on the paper's spectral grid
(32 x 25 km/s LSRK from 4686.4 km/s: the pre-averaged MSs of regrid_ms.py NCHAN=34 WIDTH_KMS=25), 0.04" pixels, 51"
field. Multiscale (0, 1, 3, 6 x the beam), auto-multithresh (short-baseline parameters) at nsigma 3, then finished
inside the growing astrodendro mask at nsigma 1.5 (msclean_complete.finish); restored with the common beam. Works on
/scratch (tclean rewrites its images every major cycle), FITS copied to results/NGC3110/joint_natural/.

    OMP_NUM_THREADS=32 /scratch/alahiry/conda/envs/deconvolver2d1d/bin/python scripts/experiments/joint_natural.py
"""
import glob
import json
import os
import shutil
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("scripts", "simple"):
    sys.path.insert(0, os.path.join(REPO, p))
from casa_logs import redirect_casa_logs  # noqa: E402
redirect_casa_logs("joint_natural_NGC3110.log")
from casatasks import tclean, exportfits  # noqa: E402
from msclean_complete import finish  # noqa: E402

ROBUST = os.environ.get("ROBUST")                                      # Briggs robust; unset: natural weighting
CFG = os.environ.get("CONFIGS", "joint")                               # "extended": the extended visibilities alone
G20 = os.environ.get("GRID") == "20kms"                                # the 64 x 20 km/s grid of the ladder
NAME = (f"{CFG}_natural" if ROBUST is None else f"{CFG}_robust{ROBUST}") + ("_20kms" if G20 else "")
R = os.path.join(REPO, "results/NGC3110"); OUT = os.path.join(R, NAME); os.makedirs(OUT, exist_ok=True)
WK = f"/scratch/alahiry/{NAME}"; os.makedirs(WK, exist_ok=True)
GC = json.load(open(os.path.join(R, "config.json")))
RG = json.load(open(f"/scratch/alahiry/regrid/NGC3110_{'64x20' if G20 else '34x25'}kms/ms.json"))
L = json.load(open(os.path.join(R, "line_channels.json")))["line"]
if G20:
    _g = json.load(open(os.path.join(R, "consistency_20kms_0.08/grid.json")))
    L = _g["line"]; GC.update(start_kms=_g["start_kms"], width_kms=_g["width_kms"], nchan=_g["nchan"])
CELL, IMSIZE = 0.04, 1280
GEO = {None: 0.70, "0": 0.176, "-0.5": 0.156}[ROBUST] if CFG == "joint" else 0.195   # beam geometric mean, untapered
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
KW = dict(vis=RG["ms"]["extended"] if CFG == "extended" else RG["ms"]["compact"] + RG["ms"]["extended"], spw=RG["spw"], datacolumn="data", imsize=IMSIZE,
          cell=f"{CELL}arcsec", phasecenter=GC["phasecenter"], specmode="cube", start=f"{GC['start_kms']}km/s",
          width=f"{GC['width_kms']}km/s", nchan=GC["nchan"], restfreq=f"{GC['restfreq_ghz']}GHz", outframe="LSRK",
          veltype="radio", gridder="standard", deconvolver="multiscale",
          **(dict(weighting="natural") if ROBUST is None else dict(weighting="briggs", robust=float(ROBUST))),
          scales=[int(round(f * GEO / CELL)) for f in (0, 1, 3, 6)], pblimit=-0.05, pbcor=False, restoringbeam="common",
          fullsummary=True)
os.chdir(WK)                                                            # tclean's weight scratch tables go here
c1, c2 = os.path.join(WK, "clean"), os.path.join(WK, "clean_finished")
if not os.path.exists(os.path.join(WK, "clean.done")):
    os.makedirs(c1, exist_ok=True); base = os.path.join(c1, "_tclean")
    for p in glob.glob(base + ".*"):
        shutil.rmtree(p)
    t0 = time.time()
    AM = (dict(sidelobethreshold=2.0, noisethreshold=4.25, lownoisethreshold=1.5, negativethreshold=15.0, minbeamfrac=0.3)
          if GEO > 0.5 else                                              # short- / long-baseline auto-multithresh
          dict(sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=7.0, minbeamfrac=0.3))
    rec = tclean(imagename=base, niter=1_000_000, nsigma=3.0, usemask="auto-multithresh", growiterations=75, **AM, **KW)
    log(f"first pass: stopcode {rec.get('stopcode')}, {rec.get('iterdone')} iterations, {time.time() - t0:.0f} s")
    open(os.path.join(WK, "clean.done"), "w").write(time.strftime("%F %H:%M"))
if not os.path.exists(os.path.join(WK, "finish.done")):
    t0 = time.time(); finish(c1, c2, KW, L, CELL, nsigma=1.5, label=f"NGC3110 {NAME}")
    log(f"finished: {time.time() - t0:.0f} s"); open(os.path.join(WK, "finish.done"), "w").write(time.strftime("%F %H:%M"))
RB = os.environ.get("RESTORE_EXTENDED") == "1"   # restore with the extended clean beam (no taper: the PSF is a bit finer)
if RB:
    from astropy.io import fits as _f
    h = _f.getheader(os.path.join(R, "extended/msclean_finished/image.fits"))
    BE = [f"{h['BMAJ'] * 3600}arcsec", f"{h['BMIN'] * 3600}arcsec", f"{h['BPA']}deg"]
    base = os.path.join(c2, "_tclean")                                 # restoration only, on the finished run
    tclean(**dict(KW, imagename=base, restoringbeam=BE), niter=0, calcres=False, calcpsf=False, restoration=True)
    exportfits(imagename=f"{base}.image", fitsimage=os.path.join(c2, "image.fits"), velocity=True, overwrite=True)
    log(f"restored with the extended clean beam {BE}")
for f in glob.glob(os.path.join(c2, "*.fits")) + glob.glob(os.path.join(c2, "*.npy")) + glob.glob(os.path.join(c2, "*.json")):
    shutil.copy(f, OUT)
json.dump(dict(weighting="natural" if ROBUST is None else f"briggs {ROBUST}", uvtaper=None,
               restoring_beam="extended clean beam" if RB else "common (fitted PSF)", cell=CELL, imsize=IMSIZE, scales_px=KW["scales"], ms=KW["vis"],
               start_kms=GC["start_kms"], width_kms=GC["width_kms"], nchan=GC["nchan"]),
          open(os.path.join(OUT, "info.json"), "w"), indent=1)
log("DONE")
