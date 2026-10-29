#!/usr/bin/env python
"""
The inputs of the lambda/B_max version of the NGC 3110 compact / extended / joint consistency figure
(fig_consistency_overview3.py BEAM=lbmax), on the 64 x 20 km/s, 0.08" (640 px) grid of consistency_20kms_0.08:

  1. a circular uv taper of the EXTENDED visibilities alone (natural weighting) whose PSF matches the compact
     array's lambda/B_max beam (0.604" circular): secant search until the PSF's geometric mean agrees to 2%
     -> results/NGC3110/consistency_20kms_lbmax/ext_taper.json (uvtaper, target_beam), read by consistency_target.py
        (TAPER_JSON, TARGET_RESTORE=1) for the tapered extended dirty cube and MS-CLEAN
  2. the dirty cube and PSF of the JOINT visibilities with the weighting and taper of the lambda/B_max step of the
     resolution ladder (ladder_joint_20kms/b05/info.json: robust 0.5 + its taper) -> dirty_joint_lbmax/{dirty,psf}.fits

    /scratch/alahiry/conda/envs/deconvolver2d1d/bin/python scripts/experiments/lbmax_extended_taper.py
"""
import glob
import json
import os
import shutil
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
from casa_logs import redirect_casa_logs  # noqa: E402
redirect_casa_logs("lbmax_extended_taper.log")
import casatools  # noqa: E402
from casatasks import tclean, exportfits  # noqa: E402

R = os.path.join(REPO, "results", "NGC3110"); OUT = os.path.join(R, "consistency_20kms_lbmax"); os.makedirs(OUT, exist_ok=True)
WK = "/scratch/alahiry/consistency_lbmax"; os.makedirs(WK, exist_ok=True)
GC = json.load(open(os.path.join(R, "config.json"))); G = json.load(open(os.path.join(R, "consistency_20kms_0.08/grid.json")))
RG = json.load(open("/scratch/alahiry/regrid/NGC3110_64x20kms/ms.json"))
LB = float(GC["lbmax_compact_arcsec"]); TGT = (LB, LB, 0.0)
KW = dict(spw=RG["spw"], datacolumn="data", imsize=640, cell="0.08arcsec", phasecenter=GC["phasecenter"], specmode="cube",
          start=f"{G['start_kms']}km/s", width=f"{G['width_kms']}km/s", nchan=G["nchan"], restfreq=f"{GC['restfreq_ghz']}GHz",
          outframe="LSRK", veltype="radio", gridder="standard", pblimit=-0.05, pbcor=False)
qa, ia = casatools.quanta(), casatools.image()
tstr = lambda t: [f"{t[0]:.4f}arcsec", f"{t[1]:.4f}arcsec", f"{t[2]:.2f}deg"]


def psf_beam(vis, taper, tag, **w):
    base = os.path.join(WK, tag)
    for p in glob.glob(base + ".*"):
        shutil.rmtree(p)
    v0 = G["start_kms"] + G["width_kms"] * (G["nchan"] // 2 - 1)
    tclean(vis=vis, imagename=base, **dict(KW, nchan=1, start=f"{v0}km/s"), niter=0, restoration=False, calcres=False,
           uvtaper=tstr(taper) if taper else [], **w)
    ia.open(base + ".psf"); b = ia.restoringbeam(); ia.close()
    b = b["beams"]["*0"]["*0"] if "beams" in b else b
    return tuple(qa.convert(b[k], u)["value"] for k, u in (("major", "arcsec"), ("minor", "arcsec"), ("positionangle", "deg")))


# ======================================================================== 1. extended alone, natural, tapered to lambda/B_max
TJ = os.path.join(OUT, "ext_taper.json")
if not os.path.exists(TJ):
    EXT = RG["ms"]["extended"]; P0 = psf_beam(EXT, None, "ext_nat", weighting="natural")
    geo = lambda P: float(np.sqrt(P[0] * P[1])); tries = []     # one circular taper: its PSF's geometric mean = LB
    run = lambda t, i: (lambda P: (tries.append(dict(taper=[t, t, 0.0], psf=P)), P)[1])(
        psf_beam(EXT, (t, t, 0.0), f"ext_t{i}", weighting="natural"))
    t0, t1 = 0.40, 0.56; f0, f1 = geo(run(t0, 0)) - LB, geo(run(t1, 1)) - LB
    for it in range(2, 9):                                       # secant on the circular taper FWHM
        t2 = float(np.clip(t1 - f1 * (t1 - t0) / (f1 - f0), 0.05, 2.0)); P = run(t2, it); f2 = geo(P) - LB
        print(f"try {it}: taper {t2:.3f} -> PSF {P[0]:.3f} x {P[1]:.3f} (geo {geo(P):.3f}, target {LB:.3f})", flush=True)
        t0, f0, t1, f1 = t1, f1, t2, f2
        if abs(f2 / LB) < 0.02:
            break
    taper = (t1, t1, 0.0)
    json.dump(dict(uvtaper=tstr(taper), target_beam=list(TGT), psf=list(P), psf_natural=list(P0), tries=tries),
              open(TJ, "w"), indent=1)
print("extended taper:", json.load(open(TJ))["uvtaper"], "PSF", json.load(open(TJ))["psf"], flush=True)

# ======================================================================== 2. joint dirty at the ladder's lambda/B_max weighting
DJ = os.path.join(OUT, "dirty_joint_lbmax")
if not os.path.exists(os.path.join(DJ, "dirty.fits")):
    os.makedirs(DJ, exist_ok=True)
    info = json.load(open(os.path.join(R, "ladder_joint_20kms/b05/info.json")))
    base = os.path.join(WK, "joint_dirty")
    for p in glob.glob(base + ".*"):
        shutil.rmtree(p)
    tclean(vis=RG["ms"]["compact"] + RG["ms"]["extended"], imagename=base, **KW, niter=0, restoration=False,
           weighting="briggs", robust=info["robust"], uvtaper=tstr(info["uvtaper"]))
    exportfits(imagename=base + ".residual", fitsimage=os.path.join(DJ, "dirty.fits"), velocity=True, overwrite=True)
    exportfits(imagename=base + ".psf", fitsimage=os.path.join(DJ, "psf.fits"), velocity=True, overwrite=True)
    json.dump(dict(robust=info["robust"], uvtaper=tstr(info["uvtaper"]), beam=info["beam"], psf=info["psf"]),
              open(os.path.join(DJ, "setup.json"), "w"), indent=1)
print("DONE", flush=True)
