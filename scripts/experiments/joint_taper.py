#!/usr/bin/env python
"""
Joint compact + extended imaging of NGC 3110 at the compact beam, as the bottom row of the consistency figure:
all three pre-averaged MSs together (regrid_ms.py: 64 x 20 km/s, LSRK), natural weighting, a uv taper that brings the
joint PSF to the compact clean beam, multiscale CLEAN (0, 1.2", 3.6") with the short-baseline auto-multithresh at
nsigma 3, then finished inside the growing astrodendro mask at nsigma 1.5 (msclean_complete.finish), restored with the
compact clean beam. Same grid as the SET folder (CELL, IMSIZE).

  1. joint natural PSF (niter 0)   -> its fitted beam Bj
  2. taper t = sqrt(B_compact^2 - Bj^2) per axis at the compact PA; PSF with the taper; one correction step on t so
     the tapered PSF matches the compact beam (to ~1%)        -> SET/joint_taper.json
  3. dirty cube + PSF                                         -> SET/dirty_joint_taper/{dirty,psf}.fits
  4. MS-CLEAN + finishing                                     -> SET/clean_joint_taper(_finished)/
  5. astrodendro masks on the finished image at >= 12 and >= 16 sigma

    SET=results/NGC3110/consistency_20kms_0.08 CELL=0.08 IMSIZE=640 OMP_NUM_THREADS=16 \
        /scratch/alahiry/conda/envs/deconvolver2d1d/bin/python scripts/experiments/joint_taper.py
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
from casa_logs import redirect_casa_logs  # noqa: E402
redirect_casa_logs("joint_taper_NGC3110.log")
import casatools  # noqa: E402
from casatasks import tclean, exportfits  # noqa: E402
from astropy.io import fits  # noqa: E402
from msclean_complete import finish, dendro_mask  # noqa: E402

SET = os.path.join(REPO, os.environ["SET"]); CELL, IMSIZE = float(os.environ["CELL"]), int(os.environ["IMSIZE"])
RG = json.load(open("/scratch/alahiry/regrid/NGC3110_64x20kms/ms.json"))
GC = json.load(open(os.path.join(REPO, "results/NGC3110/config.json")))
grid = json.load(open(os.path.join(SET, "grid.json"))); L = grid["line"]
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
done = lambda s: os.path.exists(os.path.join(SET, f"joint_{s}.done"))
mark = lambda s: open(os.path.join(SET, f"joint_{s}.done"), "w").write(time.strftime("%Y-%m-%d %H:%M"))
qa, ia = casatools.quanta(), casatools.image()
hC = fits.getheader(os.path.join(SET, "clean_compact_finished/image.fits"))
BC = (hC["BMAJ"] * 3600, hC["BMIN"] * 3600, hC["BPA"])
VIS = RG["ms"]["compact"] + RG["ms"]["extended"]
KW = dict(vis=VIS, spw=RG["spw"], datacolumn="data", imsize=IMSIZE, cell=f"{CELL}arcsec", phasecenter=GC["phasecenter"],
          specmode="cube", start=f"{RG['start_kms']}km/s", width=f"{RG['width_kms']}km/s", nchan=RG["nchan"],
          restfreq=f"{GC['restfreq_ghz']}GHz", outframe="LSRK", veltype="radio", gridder="standard", weighting="natural",
          deconvolver="multiscale", scales=[int(round(s / CELL)) for s in (0.0, 1.2, 3.6)], pblimit=-0.05, pbcor=False,
          fullsummary=True)


def psf_beam(taper=None):
    """the fitted beam of the joint PSF (median channel), arcsec / deg"""
    d = os.path.join(SET, "_joint_psf"); os.makedirs(d, exist_ok=True)
    for p in glob.glob(os.path.join(d, "_t.*")):
        shutil.rmtree(p)
    kw = dict(KW, imagename=os.path.join(d, "_t"), niter=0, restoration=False, calcres=False)
    if taper:
        kw["uvtaper"] = taper
    tclean(**kw)
    ia.open(os.path.join(d, "_t.psf")); b = ia.restoringbeam(); ia.close()
    if "beams" in b:
        bb = b["beams"]; ch = [bb[k]["*0"] for k in sorted(bb, key=lambda s: int(s.strip("*")))]
        b = ch[len(ch) // 2]
    return (qa.convert(b["major"], "arcsec")["value"], qa.convert(b["minor"], "arcsec")["value"], b["positionangle"]["value"])


TJ = os.path.join(SET, "joint_taper.json")
if not os.path.exists(TJ):
    Bj = psf_beam(); log(f"joint natural PSF {Bj[0]:.3f}\" x {Bj[1]:.3f}\" @ {Bj[2]:.1f}; compact beam {BC[0]:.3f} x {BC[1]:.3f}")
    t = [np.sqrt(max(BC[0] ** 2 - Bj[0] ** 2, 0.01)), np.sqrt(max(BC[1] ** 2 - Bj[1] ** 2, 0.01))]
    tries = []
    for it in range(3):
        tap = [f"{t[0]:.4f}arcsec", f"{t[1]:.4f}arcsec", f"{BC[2]:.2f}deg"]
        Bt = psf_beam(tap); tries.append(dict(taper=list(t), psf_beam=Bt))
        log(f"  taper {t[0]:.3f} x {t[1]:.3f} -> PSF {Bt[0]:.3f} x {Bt[1]:.3f} @ {Bt[2]:.1f}")
        if abs(Bt[0] / BC[0] - 1) < 0.01 and abs(Bt[1] / BC[1] - 1) < 0.01:
            break
        t = [t[0] * (BC[0] / Bt[0]) ** 2, t[1] * (BC[1] / Bt[1]) ** 2]            # widths scale ~ quadratically near target
    json.dump(dict(joint_natural_beam=Bj, target_beam=BC, uvtaper=tap, tries=tries), open(TJ, "w"), indent=1)
TAP = json.load(open(TJ))["uvtaper"]
KWT = dict(KW, uvtaper=TAP, restoringbeam=[f"{BC[0]}arcsec", f"{BC[1]}arcsec", f"{BC[2]}deg"])

if not done("dirty"):
    d = os.path.join(SET, "dirty_joint_taper"); os.makedirs(d, exist_ok=True); base = os.path.join(d, "_tclean")
    for p in glob.glob(base + ".*"):
        shutil.rmtree(p)
    tclean(imagename=base, niter=0, restoration=False, **{k: v for k, v in KWT.items() if k != "restoringbeam"})
    for ext in ("residual", "psf"):
        exportfits(imagename=f"{base}.{ext}", fitsimage=os.path.join(d, f"{ext}.fits"), velocity=True, overwrite=True)
    os.replace(os.path.join(d, "residual.fits"), os.path.join(d, "dirty.fits"))
    log("joint dirty done"); mark("dirty")
if not done("clean"):
    d = os.path.join(SET, "clean_joint_taper"); os.makedirs(d, exist_ok=True); base = os.path.join(d, "_tclean")
    for p in glob.glob(base + ".*"):
        shutil.rmtree(p)
    t0 = time.time()
    rec = tclean(imagename=base, niter=1_000_000, nsigma=3.0, usemask="auto-multithresh", sidelobethreshold=2.0,
                 noisethreshold=4.25, lownoisethreshold=1.5, negativethreshold=15.0, minbeamfrac=0.3, growiterations=75, **KWT)
    for ext in ("image", "model", "residual", "psf", "mask"):
        exportfits(imagename=f"{base}.{ext}", fitsimage=os.path.join(d, f"{ext}.fits"), velocity=True, overwrite=True)
    log(f"joint MS-CLEAN first pass: stopcode {rec.get('stopcode')}, {rec.get('iterdone')} iterations, {time.time() - t0:.0f} s")
    finish(d, d + "_finished", KWT, L, CELL, nsigma=1.5, label="NGC3110 joint (tapered)")
    mark("clean")
if not done("masks"):
    d = os.path.join(SET, "clean_joint_taper_finished")
    rd = lambda p: np.nan_to_num(fits.getdata(p)).astype(np.float64).squeeze()
    img, res = rd(os.path.join(d, "image.fits")), rd(os.path.join(d, "residual.fits"))
    bpx = np.pi * BC[0] * BC[1] / (4 * np.log(2)) / CELL ** 2
    for mv in (12, 16):
        M, _ = dendro_mask(img, res, bpx, L, min_value=mv)
        np.save(os.path.join(d, f"dendro_mask_min{mv}.npy"), M.astype(np.uint8))
        log(f"joint mask >= {mv} sigma: {M[L].any(0).sum() * CELL ** 2:.0f} arcsec2")
    mark("masks")
log("DONE")
