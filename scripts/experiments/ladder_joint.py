#!/usr/bin/env python
"""
The reference row of the resolution ladder (fig_resolution_ladder.py, REF=joint): the joint compact + extended
visibilities of NGC 3110 imaged at every ladder beam b, so each comparison beam is a real image of the data rather than
the extended CLEAN smoothed to it.

  weighting   the most natural Briggs robust (2 = natural, 1, 0.5, 0, -0.5, -1, -2) whose PSF fits inside b in every
              direction (with 3% to spare), from single-channel niter=0 PSFs (cached in psf_table.json)
  uv taper    tuned so the PSF matches b: Gaussian covariances add (tapered PSF ~ PSF (*) taper), so the taper's
              covariance is corrected by (b - measured PSF) until both axes are within 2%; none when b is at or below
              the finest robust -2 PSF (the extended clean beam), which is then the restoring beam regardless
  MS-CLEAN    multiscale (0, 1, 3, 6 x b's geometric FWHM), auto-multithresh (short-baseline parameters for b > 0.5",
              long-baseline ones below), nsigma 3, then finished inside the growing astrodendro mask at nsigma 1.5
              (msclean_complete.finish); restored with b itself, so image.fits is at exactly b
  masks       astrodendro (msclean_complete.dendro_mask) of the finished image at min_value 8 / 12 / 16 / 25 sigma

Grid: 0.04" pixels, 1280 px (51.2"), the ladder's 32 x 25 km/s LSRK channels (start 4686.4 km/s; the MSs are
pre-averaged with one channel of margin each side: regrid_ms.py NCHAN=34 WIDTH_KMS=25).

    I=<beam index> OMP_NUM_THREADS=8 /scratch/alahiry/conda/envs/deconvolver2d1d/bin/python scripts/experiments/ladder_joint.py
    -> results/NGC3110/ladder_joint/b<ii>/{clean, clean_finished/, info.json, dendro_mask_min<N>.npy}
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

I = int(os.environ["I"]); CELL, IMSIZE = 0.04, 1280
# GRID=20kms: the 64 x 20 km/s channels of results/NGC3110/consistency_20kms_* (the pixel-scale runs); the weighting and
# taper found for each beam on the 25 km/s grid are reused (INFO_FROM), and tclean works on /scratch (FITS copied back)
G20 = os.environ.get("GRID") == "20kms"
T = os.environ.get("T", "NGC3110")              # other targets: the 20 km/s gallery products (results/<T>/g20), G20 only
# VARIANT=<name>: the reference-accuracy test -- the same beam re-imaged into ladder_joint_20kms_<name>/ with
# ROBUST_FORCE=<r> (that Briggs weighting, its own taper) and/or FINISH_NSIG=<n> (the finishing CLEAN's depth, default 1.5;
# without ROBUST_FORCE the weighting and taper of the main ladder are reused)
VAR = os.environ.get("VARIANT", ""); RFORCE = os.environ.get("ROBUST_FORCE"); FNSIG = float(os.environ.get("FINISH_NSIG", 1.5))
CHK = os.environ.get("CHANKMS")                 # 10, 5: the same beam on finer channels (results/<T>/g<CHANKMS>/ grid)
R = os.path.join(REPO, "results", T)
OUT = os.path.join(R, (f"ladder_joint_{CHK}kms" if CHK else "ladder_joint_20kms" if G20 else "ladder_joint") + (f"_{VAR}" if VAR else ""))
D = os.path.join(OUT, f"b{I:02d}"); os.makedirs(D, exist_ok=True)
WK = (f"/scratch/alahiry/ladder_joint_{CHK + 'kms' if CHK else '20kms'}{'_' + VAR if VAR else ''}/{'' if T == 'NGC3110' else T + '/'}b{I:02d}"
      if G20 else D)
os.makedirs(WK, exist_ok=True)
os.chdir(WK)                    # tclean's Briggs weights go to a scratch table in the working directory: one per job
redirect_casa_logs(f"ladder_joint_{T}_b{I:02d}.log")
import casatools  # noqa: E402
from casatasks import tclean, exportfits  # noqa: E402
from astropy.io import fits  # noqa: E402
from msclean_complete import finish, dendro_mask  # noqa: E402

GC = json.load(open(os.path.join(R, "config.json")))
RG = json.load(open(f"/scratch/alahiry/regrid/{T}_{int(round(float(os.environ.get('BAND', 1280)) / float(CHK))) + 4}x{CHK}kms/ms.json" if CHK else
                    f"/scratch/alahiry/regrid/{T}_{('64x20' if T == 'NGC3110' else '68x20') if G20 else '34x25'}kms/ms.json"))
L = json.load(open(os.path.join(R, "line_channels.json")))["line"]
if G20:
    _g = json.load(open(os.path.join(R, os.environ.get("GDIR", f"g{CHK}") + "/grid.json" if CHK else
                                     "consistency_20kms_0.08/grid.json" if T == "NGC3110" else "g20/grid.json")))
    L = _g["line"]; GC = dict(json.load(open(os.path.join(R, "config.json"))), start_kms=_g["start_kms"],
                               width_kms=_g["width_kms"], nchan=_g["nchan"])
    _src = (None if RFORCE else os.path.join(R, "ladder_joint_20kms", f"b{I:02d}", "info.json")) if (VAR or CHK) else \
        os.path.join(R, "ladder_joint", f"b{I:02d}", "info.json")
    if not os.path.exists(os.path.join(D, "info.json")) and _src and os.path.exists(_src):
        shutil.copy(_src, os.path.join(D, "info.json"))
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {T} b{I:02d}: {m}", flush=True)
beam = lambda p: (lambda h: (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"]))(fits.getheader(p))

# the ladder's beams, exactly as fig_resolution_ladder.py builds them (NSIDE = 4)
BC = beam(os.path.join(R, "compact/msclean/image.fits")); BE = beam(os.path.join(R, "extended/msclean_finished/image.fits"))
LB = float(GC["lbmax_compact_arcsec"])
circ = lambda a, b: [(f, f, 0.0) for f in np.geomspace(a, b, 6)[1:-1]]
BEAMS = [BC] + circ(BC[1], LB) + [(LB, LB, 0.0)] + circ(LB, BE[0]) + [BE]
B = tuple(float(v) for v in BEAMS[I]); GEO = float(np.sqrt(B[0] * B[1]))
VIS = RG["ms"]["compact"] + RG["ms"]["extended"]
KW = dict(vis=VIS, spw=RG["spw"], datacolumn="data", imsize=IMSIZE, cell=f"{CELL}arcsec", phasecenter=GC["phasecenter"],
          specmode="cube", start=f"{GC['start_kms']}km/s", width=f"{GC['width_kms']}km/s", nchan=GC["nchan"],
          restfreq=f"{GC['restfreq_ghz']}GHz", outframe="LSRK", veltype="radio", gridder="standard", pblimit=-0.05,
          pbcor=False, fullsummary=True)
qa, ia = casatools.quanta(), casatools.image()


def cov(b):
    """covariance (arcsec^2) of beam b = (FWHM major, minor, PA deg E of N) in (east, north)"""
    s = np.array([b[0], b[1]]) / np.sqrt(8 * np.log(2)); t = np.radians(b[2])
    u = np.array([np.sin(t), np.cos(t)]); v = np.array([np.cos(t), -np.sin(t)])
    return s[0] ** 2 * np.outer(u, u) + s[1] ** 2 * np.outer(v, v)


def beam_of(C):
    """inverse of cov"""
    w, V = np.linalg.eigh(C); w = np.clip(w, 1e-8, None) * 8 * np.log(2)
    u = V[:, 1]; pa = np.degrees(np.arctan2(u[0], u[1]))
    pa = (pa + 90) % 180 - 90
    return float(np.sqrt(w[1])), float(np.sqrt(w[0])), float(pa)


def wkw(r):
    return dict(weighting="natural") if r == 2 else dict(weighting="briggs", robust=r)


def psf_beam(r, taper=None, tag="x"):
    """fitted PSF beam of one line channel (niter 0, 640 px), arcsec / deg"""
    d = os.path.join(D, "_psf"); os.makedirs(d, exist_ok=True); base = os.path.join(d, tag)
    for p in glob.glob(base + ".*"):
        shutil.rmtree(p)
    v0 = GC["start_kms"] + GC["width_kms"] * (GC["nchan"] // 2 - 1)       # a line channel on the MSs' own channel grid
    kw = dict(KW, imagename=base, imsize=640, nchan=1, start=f"{v0}km/s", niter=0, restoration=False,
              calcres=False, **wkw(r))
    if taper:
        kw["uvtaper"] = [f"{taper[0]:.4f}arcsec", f"{taper[1]:.4f}arcsec", f"{taper[2]:.2f}deg"]
    tclean(**kw)
    ia.open(base + ".psf"); b = ia.restoringbeam(); ia.close()
    if "beams" in b:
        b = b["beams"]["*0"]["*0"]
    out = (qa.convert(b["major"], "arcsec")["value"], qa.convert(b["minor"], "arcsec")["value"],
           qa.convert(b["positionangle"], "deg")["value"])
    if out[1] < 0.01:
        raise RuntimeError(f"PSF fit failed (empty channel?): {out}")
    return out


ROBUST = [float(RFORCE)] if RFORCE else [2, 1, 0.5, 0, -0.5, -1, -2]
INFO = os.path.join(D, "info.json")
if not os.path.exists(INFO):
    tab = os.path.join(OUT, "psf_table.json")
    if os.path.exists(tab):
        PT = {float(k): v for k, v in json.load(open(tab)).items()}
    else:
        PT = {r: psf_beam(r, tag=f"r{r}") for r in ROBUST}
        json.dump({str(k): v for k, v in PT.items()}, open(tab, "w"), indent=1)
    fits_in = lambda P: np.linalg.eigvalsh(cov(B) - cov((P[0] * 1.03, P[1] * 1.03, P[2]))).min() > 0
    r0 = next((r for r in ROBUST if fits_in(PT[r])), -2)
    best = None
    # tapering the joint PSF along one axis also widens it along the other (the compact pedestal takes over), so a
    # weighting can fit inside b and still not reach it: then the next, more uniform robust value
    for r in [q for q in ROBUST if q <= r0]:
        P0 = PT[r]; taper, P, tries = None, P0, []
        if not fits_in(P0):
            break
        th = np.radians(P0[2]); dirs = (np.array([np.sin(th), np.cos(th)]), np.array([np.cos(th), -np.sin(th)]))
        along = lambda b: np.array([np.sqrt(8 * np.log(2) * d @ cov(b) @ d) for d in dirs])   # FWHM along u, v
        tgt = along(B); t2 = np.clip(tgt ** 2 - along(P0) ** 2, 1e-4, None)          # Gaussian-additive first guess
        hist = []
        for it in range(7):
            tu, tv = np.sqrt(t2)
            taper = (float(tu), float(tv), float(P0[2])) if tu >= tv else (float(tv), float(tu), float(P0[2] + 90))
            P = psf_beam(r, taper, tag=f"r{r}_t{it}"); m = along(P); hist.append((t2.copy(), m ** 2))
            err = float(np.abs(m / tgt - 1).max()); tries.append(dict(robust=r, taper=taper, psf=P, err=err))
            log(f"robust {r}: taper {taper[0]:.3f} x {taper[1]:.3f} @ {taper[2]:.0f} -> PSF {P[0]:.3f} x {P[1]:.3f} @ {P[2]:.0f}"
                f" (target {B[0]:.3f} x {B[1]:.3f}); along PSF axes {m[0]:.3f}, {m[1]:.3f} vs {tgt[0]:.3f}, {tgt[1]:.3f}")
            if best is None or err < best["err"]:
                best = dict(robust=r, uvtaper=taper, psf=P, psf_untapered=P0, err=err)
            if err < 0.03:
                break
            new = t2 * (tgt / m) ** 4                                                  # t ~ (target / PSF)^2 x t
            if len(hist) >= 2:
                (a0, f0), (a1, f1) = hist[-2], hist[-1]
                for k in range(2):
                    if abs(f1[k] - f0[k]) > 1e-6 and abs(a1[k] - a0[k]) > 1e-8:
                        new[k] = a1[k] + (tgt[k] ** 2 - f1[k]) * (a1[k] - a0[k]) / (f1[k] - f0[k])
            t2 = np.clip(new, 1e-4, 4 * t2 + 0.01)
        best.setdefault("tries", []).extend(tries)
        if best["err"] < 0.03:
            break
    if best is None:                                                           # nothing fits inside b
        r, P0 = -2, PT[-2]
        best = dict(robust=r, uvtaper=None, psf=P0, psf_untapered=P0, err=None, tries=[])
        log(f"no weighting reaches {B[0]:.3f} x {B[1]:.3f}: robust {r} untapered, PSF {P0[0]:.3f} x {P0[1]:.3f}")
    log(f"chosen: robust {best['robust']}, taper {best['uvtaper']}, PSF {best['psf']}, max axis error {best['err']}")
    json.dump(dict(index=I, beam=B, **best), open(INFO, "w"), indent=1)
    shutil.rmtree(os.path.join(D, "_psf"), ignore_errors=True)
info = json.load(open(INFO))
KWT = dict(KW, restoringbeam=[f"{B[0]}arcsec", f"{B[1]}arcsec", f"{B[2]}deg"], deconvolver="multiscale",
           scales=sorted({int(round(f * GEO / CELL)) for f in (0, 1, 3, 6)}), **wkw(info["robust"]))
if info["uvtaper"]:
    t = info["uvtaper"]; KWT["uvtaper"] = [f"{t[0]:.4f}arcsec", f"{t[1]:.4f}arcsec", f"{t[2]:.2f}deg"]
AM = (dict(sidelobethreshold=2.0, noisethreshold=4.25, lownoisethreshold=1.5, negativethreshold=15.0, minbeamfrac=0.3)
      if GEO > 0.5 else
      dict(sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=7.0, minbeamfrac=0.3))

c1 = os.path.join(WK, "clean")
if not os.path.exists(os.path.join(D, "clean.done")):
    os.makedirs(c1, exist_ok=True); base = os.path.join(c1, "_tclean")
    for p in glob.glob(base + ".*"):
        shutil.rmtree(p)
    t0 = time.time()
    rec = tclean(imagename=base, niter=1_000_000, nsigma=3.0, usemask="auto-multithresh", growiterations=75, **AM, **KWT)
    log(f"MS-CLEAN first pass: stopcode {rec.get('stopcode')}, {rec.get('iterdone')} iterations, {time.time() - t0:.0f} s; "
        f"scales {KWT['scales']} px")
    open(os.path.join(D, "clean.done"), "w").write(time.strftime("%Y-%m-%d %H:%M"))
c2 = os.path.join(WK, "clean_finished")
if not os.path.exists(os.path.join(D, "finish.done")):
    t0 = time.time()
    finish(c1, c2, KWT, L, CELL, nsigma=FNSIG, label=f"{T} joint b{I:02d} {VAR}")
    log(f"finished: {time.time() - t0:.0f} s")
    open(os.path.join(D, "finish.done"), "w").write(time.strftime("%Y-%m-%d %H:%M"))
rd = lambda p: np.nan_to_num(fits.getdata(p)).astype(np.float64).squeeze()
img, res = rd(os.path.join(c2, "image.fits")), rd(os.path.join(c2, "residual.fits"))
bpx = np.pi * B[0] * B[1] / (4 * np.log(2)) / CELL ** 2
if G20:                                                                   # FITS back to the results folder
    os.makedirs(os.path.join(D, "clean_finished"), exist_ok=True)
    for f in glob.glob(os.path.join(c2, "*.fits")) + glob.glob(os.path.join(c2, "*.npy")) + glob.glob(os.path.join(c2, "*.json")):
        shutil.copy(f, os.path.join(D, "clean_finished"))
for mv in (() if G20 else (8, 12, 16, 25)):
    f = os.path.join(D, f"dendro_mask_min{mv}.npy")
    if not os.path.exists(f):
        M, mi = dendro_mask(img, res, bpx, L, min_value=mv)
        np.save(f, M.astype(np.uint8))
        log(f"mask >= {mv} sigma: {M[L].any(0).sum() * CELL ** 2:.1f} arcsec2, {mi['structures']} structures")
for p in glob.glob(os.path.join(c1, "_tclean.*")) + glob.glob(os.path.join(c2, "_tclean.*")):
    shutil.rmtree(p)                                                    # the FITS exports are kept
log("DONE")
