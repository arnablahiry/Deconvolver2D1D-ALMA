#!/usr/bin/env python
"""
CASA steps of the 20 km/s gallery (paper Fig. 5) for one target T, on the setup of the NGC 3110 accuracy tests:
64 channels of 20 km/s centred on v_sys, imaged from visibilities pre-averaged onto 68 such channels (a margin of two
on each side, so that no imaged channel falls at the edge of the averaged band), 51" field.

  STEP=prep     regrid (regrid_ms.py, NCHAN=68), the channel grid (grid.json: line / line-free channels on the 64), and
                the compact dirty cube (0.08", 640 px) and PSF (1280 px) for the 2D-1D deconvolution
  STEP=compact  compact MS-CLEAN on the same 0.08" grid (multiscale 0, 1.2", 3.6"; natural; auto-multithresh nsigma 3,
                then finished inside the growing astrodendro mask at nsigma 1.5): the deconvolution's support mask
  STEP=joint    compact + extended MS-CLEAN at the extended clean beam: the most natural Briggs robust whose PSF fits
                inside that beam, a uv taper matched to it (the PSF re-imaged until both axes agree to 3%), 0.04"
                pixels (1280 px), multiscale (0, 1, 3, 6 beams), auto-multithresh (long-baseline parameters) then the
                same finishing; restored with the extended clean beam

tclean works in /scratch/alahiry/gallery20/<T>/; FITS and masks are copied to results/<T>/g20/.

    T=IC5179 STEP=prep OMP_NUM_THREADS=8 /scratch/alahiry/conda/envs/deconvolver2d1d/bin/python scripts/pipeline/gallery20_casa.py
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("scripts", "simple"):
    sys.path.insert(0, os.path.join(REPO, p))
from casa_logs import redirect_casa_logs  # noqa: E402

T, STEP = os.environ["T"], os.environ["STEP"]
# CHANKMS (default 20): the channel width, the same 1280 km/s band (64 x 20, 128 x 10, 256 x 5 km/s, + 4 channels of
# margin in the pre-averaged MSs), products in results/<T>/g<CHANKMS>/
DV = float(os.environ.get("CHANKMS", 20)); BAND = float(os.environ.get("BAND", 1280))   # BAND/DV: a multiple of 32 (CDF 9/7)
NCH = int(round(BAND / DV)); NRG = NCH + 4
# GCELL (default 0.08): the pixel of the compact products (51.2" field; 0.16": 320 px, for the many-channel cubes)
GCELL = float(os.environ.get("GCELL", 0.08)); NPX = int(round(51.2 / GCELL))
GS = f"g{DV:g}" + ("" if GCELL == 0.08 else f"_c{GCELL:g}")
R = os.path.join(REPO, "results", T); OUT = os.path.join(R, GS); os.makedirs(OUT, exist_ok=True)
WK = f"/scratch/alahiry/gallery20/{T}{'' if GS == 'g20' else '_' + GS}/{STEP}"; os.makedirs(WK, exist_ok=True); os.chdir(WK)
RGD = f"/scratch/alahiry/regrid/{T}_{NRG}x{DV:g}kms"
redirect_casa_logs(f"gallery20_{T}_{STEP}.log")
import casatools  # noqa: E402
from casatasks import tclean, exportfits  # noqa: E402
from astropy.io import fits  # noqa: E402
from msclean_complete import finish  # noqa: E402

GC = json.load(open(os.path.join(R, "config.json")))
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {T} {STEP}: {m}", flush=True)
qa, ia = casatools.quanta(), casatools.image()
start = GC["vsys_kms"] - (NCH - 1) / 2 * DV                              # the 64 imaged channels
beam = lambda p: (lambda h: (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"]))(fits.getheader(p))


def kw(cfgs, cell, imsize, **x):
    RG = json.load(open(os.path.join(RGD, "ms.json")))
    return dict(vis=sum((RG["ms"][c] for c in cfgs), []), spw=RG["spw"], datacolumn="data", imsize=imsize, cell=f"{cell}arcsec",
                phasecenter=GC["phasecenter"], specmode="cube", start=f"{start}km/s", width=f"{DV}km/s", nchan=NCH,
                restfreq=f"{GC['restfreq_ghz']}GHz", outframe="LSRK", veltype="radio", gridder="standard", pblimit=-0.05,
                pbcor=False, **x)


def export_all(src, dst):
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(os.path.join(src, "*.fits")) + glob.glob(os.path.join(src, "*.npy")) + glob.glob(os.path.join(src, "*.json")):
        shutil.copy(f, dst)


if STEP == "prep":
    if not os.path.exists(os.path.join(RGD, "ms.json")):
        subprocess.run([sys.executable, os.path.join(REPO, "scripts/experiments/regrid_ms.py"), T], check=True,
                       env=dict(os.environ, NCHAN=str(NRG), WIDTH_KMS=str(DV)))
    # line channels: the velocity range of the 25 km/s line channels (line_channels.json), on the 64 x 20 km/s grid
    L25 = json.load(open(os.path.join(R, "line_channels.json")))["line"]
    v25 = GC["vsys_kms"] + GC["width_kms"] * (np.arange(GC["nchan"]) - (GC["nchan"] - 1) / 2)
    lo, hi = v25[min(L25)] - GC["width_kms"] / 2, v25[max(L25)] + GC["width_kms"] / 2
    v = start + DV * np.arange(NCH)
    line = [int(i) for i in np.nonzero((v + DV / 2 > lo) & (v - DV / 2 < hi))[0]]
    free = [i for i in range(NCH) if i not in line and 1 < i < NCH - 2]
    json.dump(dict(cell=GCELL, imsize=NPX, nchan=NCH, width_kms=DV, start_kms=start, regrid_nchan=NRG, line=line,
                   line_free=free), open(os.path.join(OUT, "grid.json"), "w"), indent=1)
    log(f"{len(line)} line / {len(free)} line-free channels")
    D = os.path.join(OUT, "deconv_inputs"); os.makedirs(D, exist_ok=True)
    if not os.path.exists(os.path.join(D, "dirty_beam_cube.fits")):     # dirty at 2x the field, cropped; the PSF at 2x
        base = os.path.join(WK, "_d")
        for p in glob.glob(base + ".*"):
            shutil.rmtree(p)
        tclean(imagename=base, niter=0, restoration=False, **kw(["compact"], GCELL, 2 * NPX, weighting="natural"))
        rg = casatools.regionmanager()
        ia.open(base + ".residual"); sub = ia.subimage(region=rg.box(blc=[NPX // 2, NPX // 2], trc=[NPX // 2 + NPX - 1, NPX // 2 + NPX - 1]))
        sub.setbrightnessunit("Jy/beam"); sub.tofits(os.path.join(D, "dirty_cube.fits"), overwrite=True, velocity=True)
        sub.done(); ia.close()
        ia.open(base + ".psf"); ia.tofits(os.path.join(D, "dirty_beam_cube.fits"), overwrite=True, velocity=True); ia.close()
        for p in glob.glob(base + ".*"):
            shutil.rmtree(p)
    log("prep done")

elif STEP == "compact":
    G = json.load(open(os.path.join(OUT, "grid.json")))
    K = kw(["compact"], GCELL, NPX, weighting="natural", deconvolver="multiscale",
           scales=[0, int(round(1.2 / GCELL)), int(round(3.6 / GCELL))], fullsummary=True)
    c1, c2 = os.path.join(WK, "clean"), os.path.join(WK, "clean_finished")
    if not os.path.exists(os.path.join(WK, "clean.done")):
        os.makedirs(c1, exist_ok=True); base = os.path.join(c1, "_tclean")
        rec = tclean(imagename=base, niter=1_000_000, nsigma=3.0, usemask="auto-multithresh", sidelobethreshold=2.0,
                     noisethreshold=4.25, lownoisethreshold=1.5, negativethreshold=15.0, minbeamfrac=0.3, growiterations=75, **K)
        log(f"first pass {rec.get('stopcode')} {rec.get('iterdone')}"); open(os.path.join(WK, "clean.done"), "w").write("")
    finish(c1, c2, K, G["line"], GCELL, nsigma=1.5, label=f"{T} compact {DV:g} km/s")
    export_all(c2, os.path.join(OUT, "clean_compact_finished")); log("compact done")

elif STEP == "joint":
    G = json.load(open(os.path.join(OUT, "grid.json")))
    BE = beam(os.path.join(R, "extended/msclean_finished/image.fits"))
    CELL = 0.04; INFO = os.path.join(OUT, "joint_info.json")

    def cov(b):
        s = np.array([b[0], b[1]]) / np.sqrt(8 * np.log(2)); t = np.radians(b[2])
        u = np.array([np.sin(t), np.cos(t)]); v = np.array([np.cos(t), -np.sin(t)])
        return s[0] ** 2 * np.outer(u, u) + s[1] ** 2 * np.outer(v, v)

    def psf_beam(r, taper=None, tag="p"):
        base = os.path.join(WK, "_psf_" + tag)
        for p in glob.glob(base + ".*"):
            shutil.rmtree(p)
        x = dict(weighting="natural") if r == 2 else dict(weighting="briggs", robust=r)
        k = dict(kw(["compact", "extended"], CELL, 640, **x), nchan=1, start=f"{start + DV * G['line'][len(G['line']) // 2]}km/s")
        if taper:
            k["uvtaper"] = [f"{taper[0]:.4f}arcsec", f"{taper[1]:.4f}arcsec", f"{taper[2]:.2f}deg"]
        tclean(imagename=base, niter=0, restoration=False, calcres=False, **k)
        ia.open(base + ".psf"); b = ia.restoringbeam(); ia.close()
        b = b["beams"]["*0"]["*0"] if "beams" in b else b
        out = (qa.convert(b["major"], "arcsec")["value"], qa.convert(b["minor"], "arcsec")["value"],
               qa.convert(b["positionangle"], "deg")["value"])
        assert out[1] > 0.01, f"PSF fit failed: {out}"
        return out

    if not os.path.exists(INFO):
        best = None
        for r in (0.5, 0, -0.5, -1, -2):                                 # the most natural robust that can reach the beam
            P0 = psf_beam(r, tag=f"r{r}")
            if np.linalg.eigvalsh(cov(BE) - cov((P0[0] * 1.02, P0[1] * 1.02, P0[2]))).min() <= 0:
                continue
            th = np.radians(P0[2]); dirs = (np.array([np.sin(th), np.cos(th)]), np.array([np.cos(th), -np.sin(th)]))
            along = lambda b: np.array([np.sqrt(8 * np.log(2) * d @ cov(b) @ d) for d in dirs])
            tgt = along(BE); t2 = np.clip(tgt ** 2 - along(P0) ** 2, 1e-4, None); hist = []
            for it in range(7):
                tu, tv = np.sqrt(t2)
                taper = (float(tu), float(tv), float(P0[2])) if tu >= tv else (float(tv), float(tu), float(P0[2] + 90))
                P = psf_beam(r, taper, tag=f"r{r}t{it}"); m = along(P); hist.append((t2.copy(), m ** 2))
                err = float(np.abs(m / tgt - 1).max())
                if best is None or err < best["err"]:
                    best = dict(robust=r, uvtaper=taper, psf=P, err=err)
                log(f"robust {r} taper {taper[0]:.3f} x {taper[1]:.3f} -> PSF {P[0]:.3f} x {P[1]:.3f} (target {BE[0]:.3f} x {BE[1]:.3f})")
                if err < 0.03:
                    break
                new = t2 * (tgt / m) ** 4
                if len(hist) >= 2:
                    (a0, f0), (a1, f1) = hist[-2], hist[-1]
                    for k_ in range(2):
                        if abs(f1[k_] - f0[k_]) > 1e-6 and abs(a1[k_] - a0[k_]) > 1e-8:
                            new[k_] = a1[k_] + (tgt[k_] ** 2 - f1[k_]) * (a1[k_] - a0[k_]) / (f1[k_] - f0[k_])
                t2 = np.clip(new, 1e-4, 4 * t2 + 0.01)
            if best["err"] < 0.03:
                break
        if best is None:                                                  # nothing fits: the finest weighting, no taper
            best = dict(robust=-2, uvtaper=None, psf=psf_beam(-2, tag="r-2"), err=None)
        json.dump(dict(beam=BE, **best), open(INFO, "w"), indent=1)
    info = json.load(open(INFO)); r = info["robust"]
    geo = float(np.sqrt(BE[0] * BE[1]))
    K = kw(["compact", "extended"], CELL, 1280, deconvolver="multiscale", fullsummary=True,
           scales=sorted({int(round(f * geo / CELL)) for f in (0, 1, 3, 6)}),
           restoringbeam=[f"{BE[0]}arcsec", f"{BE[1]}arcsec", f"{BE[2]}deg"],
           **(dict(weighting="natural") if r == 2 else dict(weighting="briggs", robust=r)))
    if info["uvtaper"]:
        t = info["uvtaper"]; K["uvtaper"] = [f"{t[0]:.4f}arcsec", f"{t[1]:.4f}arcsec", f"{t[2]:.2f}deg"]
    c1, c2 = os.path.join(WK, "clean"), os.path.join(WK, "clean_finished")
    if not os.path.exists(os.path.join(WK, "clean.done")):
        os.makedirs(c1, exist_ok=True); base = os.path.join(c1, "_tclean")
        rec = tclean(imagename=base, niter=1_000_000, nsigma=3.0, usemask="auto-multithresh", sidelobethreshold=3.0,
                     noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=7.0, minbeamfrac=0.3, growiterations=75, **K)
        log(f"first pass {rec.get('stopcode')} {rec.get('iterdone')}"); open(os.path.join(WK, "clean.done"), "w").write("")
    finish(c1, c2, K, G["line"], CELL, nsigma=1.5, label=f"{T} joint at the extended beam")
    export_all(c2, os.path.join(OUT, "joint_ext")); log("joint done")
