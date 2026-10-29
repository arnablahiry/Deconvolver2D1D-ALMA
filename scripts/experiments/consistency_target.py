#!/usr/bin/env python
"""
The NGC 3110 compact-vs-extended consistency analysis (both arrays at the compact beam) for another target of the sample,
from its results/<T>/config.json (MSs, spw selections, fields, phase centre, 32 x 25 km/s channels). Into
results/<T>/consistency/:

  dirty_compact/, dirty_extended_taper/   tclean niter=0 (natural; the extended one uv-tapered): dirty.fits, psf.fits
  clean_compact/ -> clean_compact_finished/
        multiscale (0, 30, 90 px), natural, nsigma 3 auto-multithresh (short-baseline values), finished inside the
        growing astrodendro mask at nsigma 1.5 (scripts/msclean_complete.finish)
  clean_extended_taper/ -> clean_extended_taper_cmask/
        the same with uvtaper = sqrt(compact clean beam^2 - extended natural beam^2) per axis at the compact beam's PA
        and restored with the compact clean beam; first pass nsigma 3 auto-multithresh (long-baseline values), then
        continued inside the compact run's mask (where to clean only; fluxes from the extended data) down to an absolute
        threshold of 1.5 x the line-free noise of its residual (tclean's nsigma is inflated by sidelobes there)
  masks: astrodendro on the final tapered extended image, >= 8 and >= 12 sigma (dendro_mask_min{8,12}.npy)

Grid: 1280 px x 0.04" (51"). Steps already done are skipped (each writes <step>.done).

    /scratch/alahiry/conda/envs/deconvolver2d1d/bin/python scripts/experiments/consistency_target.py NGC1614
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
for p in ("scripts", "simple"):
    sys.path.insert(0, os.path.join(REPO, p))
ap = argparse.ArgumentParser(); ap.add_argument("target"); a = ap.parse_args()
# optional overrides (defaults = the 32 x 25 km/s, 0.04" set): CELL (arcsec), IMSIZE, NCHAN, WIDTH_KMS, SPW_FULL=1 (whole
# spectral windows, for a wider band), TAPER_JSON (a tuned uvtaper), SUBDIR (output folder under results/<T>/)
T = a.target; R = os.path.join(REPO, "results", T)
OUT = os.path.join(R, os.environ.get("SUBDIR", "consistency")); os.makedirs(OUT, exist_ok=True)
from casa_logs import redirect_casa_logs  # noqa: E402
redirect_casa_logs(f"consistency_{T}.log")
import casatools  # noqa: E402
from casatasks import tclean, exportfits  # noqa: E402
from astropy.io import fits  # noqa: E402
from msclean_complete import finish, dendro_mask  # noqa: E402

GC = json.load(open(os.path.join(R, "config.json")))
CELL = float(os.environ.get("CELL", 0.04)); IMSIZE = int(os.environ.get("IMSIZE", 1280))
SCALES = [int(round(s_ / CELL)) for s_ in (0.0, 1.2, 3.6)]                       # point, 1.2", 3.6"
if "NCHAN" in os.environ:                                                    # a new spectral grid centred on v_sys
    GC["nchan"], GC["width_kms"] = int(os.environ["NCHAN"]), float(os.environ["WIDTH_KMS"])
    GC["start_kms"] = GC["vsys_kms"] - (GC["nchan"] - 1) / 2 * GC["width_kms"]
    vel_ = GC["start_kms"] + GC["width_kms"] * np.arange(GC["nchan"])
    lw = json.load(open(os.path.join(R, "line_channels.json"))).get("window_kms")
    if lw is None:                                                           # the line window of the original grid
        L0 = json.load(open(os.path.join(R, "line_channels.json")))["line"]
        c0 = json.load(open(os.path.join(R, "config.json")))
        v0 = c0["start_kms"] + c0["width_kms"] * np.arange(c0["nchan"])
        lw = (float(v0[L0[0]]) - c0["width_kms"] / 2, float(v0[L0[-1]]) + c0["width_kms"] / 2)
    L = [int(i) for i in np.nonzero((vel_ >= lw[0]) & (vel_ <= lw[1]))[0]]
else:
    L = json.load(open(os.path.join(R, "line_channels.json")))["line"]
if os.environ.get("REGRID"):                                                # pre-averaged MSs (regrid_ms.py)
    RG = json.load(open(os.path.join(os.environ["REGRID"], "ms.json")))
    for cfg_ in ("compact", "extended"):
        GC["configs"][cfg_]["ms"] = RG["ms"][cfg_]
        GC["configs"][cfg_]["spw"] = [RG["spw"]] * len(RG["ms"][cfg_])
        GC["configs"][cfg_]["field"] = [""] * len(RG["ms"][cfg_])       # one field left in each
if os.environ.get("SPW_FULL"):
    for cfg_ in ("compact", "extended"):
        GC["configs"][cfg_]["spw"] = [s_.split(":")[0] for s_ in GC["configs"][cfg_]["spw"]]
F = [c for c in range(GC["nchan"]) if c not in L and 1 < c < GC["nchan"] - 2]
json.dump(dict(cell=CELL, imsize=IMSIZE, scales_px=SCALES, nchan=GC["nchan"], width_kms=GC["width_kms"],
               start_kms=GC["start_kms"], line=L, line_free=F), open(os.path.join(OUT, "grid.json"), "w"), indent=1)
AUTO = dict(compact=dict(sidelobethreshold=2.0, noisethreshold=4.25, lownoisethreshold=1.5, negativethreshold=15.0),
            extended=dict(sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5, negativethreshold=7.0))
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {T}: {m}", flush=True)
rd = lambda p: np.nan_to_num(fits.getdata(p)).astype(np.float64).squeeze()
bm = lambda p: (lambda h: (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"]))(fits.getheader(p))
done = lambda s: os.path.exists(os.path.join(OUT, s + ".done"))
mark = lambda s: open(os.path.join(OUT, s + ".done"), "w").write(time.strftime("%Y-%m-%d %H:%M"))


def common(cfg, **extra):
    cc = GC["configs"][cfg]
    kw = dict(vis=cc["ms"], field=cc["field"] if len(set(cc["field"])) > 1 else cc["field"][0], spw=cc["spw"],
              datacolumn="data", imsize=IMSIZE, cell=f"{CELL}arcsec", phasecenter=GC["phasecenter"], specmode="cube",
              start=f"{GC['start_kms']}km/s", width=f"{GC['width_kms']}km/s", nchan=GC["nchan"],
              restfreq=f"{GC['restfreq_ghz']}GHz", outframe="LSRK", veltype="radio", gridder="standard",
              weighting="natural", deconvolver="multiscale", scales=SCALES, pblimit=-0.05, pbcor=False,
              restoringbeam="common", fullsummary=True)
    kw.update(extra)
    return kw


def run_dir(name):
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    for p in glob.glob(os.path.join(d, "_tclean.*")):
        shutil.rmtree(p)
    return d, os.path.join(d, "_tclean")


def export(d, base, exts):
    for ext in exts:
        exportfits(imagename=f"{base}.{ext}", fitsimage=os.path.join(d, f"{ext}.fits"), velocity=True, overwrite=True)


def dirty(name, kw):
    """the dirty cube (= the residual of a niter=0 run) and the PSF"""
    d, base = run_dir(name)
    tclean(imagename=base, niter=0, restoration=False, **kw)
    export(d, base, ("residual", "psf"))
    os.replace(os.path.join(d, "residual.fits"), os.path.join(d, "dirty.fits"))


def taper_kw():
    BC = bm(os.path.join(OUT, "clean_compact_finished/image.fits"))
    if os.environ.get("TAPER_JSON"):                                         # a tuned taper (NGC 3110)
        tj = json.load(open(os.environ["TAPER_JSON"])); tp = tj["uvtaper"]
        if os.environ.get("TARGET_RESTORE") == "1":                          # restore with the taper's target beam
            BC = tuple(tj["target_beam"])                                    # (e.g. lambda/B_max) instead
        return dict(uvtaper=tp, restoringbeam=[f"{BC[0]}arcsec", f"{BC[1]}arcsec", f"{BC[2]}deg"]), BC
    BEn = json.load(open(os.path.join(R, "config.json")))["beams_natural"]["extended"]
    ta, tb_ = np.sqrt(max(BC[0] ** 2 - BEn[0] ** 2, 0.01)), np.sqrt(max(BC[1] ** 2 - BEn[1] ** 2, 0.01))
    return dict(uvtaper=[f"{ta:.4f}arcsec", f"{tb_:.4f}arcsec", f"{BC[2]:.2f}deg"],
                restoringbeam=[f"{BC[0]}arcsec", f"{BC[1]}arcsec", f"{BC[2]}deg"]), BC


line_v = list(L)
# ---------------------------------------------------------------- compact
if not done("clean_compact"):
    kw = common("compact"); d, base = run_dir("clean_compact")
    log("compact MS-CLEAN, first pass"); t0 = time.time()
    rec = tclean(imagename=base, niter=1_000_000, nsigma=3.0, usemask="auto-multithresh", minbeamfrac=0.3,
                 growiterations=75, **AUTO["compact"], **kw)
    export(d, base, ("image", "model", "residual", "psf", "mask"))
    log(f"  stopcode {rec.get('stopcode')}, {rec.get('iterdone')} iterations, {time.time() - t0:.0f} s; finishing")
    finish(d, d + "_finished", kw, line_v, CELL, nsigma=1.5, label=f"{T} compact")
    json.dump(dict(scales_px=SCALES, cell=CELL, imsize=IMSIZE, automask=AUTO["compact"]),
              open(os.path.join(d, "setup.json"), "w"), indent=1)
    mark("clean_compact")
if not done("dirty_compact"):
    log("compact dirty"); dirty("dirty_compact", common("compact")); mark("dirty_compact")
# ---------------------------------------------------------------- extended, tapered to the compact beam
tk, BC = taper_kw()
if not done("dirty_extended_taper"):
    log(f"extended dirty, uvtaper {tk['uvtaper']}"); dirty("dirty_extended_taper", common("extended", **tk)); mark("dirty_extended_taper")
EXT_DIRECT = os.environ.get("EXT_DIRECT") == "1"       # no first pass: clean in the compact mask from the start
if EXT_DIRECT and not done("clean_extended_taper"):
    mark("clean_extended_taper")
if not done("clean_extended_taper"):
    kw = common("extended", **tk); d, base = run_dir("clean_extended_taper")
    log("tapered extended MS-CLEAN, first pass"); t0 = time.time()
    rec = tclean(imagename=base, niter=1_000_000, nsigma=3.0, usemask="auto-multithresh", minbeamfrac=0.3,
                 growiterations=75, **AUTO["extended"], **kw)
    export(d, base, ("image", "model", "residual", "psf", "mask"))
    log(f"  stopcode {rec.get('stopcode')}, {rec.get('iterdone')} iterations, {time.time() - t0:.0f} s")
    json.dump(dict(scales_px=SCALES, cell=CELL, imsize=IMSIZE, automask=AUTO["extended"], **tk),
              open(os.path.join(d, "setup.json"), "w"), indent=1)
    mark("clean_extended_taper")
if not done("clean_extended_taper_cmask"):
    src = os.path.join(OUT, "clean_extended_taper"); d, base = run_dir("clean_extended_taper_cmask")
    if EXT_DIRECT:                                    # fresh: PSF and dirty image of this run, noise from the dirty cube
        tclean(imagename=base, niter=0, **common("extended", **tk))
        ia0 = casatools.image(); ia0.open(base + ".residual")
        r0 = np.asarray(ia0.getchunk(), dtype=np.float64)[:, :, 0, :].transpose(2, 1, 0); ia0.close()
        fr = r0[F] - r0[F].mean(0)
        sig = float(np.median([1.4826 * np.median(np.abs(x - np.median(x))) for x in fr]))
    else:
        for p in glob.glob(os.path.join(src, "_tclean.*")):
            if not p.endswith((".mask", ".usermask")) and os.path.isdir(p):
                shutil.copytree(p, os.path.join(d, os.path.basename(p)))
        sig = float(np.std(rd(os.path.join(src, "residual.fits"))[F]))
    thr = 1.5 * sig * 1e3
    ia = casatools.image(); ia.open(base + ".image"); cs = ia.coordsys().torecord(); ia.close()
    m = np.load(os.path.join(OUT, "clean_compact_finished/mask_final.npy")).astype(np.float32)
    ia.fromarray(outfile=base + ".usermask", pixels=m.transpose(2, 1, 0)[:, :, None, :], csys=cs, overwrite=True); ia.close()
    log(f"tapered extended MS-CLEAN in the compact mask, threshold {thr:.3f} mJy (1.5 x line-free {1e3 * sig:.3f})")
    t0 = time.time()
    rec = tclean(imagename=base, niter=1_000_000, threshold=f"{thr}mJy", nsigma=0.0, usemask="user", mask=base + ".usermask",
                 calcpsf=False, calcres=False, **common("extended", **tk))
    export(d, base, ("image", "model", "residual", "psf", "mask"))
    with fits.open(os.path.join(d, "image.fits")) as hi, fits.open(os.path.join(d, "residual.fits")) as hr:
        fits.writeto(os.path.join(d, "emission.fits"), (hi[0].data - hr[0].data).astype(np.float32), hi[0].header, overwrite=True)
    json.dump(dict(stopcode=rec.get("stopcode"), iterations=rec.get("iterdone"), threshold_mJy=thr, sigma_line_free_mJy=1e3 * sig,
                   mask="clean_compact_finished/mask_final.npy", seconds=round(time.time() - t0), **tk),
              open(os.path.join(d, "run.json"), "w"), indent=1)
    log(f"  stopcode {rec.get('stopcode')}, {rec.get('iterdone')} iterations, {time.time() - t0:.0f} s")
    mark("clean_extended_taper_cmask")
# ---------------------------------------------------------------- tight masks on the final tapered extended image
if not done("masks"):
    d = os.path.join(OUT, "clean_extended_taper_cmask")
    img, res = rd(os.path.join(d, "image.fits")), rd(os.path.join(d, "residual.fits"))
    bpx = np.pi * BC[0] * BC[1] / (4 * np.log(2)) / CELL ** 2
    for mv in (8, 12):
        M, info = dendro_mask(img, res, bpx, line_v, min_value=mv)
        np.save(os.path.join(d, f"dendro_mask_min{mv}.npy"), M.astype(np.uint8))
        log(f"mask >= {mv} sigma: {int(M[L].any(0).sum() * CELL ** 2)} arcsec2")
    mark("masks")
log("DONE")
