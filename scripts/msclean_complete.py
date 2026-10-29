#!/usr/bin/env python
"""
Finish an MS-CLEAN that stopped early (scripts/msclean_cube.py), cleaning ONLY emission: alternate

  1. a tight astrodendro mask of the current CLEAN image (components * beam + residual), in units of a PER-CHANNEL
     sigma (the robust rms of that channel's residual, which in the line channels includes the sidelobes of what is
     still uncleaned), keeping trunk structures (>= MIN_VALUE sigma, split MIN_DELTA, >= MIN_BEAMS beams) whose peak
     beats the brightest structure the same dendrogram finds in the NEGATED image (noise and sidelobes only) --
     the recipe of scripts/superres_figures/fig61_dendro_mask_extended.py;
  2. tclean continued from the current model inside the union of all masks so far, to NSIGMA x the residual's
     robust rms per channel,

until the mask stops growing (< GROW_STOP of new voxels) or MAX_ROUNDS. Cleaning lowers the sidelobes, so the
per-channel sigma drops and the next mask reaches fainter (real) emission; the negated-image bar keeps noise and
sidelobes out of it. Starts from a copy of data/<galaxy>/<config>/<FROM>/_tclean.* (the original is not touched).

--smooth S1 S2 ... (arcsec): each round, also the same dendrogram mask of the image and residual smoothed to circular
beams of these FWHMs (compact_vs_msclean_dendro/beams.to_beam), OR-ed into the mask: faint extended emission below the
full-resolution bars but significant once smoothed (its own per-channel sigma, its own negated-image bar) gets cleaned.

Writes data/<galaxy>/<config>/<OUT>/{image,model,residual,mask,psf}.fits, emission.fits (= image - residual =
components convolved with the common clean beam, Jy/beam: the emission only), rounds.json and the CASA log.
Run with the CASA env:

    /scratch/alahiry/conda/envs/deconvolver2d1d/bin/python scripts/msclean_complete.py ngc3110 extended \
        [--from ms_clean] [--out msclean_complete] [--nsigma 1.5] [--max-rounds 6] [--smooth 0.6 1.2]
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "simple"))
from msclean_cube import (GALAXIES, IMSIZE, CELL_ARCSEC, RESTFREQ_GHZ, NCHAN, WIDTH_KMS, WEIGHTING,  # noqa: E402
                          phasecenter_string, line_setup)
from casa_logs import redirect_casa_logs  # noqa: E402


def get(ia, path):
    """CASA image -> (chan, y, x) float64"""
    ia.open(path); a = ia.getchunk(); ia.close()
    return np.asarray(a, dtype=np.float64)[:, :, 0, :].transpose(2, 1, 0)


def dendro_mask(img, res, beam_px, line, min_value=3.0, min_delta=1.0, min_beams=2.0):
    """fig61's mask: per-channel sigma from the residual, peak bar from the negated image. Returns mask, info."""
    from astrodendro import Dendrogram
    rms = 1.4826 * np.median(np.abs(res - np.median(res, axis=(1, 2), keepdims=True)), axis=(1, 2))
    snr = img / rms[:, None, None]
    kw = dict(min_value=min_value, min_delta=min_delta, min_npix=max(1, int(round(min_beams * beam_px))))
    neg = sorted((s.get_peak()[1] for s in Dendrogram.compute(-snr, **kw).trunk), reverse=True)
    peak = max(5.0, np.ceil(neg[0]) + 1) if neg else 5.0
    keep = [s for s in Dendrogram.compute(snr, **kw).trunk if s.get_peak()[1] >= peak]
    m = np.zeros(img.shape, bool)
    for s in keep:
        m[s.indices(subtree=True)] = True
    return m, dict(peak_bar=float(peak), negated_peaks=[round(float(v), 2) for v in neg[:6]], structures=len(keep),
                   sigma_line_mJy=[round(1e3 * float(rms[c]), 4) for c in line])


def finish(src, out, common, line, cell_arcsec, nsigma=1.5, niter=1_000_000, max_rounds=6, grow_stop=0.02, label="",
           smooth=()):
    """Continue the tclean run whose images are src/_tclean.* (copied to out/, src untouched) inside the growing
    dendrogram mask until it stops growing; `common` = the tclean keywords of that run except imagename, niter,
    nsigma and the mask. Writes out/{image,model,residual,psf,mask,emission}.fits, mask_round<k>.npy (the union used
    in round k), mask_final.npy and rounds.json; returns the rounds."""
    import casatools
    from casatasks import tclean, exportfits
    base = os.path.join(out, "_tclean")
    os.makedirs(out, exist_ok=True)
    for p in glob.glob(base + ".*"):
        shutil.rmtree(p)
    for p in glob.glob(os.path.join(src, "_tclean.*")):
        shutil.copytree(p, os.path.join(out, os.path.basename(p)))
    common = dict(common, imagename=base)
    nchan = int(common["nchan"]); imsize = int(common["imsize"])
    free = [i for i in range(nchan) if i not in line]
    dv = float(str(common["width"]).replace("km/s", ""))

    ia = casatools.image()
    ia.open(f"{base}.image"); b = ia.restoringbeam(); cs = ia.coordsys(); ia.close()
    if "major" not in b:                                                  # per-channel beams: take the median
        bb = b["beams"]; ch = [bb[k]["*0"] for k in sorted(bb, key=lambda s: int(s.strip("*")))]
        b = ch[len(ch) // 2]
    qa = casatools.quanta()
    bmaj, bmin = qa.convert(b["major"], "arcsec")["value"], qa.convert(b["minor"], "arcsec")["value"]
    beam_px = np.pi * bmaj * bmin / (4 * np.log(2)) / cell_arcsec ** 2
    B = (bmaj, bmin, qa.convert(b["positionangle"], "deg")["value"])
    if smooth:
        sys.path.insert(0, os.path.join(REPO, "scripts", "compact_vs_msclean_dendro"))
        import beams
        smooth = [s_ for s_ in smooth if s_ > bmaj]                       # coarser than the clean beam only
    print(f"{label}: continuing {src} -> {out}; scales {common.get('scales')}, nsigma {nsigma} inside the growing "
          f"dendrogram mask; beam {bmaj:.3f}\" x {bmin:.3f}\" ({beam_px:.0f} px)", flush=True)

    rounds, union = [], np.zeros((nchan, imsize, imsize), bool)
    for k in range(max_rounds + 1):
        res, mod = get(ia, f"{base}.residual"), get(ia, f"{base}.model")
        img = get(ia, f"{base}.image")
        state = dict(round=k, comps_line_Jy_kms=float(mod[line].sum() * dv),
                     resid_rms_line_mJy=1e3 * float(res[line].std()),
                     resid_rms_free_mJy=1e3 * float(res[free].std()) if free else None,
                     resid_max_line_mJy=1e3 * float(res[line].max()),
                     resid_flux_line_Jy_kms=float(res[line].sum() / beam_px * dv))
        if k == max_rounds:
            rounds.append(state); print("round", k, "(final)", state, flush=True); break
        t0 = time.time()
        m, info = dendro_mask(img, res, beam_px, line)
        for s_ in smooth:                                                 # the same mask at coarser resolutions
            bs = (s_, s_, 0.0)
            ms, i_s = dendro_mask(beams.to_beam(img, B, bs, cell_arcsec), beams.to_beam(res, B, bs, cell_arcsec),
                                  beams.omega(bs) / cell_arcsec ** 2, line)
            info[f"smooth_{s_:g}"] = dict(structures=i_s["structures"], peak_bar=i_s["peak_bar"],
                                          new_voxels=int((ms & ~m).sum()))
            m |= ms
        new = int((m & ~union).sum()); union |= m
        state.update(mask_info=info, mask_new_voxels=new, mask_voxels=int(union.sum()), mask_seconds=round(time.time() - t0))
        print("round", k, state, flush=True)
        if k > 0 and new < grow_stop * union.sum():
            state["stopped"] = f"mask grew by {new} voxels (< {grow_stop:.0%})"; rounds.append(state)
            print(state["stopped"], flush=True); break
        np.save(os.path.join(out, f"mask_round{k}.npy"), union.astype(np.uint8))
        np.save(os.path.join(out, "mask_final.npy"), union.astype(np.uint8))
        for p in (f"{base}.usermask", f"{base}.mask"):
            if os.path.exists(p):
                shutil.rmtree(p)
        ia.fromarray(outfile=f"{base}.usermask", pixels=union.astype(np.float32).transpose(2, 1, 0)[:, :, None, :],
                     csys=cs.torecord(), overwrite=True); ia.close()
        t0 = time.time()
        rec = tclean(niter=niter, nsigma=nsigma, usemask="user", mask=f"{base}.usermask",
                     calcpsf=False, calcres=False, **common)
        state.update(stopcode=rec.get("stopcode"), iterations=rec.get("iterdone"), major_cycles=rec.get("nmajordone"),
                     clean_seconds=round(time.time() - t0))
        print(f"  tclean: stopcode {state['stopcode']}, {state['iterations']} iterations, {state['major_cycles']} "
              f"major cycles, {state['clean_seconds']} s", flush=True)
        rounds.append(state)
        json.dump(rounds, open(os.path.join(out, "rounds.json"), "w"), indent=1)

    json.dump(rounds, open(os.path.join(out, "rounds.json"), "w"), indent=1)
    for ext in ("image", "model", "residual", "psf", "mask"):
        exportfits(imagename=f"{base}.{ext}", fitsimage=os.path.join(out, f"{ext}.fits"), velocity=True, overwrite=True)
    from astropy.io import fits
    with fits.open(os.path.join(out, "image.fits")) as hi, fits.open(os.path.join(out, "residual.fits")) as hr:
        em = hi[0].data - hr[0].data
        hdr = hi[0].header.copy(); hdr["HISTORY"] = "emission only: image - residual = CLEAN components * clean beam"
        fits.writeto(os.path.join(out, "emission.fits"), em.astype(np.float32), hdr, overwrite=True)
    print(f"done: {len(rounds)} rounds; FITS (image, model, residual, mask, psf, emission) in {out}", flush=True)
    return rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("galaxy", choices=sorted(GALAXIES)); ap.add_argument("config", choices=("extended", "compact"))
    ap.add_argument("--from", dest="src", default="ms_clean"); ap.add_argument("--out", default="msclean_complete")
    ap.add_argument("--scales", type=int, nargs="+", default=[0, 10, 30, 60, 120])
    ap.add_argument("--nsigma", type=float, default=1.5); ap.add_argument("--niter", type=int, default=1_000_000)
    ap.add_argument("--max-rounds", type=int, default=6); ap.add_argument("--grow-stop", type=float, default=0.02)
    ap.add_argument("--smooth", type=float, nargs="*", default=[], help="also mask at these circular beams, arcsec")
    a = ap.parse_args()

    redirect_casa_logs(f"msclean_{a.galaxy}_{a.config}_{a.out}.log")
    g = GALAXIES[a.galaxy]; paths = g["configs"][a.config]
    src = os.path.join(REPO, "data", a.galaxy, a.config, a.src)
    out = os.path.join(REPO, "data", a.galaxy, a.config, a.out)
    ref = g["configs"]["extended"][0]
    phasecenter = phasecenter_string(ref, g["field"])
    _, vsys = line_setup(ref, g["field"])
    start = vsys - (NCHAN - 1) / 2 * WIDTH_KMS
    spws = [str(line_setup(p, g["field"])[0]) for p in paths]
    common = dict(vis=list(paths), field=g["field"], spw=spws, datacolumn="data", imsize=IMSIZE,
                  cell=f"{CELL_ARCSEC}arcsec", phasecenter=phasecenter, specmode="cube", start=f"{start}km/s",
                  width=f"{WIDTH_KMS}km/s", nchan=NCHAN, restfreq=f"{RESTFREQ_GHZ}GHz", outframe="LSRK",
                  veltype="radio", gridder="standard", weighting=WEIGHTING, deconvolver="multiscale",
                  scales=list(a.scales), restoringbeam="common", pbcor=False, fullsummary=True)
    vel = vsys + WIDTH_KMS * (np.arange(NCHAN) - (NCHAN - 1) / 2)
    line = [i for i, v in enumerate(vel) if 4750 <= v <= 5280] if a.galaxy == "ngc3110" else list(range(NCHAN))
    finish(src, out, common, line, CELL_ARCSEC, a.nsigma, a.niter, a.max_rounds, a.grow_stop, f"{a.galaxy} {a.config}",
           a.smooth)


if __name__ == "__main__":
    main()
