#!/usr/bin/env python
"""
Briggs-weighted CO(2-1) cube of one galaxy/configuration of the sample, for the
integrated spectra of scripts/sample/fig_compact_vs_extended.py.

Same targets, fields, phase centre, weighting (robust 0.5) and auto-multithresh
as image_co21_briggs.py, but specmode="cube": NCHAN channels of WIDTH_KMS
centred on the CO(2-1) frequency of v_sys (frequency axis, LSRK), on a cell
CELL_FACTOR x coarser than the quick-look image (the spectrum is summed over an
aperture, so ~2.5 pixels per beam is enough).

    python scripts/sample/image_co21_cube.py IC5179 extended

Writes data/sample/<galaxy>/<config>/co21_cube/{image,residual,model}.fits and meta.json;
CASA log in casa_logs/sample_co21_cube_<galaxy>_<config>.log.
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_co21_briggs import (AUTOMASK, C_KMS, CONFIGS, DISH_M, FOV_PB, OUT_ROOT, PIX_PER_BEAM, SCALES_BEAMS,
                               REPO_ROOT, RESTFREQ_GHZ, ROBUST, TARGETS, max_uv_m, ms_paths,
                               pointings, science_fields, systemic_velocity)

sys.path.insert(0, os.path.join(REPO_ROOT, "simple"))

HALF_RANGE_KMS = 500.0
WIDTH_KMS = 20.0
NCHAN = int(round(2 * HALF_RANGE_KMS / WIDTH_KMS)) + 1
CELL_FACTOR = 2.0
NSIGMA = 3.0
SEL_MARGIN_CHAN = 5


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("galaxy", choices=list(TARGETS))
    ap.add_argument("config", choices=CONFIGS)
    ap.add_argument("--dirty", action="store_true", help="niter=0: dirty product only, in <product>_dirty/")
    args = ap.parse_args()

    from casa_logs import redirect_casa_logs
    redirect_casa_logs(f"sample_co21_cube{'_dirty' if args.dirty else ''}_{args.galaxy}_{args.config}.log")
    import casatools
    from casatasks import exportfits, tclean

    g = TARGETS[args.galaxy]
    data = g.get("data", args.galaxy)
    paths = ms_paths(data, args.config)
    if not paths:
        raise SystemExit(f"no MS for {args.galaxy} {args.config}")
    ref_paths = ms_paths(data, "extended") or paths

    md = casatools.msmetadata()
    md.open(ref_paths[0])
    ref_ids = science_fields(md, g["fields"])
    names = sorted(set(md.namesforfields(ref_ids)))
    pts = pointings(md, ref_ids)
    md.close()
    ra0 = np.angle(np.exp(1j * pts[:, 0]).mean())
    dec0 = pts[:, 1].mean()
    qa = casatools.quanta()
    phasecenter = (f"J2000 {qa.time(qa.quantity(ra0, 'rad'), prec=11)[0]} "
                   f"{qa.angle(qa.quantity(dec0, 'rad'), prec=10)[0]}")

    vsys = systemic_velocity(ref_paths[0], names)
    f_line = RESTFREQ_GHZ / (1 + vsys / C_KMS)                # GHz, optical v_sys as in the images
    dnu = f_line * WIDTH_KMS / C_KMS                          # GHz per channel
    # data selection is in the MS (TOPO) frame, the cube in LSRK: keep a SEL_MARGIN_CHAN margin
    f_lo = f_line - HALF_RANGE_KMS / C_KMS * f_line - SEL_MARGIN_CHAN * dnu
    f_hi = f_line + HALF_RANGE_KMS / C_KMS * f_line + SEL_MARGIN_CHAN * dnu

    spws, uvmax, nptg = [], 0.0, 0
    for p in paths:
        md.open(p)
        ids = science_fields(md, g["fields"])
        nptg = max(nptg, len(ids))
        sel = []
        for s in md.spwsforintent("OBSERVE_TARGET#ON_SOURCE"):
            f = md.chanfreqs(s) / 1e9
            if len(f) >= 8 and f.max() > f_lo and f.min() < f_hi:
                sel.append(f"{s}:{max(f_lo, f.min()):.6f}~{min(f_hi, f.max()):.6f}GHz")
        md.close()
        spws.append(",".join(sel))
        uvmax = max(uvmax, max_uv_m(p, ids))

    lam_m = C_KMS * 1e3 / (f_line * 1e9)
    cell = float(f"{np.degrees(lam_m / uvmax) * 3600 / PIX_PER_BEAM * CELL_FACTOR:.2g}")
    pb_fwhm = np.degrees(1.13 * lam_m / DISH_M) * 3600
    dx = np.degrees((pts[:, 0] - ra0) * np.cos(dec0)) * 3600
    dy = np.degrees(pts[:, 1] - dec0) * 3600
    extent = max(np.ptp(dx), np.ptp(dy)) if len(pts) > 1 else 0.0
    imsize = int(casatools.synthesisutils().getOptimumSize(int(np.ceil((extent + FOV_PB * pb_fwhm) / cell))))
    mosaic = nptg > 1
    start = f_line - (NCHAN - 1) / 2 * dnu

    out = os.path.join(OUT_ROOT, args.galaxy, args.config, "co21_cube" + ("_dirty" if args.dirty else ""))
    os.makedirs(out, exist_ok=True)
    base = os.path.join(out, "_tclean")
    for p in glob.glob(base + ".*"):
        shutil.rmtree(p)
    print(f"{args.galaxy} {args.config}: {len(paths)} MS, fields {names}, spw {spws}, v_sys {vsys:.1f} km/s, "
          f"f_line {f_line:.5f} GHz, {NCHAN} x {WIDTH_KMS} km/s, cell {cell}\", imsize {imsize}, "
          f"{'mosaic' if mosaic else 'standard'}", flush=True)

    scales = sorted({int(round(k * PIX_PER_BEAM / CELL_FACTOR)) for k in SCALES_BEAMS})    # multiscale CLEAN, in pixels
    clean = (dict(niter=0) if args.dirty else
             dict(niter=1_000_000, nsigma=NSIGMA, usemask="auto-multithresh", **AUTOMASK[args.config]))
    rec = tclean(vis=paths, field=",".join(names), spw=spws, datacolumn="data",
                 imagename=base, imsize=imsize, cell=f"{cell}arcsec", phasecenter=phasecenter,
                 specmode="cube", nchan=NCHAN, start=f"{start:.9f}GHz", width=f"{dnu:.9f}GHz",
                 outframe="LSRK", restfreq=f"{RESTFREQ_GHZ}GHz",
                 gridder="mosaic" if mosaic else "standard", weighting="briggs", robust=ROBUST,
                 deconvolver="multiscale", scales=scales, **clean, restoringbeam="common", pbcor=False, fullsummary=False)
    print(f"stopcode {rec.get('stopcode')}, {rec.get('iterdone')} iterations, "
          f"{rec.get('nmajordone')} major cycles", flush=True)

    for ext in ("image", "residual", "model"):      # image - residual = clean model (x) clean beam
        exportfits(imagename=f"{base}.{ext}", fitsimage=os.path.join(out, f"{ext}.fits"), overwrite=True)
    ia = casatools.image()
    ia.open(f"{base}.image")
    b = ia.restoringbeam()
    ia.close()
    meta = dict(galaxy=args.galaxy, config=args.config, ms=paths, fields=names, spw=spws,
                phasecenter=phasecenter, vsys_kms=vsys, f_line_ghz=f_line, width_kms=WIDTH_KMS,
                nchan=NCHAN, cell_arcsec=cell, imsize=imsize, mosaic=mosaic, scales_px=scales,
                beam_arcsec=[b["major"]["value"], b["minor"]["value"], b["positionangle"]["value"]],
                stopcode=rec.get("stopcode"), iterdone=rec.get("iterdone"))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    for p in glob.glob(base + ".*"):
        shutil.rmtree(p)
    print(f"beam {b['major']['value']:.3f}\" x {b['minor']['value']:.3f}\"; FITS in {os.path.relpath(out, REPO_ROOT)}")


if __name__ == "__main__":
    main()
