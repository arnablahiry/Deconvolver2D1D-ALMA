#!/usr/bin/env python
"""
Briggs-weighted CO(2-1) quick-look image of one galaxy/configuration of the
sample under /nas/datasets/ALMA_visibilities -- the input to
scripts/sample/fig_compact_vs_extended.py.

One tclean specmode="mfs" plane over the CO(2-1) channels (v_sys +- 400 km/s,
v_sys from the MS's SOURCE table), science fields only, robust 0.5,
auto-multithresh with the ALMA pipeline 12m thresholds, multiscale CLEAN
(0, 1, 3, 6 x beam). Multi-pointing targets
(IC4518 E/W, MCG-02-33-098, NGC3256) are imaged with the mosaic gridder;
IC4518E / IC4518W image one member each as a single pointing. Both
configurations of a galaxy share the extended MS's phase centre so the two
images line up.

    python scripts/sample/image_co21_briggs.py IC5179 compact

Writes data/sample/<galaxy>/<config>/co21_briggs/{image,residual,pb}.fits
and meta.json (beam, cell, rms, phase centre); CASA log in
casa_logs/sample_co21_briggs_<galaxy>_<config>.log.
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "simple"))

VIS_ROOT = "/nas/datasets/ALMA_visibilities"
OUT_ROOT = os.path.join(REPO_ROOT, "data", "sample")

# Order of the user's target table (+ NGC3110, NGC2369, the repo's original galaxies). fields=None: every OBSERVE_TARGET field.
SAMPLE = {
    "ESO297-G011": dict(label="ESO 297-G011", fields=["ESO297-G011"]),   # ESO297-G012 is a separate pointing
    "NGC1614": dict(label="NGC 1614", fields=None),
    "MCG-02-33-098": dict(label="MCG-02-33-098", fields=None),
    "IC4518": dict(label="IC 4518 E/W", fields=None),                     # IC4518E + IC4518W
    "IRASF17138-1017": dict(label="IRAS F17138-1017", fields=None),
    "NGC7130": dict(label="NGC 7130", fields=None),
    "IC5179": dict(label="IC 5179", fields=None),
    "NGC5135": dict(label="NGC 5135", fields=None),
    "ESO320-G030": dict(label="ESO 320-G030", fields=None),
    "IC4687": dict(label="IC 4687", fields=None),
    "NGC3256": dict(label="NGC 3256", fields=None),
    "NGC3110": dict(label="NGC 3110", fields=None),
    "NGC2369": dict(label="NGC 2369", fields=None),
}
# Single-pointing images of the two members of a multi-field target; "data" is the
# folder under VIS_ROOT they are imaged from.
SPLIT = {
    "IC4518W": dict(label="IC 4518W", fields=["IC4518W"], data="IC4518"),
    "IC4518E": dict(label="IC 4518E", fields=["IC4518E"], data="IC4518"),
}
TARGETS = {**SAMPLE, **SPLIT}
CONFIGS = ("compact", "extended")

RESTFREQ_GHZ = 230.538
HALF_WIDTH_KMS = 400.0
ROBUST = 0.5
NSIGMA = 2.5
# multiscale CLEAN scales in beams (point, 1, 3, 6 x beam, as scripts/msclean_cube.py)
SCALES_BEAMS = (0, 1, 3, 6)
PIX_PER_BEAM = 5
C_KMS = 299792.458
DISH_M = 12.0
FOV_PB = 1.6               # image width beyond the outermost pointings, in primary-beam FWHM
# ALMA pipeline auto-multithresh, 12m: short / long baselines (75th pct > 400 m)
AUTOMASK = dict(
    compact=dict(sidelobethreshold=2.0, noisethreshold=4.25, lownoisethreshold=1.5,
                 negativethreshold=0.0, minbeamfrac=0.3, growiterations=75),
    extended=dict(sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5,
                  negativethreshold=0.0, minbeamfrac=0.3, growiterations=75),
)


def ms_paths(galaxy, config):
    d = os.path.join(VIS_ROOT, galaxy, config)
    paths = sorted(glob.glob(os.path.join(d, "*.ms.split.cal")) + glob.glob(os.path.join(d, "uid___*.ms")))
    return paths    # NGC5135/extended/calibrated.ms is a concat() duplicate of its .ms.split.cal


def science_fields(md, wanted):
    ids = sorted(int(i) for i in md.fieldsforintent("OBSERVE_TARGET#ON_SOURCE"))
    if wanted is not None:
        ids = [i for i in ids if md.namesforfields(i)[0] in wanted]
    return ids


def systemic_velocity(msname, names):
    import casatools
    tb = casatools.table()
    tb.open(os.path.join(msname, "SOURCE"))
    try:
        v = [float(tb.getcell("SYSVEL", i)[0]) / 1e3 for i in range(tb.nrows())
             if tb.getcell("NAME", i) in names and tb.iscelldefined("SYSVEL", i) and len(tb.getcell("SYSVEL", i))]
    finally:
        tb.close()
    if not v:
        raise ValueError(f"no SYSVEL for {names} in {msname}/SOURCE")
    return float(np.mean(v))


def line_spw_selection(md, vsys):
    """'spw:f0~f1GHz,...' over every spw overlapping v_sys +- HALF_WIDTH_KMS (optical convention)."""
    f_hi = RESTFREQ_GHZ / (1 + (vsys - HALF_WIDTH_KMS) / C_KMS)
    f_lo = RESTFREQ_GHZ / (1 + (vsys + HALF_WIDTH_KMS) / C_KMS)
    sel = []
    for s in md.spwsforintent("OBSERVE_TARGET#ON_SOURCE"):
        f = md.chanfreqs(s) / 1e9
        if len(f) < 8 or f.max() < f_lo or f.min() > f_hi:
            continue
        sel.append(f"{s}:{max(f_lo, f.min()):.6f}~{min(f_hi, f.max()):.6f}GHz")
    if not sel:
        raise ValueError(f"no spw covers {f_lo:.3f}-{f_hi:.3f} GHz")
    return ",".join(sel), 0.5 * (f_lo + f_hi)


def pointings(md, ids):
    """(ra, dec) radians of each pointing."""
    out = []
    for i in ids:
        d = md.phasecenter(i)
        out.append((d["m0"]["value"], d["m1"]["value"]))
    return np.array(out)


def max_uv_m(msname, ids):
    import casatools
    tb = casatools.table()
    tb.open(msname)
    try:
        q = tb.query(f"FIELD_ID IN [{','.join(map(str, ids))}] AND ANTENNA1!=ANTENNA2", columns="UVW")
        uvw = q.getcol("UVW", rowincr=7)
        q.close()
    finally:
        tb.close()
    return float(np.sqrt(uvw[0] ** 2 + uvw[1] ** 2).max())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("galaxy", choices=list(TARGETS))
    ap.add_argument("config", choices=CONFIGS)
    ap.add_argument("--dirty", action="store_true", help="niter=0: dirty product only, in <product>_dirty/")
    args = ap.parse_args()

    from casa_logs import redirect_casa_logs
    redirect_casa_logs(f"sample_co21_briggs{'_dirty' if args.dirty else ''}_{args.galaxy}_{args.config}.log")
    import casatools
    from casatasks import tclean, exportfits

    g = TARGETS[args.galaxy]
    data = g.get("data", args.galaxy)
    paths = ms_paths(data, args.config)
    if not paths:
        raise SystemExit(f"no MS for {args.galaxy} {args.config}")
    ref_paths = ms_paths(data, "extended") or paths

    md = casatools.msmetadata()
    # Phase centre / mosaic extent from the extended MS (or this one if extended is missing).
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

    spws, uvmax, nptg = [], 0.0, 0
    for p in paths:
        md.open(p)
        ids = science_fields(md, g["fields"])
        nptg = max(nptg, len(ids))
        vsys = systemic_velocity(p, names)
        sel, f_line = line_spw_selection(md, vsys)
        md.close()
        spws.append(sel)
        uvmax = max(uvmax, max_uv_m(p, ids))

    lam_m = C_KMS * 1e3 / (f_line * 1e9)
    cell = np.degrees(lam_m / uvmax) * 3600 / PIX_PER_BEAM
    cell = float(f"{cell:.2g}")
    pb_fwhm = np.degrees(1.13 * lam_m / DISH_M) * 3600
    dx = np.degrees((pts[:, 0] - ra0) * np.cos(dec0)) * 3600
    dy = np.degrees(pts[:, 1] - dec0) * 3600
    extent = max(np.ptp(dx), np.ptp(dy)) if len(pts) > 1 else 0.0
    fov = extent + FOV_PB * pb_fwhm
    imsize = int(casatools.synthesisutils().getOptimumSize(int(np.ceil(fov / cell))))
    mosaic = nptg > 1

    out = os.path.join(OUT_ROOT, args.galaxy, args.config, "co21_briggs" + ("_dirty" if args.dirty else ""))
    os.makedirs(out, exist_ok=True)
    base = os.path.join(out, "_tclean")
    for p in glob.glob(base + ".*"):
        shutil.rmtree(p)
    print(f"{args.galaxy} {args.config}: {len(paths)} MS, fields {names} ({nptg} pointings), spw {spws}, "
          f"uvmax {uvmax:.0f} m, cell {cell}\", imsize {imsize}, "
          f"{'mosaic' if mosaic else 'standard'}, phase centre {phasecenter}", flush=True)

    scales = sorted({int(round(k * PIX_PER_BEAM)) for k in SCALES_BEAMS})    # multiscale CLEAN, in pixels
    clean = (dict(niter=0) if args.dirty else
             dict(niter=1_000_000, nsigma=NSIGMA, usemask="auto-multithresh", **AUTOMASK[args.config]))
    rec = tclean(vis=paths, field=",".join(names), spw=spws, datacolumn="data",
                 imagename=base, imsize=imsize, cell=f"{cell}arcsec", phasecenter=phasecenter,
                 specmode="mfs", gridder="mosaic" if mosaic else "standard",
                 weighting="briggs", robust=ROBUST, deconvolver="multiscale", scales=scales,
                 **clean, pbcor=False, fullsummary=False)
    print(f"stopcode {rec.get('stopcode')}, {rec.get('iterdone')} iterations, "
          f"{rec.get('nmajordone')} major cycles", flush=True)

    for ext in ("image", "residual", "pb"):
        exportfits(imagename=f"{base}.{ext}", fitsimage=os.path.join(out, f"{ext}.fits"), overwrite=True)

    ia = casatools.image()
    ia.open(f"{base}.image")
    b = ia.restoringbeam()
    ia.close()
    ia.open(f"{base}.residual")
    res = np.squeeze(ia.getchunk())
    ia.close()
    ia.open(f"{base}.pb")
    pb = np.squeeze(ia.getchunk())
    ia.close()
    rms = float(1.4826 * np.median(np.abs(res[pb > 0.5])))
    meta = dict(galaxy=args.galaxy, config=args.config, ms=paths, fields=names, spw=spws,
                phasecenter=phasecenter, cell_arcsec=cell, imsize=imsize, mosaic=mosaic,
                beam_arcsec=[b["major"]["value"], b["minor"]["value"], b["positionangle"]["value"]],
                rms_jy_beam=rms, robust=ROBUST, half_width_kms=HALF_WIDTH_KMS, scales_px=scales,
                stopcode=rec.get("stopcode"), iterdone=rec.get("iterdone"))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    for p in glob.glob(base + ".*"):
        shutil.rmtree(p)
    print(f"beam {b['major']['value']:.3f}\" x {b['minor']['value']:.3f}\", rms {rms * 1e3:.3f} mJy/beam; "
          f"FITS in {os.path.relpath(out, REPO_ROOT)}")


if __name__ == "__main__":
    main()
