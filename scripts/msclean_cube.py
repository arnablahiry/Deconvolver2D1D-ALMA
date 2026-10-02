#!/usr/bin/env python
"""
Multiscale CLEAN (tclean deconvolver="multiscale") of one galaxy/configuration's
CO(2-1) cube -- the standard CASA reference to compare the 2D-1D model against.

Same pixel grid, phase centre, spectral axis and weighting as the dirty cube of
deconvolution_2d1d.ipynb (scripts/build_deconvolution_notebook.py), so every
pixel and channel lines up with data/<galaxy>/<config>/cube/*.fits.

    python scripts/msclean_cube.py ngc3110 extended

Writes data/<galaxy>/<config>/ms_clean/{image,model,residual,psf,mask}.fits
(image restored with one common beam, Jy/beam; model in Jy/pixel) and the CASA
log casa_logs/msclean_<galaxy>_<config>.log.
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "simple"))

from casa_logs import redirect_casa_logs

VIS_ROOT = "/nas/datasets/ALMA_visibilities"
GALAXIES = dict(
    ngc3110=dict(field="ngc3110", configs=dict(
        extended=[f"{VIS_ROOT}/gal1/NGC3110_12m_extended_X1371/uid___A002_Xc7cf4e_X7866.ms.split.cal"],
        compact=[f"{VIS_ROOT}/gal1/NGC3110_12m_compact_X1373/uid___A002_Xcf92df_X4e8e.ms.split.cal",
                 f"{VIS_ROOT}/gal1/NGC3110_12m_compact_X1373/uid___A002_Xd09670_X3f8d.ms.split.cal"])),
    ngc2369=dict(field="ngc2369", configs=dict(
        extended=[f"{VIS_ROOT}/gal2/NGC2369_12m_extended_X136b/uid___A002_Xc845c0_X1710.ms.split.cal"],
        compact=[f"{VIS_ROOT}/gal2/NGC2369_12m_compact_X136d/uid___A002_Xcd07af_X20e.ms.split.cal"])),
)

# Grid of the notebook's dirty cubes.
IMSIZE = 800
CELL_ARCSEC = 0.02
RESTFREQ_GHZ = 230.538
NCHAN = 32
WIDTH_KMS = 25.0
WEIGHTING = "natural"

# Multiscale: point + 1, 3, 6 x the extended beam FWHM (~0.2" = 10 px), all
# below the extended LRS (2.4"); override with --scales for the compact data.
SCALES_PX = [0, 10, 30, 60]
NSIGMA = 3.0
# auto-multithresh, ALMA pipeline values for 12m long-baseline cubes
AUTOMASK = dict(sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5,
                negativethreshold=7.0, minbeamfrac=0.3, growiterations=75)


def field_id(md, field):
    ids = md.fieldsforname(field)
    if len(ids) == 0:
        raise ValueError(f"no field {field!r}; fields: {md.fieldnames()}")
    return int(ids[0])


def phasecenter_string(msname, field):
    import casatools
    md = casatools.msmetadata()
    md.open(msname)
    try:
        d = md.phasecenter(field_id(md, field))
    finally:
        md.close()
    qa = casatools.quanta()
    return f"{d['refer']} {qa.time(d['m0'], prec=11)[0]} {qa.angle(d['m1'], prec=10)[0]}"


def line_setup(msname, field):
    """(spw id, systemic velocity km/s) of CO(2-1) from the MS's SOURCE table."""
    import casatools
    tb = casatools.table()
    tb.open(os.path.join(msname, "SOURCE"))
    try:
        for i in range(tb.nrows()):
            if tb.getcell("NAME", i) != field or not tb.iscelldefined("REST_FREQUENCY", i):
                continue
            rest = tb.getcell("REST_FREQUENCY", i)
            if len(rest) and abs(rest[0] / 1e9 - RESTFREQ_GHZ) < 1e-3:
                return int(tb.getcell("SPECTRAL_WINDOW_ID", i)), float(tb.getcell("SYSVEL", i)[0]) / 1e3
    finally:
        tb.close()
    raise ValueError(f"no {RESTFREQ_GHZ} GHz line for {field!r} in {msname}/SOURCE")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("galaxy", choices=sorted(GALAXIES))
    ap.add_argument("config", choices=("extended", "compact"))
    ap.add_argument("--scales", type=int, nargs="+", default=SCALES_PX, help="pixels")
    ap.add_argument("--nsigma", type=float, default=NSIGMA)
    ap.add_argument("--niter", type=int, default=1_000_000)
    args = ap.parse_args()

    import casatools
    from casatasks import tclean, exportfits

    g = GALAXIES[args.galaxy]
    paths = g["configs"][args.config]
    redirect_casa_logs(f"msclean_{args.galaxy}_{args.config}.log")

    # Both configurations use the extended MS's phase centre and v_sys, as in the notebook.
    ref = g["configs"]["extended"][0]
    phasecenter = phasecenter_string(ref, g["field"])
    _, vsys = line_setup(ref, g["field"])
    start = vsys - (NCHAN - 1) / 2 * WIDTH_KMS
    spws = [str(line_setup(p, g["field"])[0]) for p in paths]

    out = os.path.join(REPO_ROOT, "data", args.galaxy, args.config, "ms_clean")
    os.makedirs(out, exist_ok=True)
    base = os.path.join(out, "_tclean")
    for p in glob.glob(base + ".*"):
        shutil.rmtree(p)
    print(f"{args.galaxy} {args.config}: {len(paths)} MS, spw {spws}, phase centre {phasecenter}, "
          f"start {start:.1f} km/s, scales {args.scales} px, nsigma {args.nsigma}", flush=True)

    rec = tclean(vis=list(paths), field=g["field"], spw=spws, datacolumn="data",
                 imagename=base, imsize=IMSIZE, cell=f"{CELL_ARCSEC}arcsec",
                 phasecenter=phasecenter, specmode="cube", start=f"{start}km/s",
                 width=f"{WIDTH_KMS}km/s", nchan=NCHAN, restfreq=f"{RESTFREQ_GHZ}GHz",
                 outframe="LSRK", veltype="radio", gridder="standard", weighting=WEIGHTING,
                 deconvolver="multiscale", scales=list(args.scales), niter=args.niter,
                 nsigma=args.nsigma, usemask="auto-multithresh", **AUTOMASK,
                 restoringbeam="common", pbcor=False, fullsummary=True)
    print(f"stopcode {rec.get('stopcode')}, {rec.get('iterdone')} iterations, "
          f"{rec.get('nmajordone')} major cycles", flush=True)

    for ext in ("image", "model", "residual", "psf", "mask"):
        exportfits(imagename=f"{base}.{ext}", fitsimage=os.path.join(out, f"{ext}.fits"),
                   velocity=True, overwrite=True)

    ia = casatools.image()
    ia.open(f"{base}.image")
    b = ia.restoringbeam()
    ia.close()
    ia.open(f"{base}.residual")
    res = ia.getchunk()
    ia.close()
    rms = np.std(res, axis=(0, 1)).ravel()
    print(f"common beam {b['major']['value']:.3f}\" x {b['minor']['value']:.3f}\" "
          f"@ {b['positionangle']['value']:.1f} deg; residual rms per channel "
          f"{rms.min() * 1e3:.3f}-{rms.max() * 1e3:.3f} mJy/beam; FITS in {os.path.relpath(out, REPO_ROOT)}")


if __name__ == "__main__":
    main()
