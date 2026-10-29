#!/usr/bin/env python
"""
NGC 3110's extended data imaged at the compact resolution: natural weighting with a uv taper chosen so the extended
PSF's fitted beam matches the compact clean beam, then multiscale CLEAN restored with exactly that beam, finished
inside the growing astrodendro mask (msclean_complete.finish). The tapered extended and the compact CLEAN then share
one restoring beam and can be compared image to image. The taper only reweights: it does not add the short spacings
the extended array lacks, so structure larger than its shortest baselines can still be missing.

  1. taper: tclean niter = 0 (PSF only) with an elliptical uvtaper (image-plane FWHM, the compact beam's PA); the
     taper is adjusted (quadrature correction of each axis towards the target) until the fitted PSF beam is within
     2% of the compact clean beam's axes, at most 4 tries
  2. tclean multiscale, scales 0, 60, 180 px (0, 1, 3 compact beams), nsigma 3, auto-multithresh with the ALMA
     pipeline's short-baseline values (as the compact CLEAN), restoringbeam = the compact clean beam
  3. msclean_complete.finish: inside the growing astrodendro mask, nsigma 1.5

Writes data/ngc3110/extended/msclean_taper{,_finished}/ (FITS as msclean_cube / msclean_complete) and taper.json.
CASA env:

    python scripts/experiments/msclean_extended_taper.py
"""
import glob
import json
import os
import shutil
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("scripts", "simple"):
    sys.path.insert(0, os.path.join(REPO, p))
from msclean_cube import GALAXIES, IMSIZE, CELL_ARCSEC, RESTFREQ_GHZ, NCHAN, WIDTH_KMS, phasecenter_string, line_setup  # noqa: E402
from msclean_complete import finish  # noqa: E402
from casa_logs import redirect_casa_logs  # noqa: E402

import casatools  # noqa: E402
from casatasks import tclean, exportfits  # noqa: E402
from astropy.io import fits  # noqa: E402

redirect_casa_logs("msclean_ngc3110_extended_taper.log")
h = fits.getheader(os.path.join(REPO, "results/NGC3110/compact/msclean/image.fits"))
TARGET = (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"])                      # the compact clean beam
OUT = os.path.join(REPO, "data/ngc3110/extended/msclean_taper"); os.makedirs(OUT, exist_ok=True)
base = os.path.join(OUT, "_tclean")

g = GALAXIES["ngc3110"]; paths = g["configs"]["extended"]; ref = paths[0]
_, vsys = line_setup(ref, g["field"])
common = dict(vis=list(paths), field=g["field"], spw=[str(line_setup(p, g["field"])[0]) for p in paths],
              datacolumn="data", imsize=IMSIZE, cell=f"{CELL_ARCSEC}arcsec", phasecenter=phasecenter_string(ref, g["field"]),
              specmode="cube", start=f"{vsys - (NCHAN - 1) / 2 * WIDTH_KMS}km/s", width=f"{WIDTH_KMS}km/s", nchan=NCHAN,
              restfreq=f"{RESTFREQ_GHZ}GHz", outframe="LSRK", veltype="radio", gridder="standard", weighting="natural",
              deconvolver="multiscale", scales=[0, 60, 180], pbcor=False, fullsummary=True,
              restoringbeam=[f"{TARGET[0]}arcsec", f"{TARGET[1]}arcsec", f"{TARGET[2]}deg"])


def clear():
    for p in glob.glob(base + ".*"):
        shutil.rmtree(p)


def psf_beam():
    """median-channel fitted beam of the PSF, arcsec / deg"""
    ia, qa = casatools.image(), casatools.quanta()
    ia.open(base + ".psf"); b = ia.restoringbeam(); ia.close()
    if "beams" in b:
        bb = b["beams"]; b = [bb[k]["*0"] for k in sorted(bb, key=lambda s: int(s.strip("*")))][NCHAN // 2]
    return (qa.convert(b["major"], "arcsec")["value"], qa.convert(b["minor"], "arcsec")["value"],
            qa.convert(b["positionangle"], "deg")["value"])


# ---------------------------------------------------------------- 1. the taper
taper, tries = [TARGET[0], TARGET[1]], []
for it in range(4):
    clear()
    tclean(imagename=base, niter=0, uvtaper=[f"{taper[0]:.4f}arcsec", f"{taper[1]:.4f}arcsec", f"{TARGET[2]:.2f}deg"],
           calcres=False, **common)
    b = psf_beam(); tries.append(dict(taper=list(taper), psf_beam=list(b)))
    print(f"taper {taper[0]:.3f}\" x {taper[1]:.3f}\" -> PSF beam {b[0]:.3f}\" x {b[1]:.3f}\" PA {b[2]:.1f} "
          f"(target {TARGET[0]:.3f} x {TARGET[1]:.3f} PA {TARGET[2]:.1f})", flush=True)
    if abs(b[0] / TARGET[0] - 1) < 0.02 and abs(b[1] / TARGET[1] - 1) < 0.02:
        break
    taper = [float(np.sqrt(max(t ** 2 + T ** 2 - m ** 2, 0.01))) for t, T, m in zip(taper, TARGET[:2], b[:2])]
UVTAPER = [f"{taper[0]:.4f}arcsec", f"{taper[1]:.4f}arcsec", f"{TARGET[2]:.2f}deg"]
json.dump(dict(target_beam=TARGET, uvtaper=UVTAPER, tries=tries), open(os.path.join(OUT, "taper.json"), "w"), indent=1)

# ---------------------------------------------------------------- 2. multiscale CLEAN at the compact resolution
clear()
common["uvtaper"] = UVTAPER
rec = tclean(imagename=base, niter=1_000_000, nsigma=3.0, usemask="auto-multithresh", sidelobethreshold=2.0,
             noisethreshold=4.25, lownoisethreshold=1.5, negativethreshold=15.0, minbeamfrac=0.3, growiterations=75,
             **common)
print(f"tclean: stopcode {rec.get('stopcode')}, {rec.get('iterdone')} iterations, {rec.get('nmajordone')} major cycles",
      flush=True)
for ext in ("image", "model", "residual", "psf", "mask"):
    exportfits(imagename=f"{base}.{ext}", fitsimage=os.path.join(OUT, f"{ext}.fits"), velocity=True, overwrite=True)

# ---------------------------------------------------------------- 3. finished inside the growing astrodendro mask
vel = vsys + WIDTH_KMS * (np.arange(NCHAN) - (NCHAN - 1) / 2)
line = [i for i, v in enumerate(vel) if 4750 <= v <= 5280]
finish(OUT, OUT + "_finished", {k: v for k, v in common.items() if k != "imagename"}, line, CELL_ARCSEC, nsigma=1.5,
       label="ngc3110 extended (tapered)")
print("DONE", flush=True)
