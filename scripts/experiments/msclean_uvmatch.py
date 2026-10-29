#!/usr/bin/env python
"""
NGC 3110's compact and extended data imaged from the same spatial frequencies only: both restricted to the uv range
both arrays sample (UVRANGE, default 83~341 klambda: the extended array's 5th percentile to the compact array's
longest baseline), natural weighting, the narrower of the two PSFs tapered to the wider one, multiscale CLEAN restored
with that one beam and finished inside the growing astrodendro mask. The two images then contain the same Fourier
components, so any difference beyond noise and flux calibration is in the data or the imaging, not the coverage.

    python scripts/experiments/msclean_uvmatch.py psf                 # both PSFs -> data/ngc3110/uvmatch.json
    python scripts/experiments/msclean_uvmatch.py clean compact|extended

Writes data/ngc3110/<config>/msclean_uvmatch{,_finished}/ (FITS as msclean_cube / msclean_complete). CASA env.
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
from casa_logs import redirect_casa_logs  # noqa: E402

UVRANGE = os.environ.get("UVRANGE", "83~341klambda")
JS = os.path.join(REPO, "data/ngc3110/uvmatch.json")
mode = sys.argv[1]; cfg = sys.argv[2] if len(sys.argv) > 2 else None
redirect_casa_logs(f"msclean_ngc3110_uvmatch_{mode}{'_' + cfg if cfg else ''}.log")
import casatools  # noqa: E402
from casatasks import tclean, exportfits  # noqa: E402

g = GALAXIES["ngc3110"]; ref = g["configs"]["extended"][0]
_, vsys = line_setup(ref, g["field"])


def common(c):
    paths = g["configs"][c]
    return dict(vis=list(paths), field=g["field"], spw=[str(line_setup(p, g["field"])[0]) for p in paths],
                datacolumn="data", imsize=IMSIZE, cell=f"{CELL_ARCSEC}arcsec", phasecenter=phasecenter_string(ref, g["field"]),
                specmode="cube", start=f"{vsys - (NCHAN - 1) / 2 * WIDTH_KMS}km/s", width=f"{WIDTH_KMS}km/s", nchan=NCHAN,
                restfreq=f"{RESTFREQ_GHZ}GHz", outframe="LSRK", veltype="radio", gridder="standard", weighting="natural",
                uvrange=UVRANGE, deconvolver="multiscale", pbcor=False, fullsummary=True)


def psf_beam(base):
    ia, qa = casatools.image(), casatools.quanta()
    ia.open(base + ".psf"); b = ia.restoringbeam(); ia.close()
    if "beams" in b:
        bb = b["beams"]; b = [bb[k]["*0"] for k in sorted(bb, key=lambda s: int(s.strip("*")))][NCHAN // 2]
    return [qa.convert(b["major"], "arcsec")["value"], qa.convert(b["minor"], "arcsec")["value"],
            qa.convert(b["positionangle"], "deg")["value"]]


def psf(c, base, taper=None):
    for p in glob.glob(base + ".*"):
        shutil.rmtree(p)
    kw = dict(uvtaper=[f"{taper[0]:.4f}arcsec", f"{taper[1]:.4f}arcsec", f"{taper[2]:.2f}deg"]) if taper else {}
    tclean(imagename=base, niter=0, calcres=False, **kw, **common(c))
    return psf_beam(base)


if mode == "psf":
    out = {}
    for c in ("compact", "extended"):
        d = os.path.join(REPO, "data/ngc3110", c, "uvmatch_psf"); os.makedirs(d, exist_ok=True)
        out[c] = psf(c, os.path.join(d, "_tclean"))
        print(f"{c}: {UVRANGE}, natural: PSF beam {out[c][0]:.3f}\" x {out[c][1]:.3f}\" PA {out[c][2]:.1f}", flush=True)
    big = max(out, key=lambda c: out[c][0] * out[c][1])
    json.dump(dict(uvrange=UVRANGE, psf_beams=out, target_config=big, target_beam=out[big]), open(JS, "w"), indent=1)
    print(f"target beam: {big}'s {out[big]}", flush=True)
    sys.exit()

# ---------------------------------------------------------------- clean <config>
J = json.load(open(JS)); TARGET = J["target_beam"]
OUT = os.path.join(REPO, "data/ngc3110", cfg, "msclean_uvmatch"); os.makedirs(OUT, exist_ok=True)
base = os.path.join(OUT, "_tclean")
kw = common(cfg)
taper = None
if cfg != J["target_config"]:                                            # taper this one's PSF to the target beam
    b = J["psf_beams"][cfg]; taper = [float(np.sqrt(max(T ** 2 - m ** 2, 0.01))) for T, m in zip(TARGET[:2], b[:2])] + [TARGET[2]]
    for it in range(4):
        b = psf(cfg, base, taper)
        print(f"{cfg}: taper {taper[0]:.3f}\" x {taper[1]:.3f}\" -> PSF {b[0]:.3f}\" x {b[1]:.3f}\" PA {b[2]:.1f} "
              f"(target {TARGET[0]:.3f} x {TARGET[1]:.3f} PA {TARGET[2]:.1f})", flush=True)
        if abs(b[0] / TARGET[0] - 1) < 0.02 and abs(b[1] / TARGET[1] - 1) < 0.02:
            break
        taper = [float(np.sqrt(max(t ** 2 + T ** 2 - m ** 2, 0.01))) for t, T, m in zip(taper[:2], TARGET[:2], b[:2])] + [TARGET[2]]
    kw["uvtaper"] = [f"{taper[0]:.4f}arcsec", f"{taper[1]:.4f}arcsec", f"{taper[2]:.2f}deg"]
bpx = int(round(np.sqrt(TARGET[0] * TARGET[1]) / CELL_ARCSEC / 5) * 5)
kw.update(scales=[0, bpx, 3 * bpx], restoringbeam=[f"{TARGET[0]}arcsec", f"{TARGET[1]}arcsec", f"{TARGET[2]}deg"])
json.dump(dict(uvrange=UVRANGE, target_beam=TARGET, uvtaper=kw.get("uvtaper"), scales=kw["scales"]),
          open(os.path.join(OUT, "uvmatch.json"), "w"), indent=1)
for p in glob.glob(base + ".*"):
    shutil.rmtree(p)
rec = tclean(imagename=base, niter=1_000_000, nsigma=3.0, usemask="auto-multithresh", sidelobethreshold=2.0,
             noisethreshold=4.25, lownoisethreshold=1.5, negativethreshold=15.0, minbeamfrac=0.3, growiterations=75, **kw)
print(f"{cfg} tclean: stopcode {rec.get('stopcode')}, {rec.get('iterdone')} iterations, {rec.get('nmajordone')} major cycles",
      flush=True)
for ext in ("image", "model", "residual", "psf", "mask"):
    exportfits(imagename=f"{base}.{ext}", fitsimage=os.path.join(OUT, f"{ext}.fits"), velocity=True, overwrite=True)
from msclean_complete import finish  # noqa: E402
vel = vsys + WIDTH_KMS * (np.arange(NCHAN) - (NCHAN - 1) / 2)
line = [i for i, v in enumerate(vel) if 4750 <= v <= 5280]
finish(OUT, OUT + "_finished", kw, line, CELL_ARCSEC, nsigma=1.5, label=f"ngc3110 {cfg} (uv-matched)")
print("DONE", flush=True)
