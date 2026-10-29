#!/usr/bin/env python
"""
Compact-configuration MS-CLEAN of NGC 3110 with a different visibility weighting: the run of consistency_20kms_0.08
(clean_compact -> clean_compact_finished; natural weighting) repeated with Briggs robust ROBUST, everything else the
same -- 64 x 20 km/s channels from the pre-averaged MSs, 0.08" pixels, 640 px (51"), multiscale (0, 1.2", 3.6"),
auto-multithresh (short-baseline values) to nsigma 3, then finished inside the growing astrodendro mask at nsigma 1.5
(msclean_complete.finish). For the comparison of MS-CLEAN's own model and resolution with the 2D-1D model
(fig_power_msclean_weighting.py).

    ROBUST=-1 OMP_NUM_THREADS=8 /scratch/alahiry/conda/envs/deconvolver2d1d/bin/python scripts/experiments/compact_weighting_clean.py
    -> results/NGC3110/compact_weighting_20kms/r<ROBUST>/{clean,clean_finished}/
ROBUST=natural: natural weighting. SCALES=0,2,4,8,16,32,64 (pixels): another multiscale set, e.g. the dyadic scales of the
2D-1D starlet (J = 6 on 0.08" pixels) -> .../<r<ROBUST> or natural>_s<SCALES joined by ->/
"""
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("scripts", "simple"):
    sys.path.insert(0, os.path.join(REPO, p))
NAT = os.environ["ROBUST"] == "natural"; ROB = 2.0 if NAT else float(os.environ["ROBUST"])
SC = [int(v) for v in os.environ["SCALES"].split(",")] if os.environ.get("SCALES") else None
TAG = ("natural" if NAT else f"r{ROB:g}") + (f"_s{'-'.join(map(str, SC))}" if SC else "")
R = os.path.join(REPO, "results", "NGC3110"); OUT = os.path.join(R, "compact_weighting_20kms", TAG)
os.makedirs(OUT, exist_ok=True)
from casa_logs import redirect_casa_logs  # noqa: E402
redirect_casa_logs(f"compact_weighting_{TAG}.log")
from casatasks import tclean, exportfits  # noqa: E402
from msclean_complete import finish  # noqa: E402

GC = json.load(open(os.path.join(R, "config.json"))); G = json.load(open(os.path.join(R, "consistency_20kms_0.08/grid.json")))
RG = json.load(open("/scratch/alahiry/regrid/NGC3110_64x20kms/ms.json"))
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {TAG}: {m}", flush=True)
K = dict(vis=RG["ms"]["compact"], spw=RG["spw"], datacolumn="data", imsize=G["imsize"], cell=f"{G['cell']}arcsec",
         phasecenter=GC["phasecenter"], specmode="cube", start=f"{G['start_kms']}km/s", width=f"{G['width_kms']}km/s",
         nchan=G["nchan"], restfreq=f"{GC['restfreq_ghz']}GHz", outframe="LSRK", veltype="radio", gridder="standard",
         **(dict(weighting="natural") if NAT else dict(weighting="briggs", robust=ROB)), deconvolver="multiscale",
         scales=SC or G["scales_px"], pblimit=-0.05, pbcor=False,
         restoringbeam="common", fullsummary=True)
c1, c2 = os.path.join(OUT, "clean"), os.path.join(OUT, "clean_finished")
if not os.path.exists(os.path.join(OUT, "clean.done")):
    os.makedirs(c1, exist_ok=True); base = os.path.join(c1, "_tclean"); t0 = time.time()
    rec = tclean(imagename=base, niter=1_000_000, nsigma=3.0, usemask="auto-multithresh", sidelobethreshold=2.0,
                 noisethreshold=4.25, lownoisethreshold=1.5, negativethreshold=15.0, minbeamfrac=0.3, growiterations=75, **K)
    for ext in ("image", "model", "residual", "psf", "mask"):
        exportfits(imagename=f"{base}.{ext}", fitsimage=os.path.join(c1, f"{ext}.fits"), velocity=True, overwrite=True)
    log(f"first pass: stopcode {rec.get('stopcode')}, {rec.get('iterdone')} iterations, {time.time() - t0:.0f} s")
    open(os.path.join(OUT, "clean.done"), "w").write("")
if not os.path.exists(os.path.join(c2, "image.fits")):
    finish(c1, c2, K, G["line"], G["cell"], nsigma=1.5, label=f"NGC3110 compact {TAG}")
log("DONE")
