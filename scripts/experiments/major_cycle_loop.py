#!/usr/bin/env python
"""
Outer major cycles around the 2D-1D solver (the "minor cycle"), like CLEAN's loop.

Cycle k, starting from the converged compact model x_0 (results/<T>/compact/deconvolved/model.npy):
  1. CASA major cycle on the real visibilities: Rt_k = A^H W (V - A x_k)        (tclean niter=0, startmodel = x_k)
  2. corrected data term on the solver's grid:   d'_k = N x_k + Rt_k             (N: the solver's fast operator)
     so the solver's gradient at x_k, N x_k - d'_k = -Rt_k, is the TRUE one
  3. re-solve with d'_k (same recipe: pdtau, k = 5, lambdas from the real dirty cube), warm-started at x_k -> x_{k+1}
After the last cycle one more CASA call measures the final true residual. Reports per cycle the true residual rms in
the line channels (whole field and inside the finished extended-CLEAN mask, in units of the line-free sigma), the
model flux and the change of the model.

Writes results/<T>/compact/deconvolved_majorcycle/{model.npy, cycles.json} and the CASA log.

    CUDA_VISIBLE_DEVICES=1 /scratch/alahiry/conda/envs/deconvolver2d1d/bin/python scripts/experiments/major_cycle_loop.py \
        [NGC3110] [--cycles 3] [--n-iter 8000]
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
for p in ("simple", "scripts/experiments"):
    sys.path.insert(0, os.path.join(REPO, p))
RES = os.path.join(REPO, "results")

ap = argparse.ArgumentParser()
ap.add_argument("target", nargs="?", default="NGC3110")
ap.add_argument("--cycles", type=int, default=3); ap.add_argument("--n-iter", type=int, default=8000)
ap.add_argument("--control", action="store_true", help="re-solve on the ORIGINAL dirty cube (no major-cycle correction), "
                "to separate the effect of restarting the solver from that of the correction")
a = ap.parse_args()
T = a.target; R = os.path.join(RES, T)
OUT = os.path.join(R, "compact", "deconvolved_majorcycle" + ("_control" if a.control else "")); os.makedirs(OUT, exist_ok=True)
GC = json.load(open(os.path.join(R, "config.json")))
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

import casatools  # noqa: E402
from casatasks import tclean  # noqa: E402
from casa_logs import redirect_casa_logs  # noqa: E402
redirect_casa_logs(f"major_cycle_loop_{T}.log")
cc = GC["configs"]["compact"]
KW = dict(vis=cc["ms"], field=cc["field"] if len(set(cc["field"])) > 1 else cc["field"][0], spw=cc["spw"],
          datacolumn="data", imsize=GC["imsize"], cell=f"{GC['cell_arcsec']}arcsec", phasecenter=GC["phasecenter"],
          specmode="cube", start=f"{GC['start_kms']}km/s", width=f"{GC['width_kms']}km/s", nchan=GC["nchan"],
          restfreq=f"{GC['restfreq_ghz']}GHz", outframe="LSRK", veltype="radio", gridder="standard",
          weighting="natural", niter=0, restoration=False, pbcor=False)
ia = casatools.image()
CS = None


def casa_residual(x, tag):
    """the true residual A^H W (V - A x) on the solver's 0.02" grid (Jy/beam), x in Jy/pixel on that grid"""
    global CS
    get = lambda p: np.asarray((ia.open(p), ia.getchunk(), ia.close())[1], dtype=np.float64)[:, :, 0, :].transpose(2, 1, 0)
    if CS is None:                                    # the grid's coordinates, from one dirty run
        base = os.path.join(OUT, "_grid")
        for p in glob.glob(base + ".*"):
            shutil.rmtree(p)
        tclean(imagename=base, **KW)
        ia.open(base + ".residual"); CS = ia.coordsys().torecord(); ia.close()
    mpath = os.path.join(OUT, "_model.image")
    if os.path.exists(mpath):
        shutil.rmtree(mpath)
    ia.fromarray(outfile=mpath, pixels=x.transpose(2, 1, 0)[:, :, None, :].astype(np.float32), csys=CS, overwrite=True)
    ia.setbrightnessunit("Jy/pixel"); ia.close()
    base = os.path.join(OUT, "_resid")
    for p in glob.glob(base + ".*"):
        shutil.rmtree(p)
    t0 = time.time(); tclean(imagename=base, startmodel=mpath, **KW)
    r = get(base + ".residual"); log(f"  CASA major cycle ({tag}): {time.time() - t0:.0f} s")
    return r


import common  # noqa: E402
common.DATA = RES
for c, b in GC["beams_natural"].items():
    common.BEAMS[(T, c)] = tuple(b)
G = common.G
P = common.problem(T, "compact", k=5.0)
L, NC = P.line, P.noise
M = np.load(os.path.join(R, "extended/msclean_finished/mask_final.npy")).astype(bool)


def stats(r, x, tag):
    sig = float(np.std(r[NC]))
    rf = float(np.sqrt(np.mean(r[L] ** 2))); rm = float(np.sqrt(np.mean(r[L][M[L]] ** 2)))
    s = dict(tag=tag, sigma_line_free_mJy=1e3 * sig, true_resid_rms_line_field_mJy=1e3 * rf,
             true_resid_rms_line_field_sigma=rf / sig, true_resid_rms_line_mask_mJy=1e3 * rm,
             true_resid_rms_line_mask_sigma=rm / sig, max_abs_resid_line_mJy=1e3 * float(np.abs(r[L]).max()),
             flux_line_Jy_kms=float(x[L].sum() * GC["width_kms"]))
    log(f"  {tag}: true residual rms (line) field {s['true_resid_rms_line_field_sigma']:.2f} sigma, in the bright mask "
        f"{s['true_resid_rms_line_mask_sigma']:.2f} sigma, max {s['max_abs_resid_line_mJy']:.2f} mJy/beam; "
        f"flux {s['flux_line_Jy_kms']:.1f} Jy km/s")
    return s


x_fine = np.load(os.path.join(R, "compact/deconvolved/model.npy")).astype(np.float64)   # x_0, 0.02" grid
rows = []
for k in range(a.cycles + 1):
    Rt = casa_residual(x_fine, f"cycle {k}")
    rows.append(stats(Rt, x_fine, f"cycle {k} (model x_{k})"))
    json.dump(rows, open(os.path.join(OUT, "cycles.json"), "w"), indent=1)
    if k == a.cycles:
        break
    # corrected data term on the solver's (binned) grid: d' = N x + rebin(Rt)
    xc = G.to_t(common.to_native(P, x_fine))
    b_ = P.bin; ny_, nx_ = Rt.shape[1] // b_ * b_, Rt.shape[2] // b_ * b_      # block mean, as gpu_pd.rebin does for d
    Rt_c = Rt[:, :ny_, :nx_].reshape(Rt.shape[0], ny_ // b_, b_, nx_ // b_, b_).mean((2, 4))
    d_new = P.op.d.cpu().numpy().astype(np.float64) if a.control else (P.op.apply(xc).cpu().numpy().astype(np.float64) + Rt_c)
    Pk = common.problem(T, "compact", k=5.0, data=d_new)
    t0 = time.time()
    x = common.solve(Pk, "pdtau", n_iter=a.n_iter, x0=xc, log=lambda m: log("    " + m))
    new_fine = common.to_saved(Pk, x).astype(np.float64)
    dx = float(np.linalg.norm(new_fine[L] - x_fine[L]) / np.linalg.norm(x_fine[L]))
    log(f"  re-solve {k + 1}: {time.time() - t0:.0f} s, model change ||dx||/||x|| = {100 * dx:.2f}%")
    rows[-1]["model_change_next_pct"] = 100 * dx
    x_fine = new_fine
    np.save(os.path.join(OUT, "model.npy"), x_fine.astype(np.float32))
    del Pk, x
json.dump(rows, open(os.path.join(OUT, "cycles.json"), "w"), indent=1)
log("done")
