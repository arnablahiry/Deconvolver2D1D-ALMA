#!/usr/bin/env python
"""
The resolution ladder (fig_resolution_ladder.py REF=joint GRID=20kms; masks.npz) for every alternative 2D-1D model of
target T, to test the method's choices: k (the threshold multiple), the support mask (grown, none), the 2D dictionary
(starlet, FIRE learnlet), the 1D velocity transform (L1D = 0: the 2D dictionary per channel), and the null test on the
negated data (NEGATE: positivity makes any model flux false). Models: results/NGC3110/pixel_scale/model_0.08*.npy, or
results/<T>/g20/model_0.08_support*.npy for the other targets.

  per model and comparison beam: cube RMSE and total flux (%), moment-0 RMSE (%), moment-1 and -2 RMSE (km/s), as Fig. 4
  per model: the model flux in the line and in the line-free channels (Jy km/s; for the negated data, all of it false)
  reference variants (ladder_joint_20kms_<VARIANT>/bXX, ladder_joint.py VARIANT=...): how much the joint image itself
  changes with its weighting or CLEAN depth (RMSE between the two references), and the main model against each

    python scripts/compact_vs_msclean_dendro/ablation_eval.py [T]
    -> results/<T>/figures/resolution_ladder_joint20/ablation_eval.json
"""
import glob
import json
import os
import sys

import numpy as np
from astropy.io import fits
from scipy.ndimage import convolve1d

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
from restore import convolve_with_beam  # noqa: E402

T = sys.argv[1] if len(sys.argv) > 1 else "NGC3110"; S3110 = T == "NGC3110"
R = os.path.join(REPO, "results", T); OUT = os.path.join(R, "figures", "resolution_ladder_joint20")
LAD = json.load(open(os.path.join(OUT, "resolution_ladder.json"))); MK = np.load(os.path.join(OUT, "masks.npz"))
L = [int(c) for c in MK["line"]]; CELL = 0.08
G = json.load(open(os.path.join(R, "consistency_20kms_0.08/grid.json" if S3110 else "g20/grid.json")))
DV = G["width_kms"]; VL = (G["start_kms"] + DV * np.arange(G["nchan"]))[L]; F = G["line_free"]
BEAMS = [(r["bmaj"], r["bmin"], r["bpa"]) for r in LAD["rows"]]
rd = lambda p: np.nan_to_num(fits.getdata(p)).astype(np.float64).squeeze()
pat = os.path.join(R, "pixel_scale/model_0.08*.npy") if S3110 else os.path.join(R, "g20/model_0.08_support*.npy")
MODELS = {os.path.basename(p)[len("model_0.08"):-4].lstrip("_") or "nosupport": p for p in sorted(glob.glob(pat))}
MAIN = "support"


def crop(p):
    k = np.array([0.5, 1.0, 0.5]); x = rd(p)[L]
    return convolve1d(convolve1d(x, k, axis=-1, mode="constant"), k, axis=-2, mode="constant")[..., ::2, ::2] / 4.0


def moments(cube, M, thr):
    w = np.where(cube >= thr[:, None, None], cube, 0.0); s = w.sum(0); ok = M.any(0) & (s > 0)
    m1 = np.where(ok, (w * VL[:, None, None]).sum(0) / np.where(ok, s, 1), np.nan)
    m2 = np.where(ok, np.sqrt(np.clip((w * (VL[:, None, None] - np.nan_to_num(m1)) ** 2).sum(0) / np.where(ok, s, 1), 0, None)), np.nan)
    return m1, m2, s


def metrics(C, I, M, thr, j1, j2, j0):
    c, i = C[M], I[M]; c1, c2, c0 = moments(C, M, thr); v = np.isfinite(c1) & np.isfinite(j1)
    return dict(rmse=float(100 * np.linalg.norm(c - i) / np.linalg.norm(i)), flux=float(100 * (c.sum() - i.sum()) / i.sum()),
                mom0=float(100 * np.linalg.norm((c0 - j0)[v]) / np.linalg.norm(j0[v])),
                mom1=float(np.sqrt(np.mean((c1 - j1)[v] ** 2))), mom2=float(np.sqrt(np.mean((c2 - j2)[v] ** 2))))


X = {}
out = dict(target=T, models={}, rows=[], variants={})
for name, p in MODELS.items():
    x = np.load(p).astype(np.float64)
    out["models"][name] = dict(path=os.path.relpath(p, REPO), flux_line=float(x[L].sum() * DV), flux_line_free=float(x[F].sum() * DV),
                               flux_all=float(x.sum() * DV))
    X[name] = x[L]
for ib, b in enumerate(BEAMS):
    jd = os.path.join(R, f"ladder_joint_20kms/b{ib:02d}/clean_finished"); M = MK[f"b{ib:02d}"]
    I = crop(os.path.join(jd, "image.fits")); res = crop(os.path.join(jd, "residual.fits"))
    thr = 3 * 1.4826 * np.median(np.abs(res - np.median(res, axis=(1, 2), keepdims=True)), axis=(1, 2))
    j1, j2, j0 = moments(I, M, thr)
    row = dict(beam=b, **{n: metrics(convolve_with_beam(x, CELL, *b), I, M, thr, j1, j2, j0) for n, x in X.items() if "neg" not in n})
    out["rows"].append(row)
    for vd in sorted(glob.glob(os.path.join(R, f"ladder_joint_20kms_*/b{ib:02d}/clean_finished/image.fits"))):
        var = vd.split("ladder_joint_20kms_")[1].split("/")[0]; Iv = crop(vd)
        out["variants"].setdefault(var, {})[f"b{ib:02d}"] = dict(
            beam=b, ref_vs_ref_rmse=float(100 * np.linalg.norm((Iv - I)[M]) / np.linalg.norm(I[M])),
            ref_vs_ref_flux=float(100 * ((Iv - I)[M].sum()) / I[M].sum()),
            main_model_vs_variant=metrics(convolve_with_beam(X[MAIN], CELL, *b), Iv, M, thr, *moments(Iv, M, thr)))
    print(f"b{ib:02d} {np.sqrt(b[0] * b[1]):.2f}\": " + "  ".join(f"{n} {m['rmse']:.1f}%/{m['mom1']:.1f}" for n, m in row.items() if n != "beam"),
          flush=True)
json.dump(out, open(os.path.join(OUT, "ablation_eval.json"), "w"), indent=1)
for n, m in out["models"].items():
    print(f"  {n:28s} flux: line {m['flux_line']:8.1f}, line-free {m['flux_line_free']:7.1f}, all {m['flux_all']:8.1f} Jy km/s")
for v, d in out["variants"].items():
    for b_, e in d.items():
        print(f"  reference variant {v} {b_}: vs main reference RMSE {e['ref_vs_ref_rmse']:.1f}%, flux {e['ref_vs_ref_flux']:+.1f}%; "
              f"main model vs it RMSE {e['main_model_vs_variant']['rmse']:.1f}%")
