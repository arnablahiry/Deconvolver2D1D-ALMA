#!/usr/bin/env python
"""
The resolution ladder of NGC 3110 (fig_resolution_ladder.py REF=joint GRID=20kms) for compact MS-CLEAN treated like
the 2D-1D model: its CLEAN components alone (Jy/pixel), convolved with each of the eleven comparison beams and compared
with the joint image at that beam inside the ladder's mask (masks.npz) -- cube RMSE and total flux, as Fig. 4a.
Models: the 2D-1D model (a check: must reproduce resolution_ladder.json), natural-weighted compact MS-CLEAN
(consistency_20kms_0.08/clean_compact_finished) and the Briggs-weighted runs of compact_weighting_clean.py that exist.

    python scripts/compact_vs_msclean_dendro/ladder_msclean_model.py
    -> results/NGC3110/figures/resolution_ladder_joint20/ladder_msclean_model.{png,pdf,json}
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import convolve1d

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
from restore import convolve_with_beam  # noqa: E402

R = os.path.join(REPO, "results", "NGC3110"); OUT = os.path.join(R, "figures", "resolution_ladder_joint20")
LAD = json.load(open(os.path.join(OUT, "resolution_ladder.json"))); MK = np.load(os.path.join(OUT, "masks.npz"))
L = [int(c) for c in MK["line"]]; CELL = 0.08; LB = float(LAD["lbmax_compact"])
BEAMS = [(r["bmaj"], r["bmin"], r["bpa"]) for r in LAD["rows"]]
rd = lambda p: np.nan_to_num(fits.getdata(p)).astype(np.float64).squeeze()
bm = lambda p: (lambda h: (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"]))(fits.getheader(p))


def crop(p):                                       # joint image, 0.04" -> 0.08" (averaged, Jy/beam), line channels
    k = np.array([0.5, 1.0, 0.5]); x = rd(p)[L]
    return convolve1d(convolve1d(x, k, axis=-1, mode="constant"), k, axis=-2, mode="constant")[..., ::2, ::2] / 4.0


MODELS = {"2D–1D": np.load(os.path.join(R, "pixel_scale/model_0.08_support.npy")).astype(np.float64)[L]}
_p = os.path.join(R, "pixel_scale/model_0.08_support_L0.npy")             # the ablation: the 2D starlet per channel
if os.path.exists(_p):
    MODELS["2D per channel"] = np.load(_p).astype(np.float64)[L]
CB = {}
SS = "s0-2-4-8-16-32-64"                                                 # MS-CLEAN with the starlet's dyadic scales
for name, d in [("MS-CLEAN natural", os.path.join(R, "consistency_20kms_0.08/clean_compact_finished"))] + \
               [(f"MS-CLEAN robust {r:g}", os.path.join(R, f"compact_weighting_20kms/r{r:g}/clean_finished")) for r in (0.5, 0, -1, -2)] + \
               [(f"MS-CLEAN {w}, starlet scales", os.path.join(R, f"compact_weighting_20kms/{t}_{SS}/clean_finished"))
                for w, t in (("natural", "natural"), ("robust 0", "r0"))]:
    if os.path.exists(os.path.join(d, "model.fits")):
        MODELS[name] = rd(os.path.join(d, "model.fits"))[L]; CB[name] = bm(os.path.join(d, "image.fits"))
GG = json.load(open(os.path.join(R, "consistency_20kms_0.08/grid.json")))
VL = (GG["start_kms"] + GG["width_kms"] * np.arange(GG["nchan"]))[L]


def moments(cube, M, thr):                         # as fig_resolution_ladder_moments.py
    w = np.where(cube >= thr[:, None, None], cube, 0.0); s = w.sum(0); ok = M.any(0) & (s > 0)
    m1 = np.where(ok, (w * VL[:, None, None]).sum(0) / np.where(ok, s, 1), np.nan)
    m2 = np.where(ok, np.sqrt(np.clip((w * (VL[:, None, None] - np.nan_to_num(m1)) ** 2).sum(0) / np.where(ok, s, 1), 0, None)), np.nan)
    return m1, m2, s


rows = []
for ib, b in enumerate(BEAMS):
    I = crop(os.path.join(R, f"ladder_joint_20kms/b{ib:02d}/clean_finished/image.fits")); M = MK[f"b{ib:02d}"]; i = I[M]
    res = crop(os.path.join(R, f"ladder_joint_20kms/b{ib:02d}/clean_finished/residual.fits"))
    thr = 3 * 1.4826 * np.median(np.abs(res - np.median(res, axis=(1, 2), keepdims=True)), axis=(1, 2))
    j1, j2, j0 = moments(I, M, thr)
    row = dict(beam=b)
    for name, x in MODELS.items():
        C = convolve_with_beam(x, CELL, *b); c = C[M]; c1, c2, c0 = moments(C, M, thr)
        v = np.isfinite(c1) & np.isfinite(j1)
        row[name] = dict(rmse=float(100 * np.linalg.norm(c - i) / np.linalg.norm(i)), flux=float(100 * (c.sum() - i.sum()) / i.sum()),
                         mom0=float(100 * np.linalg.norm((c0 - j0)[v]) / np.linalg.norm(j0[v])),
                         mom1=float(np.sqrt(np.mean((c1 - j1)[v] ** 2))), mom2=float(np.sqrt(np.mean((c2 - j2)[v] ** 2))))
    rows.append(row)
    print(f"b{ib:02d} {b[0]:.2f}x{b[1]:.2f}: " + ", ".join(f"{k} {v['rmse']:.1f}% ({v['flux']:+.1f}%) m0 {v['mom0']:.1f}% m1 {v['mom1']:.1f} m2 {v['mom2']:.1f}"
                                                       for k, v in row.items()
                                                       if k != "beam") + f"  [ladder 2D-1D {LAD['rows'][ib]['rmse']:.1f}%]", flush=True)
json.dump(dict(beams=BEAMS, clean_beams=CB, rows=rows), open(os.path.join(OUT, "ladder_msclean_model.json"), "w"), indent=1)

# ======================================================================== figure (A&A column), as Fig. 4a
FS, FT, FL = 8, 7, 6
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": FS, "axes.labelsize": FS,
                     "xtick.labelsize": FT, "ytick.labelsize": FT, "xtick.direction": "in", "ytick.direction": "in",
                     "axes.linewidth": 0.6, "legend.fontsize": FL, "lines.linewidth": 1.0, "lines.markersize": 2.8})
FW = 88 / 25.4; fig = plt.figure(figsize=(FW, 3.6))
nb = len(BEAMS); x = np.arange(nb); ILB = int(np.argmin([abs(b[0] - LB) + abs(b[1] - LB) for b in BEAMS]))
tl = [f"{b[0]:.2f}×{b[1]:.2f}" if abs(b[0] - b[1]) > 1e-3 else f"{b[0]:.2f}" for b in BEAMS]
COL = {"2D–1D": "#2a6fdb", "2D per channel": "#7fb2ff", "MS-CLEAN natural, starlet scales": "#555555",
       "MS-CLEAN robust 0, starlet scales": "#999999", **dict(zip([f"MS-CLEAN {w}" for w in ("natural", "robust 0.5", "robust 0", "robust -1", "robust -2")],
                                     plt.get_cmap("Oranges")(np.linspace(0.95, 0.4, 5))))}
LEG = dict(frameon=True, framealpha=0.95, edgecolor="0.6", fancybox=False, borderpad=0.35, labelspacing=0.25, handlelength=2.0)
for j, (key, ylab) in enumerate((("rmse", "cube RMSE (%)"), ("flux", "total flux, compact − joint (%)"))):
    a = fig.add_axes([0.15, 0.56 - j * 0.42, 0.82, 0.38])
    for name in MODELS:
        lab = name + (f" ({CB[name][0]:.2f}″×{CB[name][1]:.2f}″)" if name in CB else "")
        a.plot(x, [r[name][key] for r in rows], "-o", color=COL[name], lw=1.3 if name == "2D–1D" else 1.0, label=lab)
    a.axvline(ILB, color="0.5", lw=0.6, ls=":"); a.axhline(0, color="0.5", lw=0.5)
    if key == "rmse":
        a.axhline(10, color="0.45", lw=0.6, ls="--"); a.legend(loc="upper left", **LEG); a.set_ylim(bottom=0)
    a.set_xlim(-0.5, nb - 0.5); a.set_xticks(x); a.set_ylabel(ylab)
    a.set_xticklabels(tl if j else [], rotation=45, ha="right", rotation_mode="anchor", fontsize=FL)
    a.tick_params(which="both", top=True, right=True)
fig.text(0.56, 0.01, "comparison beam FWHM (″)", ha="center", va="bottom")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"ladder_msclean_model.{ext}"), dpi=300 if ext == "png" else None)
print("saved", os.path.join(OUT, "ladder_msclean_model.png"))
