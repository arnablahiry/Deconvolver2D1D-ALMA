#!/usr/bin/env python
"""
NGC 3110: compact 2D-1D against compact MS-CLEAN treated in the same way -- its CLEAN components convolved with each
comparison beam -- for natural weighting and for the best Briggs weighting (robust 0; robust -1 and -2 give the same
beam and results), all against the joint compact + extended data at that beam inside the ladder mask.

  top     moment 0 at the 0.40" beam (the ladder's step finer than lambda/B_max where 2D-1D is still < 10%): joint,
          2D-1D, MS-CLEAN (robust 0) model, and the two differences in % of the joint map
  (a)     cube RMSE against the comparison beam (Eq. 2)
  (b)     moment-1 RMSE (km/s)
  (c)     the finest comparison beam at which each model is accurate to 5% and to 10% (cube RMSE, interpolated in
          log beam along the ladder); for the restored MS-CLEAN images, their clean beams: the resolution they deliver

Reads ladder_msclean_model.json (ladder_msclean_model.py).
    python scripts/compact_vs_msclean_dendro/fig_msclean_comparison.py
    -> results/NGC3110/figures/resolution_ladder_joint20/msclean_comparison.{png,pdf}
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.patches import Ellipse
from scipy.ndimage import convolve1d

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
from restore import convolve_with_beam  # noqa: E402

R = os.path.join(REPO, "results", "NGC3110"); OUT = os.path.join(R, "figures", "resolution_ladder_joint20")
D = json.load(open(os.path.join(OUT, "ladder_msclean_model.json"))); LAD = json.load(open(os.path.join(OUT, "resolution_ladder.json")))
MK = np.load(os.path.join(OUT, "masks.npz")); L = [int(c) for c in MK["line"]]; CELL, DV = 0.08, 20.0
BEAMS = [tuple(b) for b in D["beams"]]; nb = len(BEAMS); LB = float(LAD["lbmax_compact"])
ILB = int(np.argmin([abs(b[0] - LB) + abs(b[1] - LB) for b in BEAMS]))
geo = np.array([np.sqrt(b[0] * b[1]) for b in BEAMS])
rd = lambda p: np.nan_to_num(fits.getdata(p)).astype(np.float64).squeeze()
om = lambda b: np.pi * b[0] * b[1] / (4 * np.log(2))
M2D, MN, MB0 = "2D–1D", "MS-CLEAN natural", "MS-CLEAN robust 0"
COL = {M2D: "#2a6fdb", MB0: "#e07b00", MN: "#9c4a00"}
LAB = {M2D: "2D–1D model", MB0: "MS-CLEAN model, robust 0", MN: "MS-CLEAN model, natural"}
CB = D["clean_beams"]


def crop(p):
    k = np.array([0.5, 1.0, 0.5]); x = rd(p)[L]
    return convolve1d(convolve1d(x, k, axis=-1, mode="constant"), k, axis=-2, mode="constant")[..., ::2, ::2] / 4.0


def finest(name, thr):
    """the finest beam (geometric FWHM) down to which the cube RMSE stays below thr, interpolated in log beam"""
    r = np.array([row[name]["rmse"] for row in D["rows"]]); i = int(np.nonzero(r > thr)[0][0])
    return float(np.exp(np.interp(thr, [r[i - 1], r[i]], np.log([geo[i - 1], geo[i]]))))


# ======================================================================== maps at the 0.40" beam
IB = int(np.argmin(np.abs(geo - 0.40))); b = BEAMS[IB]; M = MK[f"b{IB:02d}"]; foot = M.any(0)
I = crop(os.path.join(R, f"ladder_joint_20kms/b{IB:02d}/clean_finished/image.fits"))
mods = {M2D: np.load(os.path.join(R, "pixel_scale/model_0.08_support.npy")).astype(np.float64)[L],
        MB0: rd(os.path.join(R, "compact_weighting_20kms/r0/clean_finished/model.fits"))[L]}
m0 = lambda c: (c * M).sum(0) * DV / om(b)                               # Jy arcsec^-2 km/s inside the mask
J0 = m0(I); MAP = {k: m0(convolve_with_beam(x, CELL, *b)) for k, x in mods.items()}
pct = lambda c: np.where(foot & (J0 > 0.05 * J0[foot].max()), 100 * (c - J0) / np.where(J0 > 0, J0, 1), np.nan)
ys, xs = np.nonzero(foot); c0 = foot.shape[0] // 2; H = (max(np.abs(ys - c0).max(), np.abs(xs - c0).max()) + 6) * CELL
hp = int(round(H / CELL)); sl = np.s_[c0 - hp:c0 + hp, c0 - hp:c0 + hp]; EXT = [H, -H, -H, H]

# ======================================================================== figure (A&A text width)
FS, FT, FL = 8, 7, 6.5
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": FS, "axes.labelsize": FS,
                     "xtick.labelsize": FT, "ytick.labelsize": FT, "xtick.direction": "in", "ytick.direction": "in",
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.fontsize": FL,
                     "lines.linewidth": 1.2, "lines.markersize": 3})
FW = 184 / 25.4; ML, MR, GX = 0.45, 0.1, 0.06; P = (FW - ML - MR - 4 * GX - 0.25) / 5
HP, GB, MB_, CBH = 1.75, 0.0, 0.5, 0.06
Y_P = MB_; Y_M = Y_P + HP + GB + CBH + 0.42; FH = Y_M + P + 0.24
fig = plt.figure(figsize=(FW, FH)); inch = lambda x, y, w, h: fig.add_axes([x / FW, y / FH, w / FW, h / FH])
X = lambda j: ML + j * (P + GX) + (0.25 if j >= 3 else 0)
vmax = np.nanpercentile(J0[foot], 99.9)
for j, (img, t, cm, lo, hi) in enumerate([(J0, f"joint (all at a {b[0]:.2f}″ beam)", "inferno", 0, vmax),
                                          (MAP[M2D], "2D–1D model", "inferno", 0, vmax),
                                          (MAP[MB0], "MS-CLEAN model (robust 0)", "inferno", 0, vmax),
                                          (pct(MAP[M2D]), "2D–1D − joint", "RdBu_r", -50, 50),
                                          (pct(MAP[MB0]), "MS-CLEAN − joint", "RdBu_r", -50, 50)]):
    a = inch(X(j), Y_M, P, P); cmap = plt.get_cmap(cm).copy(); cmap.set_bad("0.93" if j >= 3 else "0.12")
    show = np.where(foot, img, np.nan)
    im = a.imshow(show[sl], origin="lower", extent=EXT, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    a.contour(foot[sl].astype(float), [0.5], colors="0.55", linewidths=0.35, extent=EXT, origin="lower")
    a.set_xticks([]); a.set_yticks([]); a.set_title(t, fontsize=FS, pad=3)
    pad = 0.07 * H; light = j >= 3
    a.add_patch(Ellipse((H - pad - b[0] / 2, -H + pad + b[0] / 2), b[0], b[1], angle=90 - b[2], color="0.15" if light else "w"))
    if j >= 3:
        name = M2D if j == 3 else MB0; r = D["rows"][IB][name]
        a.text(0.04, 0.955, f"RMSE {r['rmse']:.1f}%\nΔF {r['flux']:+.1f}%", transform=a.transAxes, ha="left", va="top",
               fontsize=FL, bbox=dict(boxstyle="square,pad=0.25", facecolor="w", edgecolor="0.6", linewidth=0.4))
    if j == 0:
        a.plot([-H + 0.1 * H, -H + 0.1 * H + 2], [H - 0.12 * H] * 2, color="w", lw=0.9)
        a.text(-H + 0.1 * H + 1, H - 0.17 * H, "2″", color="w", ha="center", va="top", fontsize=FL, fontweight="bold")
    if j in (0, 3):
        w_ = 3 * P + 2 * GX if j == 0 else 2 * P + GX
        cax = inch(X(j) + 0.1, Y_M - 0.06 - CBH, w_ - 0.2, CBH)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.outline.set_linewidth(0.4)
        cax.tick_params(labelsize=FL, length=1.5, pad=1)
        cb.set_label(r"moment 0 (Jy arcsec$^{-2}$ km s$^{-1}$)" if j == 0 else "% of the joint map", fontsize=FL, labelpad=1)
fig.text(X(0) - 0.08, Y_M + P / 2, f"moment 0, all at a {b[0]:.2f}″ beam", rotation=90, ha="right", va="center", fontsize=FS - 0.5,
         fontweight="bold")

# ---------------------------------------------------------------- (a), (b): against the comparison beam
WP = (FW - ML - MR - 2 * 0.62) / 3
x = np.arange(nb); tl = [f"{g:.2f}" for g in geo]
for j, (key, ylab, t) in enumerate((("rmse", "cube RMSE (%)", "(a) cube"), ("mom1", r"moment-1 RMSE (km s$^{-1}$)", "(b) velocity field"))):
    a = inch(ML + j * (WP + 0.62), Y_P, WP, HP)
    for name in (M2D, MB0, MN):
        a.plot(x, [row[name][key] for row in D["rows"]], "-o", color=COL[name], lw=1.6 if name == M2D else 1.1,
               ls="-" if name != MN else "--", label=LAB[name])
    a.axvline(ILB, color="0.5", lw=0.6, ls=":"); a.set_xlim(-0.5, nb - 0.5); a.set_ylim(bottom=0)
    if key == "rmse":
        a.axhline(10, color="0.6", lw=0.5, ls="--"); a.axhline(5, color="0.8", lw=0.5, ls="--")
        a.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="0.6", fancybox=False, bbox_to_anchor=(0.0, 0.88))
    a.set_xticks(x); a.set_xticklabels(tl, rotation=60, ha="right", rotation_mode="anchor", fontsize=FL - 0.5)
    for i_ in (0, ILB, nb - 1):
        a.get_xticklabels()[i_].set_fontweight("bold")
    a.tick_params(which="both", top=True, right=True); a.set_ylabel(ylab, labelpad=2)
    a.set_xlabel("comparison beam FWHM (″)", labelpad=2)
    a.text(0.03, 0.96, t, transform=a.transAxes, ha="left", va="top", fontweight="bold", fontsize=FT)
# ---------------------------------------------------------------- (c): the resolution each delivers
a = inch(ML + 2 * (WP + 0.62), Y_P, WP, HP)
ROWS = [(M2D, "2D–1D model"), (MB0, "MS-CLEAN model\nrobust 0"), (MN, "MS-CLEAN model\nnatural"),
        ("rest0", "MS-CLEAN restored\nrobust 0"), ("restN", "MS-CLEAN restored\nnatural")]
for k, (name, lab) in enumerate(ROWS):
    yk = len(ROWS) - 1 - k
    a.text(1.55, yk + 0.3, lab.replace("\n", ", "), ha="left", va="center", fontsize=FL - 0.5, color="0.25")
    if name.startswith("rest"):
        cb = CB[MB0 if name == "rest0" else MN]; g = np.sqrt(cb[0] * cb[1])
        a.plot([g], [yk], "D", color=COL[MB0 if name == "rest0" else MN], ms=4.5)
        a.text(g * 0.9, yk, f"{g:.2f}″ (its clean beam)", ha="left", va="center", fontsize=FL - 0.5)
    else:
        g5, g10 = finest(name, 5), finest(name, 10)
        a.plot([g10, g5], [yk, yk], color=COL[name], lw=2.2, solid_capstyle="butt")
        a.plot([g10], [yk], "o", color=COL[name], ms=4.5); a.plot([g5], [yk], "o", mfc="w", mec=COL[name], ms=4.5)
        a.text(g10 * 0.9, yk, f"{g5:.2f}″ / {g10:.2f}″", ha="left", va="center", fontsize=FL - 0.5)
a.set_xscale("log"); a.invert_xaxis(); a.set_xlim(1.6, 0.14); a.set_ylim(-0.55, len(ROWS) + 0.2)
a.set_xticks([1.5, 1, 0.6, 0.4, 0.3, 0.2]); a.set_xticklabels(["1.5", "1", "0.6", "0.4", "0.3", "0.2"])
a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a.set_yticks([])
a.axvline(LB, color="0.5", lw=0.6, ls=":"); a.tick_params(which="both", top=True, right=True)
a.set_xlabel("finest beam FWHM (″)", labelpad=2)
a.text(0.03, 0.975, "(c) finest beam, 5% ○ / 10% ●", transform=a.transAxes, ha="left", va="top", fontweight="bold",
       fontsize=FT - 0.5)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"msclean_comparison.{ext}"), dpi=300 if ext == "png" else None)
print("saved", os.path.join(OUT, "msclean_comparison.png"), {n: (round(finest(n, 5), 3), round(finest(n, 10), 3)) for n in (M2D, MB0, MN)})
