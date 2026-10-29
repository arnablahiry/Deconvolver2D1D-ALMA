#!/usr/bin/env python
"""
Grid of the extended MS-CLEAN of every pipeline target (results/<target>/, scripts/pipeline/run_target.py): line
moment 0 of the finished extended MS-CLEAN (its components x the astrodendro mask, convolved with the extended clean
beam; the same map as the "Extended MS-CLEAN" column of fig_gallery.py) in surface brightness (Jy arcsec^-2 km/s),
each panel on its own colour scale, no mask outline. Field of view per target as in fig_gallery.py.

    [STRETCH=power|symlog|linear] [GAMMA=0.5] [LINFRAC=0.03] [NCOL=5] [OUT=...] \
        python scripts/compact_vs_msclean_dendro/fig_extended_msclean_grid.py
    -> results/gallery/extended_msclean_all_<stretch>.{png,pdf}
STRETCH=power: x**GAMMA (default, GAMMA = 0.5); symlog: linear below LINFRAC x the panel's maximum, logarithmic above.
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

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
from restore import convolve_with_beam  # noqa: E402

RES = os.path.join(REPO, "results")
ALL = ["NGC3110", "NGC2369", "ESO297-G011", "NGC1614", "IC4518E", "IC4518W", "IRASF17138-1017", "NGC7130", "IC5179",
       "NGC5135", "IC4687", "MCG-02-33-098", "NGC3256"]
NCOL = int(os.environ.get("NCOL", 5))
STRETCH = os.environ.get("STRETCH", "power")
GAMMA, LINFRAC = float(os.environ.get("GAMMA", 0.5)), float(os.environ.get("LINFRAC", 0.03))
OUT = os.environ.get("OUT", os.path.join(RES, "gallery", f"extended_msclean_all_{STRETCH}.png"))
STXT = {"power": f"power stretch (γ = {GAMMA:g})", "symlog": f"symlog stretch (linear below {LINFRAC:g} × max)",
        "linear": "linear"}[STRETCH]


def norm(hi):
    if STRETCH == "power":
        return matplotlib.colors.PowerNorm(GAMMA, vmin=0, vmax=hi)
    if STRETCH == "symlog":
        return matplotlib.colors.SymLogNorm(LINFRAC * hi, vmin=0, vmax=hi, base=10)
    return matplotlib.colors.Normalize(0, hi)
plt.rcParams.update({"font.family": "serif", "xtick.direction": "in", "ytick.direction": "in"})
rd = lambda p: np.nan_to_num(fits.getdata(p).astype(np.float64)).squeeze()


def load(T):
    R = os.path.join(RES, T); c = json.load(open(os.path.join(R, "config.json")))
    L = json.load(open(os.path.join(R, "line_channels.json")))["line"]
    D = os.path.join(R, "extended/msclean_finished")
    cell, dv = c["cell_arcsec"], c["width_kms"]
    M = np.load(os.path.join(D, "mask_final.npy")).astype(bool)
    h = fits.getheader(os.path.join(D, "image.fits"))
    BE = (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"])
    x = rd(os.path.join(D, "model.fits")) * M
    sb = convolve_with_beam((x[L].sum(0) * dv)[None], cell, *BE)[0] / (np.pi * BE[0] * BE[1] / (4 * np.log(2)))
    foot = M[L].any(0); n = foot.shape[0]; c0 = n // 2
    if "half_arcsec" in c:
        half = float(c["half_arcsec"])
    else:
        fy, fx = np.nonzero(foot)
        half = float(min(n * cell / 2 - 0.1, max(np.abs(fx - c0).max(), np.abs(fy - c0).max()) * cell + 0.5))
    return dict(T=T, label=c["label"], mosaic=c.get("mosaic", False), cell=cell, c0=c0, half=half, BE=BE, sb=sb,
                hi=float(np.percentile(sb[foot], 99.9)))


gals = [load(t) for t in ALL]
nr = -(-len(gals) // NCOL)
S, GAP, CBH, CBP, ROWGAP = 3.0, 0.25, 0.12, 0.07, 0.55
ML, MR, MB, MT = 0.2, 0.2, 0.25, 0.85
FW, FH = ML + NCOL * S + (NCOL - 1) * GAP + MR, MB + nr * (S + CBP + CBH) + (nr - 1) * ROWGAP + MT
fig = plt.figure(figsize=(FW, FH))
inch = lambda x, y, w, h: fig.add_axes([x / FW, y / FH, w / FW, h / FH])
for i, g in enumerate(gals):
    r, c = divmod(i, NCOL)
    x0, y0 = ML + c * (S + GAP), FH - MT - S - r * (S + CBP + CBH + ROWGAP)
    a = inch(x0, y0, S, S)
    half, cell, c0 = g["half"], g["cell"], g["c0"]
    hp = int(round(half / cell)); sl = np.s_[c0 - hp:c0 + hp, c0 - hp:c0 + hp]
    im = a.imshow(g["sb"][sl], origin="lower", extent=[half, -half, -half, half], cmap="inferno", norm=norm(g["hi"]),
                  interpolation="nearest")
    a.set_xticks([]); a.set_yticks([])
    a.text(0.04, 0.96, g["label"] + ("\n(central pointing)" if g["mosaic"] else ""), transform=a.transAxes, color="w",
           ha="left", va="top", fontsize=13, fontweight="bold")
    sb_len = 1.0 if half < 7 else 2.0                                   # scale bar, top right (RA increases to the left)
    xs, ys = -half + 0.08 * half, half - 0.1 * half
    a.plot([xs + sb_len, xs], [ys, ys], color="w", lw=2.0, solid_capstyle="butt")
    a.text(xs + sb_len / 2, ys - 0.025 * half, f"{sb_len:g}″", color="w", ha="center", va="top", fontsize=10)
    BE = g["BE"]; pad = 0.06 * half                                       # beam, bottom left
    bx, by = half - pad - BE[0] / 2, -half + pad + BE[0] / 2
    a.add_patch(Ellipse((bx, by), width=BE[0], height=BE[1], angle=90 - BE[2], facecolor="w", edgecolor="w", lw=0.8))
    a.text(bx - BE[0] / 2 - 0.04 * half, by, f"{BE[0]:.2f}″ × {BE[1]:.2f}″", color="w", ha="left", va="center", fontsize=9)
    cax = inch(x0, y0 - CBP - CBH, S, CBH)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cax.tick_params(labelsize=10, pad=2, direction="out", length=3)
    if STRETCH != "symlog":
        cb.locator = matplotlib.ticker.MaxNLocator(4); cb.update_ticks()
fig.text(0.5, 1 - 0.12 / FH, "Extended-configuration MS-CLEAN, CO(2-1) moment 0", ha="center", va="top", fontsize=19,
         fontweight="bold")
fig.text(0.5, 1 - 0.47 / FH, "finished CLEAN components inside the astrodendro mask ∗ extended clean beam;  "
         r"Jy arcsec$^{-2}$ km s$^{-1}$, each panel its own scale, " + STXT, ha="center", va="top", fontsize=12)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=110)
fig.savefig(OUT.replace(".png", ".pdf"))
print("saved", OUT, "and .pdf")
