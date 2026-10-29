#!/usr/bin/env python
"""
Grid of the MS-CLEAN of pipeline targets (results/<target>/, scripts/pipeline/run_target.py), each cell a pair:
compact-configuration MS-CLEAN (left, or top) and extended-configuration MS-CLEAN (right, or bottom), line moment 0 in surface
brightness (Jy arcsec^-2 km/s) so the two beams are on the same units. Both are CLEAN components inside the extended
astrodendro mask (extended/msclean_finished/mask_final.npy) * their own clean beam, no residual: compact/msclean
(as the "MS-CLEAN compact" column of fig_gallery.py) and extended/msclean_finished (as fig_extended_msclean_grid.py).
Field of view per target as in fig_gallery.py (the line footprint of the extended mask_final + 0.5", or config.json
half_arcsec).

    [ORIENT=h|v] [NCOL=...] [TARGETS=NGC3110,...] [SCALE=panel|pair] [GAMMA=0.5] [SUBTITLE=1|0] [OUT=...] \
        python scripts/compact_vs_msclean_dendro/fig_msclean_pairs_grid.py
    -> results/gallery/msclean_compact_extended_3x3.{png,pdf}            (ORIENT=h: 9 galaxies, 3 per row, pairs side by side)
       results/gallery/msclean_compact_extended_2x6_vertical.{png,pdf}   (ORIENT=v: 12 galaxies, 6 per row, pairs stacked,
                                                                          compact / extended as row labels)
SCALE=panel: every panel its own colour scale (default); pair: compact and extended of a galaxy share one scale.
SUBTITLE=0: no line of text at the top (the paper version, ~/papers/deconvolver_paper/Figures/; the caption says it).
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
ORIENT = os.environ.get("ORIENT", "h")                                   # h: compact | extended; v: compact over extended
TOP9 = "NGC3110,NGC5135,NGC1614,ESO297-G011,NGC3256,IC4518W,NGC2369,IC4518E,NGC7130"
TARGETS = os.environ.get("TARGETS", TOP9 if ORIENT == "h" else TOP9 + ",IC5179,IRASF17138-1017,IC4687").split(",")
NCOL = int(os.environ.get("NCOL", 3 if ORIENT == "h" else 6))
NROW = -(-len(TARGETS) // NCOL)
SCALE = os.environ.get("SCALE", "panel")
GAMMA = float(os.environ.get("GAMMA", 0.5))
SUBTITLE = os.environ.get("SUBTITLE", "1") != "0"
OUT = os.environ.get("OUT", os.path.join(RES, "gallery", f"msclean_compact_extended_{NROW}x{NCOL}"
                                         f"{'' if ORIENT == 'h' else '_vertical'}.png"))

plt.rcParams.update({"font.family": "serif", "xtick.direction": "in", "ytick.direction": "in"})
rd = lambda p: np.nan_to_num(fits.getdata(p).astype(np.float64)).squeeze()
omega = lambda b: np.pi * b[0] * b[1] / (4 * np.log(2))                 # Gaussian beam area, arcsec^2


def load(T):
    R = os.path.join(RES, T); c = json.load(open(os.path.join(R, "config.json")))
    L = json.load(open(os.path.join(R, "line_channels.json")))["line"]
    cell, dv = c["cell_arcsec"], c["width_kms"]
    out = dict(T=T, label=c["label"], mosaic=c.get("mosaic", False), cell=cell)
    beam = lambda p: (lambda h: (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"]))(fits.getheader(p))
    M = np.load(os.path.join(R, "extended/msclean_finished/mask_final.npy")).astype(bool)
    for k, D in (("compact", "compact/msclean"), ("extended", "extended/msclean_finished")):
        D = os.path.join(R, D); B = beam(os.path.join(D, "image.fits"))
        x = rd(os.path.join(D, "model.fits")) * M                      # Jy/pixel components inside the mask
        out[k] = dict(sb=convolve_with_beam((x[L].sum(0) * dv)[None], cell, *B)[0] / omega(B), B=B)
    foot = M[L].any(0); n = foot.shape[0]; c0 = n // 2
    if "half_arcsec" in c:
        half = float(c["half_arcsec"])
    else:
        fy, fx = np.nonzero(foot)
        half = float(min(n * cell / 2 - 0.1, max(np.abs(fx - c0).max(), np.abs(fy - c0).max()) * cell + 0.5))
    for k in ("compact", "extended"):
        out[k]["hi"] = float(np.percentile(out[k]["sb"][foot], 99.9))
    if SCALE == "pair":
        hi = max(out["compact"]["hi"], out["extended"]["hi"])
        out["compact"]["hi"] = out["extended"]["hi"] = hi
    out.update(c0=c0, half=half)
    return out


gals = [load(t) for t in TARGETS]

# ======================================================================== layout (inches)
S, CBH, CBP, TITH = 2.6, 0.10, 0.06, 0.32
if ORIENT == "h":
    PGAP, GAP, ROWGAP = 0.32, 0.45, 0.55            # gap between a pair's panels, between galaxies, between rows
    BW, BH = 2 * S + PGAP, TITH + S + CBP + CBH     # one galaxy block (compact | extended): name, panels, colorbars
else:
    PGAP, GAP, ROWGAP = 0.30, 0.35, 0.75            # PGAP: below the compact colorbar (its tick labels)
    BW, BH = S, TITH + 2 * (S + CBP + CBH) + PGAP   # one galaxy block (compact over extended)
ML, MR, MB, MT = (0.2 if ORIENT == "h" else 0.5), 0.2, 0.35, (0.55 if ORIENT == "h" else 0.75) if SUBTITLE else 0.1
FW = ML + NCOL * BW + (NCOL - 1) * GAP + MR
FH = MB + NROW * BH + (NROW - 1) * ROWGAP + MT
fig = plt.figure(figsize=(FW, FH))
inch = lambda x, y, w, h: fig.add_axes([x / FW, y / FH, w / FW, h / FH])

for i, g in enumerate(gals):
    r, cblk = divmod(i, NCOL)
    xb = ML + cblk * (BW + GAP)
    yt = FH - MT - r * (BH + ROWGAP) - TITH                                # top of the block's first panel
    fig.text((xb + BW / 2) / FW, (yt + 0.06) / FH, g["label"] + ("  (central pointing)" if g["mosaic"] and ORIENT == "h" else ""),
             ha="center", va="bottom", fontsize=16)
    half, cell, c0 = g["half"], g["cell"], g["c0"]
    hp = int(round(half / cell)); sl = np.s_[c0 - hp:c0 + hp, c0 - hp:c0 + hp]
    for j, k in enumerate(("compact", "extended")):
        if ORIENT == "h":
            x0, y0 = xb + j * (S + PGAP), yt - S
        else:
            x0, y0 = xb, yt - S - j * (S + CBP + CBH + PGAP)
        a = inch(x0, y0, S, S); P = g[k]
        im = a.imshow(P["sb"][sl], origin="lower", extent=[half, -half, -half, half], cmap="inferno",
                      norm=matplotlib.colors.PowerNorm(GAMMA, vmin=0, vmax=P["hi"]), interpolation="nearest")
        a.set_xticks([]); a.set_yticks([])
        if ORIENT == "h":
            a.text(0.04, 0.96, k, transform=a.transAxes, color="w", ha="left", va="top", fontsize=14, fontweight="bold")
        else:
            if cblk == 0:                                               # row label, left of the first column
                fig.text((x0 - 0.12) / FW, (y0 + S / 2) / FH, k, rotation=90, ha="right", va="center", fontsize=16)
            if g["mosaic"] and k == "compact":                          # too wide for a one-column title
                a.text(0.04, 0.96, "central pointing", transform=a.transAxes, color="w", ha="left", va="top",
                       fontsize=11)
        if k == "extended":                                             # scale bar, top right (RA increases left)
            sb_len = 1.0 if half < 4 else 2.0
            xs, ys = -half + 0.08 * half, half - 0.1 * half
            a.plot([xs + sb_len, xs], [ys, ys], color="w", lw=2.0, solid_capstyle="butt")
            a.text(xs + sb_len / 2, ys - 0.03 * half, f"{sb_len:g}″", color="w", ha="center", va="top", fontsize=9)
        B = P["B"]; pad = 0.06 * half                                   # beam, bottom left
        bx, by = half - pad - B[0] / 2, -half + pad + B[0] / 2
        a.add_patch(Ellipse((bx, by), width=B[0], height=B[1], angle=90 - B[2], facecolor="w", edgecolor="w", lw=0.8))
        a.text(bx - B[0] / 2 - 0.04 * half, by, f"{B[0]:.2f}″ × {B[1]:.2f}″", color="w", ha="left", va="center",
               fontsize=11)
        cax = inch(x0, y0 - CBP - CBH, S, CBH)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cax.tick_params(labelsize=11, pad=1.5, direction="out", length=2.5)
        cb.locator = matplotlib.ticker.MaxNLocator(4); cb.update_ticks()

if SUBTITLE:
    stxt = "each panel its own scale" if SCALE == "panel" else "one scale per galaxy"
    fig.text(0.5, 1 - 0.12 / FH, r"CLEAN components inside the astrodendro mask $\ast$ each configuration's clean beam (no residual);"
             + ("  " if FW > 14 else "\n") + r"Jy arcsec$^{-2}$ km s$^{-1}$, " + stxt + f", power stretch (γ = {GAMMA:g})",
             ha="center", va="top", fontsize=11.5)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=120)
fig.savefig(OUT.replace(".png", ".pdf"))
print("saved", OUT, "and .pdf")
