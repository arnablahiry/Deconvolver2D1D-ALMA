#!/usr/bin/env python
"""
NGC 3110 in one row, for the abstract / talk: the uv coverage of both arrays (results/NGC3110/uv_coverage.npz, compact
darker) | moment 0 of the compact MS-CLEAN (restored, its clean beam) | the compact
2D-1D model * lambda/B_max (compact) | the joint compact + extended image at that beam (the reference) | 2D-1D - joint,
in % of the joint map. Data as fig_ngc3110_moments_rows.py (20 km/s line channels, 0.08" grid; joint image
ladder_joint_20kms/b<ILB>, averaged onto that grid); maps in Jy arcsec^-2 km/s on one colour scale, the lambda/B_max
panels at full opacity inside the ladder's mask at that beam (the compact CLEAN inside the compact-beam mask), faded
outside. RMSE and flux error from resolution_ladder.json (cube, inside the mask).

    python scripts/compact_vs_msclean_dendro/fig_abstract_overview.py
    -> results/NGC3110/figures/resolution_ladder_joint20/ngc3110_abstract_overview.{png,pdf}
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.colors import PowerNorm
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from matplotlib.patches import Ellipse
from scipy.ndimage import convolve1d

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
from restore import convolve_with_beam  # noqa: E402

R = os.path.join(REPO, "results/NGC3110"); OUT = os.path.join(R, "figures", "resolution_ladder_joint20")
LAD = json.load(open(os.path.join(OUT, "resolution_ladder.json"))); MK = np.load(os.path.join(OUT, "masks.npz"))
g = json.load(open(os.path.join(R, "consistency_20kms_0.08/grid.json")))
L, DV, CELL = g["line"], g["width_kms"], 0.08
HALF = float(json.load(open(os.path.join(R, "config.json"))).get("half_arcsec", 5.5))
BEAMS = [(r["bmaj"], r["bmin"], r["bpa"]) for r in LAD["rows"]]
LB = LAD["lbmax_compact"]; ILB = int(np.argmin([abs(b[0] - LB) + abs(b[1] - LB) for b in BEAMS]))
rd = lambda p: np.nan_to_num(fits.getdata(p)).astype(np.float64).squeeze()
omega = lambda b: np.pi / (4 * np.log(2)) * b[0] * b[1]                 # beam area, arcsec^2
mom0 = lambda cube, b: cube.sum(0) * DV / omega(b)                       # Jy/beam -> Jy arcsec^-2 km/s


def label(a, title, sub=None, loc="upper left"):
    """bold title (+ plain subtitle) in one white box inside the panel"""
    kids = [TextArea(title, textprops=dict(fontsize=FL, fontweight="bold"))]
    if sub:
        kids.append(TextArea(sub, textprops=dict(fontsize=FL - 0.5, linespacing=1.2)))
    box = AnchoredOffsetbox(loc=loc, child=VPacker(children=kids, align="left", pad=0, sep=1.5), pad=0.25, borderpad=0.35,
                            frameon=True)
    box.patch.set(facecolor="w", edgecolor="0.6", linewidth=0.35, alpha=0.9, boxstyle="square,pad=0")
    a.add_artist(box)


def to008(x):                                                            # 0.04" Jy/beam cube -> the 0.08" grid (aligned)
    k = np.array([0.5, 1.0, 0.5]); return convolve1d(convolve1d(x, k, axis=-1, mode="constant"), k, axis=-2, mode="constant")[..., ::2, ::2] / 4


cc = os.path.join(R, "consistency_20kms_0.08/clean_compact_finished"); hc = fits.getheader(os.path.join(cc, "image.fits"))
BC = (hc["BMAJ"] * 3600, hc["BMIN"] * 3600, hc["BPA"])
m_cc = mom0(rd(os.path.join(cc, "image.fits"))[L], BC); foot_cc = MK["b00"].any(0)
UV = np.load(os.path.join(R, "uv_coverage.npz")); COL = {"compact": "#0b3c6e", "extended": "#3f8fd8"}  # as fig_ngc3110_data
BL = BEAMS[ILB]; foot = MK[f"b{ILB:02d}"].any(0)
xc = np.load(os.path.join(R, "pixel_scale/model_0.08_support.npy")).astype(np.float64)[L]
m_2d = mom0(convolve_with_beam(xc, CELL, *BL), BL)
m_jt = mom0(to008(rd(os.path.join(R, "ladder_joint_20kms", f"b{ILB:02d}", "clean_finished", "image.fits"))[L]), BL)
diff = np.where(foot, 100 * (m_2d - m_jt) / np.where(m_jt > 0, m_jt, np.nan), np.nan)
row = LAD["rows"][ILB]; RMSE, DF = row["rmse"], row["flux"]
print(f"lambda/B_max beam {BL[0]:.3f}\", compact beam {BC[0]:.2f}\" x {BC[1]:.2f}\"; cube RMSE {RMSE:.1f}%, flux {DF:+.1f}%")

# ======================================================================== figure
FS, FL = 8, 7
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": FS, "axes.linewidth": 0.5})
FW = 184 / 25.4; ML, MR, GX, GD, GU = 0.36, 0.05, 0.04, 0.16, 0.12
P = (FW - ML - MR - 2 * GX - GD - GU) / 5
CBP, CBH, CBL, MT, MB = 0.05, 0.06, 0.32, 0.05, 0.04
FH = MT + P + CBP + CBH + CBL + MB
fig = plt.figure(figsize=(FW, FH))
inch = lambda x_, y_, w, h: fig.add_axes([x_ / FW, y_ / FH, w / FW, h / FH])
XC = lambda c: ML + c * (P + GX) + (GU - GX) * (c >= 1) + (GD - GX) * (c >= 4)
Y0 = FH - MT - P
n = m_cc.shape[0]; c0 = n // 2; hp = int(round(HALF / CELL)); sl = np.s_[c0 - hp:c0 + hp, c0 - hp:c0 + hp]
EXT = [HALF, -HALF, -HALF, HALF]
VMAX = np.nanpercentile(np.where(foot, m_jt, np.nan), 99.8); NORM = PowerNorm(0.6, vmin=0, vmax=VMAX); DL = 30
# uv coverage: extended first, compact (darker) on top
a = inch(XC(0), Y0, P, P)
for k in ("extended", "compact"):
    u, v = UV[f"{k}_u"] / 1e3, UV[f"{k}_v"] / 1e3                         # Mlambda
    a.scatter(np.r_[u, -u], np.r_[v, -v], s=0.08 if k == "extended" else 0.25, color=COL[k], alpha=0.9, rasterized=True,
              linewidths=0)
    a.plot([], [], "s", ms=4, color=COL[k], label=k.capitalize())
lim = 1.08 * np.hypot(UV["extended_u"], UV["extended_v"]).max() / 1e3
a.set_xlim(lim, -lim); a.set_ylim(-lim, lim); a.set_aspect("equal")
a.set_xlabel(r"$u$ (M$\lambda$)", fontsize=FL, labelpad=1.5); a.set_ylabel(r"$v$ (M$\lambda$)", fontsize=FL, labelpad=1)
a.tick_params(labelsize=FL - 1, direction="in", length=1.5, width=0.4, pad=1.5, top=True, right=True)
h, l = a.get_legend_handles_labels()
a.legend(h[::-1], l[::-1], loc="lower left", fontsize=FL - 1, frameon=True, framealpha=0.95, edgecolor="0.6", fancybox=False,
         handletextpad=0.2, borderaxespad=0.3, borderpad=0.3).get_frame().set_linewidth(0.35)
label(a, "uv coverage", loc="upper right")

PANELS = [("Compact CLEAN", m_cc, foot_cc, BC, None, False),
          ("2D–1D deconvolved", m_2d, foot, BL, None, False),
          ("Ground truth", m_jt, foot, BL, "compact + extended", False),
          ("Difference", diff, foot, BL, f"RMSE {RMSE:.1f}%\nflux {DF:+.1f}%", True)]
IMS = []
for c, (title, img, fp, b, sub, light) in enumerate(PANELS, start=1):
    a = inch(XC(c), Y0, P, P)
    cm = plt.get_cmap("RdBu_r" if light else "inferno").copy(); cm.set_bad((0, 0, 0, 0))   # off-mask: show what is below
    kw = dict(vmin=-DL, vmax=DL) if light else dict(norm=NORM)
    a.set_facecolor("0.93" if light else "w")
    if not light:                                                       # outside the mask: the same map, faded
        a.imshow(img[sl], origin="lower", extent=EXT, cmap=cm, interpolation="nearest", alpha=0.6, **kw)
    IMS.append(a.imshow(np.where(fp, img, np.nan)[sl], origin="lower", extent=EXT, cmap=cm, interpolation="nearest", **kw))
    a.contour(fp[sl].astype(float), levels=[0.5], colors="0.55", linewidths=0.3, extent=EXT, origin="lower")
    a.set_xticks([]); a.set_yticks([])
    fg = "k"
    label(a, title, sub)
    pad = 0.07 * HALF                                                   # beam: bottom left, no text
    a.add_patch(Ellipse((HALF - pad - b[0] / 2, -HALF + pad + b[0] / 2), width=b[0], height=b[1], angle=90 - b[2],
                        facecolor="0.15" if light else "w", edgecolor="none"))
    if c == 1:                                                          # 2" scale bar: top right, end caps, bold label below
        xe, ys, cap = -HALF + 0.06 * HALF, HALF - 0.1 * HALF, 0.035 * HALF
        a.plot([xe + 2, xe], [ys, ys], color="k", lw=0.9, solid_capstyle="butt")
        for x_ in (xe, xe + 2):
            a.plot([x_, x_], [ys - cap, ys + cap], color="k", lw=0.7)
        a.text(xe + 1, ys - 2.4 * cap, "2″", color="k", ha="center", va="top", fontsize=FL, fontweight="bold")
for c0_, c1_, im, lab, tk in ((1, 3, IMS[0], r"moment 0 (Jy arcsec$^{-2}$ km s$^{-1}$)", None),
                              (4, 4, IMS[3], "2D–1D − truth (%)", [-DL, -15, 0, 15, DL])):
    ins = 0.15
    cax = inch(XC(c0_) + ins, Y0 - CBP - CBH, XC(c1_) + P - XC(c0_) - 2 * ins, CBH)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.outline.set_linewidth(0.35)
    if tk:
        cb.set_ticks(tk)
    cax.tick_params(labelsize=FL, length=1.5, width=0.4, pad=1); cb.set_label(lab, fontsize=FL, labelpad=1.5)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"ngc3110_abstract_overview.{ext}"), dpi=300 if ext == "png" else None)
print("saved", os.path.join(OUT, "ngc3110_abstract_overview.png"), f"({FW * 25.4:.0f} x {FH * 25.4:.0f} mm)")
