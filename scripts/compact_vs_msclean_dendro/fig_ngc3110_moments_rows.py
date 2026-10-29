#!/usr/bin/env python
"""
NGC 3110 kinematics in the layout of the paper's gallery (fig_gallery_transposed.py LAYOUT=rows): row 1 moment 1, row 2
moment 2; columns compact MS-CLEAN (restored, its clean beam) | compact 2D-1D, joint, difference at lambda/B_max | the same
at the extended clean beam 

Moments as in fig_resolution_ladder_moments.py: line channels, voxels below 3 sigma of the joint image (per channel, from
its residual) dropped from both cubes, at the pixels of the ladder's mask at that beam (resolution_ladder_joint20/masks.npz);
the compact MS-CLEAN with its own 3 sigma cut, at the pixels of the compact-beam mask. Joint images: the 64 x 20 km/s
ladder (ladder_joint_20kms/b<ii>), averaged onto the 0.08" grid of the 2D-1D model.

    python scripts/compact_vs_msclean_dendro/fig_ngc3110_moments_rows.py
    -> results/NGC3110/figures/resolution_ladder_joint20/ngc3110_moments_rows.{png,pdf}
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

R = os.path.join(REPO, "results/NGC3110"); OUT = os.path.join(R, "figures", "resolution_ladder_joint20")
LAD = json.load(open(os.path.join(OUT, "resolution_ladder.json"))); MK = np.load(os.path.join(OUT, "masks.npz"))
g = json.load(open(os.path.join(R, "consistency_20kms_0.08/grid.json")))
L, DV, CELL = g["line"], g["width_kms"], 0.08
VL = (g["start_kms"] + DV * np.arange(g["nchan"]))[L]
GC = json.load(open(os.path.join(R, "config.json"))); VS = GC["vsys_kms"]; HALF = float(GC.get("half_arcsec", 5.5))
BEAMS = [(r["bmaj"], r["bmin"], r["bpa"]) for r in LAD["rows"]]
LB = LAD["lbmax_compact"]; ILB = int(np.argmin([abs(b[0] - LB) + abs(b[1] - LB) for b in BEAMS])); IEX = len(BEAMS) - 1
rd = lambda p: np.nan_to_num(fits.getdata(p)).astype(np.float64).squeeze()
sig = lambda res: 1.4826 * np.median(np.abs(res - np.median(res, axis=(1, 2), keepdims=True)), axis=(1, 2))


def to008(x):                                                            # 0.04" Jy/beam cube -> the 0.08" grid (aligned)
    k = np.array([0.5, 1.0, 0.5]); return convolve1d(convolve1d(x, k, axis=-1, mode="constant"), k, axis=-2, mode="constant")[..., ::2, ::2] / 4


def moments(cube, foot, thr):
    w = np.where(cube >= thr[:, None, None], cube, 0.0); s = w.sum(0); ok = foot & (s > 0)
    m1 = np.where(ok, (w * VL[:, None, None]).sum(0) / np.where(ok, s, 1), np.nan)
    m2 = np.where(ok, np.sqrt(np.clip((w * (VL[:, None, None] - np.nan_to_num(m1)) ** 2).sum(0) / np.where(ok, s, 1), 0, None)),
                  np.nan)
    return m1, m2


xc = np.load(os.path.join(R, "pixel_scale/model_0.08_support.npy")).astype(np.float64)[L]
COLS = {}
cc = os.path.join(R, "consistency_20kms_0.08/clean_compact_finished")
hc = fits.getheader(os.path.join(cc, "image.fits"))
BC = (hc["BMAJ"] * 3600, hc["BMIN"] * 3600, hc["BPA"]) if "BMAJ" in hc else BEAMS[0]
I0, r0 = rd(os.path.join(cc, "image.fits"))[L], rd(os.path.join(cc, "residual.fits"))[L]
COLS["cc"] = moments(I0, MK["b00"].any(0), 3 * sig(r0)); del I0, r0
STAT = {}
for key, ib in (("l", ILB), ("e", IEX)):
    jd = os.path.join(R, "ladder_joint_20kms", f"b{ib:02d}", "clean_finished")
    I, res = to008(rd(os.path.join(jd, "image.fits"))[L]), to008(rd(os.path.join(jd, "residual.fits"))[L])
    thr = 3 * sig(res); foot = MK[f"b{ib:02d}"].any(0)
    c1, c2 = moments(convolve_with_beam(xc, CELL, *BEAMS[ib]), foot, thr); j1, j2 = moments(I, foot, thr)
    COLS[key] = (c1, j1, c1 - j1, c2, j2, c2 - j2, foot)
    v = np.isfinite(c1) & np.isfinite(j1)
    STAT[key] = {m: (float(np.sqrt(np.nanmean(d[v] ** 2))), float(np.nanmedian(d[v]))) for m, d in (("m1", c1 - j1), ("m2", c2 - j2))}
    print(key, BEAMS[ib], {m: tuple(round(x, 2) for x in t) for m, t in STAT[key].items()}, flush=True)
    del I, res

if os.environ.get("LAYOUT") == "grid":
    # ==================================================================== 3 rows (2D-1D, joint, difference) x 4 columns
    # (moment 1, moment 2 at lambda/B_max | moment 1, moment 2 at the extended beam); map colorbars under the joint row,
    # difference colorbars under the difference row
    FS, FL = 7, 5.5
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": FS, "xtick.direction": "in",
                         "ytick.direction": "in", "axes.linewidth": 0.5})
    FW = 184 / 25.4; ML, MR, GX, GB, GY = 0.3, 0.06, 0.04, 0.16, 0.04
    P = (FW - ML - MR - 2 * GX - GB) / 4
    CBP, CBH, CBL, MT, MB = 0.04, 0.05, 0.3, 0.5, 0.04
    FH = MT + 3 * P + 2 * GY + 2 * (CBP + CBH + CBL) + MB
    fig = plt.figure(figsize=(FW, FH))
    inch = lambda x_, y_, w, h: fig.add_axes([x_ / FW, y_ / FH, w / FW, h / FH])
    XC = lambda c: ML + c * (P + GX) + (GB - GX) * (c >= 2)
    YT = [FH - MT - P]                                                  # bottoms of the rows, top down
    YT.append(YT[0] - GY - P)
    YT.append(YT[1] - (CBP + CBH + CBL) - P)
    n = 640; c0 = n // 2; hp = int(round(HALF / CELL)); sl = np.s_[c0 - hp:c0 + hp, c0 - hp:c0 + hp]; EXT = [HALF, -HALF, -HALF, HALF]
    DL = 20
    SPEC = [("RdBu_r", VS - 260, VS + 260, r"$v$ (km s$^{-1}$)"), ("viridis", 0, 100, r"$\sigma_v$ (km s$^{-1}$)")]
    IMS = {}
    for c, (key, mi) in enumerate((("l", 0), ("l", 1), ("e", 0), ("e", 1))):
        t = COLS[key]; foot = t[6]; ib = ILB if key == "l" else IEX; b = BEAMS[ib]; cmap, lo, hi, unit = SPEC[mi]
        for r, (img, cm, l_, h_, light) in enumerate(((t[3 * mi], cmap, lo, hi, False), (t[3 * mi + 1], cmap, lo, hi, False),
                                                     (t[3 * mi + 2], "PuOr_r", -DL, DL, True))):
            a = inch(XC(c), YT[r], P, P)
            cmo = plt.get_cmap(cm).copy(); cmo.set_bad("0.93" if light else "0.12")
            IMS[(c, r)] = a.imshow(img[sl], origin="lower", extent=EXT, cmap=cmo, vmin=l_, vmax=h_, interpolation="nearest")
            a.contour(foot[sl].astype(float), levels=[0.5], colors="0.55", linewidths=0.3, extent=EXT, origin="lower")
            a.set_xticks([]); a.set_yticks([])
            pad = 0.07 * HALF
            a.add_patch(Ellipse((HALF - pad - b[0] / 2, -HALF + pad + b[0] / 2), width=b[0], height=b[1], angle=90 - b[2],
                                facecolor="0.15" if light else "w", edgecolor="0.15" if light else "w", lw=0.3))
            if r == 2:
                rm, bi = STAT[key]["m1" if mi == 0 else "m2"]
                a.text(0.04, 0.96, f"RMSE {rm:.1f}\nbias {bi + 0.0:+.1f}".replace("-0.0", "0.0").replace("+0.0", "0.0"),
                       transform=a.transAxes, ha="left", va="top", fontsize=FL, linespacing=1.25)
            if c == 0 and r == 0:                                       # 2" scale bar, top left, end caps
                xs, ys, cap = -HALF + 0.1 * HALF, HALF - 0.14 * HALF, 0.035 * HALF
                a.plot([xs, xs + 2], [ys, ys], color="w", lw=0.8, solid_capstyle="butt")
                for xe in (xs, xs + 2):
                    a.plot([xe, xe], [ys - cap, ys + cap], color="w", lw=0.6)
                a.text(xs + 1, ys - 2.2 * cap, "2″", color="w", ha="center", va="top", fontsize=FL, fontweight="bold")
        fig.text((XC(c) + P / 2) / FW, (FH - MT + 0.03) / FH, ["moment 1", "moment 2"][mi], ha="center", va="bottom",
                 fontsize=FS - 0.5)
        for r, im, lab, tk in ((1, IMS[(c, 1)], unit, None), (2, IMS[(c, 2)], r"Δ (km s$^{-1}$)", [-DL, -10, 0, 10, DL])):
            ins = 0.12
            cax = inch(XC(c) + ins, YT[r] - CBP - CBH, P - 2 * ins, CBH)
            cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.outline.set_linewidth(0.35)
            if tk:
                cb.set_ticks(tk)
            else:
                cb.locator = matplotlib.ticker.MaxNLocator(4); cb.update_ticks()
            cax.tick_params(labelsize=FL - 0.5, length=1.5, width=0.4, pad=1); cb.set_label(lab, fontsize=FL, labelpad=1)
    for r, t in enumerate(("2D–1D", "joint", "difference")):
        fig.text((ML - 0.05) / FW, (YT[r] + P / 2) / FH, t, rotation=90, ha="right", va="center", fontsize=FS)
    for c0_, t in ((0, r"at λ/B$_{\mathbf{max}}$ (compact)"), (2, "at the extended clean beam")):
        x0, x1 = XC(c0_), XC(c0_ + 1) + P; yb = FH - MT + 0.22
        fig.add_artist(matplotlib.lines.Line2D([x0 / FW, x1 / FW], [yb / FH, yb / FH], color="0.3", lw=0.5))
        fig.text((x0 + x1) / 2 / FW, (yb + 0.03) / FH, t, ha="center", va="bottom", fontsize=FS, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"ngc3110_moments_grid.{ext}"), dpi=300 if ext == "png" else None)
    print("saved", os.path.join(OUT, "ngc3110_moments_grid.png"), f"({FW * 25.4:.0f} x {FH * 25.4:.0f} mm)"); sys.exit(0)

# ======================================================================== figure (A&A text width)
FS, FL = 7, 5.5
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": FS, "xtick.direction": "in",
                     "ytick.direction": "in", "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5})
FW = 184 / 25.4
ML, MR, GX, GB, GY = 0.2, 0.08, 0.03, 0.1, 0.06
CBP, CBH, CBL = 0.04, 0.045, 0.3                                         # colorbar pad, height, its labels
P = (FW - ML - MR - 6 * GX - 2 * GB) / 7
MT, MB = 0.42, 0.06
FH = MT + 2 * P + GY + 2 * (CBP + CBH + CBL) + MB
fig = plt.figure(figsize=(FW, FH))
inch = lambda x_, y_, w, h: fig.add_axes([x_ / FW, y_ / FH, w / FW, h / FH])
XC = lambda c: ML + c * (P + GX) + (GB - GX) * ((c >= 1) + (c >= 4))
YR = lambda r: FH - MT - (r + 1) * P - r * (GY + CBP + CBH + CBL)
n = 640; c0 = n // 2; hp = int(round(HALF / CELL)); sl = np.s_[c0 - hp:c0 + hp, c0 - hp:c0 + hp]; EXT = [HALF, -HALF, -HALF, HALF]
DL = 20


def m(c, r, img, beam, lo, hi, cmap, foot, light=False):
    a = inch(XC(c), YR(r), P, P)
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("0.93" if light else "0.12")
    im = a.imshow(img[sl], origin="lower", extent=EXT, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
    a.contour(foot[sl].astype(float), levels=[0.5], colors="0.55", linewidths=0.3, extent=EXT, origin="lower")
    a.set_xticks([]); a.set_yticks([])
    pad = 0.07 * HALF
    a.add_patch(Ellipse((HALF - pad - beam[0] / 2, -HALF + pad + beam[0] / 2), width=beam[0], height=beam[1], angle=90 - beam[2],
                        facecolor="0.15" if light else "w", edgecolor="0.15" if light else "w", lw=0.3))
    return a, im


ROWS = [("moment 1", "RdBu_r", VS - 260, VS + 260, 0, r"$v$ (km s$^{-1}$)"),
        ("moment 2", "viridis", 0, 100, 3, r"$\sigma_v$ (km s$^{-1}$)")]
for r, (name, cmap, lo, hi, o, unit) in enumerate(ROWS):
    a, IMm = m(0, r, COLS["cc"][r], BC, lo, hi, cmap, MK["b00"].any(0))
    if r == 0:                                                           # 2" scale bar, top left, end caps
        xs, ys, cap = -HALF + 0.1 * HALF, HALF - 0.14 * HALF, 0.035 * HALF
        a.plot([xs, xs + 2], [ys, ys], color="k", lw=0.8, solid_capstyle="butt")
        for xe in (xs, xs + 2):
            a.plot([xe, xe], [ys - cap, ys + cap], color="k", lw=0.6)
        a.text(xs + 1, ys - 2.2 * cap, "2″", ha="center", va="top", fontsize=FL - 0.5, fontweight="bold")
    fig.text((ML - 0.04) / FW, (YR(r) + P / 2) / FH, name, rotation=90, ha="right", va="center", fontsize=FS - 0.5)
    for key, c_, ib in (("l", 1, ILB), ("e", 4, IEX)):
        t = COLS[key]; foot = t[6]
        m(c_, r, t[o], BEAMS[ib], lo, hi, cmap, foot)
        m(c_ + 1, r, t[o + 1], BEAMS[ib], lo, hi, cmap, foot)
        a, IMd = m(c_ + 2, r, t[o + 2], BEAMS[ib], -DL, DL, "PuOr_r", foot, light=True)
        rm, bi = STAT[key]["m1" if r == 0 else "m2"]
        a.text(0.04, 0.96, f"RMSE {rm:.1f}\nbias {bi + 0.0:+.1f}".replace("-0.0", "0.0").replace("+0.0", "0.0"), transform=a.transAxes, ha="left", va="top", fontsize=FL - 0.5,
               linespacing=1.25)
    # horizontal colorbars under this row, one per column group: maps (cols 0-2, 4-5), differences (cols 3, 6)
    for c0_, c1_, im, lab, tk in ((0, 2, IMm, unit, None), (3, 3, IMd, r"Δ (km s$^{-1}$)", [-DL, 0, DL]),
                                  (4, 5, IMm, unit, None), (6, 6, IMd, r"Δ (km s$^{-1}$)", [-DL, 0, DL])):
        ins = 0.13
        cax = inch(XC(c0_) + ins, YR(r) - CBP - CBH, XC(c1_) + P - XC(c0_) - 2 * ins, CBH)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.outline.set_linewidth(0.35)
        if tk:
            cb.set_ticks(tk)
        else:
            cb.locator = matplotlib.ticker.MaxNLocator(4); cb.update_ticks()
        cax.tick_params(labelsize=FL - 0.5, length=1.5, width=0.4, pad=1); cb.set_label(lab, fontsize=FL - 0.5, labelpad=1)
TIT = ["compact\nMS-CLEAN", "2D–1D", "joint", "difference", "2D–1D", "joint", "difference"]
for c, t in enumerate(TIT):
    fig.text((XC(c) + P / 2) / FW, (FH - MT + 0.03) / FH, t, ha="center", va="bottom", fontsize=FS - 0.5, linespacing=1.0)
for c0_, t in ((1, r"at λ/B$_{\mathbf{max}}$ (compact)"), (4, "at the extended clean beam")):
    x0, x1 = XC(c0_), XC(c0_ + 2) + P; yb = FH - MT + 0.2
    fig.add_artist(matplotlib.lines.Line2D([x0 / FW, x1 / FW], [yb / FH, yb / FH], color="0.3", lw=0.5))
    fig.text((x0 + x1) / 2 / FW, (yb + 0.03) / FH, t, ha="center", va="bottom", fontsize=FS, fontweight="bold")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"ngc3110_moments_rows.{ext}"), dpi=300 if ext == "png" else None)
print("saved", os.path.join(OUT, "ngc3110_moments_rows.png"), f"({FW * 25.4:.0f} x {FH * 25.4:.0f} mm)")
