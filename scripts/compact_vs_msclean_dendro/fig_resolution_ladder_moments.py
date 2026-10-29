#!/usr/bin/env python
"""
Kinematics along the resolution ladder (companion of fig_resolution_ladder.py REF=joint): moment 1 (intensity-weighted
velocity) and moment 2 (intensity-weighted dispersion) of the compact 2D-1D model * b and of the joint compact +
extended MS-CLEAN image at b (scripts/experiments/ladder_joint.py), for every ladder beam b, both inside that beam's
tight mask (figures/resolution_ladder_joint/masks.npz: one surface-brightness level, >= 5 sigma, inside the
compact-beam mask).

  moments    per pixel of the mask's footprint (its union over the line channels), over all line channels, negative
             voxels below 3 x the joint image's noise in that channel dropped from BOTH cubes (the same cut, so the noise
             of the joint image does not widen its spectra; masking voxel by voxel with the joint image's own tight mask
             would force both cubes onto the same channels and agree by construction)
  metrics    over those pixels: RMSE of (compact - joint) and its median (bias), km/s, for moment 1 and moment 2;
             also the moment-0-weighted RMSE (bright pixels count more)

Paper layout: rows = five beams (the clean beams, lambda/B_max and one on each side); columns = moment 0 (surface
brightness), moment 1, moment 2, each as compact | joint | difference, a colorbar above every column group (the
difference columns their own). Below: (a) moment 0, RMSE and median bias in % of joint; (b) moments 1
and 2, RMSE, moment-0-weighted RMSE and median bias in km/s. Percentages for moments 1 and 2 (moment 1 relative to
the joint velocity field's 5th-95th percentile spread, moment 2 relative to the joint dispersion) are in the JSON.

    python scripts/compact_vs_msclean_dendro/fig_resolution_ladder_moments.py
    -> results/NGC3110/figures/resolution_ladder_joint/resolution_ladder_moments.{png,pdf,json}
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
import matplotlib.patheffects as pe

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
from restore import convolve_with_beam  # noqa: E402

G20 = os.environ.get("GRID") == "20kms"   # the 64 x 20 km/s, 0.08" ladder (fig_resolution_ladder.py REF=joint GRID=20kms)
TGT = os.environ.get("T", "NGC3110"); G20 = G20 or TGT != "NGC3110"
R = os.path.join(REPO, "results", TGT); OUT = os.path.join(R, "figures", "resolution_ladder_joint" + ("20" if G20 else ""))
GC = json.load(open(os.path.join(R, "config.json")))
LAD = json.load(open(os.path.join(OUT, "resolution_ladder.json")))
MK = np.load(os.path.join(OUT, "masks.npz"))
L = list(MK["line"]); CELL = float(MK["cell"])
if G20:
    _g = json.load(open(os.path.join(R, "consistency_20kms_0.08/grid.json" if TGT == "NGC3110" else "g20/grid.json")))
    DV = _g["width_kms"]; vel = _g["start_kms"] + DV * np.arange(_g["nchan"])
else:
    DV = GC["width_kms"]; vel = GC["vsys_kms"] + DV * (np.arange(GC["nchan"]) - (GC["nchan"] - 1) / 2)
VL = vel[L]
BEAMS = [(r["bmaj"], r["bmin"], r["bpa"]) for r in LAD["rows"]]
LB = LAD["lbmax_compact"]; ILB = int(np.argmin([abs(b[0] - LB) + abs(b[1] - LB) for b in BEAMS]))
rd = lambda p: np.nan_to_num(fits.getdata(p).astype(np.float64)).squeeze()
if G20:
    from scipy.ndimage import convolve1d as _c1
    xc = np.load(os.path.join(R, "pixel_scale/model_0.08_support.npy" if TGT == "NGC3110" else "g20/model_0.08_support.npy")).astype(np.float64)[L]
    n = xc.shape[-1]; JD = os.path.join(R, "ladder_joint_20kms")

    def crop(p):                                                         # 0.04" joint cube -> the 0.08" grid (aligned)
        x = rd(p)[L]; k = np.array([0.5, 1.0, 0.5]); y = _c1(_c1(x, k, axis=-1, mode="constant"), k, axis=-2, mode="constant")
        return y[..., ::2, ::2] / (1.0 if p.endswith("model.fits") else 4.0)
else:
    xc = np.load(os.path.join(R, "compact/deconvolved/model.npy")).astype(np.float64)[L]
    xc = xc.reshape(len(L), xc.shape[1] // 2, 2, xc.shape[2] // 2, 2).sum((2, 4))   # onto 0.04" pixels
    n = xc.shape[-1]; JD = os.path.join(R, "ladder_joint")
    o = (fits.getheader(os.path.join(JD, "b00/clean_finished/image.fits"))["NAXIS1"] - n) // 2
    crop = lambda p: rd(p)[L][:, o:o + n, o:o + n]
NSIG = float(os.environ.get("NSIG", 3.0))                                # voxels below NSIG x the joint noise: dropped


def moments(cube, M, thr):
    """moment 1 and 2 (km/s) and moment 0 over all line channels at the pixels of M's footprint, voxels below thr
    (per channel, Jy/beam) set to zero"""
    w = np.where(cube >= thr[:, None, None], cube, 0.0); s = w.sum(0)
    ok = M.any(0) & (s > 0)
    m1 = np.where(ok, (w * VL[:, None, None]).sum(0) / np.where(ok, s, 1), np.nan)
    m2 = np.where(ok, np.sqrt(np.clip((w * (VL[:, None, None] - np.nan_to_num(m1)) ** 2).sum(0) / np.where(ok, s, 1),
                                      0, None)), np.nan)
    return m1, m2, s * DV


omega = lambda b: np.pi * b[0] * b[1] / (4 * np.log(2))
rows, MAPS = [], []
for ib, b in enumerate(BEAMS):
    M = MK[f"b{ib:02d}"]
    C = convolve_with_beam(xc, CELL, *b); I = crop(os.path.join(JD, f"b{ib:02d}", "clean_finished", "image.fits"))
    res = crop(os.path.join(JD, f"b{ib:02d}", "clean_finished", "residual.fits"))
    thr = NSIG * 1.4826 * np.median(np.abs(res - np.median(res, axis=(1, 2), keepdims=True)), axis=(1, 2))
    c1, c2, c0 = moments(C, M, thr); j1, j2, j0 = moments(I, M, thr)                # the same cut for both cubes
    c0, j0 = c0 / omega(b), j0 / omega(b)                                # Jy arcsec^-2 km/s
    v = np.isfinite(c1) & np.isfinite(j1); wt = j0[v] / j0[v].sum()
    d0 = 100 * (c0 - j0) / np.where(j0 > 0, j0, np.nan)
    met = {"mom0_rmse_pct": float(100 * np.linalg.norm((c0 - j0)[v]) / np.linalg.norm(j0[v])),
           "mom0_bias_pct": float(np.nanmedian(d0[v]))}
    for name, a_, b_ in (("mom1", c1, j1), ("mom2", c2, j2)):
        d = (a_ - b_)[v]
        met.update({f"{name}_rmse_kms": float(np.sqrt(np.mean(d ** 2))), f"{name}_bias_kms": float(np.median(d)),
                    f"{name}_rmse_w_kms": float(np.sqrt(np.sum(wt * d ** 2)))})
    # percentages: moment 1 relative to the joint velocity field's spread (5th-95th percentile), moment 2 relative to the
    # joint dispersion (as moment 0: 100 ||c - j|| / ||j||), biases likewise
    vspread = float(np.percentile(j1[v], 95) - np.percentile(j1[v], 5))
    met.update(mom1_vspread_kms=vspread, mom1_rmse_pct=100 * met["mom1_rmse_kms"] / vspread,
               mom1_bias_pct=100 * met["mom1_bias_kms"] / vspread,
               mom2_rmse_pct=float(100 * np.linalg.norm((c2 - j2)[v]) / np.linalg.norm(j2[v])),
               mom2_bias_pct=float(np.median(100 * (c2 - j2)[v & (j2 > 0)] / j2[v & (j2 > 0)])))
    rows.append(dict(bmaj=b[0], bmin=b[1], bpa=b[2], pixels=int(v.sum()), **met))
    nan = lambda x: np.where(v, x, np.nan)
    MAPS.append((nan(c0), nan(j0), nan(d0), c1, j1, c1 - j1, c2, j2, c2 - j2))
    print(f"{np.sqrt(b[0] * b[1]):.3f}\": " + ", ".join(f"{k} {x:.2f}" for k, x in met.items()), flush=True)
json.dump(dict(galaxy=GC["name"], reference="joint MS-CLEAN (restored) at each beam", masks="resolution_ladder_joint/masks.npz",
               rows=rows), open(os.path.join(OUT, "resolution_ladder_moments.json"), "w"), indent=1)

# ======================================================================== paper figure (A&A: full text width, 17 cm)
FS, FT, FL = 8, 7, 6.5                                                  # text, tick labels, legends (pt)
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": FS, "axes.labelsize": FS,
                     "xtick.labelsize": FT, "ytick.labelsize": FT, "xtick.direction": "in", "ytick.direction": "in",
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.5,
                     "ytick.minor.width": 0.5, "xtick.major.pad": 3, "ytick.major.pad": 3, "axes.labelpad": 4,
                     "legend.fontsize": FL, "lines.linewidth": 1.0, "lines.markersize": 3})
nb = len(BEAMS); RB = sorted({0, ILB, nb - 1})                                         # the beams shown: the clean beams and lambda/B_max
RC = RB.index(ILB)                                                      # the lambda/B_max row, set apart
FW, ML, MR = 184 / 25.4, 0.6, 0.05                                     # A&A \textwidth (184 mm)
PG, GG, GRC = 0.045, 0.2, 0.14                                             # gaps: panels, groups, before Δ, around RC
P = (FW - ML - MR - 6 * PG - 2 * GG) / 9                                 # equal gaps inside a block
W = 9 * P + 6 * PG + 2 * GG
HDR, CBP, CBH, CBL = 0.15, 0.04, 0.06, 0.4                              # header, colorbar pad / height / labels
HL, GL, GAB, GT, HS, MB = 1.36, 0.38, 0.28, 0.18, 0.36, 0.2               # metric panels, gaps, tick labels, beams, bottom
Y_B = MB + HS + GT; Y_A = Y_B + HL + GAB
HM = len(RB) * P + (len(RB) - 1) * PG + 2 * GRC
y_top = Y_A + HL + GL + HM                                              # top of the maps
FH = y_top + HDR + CBP + CBH + CBL
fig = plt.figure(figsize=(FW, FH))
inch = lambda x, y, w, h: fig.add_axes([x / FW, y / FH, w / FW, h / FH])
XC = lambda c: ML + c * (P + PG) + (c // 3) * (GG - PG)                  # column c's left edge
YR = lambda r: y_top - (r + 1) * P - r * PG - (GRC if r >= RC else 0) - (GRC if r > RC else 0)      # row r's bottom edge
HALF = float(GC.get("half_arcsec", 5.5))
hp = int(round(HALF / CELL)); C0 = n // 2
sl = np.s_[C0 - hp:C0 + hp, C0 - hp:C0 + hp]; EXT = [HALF, -HALF, -HALF, HALF]
VS = GC["vsys_kms"]; DL = float(os.environ.get("DLIM", 20))
vm0 = float(np.nanpercentile(np.concatenate([MAPS[i][1][sl].ravel() for i in RB]), 99.5))
DP = float(os.environ.get("DPCT", 40))                                  # moment 0 difference: +-DP %
COLS = [("inferno", 0, vm0), ("inferno", 0, vm0), ("PuOr_r", -DP, DP),       # differences: one colour map
        ("RdBu_r", VS - 260, VS + 260), ("RdBu_r", VS - 260, VS + 260), ("PuOr_r", -DL, DL),
        ("viridis", 0, 100), ("viridis", 0, 100), ("PuOr_r", -DL, DL)]
HEAD = ["compact", "joint", "difference"] * 3
IMS = {}
for r_, ib in enumerate(RB):
    b = BEAMS[ib]; y = YR(r_)
    for c in range(9):
        a = inch(XC(c), y, P, P)
        cm, lo, hi = COLS[c]; cmap = plt.get_cmap(cm).copy(); cmap.set_bad("0.92")
        IMS[c] = a.imshow(MAPS[ib][c][sl], origin="lower", extent=EXT, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_linewidth(0.75 if r_ == RC else 0.5)                    # the lambda/B_max row stands out
        pad = 0.08 * HALF; bw = max(b[0], 0.18)                          # beam, bottom left (tiny beams kept visible)
        a.add_patch(Ellipse((HALF - pad - bw / 2, -HALF + pad + bw / 2), width=b[0], height=b[1], angle=90 - b[2],
                            facecolor="0.15", edgecolor="0.15", lw=0.4))
        if r_ == 0:                                                      # column header, above the panel
            a.text(0.5, 1.03, HEAD[c], transform=a.transAxes, ha="center", va="bottom", fontsize=FT)
        if c == 0:
            t = f"{b[0]:.2f}″ × {b[1]:.2f}″" if abs(b[0] - b[1]) > 1e-3 else f"{b[0]:.2f}″"
            nm = {0: "compact beam", ILB: r"λ/B$_{\mathbf{max}}$", nb - 1: "extended beam"}.get(ib)
            a.set_ylabel(t + (f"\n{nm}" if nm else ""), fontsize=FT, fontweight="bold" if nm else "normal", labelpad=4)
        if r_ == 0 and c == 0:                                           # 2" scale bar, top right, end caps
            xs, ys, cap = -HALF + 0.1 * HALF, HALF - 0.32 * HALF, 0.04 * HALF
            a.plot([xs, xs + 2], [ys, ys], color="k", lw=0.9, solid_capstyle="butt")
            for xe in (xs, xs + 2):
                a.plot([xe, xe], [ys - cap, ys + cap], color="k", lw=0.7)
            a.text(xs + 1, ys + 1.5 * cap, "2″", ha="center", va="bottom", fontsize=FT - 0.5, fontweight="bold")
# colorbars above the headers: one per moment pair, one per difference column
CB = [((0, 2), r"moment 0 (Jy arcsec$^{-2}$ km s$^{-1}$)"), ((2, 1), "Δ (%)"),
      ((3, 2), r"moment 1 (km s$^{-1}$)"), ((5, 1), r"Δ (km s$^{-1}$)"),
      ((6, 2), r"moment 2 (km s$^{-1}$)"), ((8, 1), r"Δ (km s$^{-1}$)")]
for (c, w_), lab in CB:
    cax = inch(XC(c), y_top + HDR + CBP, w_ * P + (w_ - 1) * PG, CBH)
    cb = fig.colorbar(IMS[c], cax=cax, orientation="horizontal"); cb.outline.set_linewidth(0.5)
    cax.xaxis.set_ticks_position("top"); cax.xaxis.set_label_position("top")
    cb.locator = (matplotlib.ticker.MaxNLocator(5, prune="upper") if w_ == 2 else matplotlib.ticker.MaxNLocator(3))
    cb.update_ticks()                                                    # (pairs: no end tick next to the Δ bar)
    cax.tick_params(labelsize=FL, length=2, width=0.5, pad=2); cb.set_label(lab, fontsize=FT, labelpad=3)

# metrics, stacked: (a) % (whole cube from resolution_ladder.json, and moment 0); (b) km/s (moments 1 and 2); the beams
# drawn to scale below
x = np.arange(nb)
tl = [f"{b[0]:.2f}″×{b[1]:.2f}″" if abs(b[0] - b[1]) > 1e-3 else f"{b[0]:.2f}″" for b in BEAMS]
CUBE = LAD["rows"]
LEG = dict(fontsize=FL, frameon=True, framealpha=0.95, edgecolor="0.6", fancybox=False, borderpad=0.4, labelspacing=0.3,
           handlelength=2.2, columnspacing=1.2)


def style(a, labels):
    a.set_xlim(-0.5, nb - 0.5); a.set_xticks(x); a.minorticks_on()
    a.tick_params(which="both", top=True, right=True); a.tick_params(which="major", length=3); a.tick_params(which="minor", length=1.5)
    a.tick_params(axis="x", which="minor", bottom=False, top=False)
    a.axhline(0, color="0.5", lw=0.5); a.axvline(ILB, color="0.5", lw=0.6, ls=":")
    if labels:
        a.set_xticklabels(tl, fontsize=FT)
        for i_ in (0, ILB, nb - 1):
            a.get_xticklabels()[i_].set_fontweight("bold")
    else:
        a.tick_params(labelbottom=False)


a = inch(ML, Y_A, W, HL); style(a, False)
a.axhline(10, color="0.45", lw=0.6, ls="--")                            # 10% reference
a.plot(x, [r["rmse"] for r in CUBE], "-D", color="k", ms=2.6, label="cube: RMSE")
a.plot(x, [r["flux"] for r in CUBE], "--D", color="k", lw=0.8, ms=2.4, mfc="none", label="cube: total flux")
a.plot(x, [r["median_dev"] for r in CUBE], ":D", color="k", lw=0.8, ms=2.4, label="cube: median deviation")
a.plot(x, [r["mom0_rmse_pct"] for r in rows], "-o", color="#7a3e9d", label="moment 0: RMSE")
a.plot(x, [r["mom0_bias_pct"] for r in rows], ":s", color="#7a3e9d", lw=0.8, ms=2.6, label="moment 0: median bias")
a.set_ylabel("compact − joint (%)"); a.set_ylim(top=a.get_ylim()[1] * 1.3); a.yaxis.set_label_coords(-0.055, 0.5)
a.legend(loc="upper left", ncol=2, **LEG)
a.text(0.0, 1.04, "(a) cube and moment 0", transform=a.transAxes, ha="left", va="bottom", fontweight="bold")
a = inch(ML, Y_B, W, HL); style(a, True)
a.axhline(5, color="0.45", lw=0.6, ls="--")                             # 5 km/s reference
for key, col, lab in (("mom1", "#2a6fdb", "moment 1"), ("mom2", "#e07b00", "moment 2")):
    a.plot(x, [r[f"{key}_rmse_kms"] for r in rows], "-o", color=col, label=f"{lab}: RMSE")
    a.plot(x, [r[f"{key}_rmse_w_kms"] for r in rows], "--o", color=col, lw=0.8, ms=2.4, mfc="none",
           label=f"{lab}: RMSE, moment-0 weighted")
    a.plot(x, [r[f"{key}_bias_kms"] for r in rows], ":s", color=col, lw=0.8, ms=2.4, label=f"{lab}: median bias")
a.set_ylabel(r"compact − joint (km s$^{-1}$)"); a.yaxis.set_label_coords(-0.055, 0.5)
a.set_ylim(top=a.get_ylim()[1] * 1.5)
a.legend(loc="upper left", ncol=2, **LEG)
a.text(0.0, 1.04, f"(b) moments 1 and 2 (channel width {DV:g} km s$^{{-1}}$)", transform=a.transAxes, ha="left",
       va="bottom", fontweight="bold")
# the beams to scale (one arcsec scale for all; RA increases left), bold outline for the named ones
st = inch(ML, MB, W, HS); cw = W / nb
st.set_xlim(-0.5, nb - 0.5); st.set_ylim(0, HS / cw); st.axis("off")
ang = lambda b: np.radians(90 + b[2])
ext_ = lambda b, c_, s_: np.hypot(b[0] / 2 * c_, b[1] / 2 * s_)
wide = max(2 * ext_(b, np.cos(ang(b)), np.sin(ang(b))) for b in BEAMS)
tall = max(2 * ext_(b, np.sin(ang(b)), np.cos(ang(b))) for b in BEAMS)
k_ = min(0.8 / wide, 0.9 * (HS / cw) / tall)                            # columns per arcsec
for i_, b in enumerate(BEAMS):
    named = i_ in (0, ILB, nb - 1)
    st.add_patch(Ellipse((i_, HS / cw / 2), width=b[0] * k_, height=b[1] * k_, angle=90 + b[2], facecolor="none",
                         edgecolor="k" if named else "0.55", lw=0.9 if named else 0.6))
fig.text((ML + W / 2) / FW, 0.03 / FH, "comparison beam (FWHM)", ha="center", va="bottom", fontsize=FS)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"resolution_ladder_moments.{ext}"), dpi=300 if ext == "png" else None)
print("saved", os.path.join(OUT, "resolution_ladder_moments.png"), f"({FW:.2f} x {FH:.2f} in)")
