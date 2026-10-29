#!/usr/bin/env python
"""
Column-width (A&A, 88 mm) version of the metric panels of the resolution-ladder figure (fig_resolution_ladder_moments.py):
(a) whole cube (RMSE, total flux, median deviation) and moment 0 (RMSE, median bias) in %, (b) moments 1 and 2 (RMSE,
moment-0-weighted RMSE, median bias) in km/s, stacked, against the comparison beam; the beams drawn to scale below.
Reads resolution_ladder.json and resolution_ladder_moments.json of the 64 x 20 km/s ladder.

    python scripts/compact_vs_msclean_dendro/fig_ladder_metrics_column.py
    -> results/NGC3110/figures/resolution_ladder_joint20/ladder_metrics_column.{png,pdf}
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TGT = os.environ.get("T", "NGC3110")
D = os.path.join(REPO, "results", TGT, "figures", os.environ.get("LADDER", "resolution_ladder_joint20"))
LABEL = json.load(open(os.path.join(REPO, "results", TGT, "config.json")))["label"]
LAD = json.load(open(os.path.join(D, "resolution_ladder.json")))
MOM = json.load(open(os.path.join(D, "resolution_ladder_moments.json")))
CUBE, rows = LAD["rows"], MOM["rows"]
BEAMS = [(r["bmaj"], r["bmin"], r["bpa"]) for r in CUBE]
LB = LAD["lbmax_compact"]; nb = len(BEAMS)
ILB = int(np.argmin([abs(b[0] - LB) + abs(b[1] - LB) for b in BEAMS]))
DV = 20

FS, FT, FL = 8, 7, 6
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": FS, "axes.labelsize": FS,
                     "xtick.labelsize": FT, "ytick.labelsize": FT, "xtick.direction": "in", "ytick.direction": "in",
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.5,
                     "ytick.minor.width": 0.5, "xtick.major.pad": 2.5, "ytick.major.pad": 2.5, "legend.fontsize": FL,
                     "lines.linewidth": 1.0, "lines.markersize": 2.8})
FW = 88 / 25.4; ML, MR = 0.45, 0.06; W = FW - ML - MR
HL, HS, GS, MB, MT = 1.6, 0.26, 0.04, 0.6, 0.06                  # panels, beam strip between them, its padding
Y_B = MB; Y_S = Y_B + HL + GS; Y_A = Y_S + HS + GS; FH = Y_A + HL + MT
fig = plt.figure(figsize=(FW, FH))
inch = lambda x, y, w, h: fig.add_axes([x / FW, y / FH, w / FW, h / FH])
x = np.arange(nb)
tl = [f"{b[0]:.2f}×{b[1]:.2f}" if abs(b[0] - b[1]) > 1e-3 else f"{b[0]:.2f}" for b in BEAMS]
LEG = dict(frameon=True, framealpha=0.95, edgecolor="0.6", fancybox=False, borderpad=0.35, labelspacing=0.25,
           handlelength=2.0, columnspacing=0.8, handletextpad=0.4)


def style(a, labels):
    a.set_xlim(-0.5, nb - 0.5); a.set_xticks(x); a.minorticks_on()
    a.tick_params(which="both", top=True, right=True); a.tick_params(which="major", length=2.5)
    a.tick_params(which="minor", length=1.3); a.tick_params(axis="x", which="minor", bottom=False, top=False)
    a.axhline(0, color="0.5", lw=0.5); a.axvline(ILB, color="0.5", lw=0.6, ls=":")
    if labels:
        a.set_xticklabels(tl, rotation=45, ha="right", rotation_mode="anchor", fontsize=FL)
        for i_ in (0, ILB, nb - 1):
            a.get_xticklabels()[i_].set_fontweight("bold")
    else:
        a.tick_params(labelbottom=False)
    a.yaxis.set_label_coords(-0.1, 0.5)


a = inch(ML, Y_A, W, HL); style(a, False)
a.axhline(10, color="0.45", lw=0.6, ls="--"); a.axhline(5, color="0.75", lw=0.5, ls="--")
a.plot(x, [r["rmse"] for r in CUBE], "-D", color="k", ms=2.4, label="cube: RMSE")
a.plot(x, [r["flux"] for r in CUBE], "--D", color="k", lw=0.8, ms=2.2, mfc="none", label="cube: total flux")
a.plot(x, [r["median_dev"] for r in CUBE], ":D", color="k", lw=0.8, ms=2.2, label="cube: median dev.")
a.plot(x, [r["mom0_rmse_pct"] for r in rows], "-o", color="#7a3e9d", label="mom. 0: RMSE")
a.plot(x, [r["mom0_bias_pct"] for r in rows], ":s", color="#7a3e9d", lw=0.8, ms=2.4, label="mom. 0: median bias")
a.set_ylabel("compact − joint (%)"); a.set_ylim(top=a.get_ylim()[1] * 1.75)
a.text(0.03, 0.95, "(a) cube and moment 0" + ("" if TGT == "NGC3110" else f" — {LABEL}"), transform=a.transAxes,
       ha="left", va="top", fontweight="bold", fontsize=FT)
a.legend(loc="upper left", bbox_to_anchor=(0.012, 0.865), ncol=2, **LEG)
a = inch(ML, Y_B, W, HL); style(a, True)
a.axhline(5, color="0.45", lw=0.6, ls="--"); a.axhline(2.5, color="0.75", lw=0.5, ls="--")
for key, col, lab in (("mom1", "#2a6fdb", "mom. 1"), ("mom2", "#e07b00", "mom. 2")):
    a.plot(x, [r[f"{key}_rmse_kms"] for r in rows], "-o", color=col, label=f"{lab}: RMSE")
    a.plot(x, [r[f"{key}_rmse_w_kms"] for r in rows], "--o", color=col, lw=0.8, ms=2.2, mfc="none",
           label=f"{lab}: weighted RMSE")
    a.plot(x, [r[f"{key}_bias_kms"] for r in rows], ":s", color=col, lw=0.8, ms=2.2, label=f"{lab}: median bias")
a.set_ylabel(r"compact − joint (km s$^{-1}$)"); a.set_ylim(top=a.get_ylim()[1] * 2.05)
a.text(0.03, 0.95, "(b) moments 1 and 2", transform=a.transAxes, ha="left", va="top", fontweight="bold", fontsize=FT)
a.legend(loc="upper left", bbox_to_anchor=(0.012, 0.865), ncol=2, **LEG)
st = inch(ML, Y_S, W, HS); cw = W / nb                                  # the beams to scale, between (a) and (b)
st.set_xlim(-0.5, nb - 0.5); st.set_ylim(0, HS / cw); st.axis("off")
ang = lambda b: np.radians(90 + b[2]); ext_ = lambda b, c_, s_: np.hypot(b[0] / 2 * c_, b[1] / 2 * s_)
wide = max(2 * ext_(b, np.cos(ang(b)), np.sin(ang(b))) for b in BEAMS)
tall = max(2 * ext_(b, np.sin(ang(b)), np.cos(ang(b))) for b in BEAMS)
k_ = min(0.85 / wide, 0.92 * (HS / cw) / tall)
for i_, b in enumerate(BEAMS):
    named = i_ in (0, ILB, nb - 1)
    st.add_patch(Ellipse((i_, HS / cw / 2), width=b[0] * k_, height=b[1] * k_, angle=90 + b[2], facecolor="none",
                         edgecolor="k" if named else "0.55", lw=0.8 if named else 0.5))
fig.text((ML + W / 2) / FW, 0.03 / FH, "comparison beam FWHM (″)", ha="center", va="bottom", fontsize=FS)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(D, f"ladder_metrics_column.{ext}"), dpi=300 if ext == "png" else None)
print("saved", os.path.join(D, "ladder_metrics_column.png"), f"({FW * 25.4:.0f} x {FH * 25.4:.0f} mm)")
