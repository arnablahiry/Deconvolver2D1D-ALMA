#!/usr/bin/env python
"""
Why 2D-1D, against the joint data: NGC 3110 deconvolved on 20, 15, 10, 5 and 2.6 km/s channels (0.16" pixels) with the
2D-1D dictionary and with the 2D starlet per channel (chanres_eval.json, chanres_eval.py).

  rows: the comparison beam (compact clean beam, lambda/B_max, 0.40", extended clean beam; the joint data's peak S/N per
  channel at that beam in the row label); columns: the RMSE of the cube, moment 0, moment 1 and moment 2, against the
  channel width; blue: 2D-1D, orange: 2D per channel. Widths whose reference at a beam is not yet made are left out.

    python scripts/compact_vs_msclean_dendro/fig_chanres.py
    -> results/NGC3110/figures/resolution_ladder_joint20/chanres_2d1d.{png,pdf}
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FD = os.path.join(REPO, "results/NGC3110/figures/resolution_ladder_joint20")
E = json.load(open(os.path.join(FD, "chanres_eval.json"))); DV = sorted(E, key=float, reverse=True)       # coarse to fine
BEAMS = [("b00", "compact clean beam"), ("b05", r"$\lambda/B_\mathrm{max}$ (0.60″)"), ("b07", "0.40″"), ("b10", "extended clean beam")]
BEAMS = [(b, t) for b, t in BEAMS if any(b in E[d] for d in DV)]
COLS = [("rmse", "cube (%)"), ("mom0", "moment 0 (%)"), ("mom1", r"moment 1 (km s$^{-1}$)"), ("mom2", r"moment 2 (km s$^{-1}$)")]
MOD = [("2D-1D", "#2a6fdb", "-", "o", "2D–1D"), ("2D per channel", "#e07b00", "--", "s", "2D starlet per channel")]

FS, FT, FL = 8, 7, 6.5
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": FS, "xtick.labelsize": FT,
                     "ytick.labelsize": FT, "xtick.direction": "in", "ytick.direction": "in", "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "legend.fontsize": FL, "lines.linewidth": 1.2, "lines.markersize": 3.2})
nr = len(BEAMS); FW = 184 / 25.4; ML, MR, MT, MB, GX, GY = 0.62, 0.06, 0.42, 0.4, 0.42, 0.14
PW = (FW - ML - MR - 3 * GX) / 4; PH = 0.82; FH = MB + nr * PH + (nr - 1) * GY + MT
fig = plt.figure(figsize=(FW, FH))
for r, (b, bl) in enumerate(BEAMS):
    xs = [i for i, d in enumerate(DV) if b in E[d]]
    snr = [E[DV[i]][b]["snr_peak_per_channel"] for i in xs]
    for c, (key, cl) in enumerate(COLS):
        a = fig.add_axes([(ML + c * (PW + GX)) / FW, (FH - MT - (r + 1) * PH - r * GY) / FH, PW / FW, PH / FH])
        y5, y0 = ([E[DV[i]][b][m][key] for i in xs] for m in ("2D-1D", "2D per channel"))
        a.fill_between(xs, y5, y0, color="#2a6fdb", alpha=0.1, lw=0)                 # the gap between the two
        for m, col, ls, mk, lab in MOD:
            a.plot(xs, [E[DV[i]][b][m][key] for i in xs], ls, marker=mk, color=col, mfc=col if m == "2D-1D" else "w", label=lab)
        a.set_xlim(-0.35, len(DV) - 0.65); a.set_xticks(range(len(DV)))
        a.set_xticklabels([f"{float(d):g}" for d in DV] if r == nr - 1 else [])
        a.set_ylim(0, a.get_ylim()[1] * 1.12); a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        a.tick_params(which="both", top=True, right=True, pad=2)
        if r == 0:
            a.set_title(cl, fontsize=FS, pad=4)
        if r == nr - 1:
            a.set_xlabel(r"channel width (km s$^{-1}$)", labelpad=2)
        if c == 0:
            fig.text((ML - 0.42) / FW, (FH - MT - (r + 0.5) * PH - r * GY) / FH,
                     f"{bl}\nS/N {max(snr):.0f}–{min(snr):.0f}", rotation=90, ha="center", va="center", fontsize=FL, linespacing=1.15)
fig.legend(*fig.axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
           fontsize=FL + 0.5, handlelength=2.4, columnspacing=2.0)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FD, f"chanres_2d1d.{ext}"), dpi=300 if ext == "png" else None)
print("saved", os.path.join(FD, "chanres_2d1d.png"), f"({FW * 25.4:.0f} x {FH * 25.4:.0f} mm)")
