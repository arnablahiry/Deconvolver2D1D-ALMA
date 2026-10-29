#!/usr/bin/env python
"""
Why 2D-1D? NGC 3110: the 2D-1D model (starlet x CDF 9/7 along velocity) against the same solver with the 2D starlet
alone in every channel (L1D = 0), on the resolution ladder (ablation_eval.json): at the data's own S/N and, where they
exist, with noise added to the dirty cube (NOISE_X = 3, 10: PSF-filtered, independent per channel).

  (a) cube RMSE   (b) moment-1 RMSE   (c) moment-2 RMSE, against the comparison beam;
  solid: 2D-1D, dashed: 2D per channel; colour: the noise level

    python scripts/compact_vs_msclean_dendro/fig_ablation_2d.py
    -> results/NGC3110/figures/resolution_ladder_joint20/ablation_2d.{png,pdf}
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "results/NGC3110/figures/resolution_ladder_joint20")
E = json.load(open(os.path.join(OUT, "ablation_eval.json"))); LB = json.load(open(os.path.join(OUT, "resolution_ladder.json")))["lbmax_compact"]
rows = E["rows"]; nb = len(rows); geo = [np.sqrt(r["beam"][0] * r["beam"][1]) for r in rows]
ILB = int(np.argmin([abs(r["beam"][0] - LB) + abs(r["beam"][1] - LB) for r in rows]))
LEVELS = [("", "data as observed", "#2a6fdb"), ("_noise3", "noise × 3", "#e07b00"), ("_noise10", "noise × 10", "#b2182b")]
have = [(s, lab, c) for s, lab, c in LEVELS if f"support{s}" in rows[0] and f"support_L0{s}" in rows[0]]

FS, FT, FL = 8, 7, 6.5
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": FS, "axes.labelsize": FS,
                     "xtick.labelsize": FT, "ytick.labelsize": FT, "xtick.direction": "in", "ytick.direction": "in",
                     "axes.linewidth": 0.6, "legend.fontsize": FL, "lines.linewidth": 1.2, "lines.markersize": 2.8})
fig, ax = plt.subplots(1, 3, figsize=(184 / 25.4, 2.5), gridspec_kw=dict(wspace=0.3, left=0.06, right=0.99, bottom=0.27, top=0.9))
x = np.arange(nb)
for a, (key, ylab, t) in zip(ax, (("rmse", "cube RMSE (%)", "(a) cube"), ("mom1", r"moment-1 RMSE (km s$^{-1}$)", "(b) velocity"),
                                  ("mom2", r"moment-2 RMSE (km s$^{-1}$)", "(c) dispersion"))):
    for s, lab, c in have:
        a.plot(x, [r[f"support{s}"][key] for r in rows], "-o", color=c, label=f"2D–1D, {lab}")
        a.plot(x, [r[f"support_L0{s}"][key] for r in rows], "--s", color=c, mfc="w", label=f"2D per channel, {lab}")
    a.axvline(ILB, color="0.5", lw=0.6, ls=":"); a.set_xlim(-0.5, nb - 0.5); a.set_ylim(bottom=0)
    a.set_xticks(x); a.set_xticklabels([f"{g:.2f}" for g in geo], rotation=60, ha="right", rotation_mode="anchor", fontsize=FL - 0.5)
    for i_ in (0, ILB, nb - 1):
        a.get_xticklabels()[i_].set_fontweight("bold")
    a.tick_params(which="both", top=True, right=True); a.set_ylabel(ylab, labelpad=2); a.set_xlabel("comparison beam FWHM (″)", labelpad=2)
    a.text(0.04, 0.95, t, transform=a.transAxes, ha="left", va="top", fontweight="bold", fontsize=FT)
ax[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.86), frameon=True, framealpha=0.95, edgecolor="0.6", fancybox=False,
             fontsize=FL - 0.5, handlelength=2.2)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"ablation_2d.{ext}"), dpi=300 if ext == "png" else None)
print("saved", os.path.join(OUT, "ablation_2d.png"), [h[1] for h in have])
