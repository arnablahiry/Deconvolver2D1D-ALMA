#!/usr/bin/env python
"""
Figure 4 of the compact-vs-extended consistency set (NGC 3110): every test in one panel, compact relative to extended
at the compact-beam resolution. Left: line flux difference (%); right: RMSE (%) with what noise alone would give.
Numbers read from the other scripts' JSON (results/NGC3110/figures/{consistency,cross_convolution,clean_bigfield}):

  data, no imaging     raw visibilities, both arrays against one model, baselines < 141 klambda (raw_visibility_compare)
  data, model-free     cross-convolved dirty cubes, 82" field (cross_convolution: E1 at 1.3", E3 at 2.8")
  data, forward model  compact MS-CLEAN model * tapered extended dirty beam vs the tapered extended dirty cube
  images (CLEAN)       compact vs tapered extended MS-CLEAN, restored and models only (fig_consistency_images,
                       compare_clean_bigfield)
The grey band is +-5%, the absolute flux calibration agreement expected between two ALMA observations.

    python scripts/experiments/fig_consistency_summary.py   -> results/NGC3110/figures/consistency/fig4_summary.{png,pdf}
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(REPO, "results/NGC3110/figures")
OUT = os.path.join(FIG, "consistency")
plt.rcParams.update({"font.family": "serif", "font.size": 13, "xtick.direction": "in", "ytick.direction": "in"})
vis = json.load(open(os.path.join(OUT, "fig1_visibilities.json")))["stats"]["compact_beam_band"]
xc = json.load(open(os.path.join(FIG, "cross_convolution/cross_convolution.json")))
img = json.load(open(os.path.join(OUT, "images.json")))
cmp_ = json.load(open(os.path.join(FIG, "clean_bigfield/compare.json")))["pairs"]

e1 = [m for m in xc["E1"] if m["K"] == 5.0][0]; e3 = xc["E3"][-1]
r = vis["extended_over_compact"]; rerr = vis["extended_over_compact_err"]
mod = cmp_["C vs T | models | T mask"]
ROWS = [  # label, group, flux %, flux err, RMSE %, noise RMSE %
    (f"visibilities, < {json.load(open(os.path.join(OUT, 'fig1_visibilities.json')))['stats']['compact_beam_klambda']:.0f} kλ "
     f"(χ²/dof {vis['compact']['chi2']:.1f} / {vis['extended']['chi2']:.1f})", "data", 100 * (1 / r - 1), 100 * rerr / r ** 2, None, None),
    (f"cross-convolved dirty cubes, {xc['effective_beam_fwhm_arcsec']:.1f}″", "data", e1["flux_pct"], None, e1["rmse_pct"], e1["noise_rmse_pct"]),
    (f"  … smoothed to {e3['total_fwhm']:.1f}″ (after the {e3['scale']:.2f} scale)", "data", e3["flux_pct"], None, e3["rmse_scaled_pct"], e3["noise_rmse_pct"]),
    ("forward model: compact model ∗ extended dirty beam vs extended dirty data", "data", img["forward_vs_data"]["flux_pct"], None,
     img["forward_vs_data"]["rmse_pct"], img["forward_vs_data"]["noise_rmse_pct"]),
    ("CLEAN images: compact vs tapered extended (restored)", "images", img["clean_vs_clean"]["flux_pct"], None,
     img["clean_vs_clean"]["rmse_pct"], img["clean_vs_clean"]["noise_rmse_pct"]),
    ("CLEAN models only (× the compact beam)", "images", mod["flux_pct"], None, mod["rmse_pct"], mod["noise_rmse_pct"]),
]
COL = {"data": "#1a9e77", "images": "#c0392b"}
fig, ax = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True, gridspec_kw=dict(wspace=0.05, width_ratios=[1, 1]))
y = np.arange(len(ROWS))[::-1]
for yi, (lab, grp, fl, fe, rm, nr) in zip(y, ROWS):
    ax[0].barh(yi, fl, color=COL[grp], alpha=0.85, height=0.6)
    if fe:
        ax[0].errorbar(fl, yi, xerr=fe, color="k", capsize=3)
    ax[0].text(fl + (1 if fl >= 0 else -1), yi, f"{fl:+.1f}%", va="center", ha="left" if fl >= 0 else "right", fontsize=11)
    if rm is not None:
        ax[1].barh(yi, rm, color=COL[grp], alpha=0.85, height=0.6)
        ax[1].plot([nr, nr], [yi - 0.35, yi + 0.35], color="k", lw=2.5)
        ax[1].text(rm + 0.6, yi, f"{rm:.1f}%  (noise {nr:.1f}%)", va="center", fontsize=11)
    else:
        ax[1].text(0.6, yi, "χ²/dof ≈ 1 against one common model: agree to noise", va="center", fontsize=11, color="0.3")
ax[0].axvspan(-5, 5, color="0.88", zorder=0); ax[0].axvline(0, color="0.4", lw=1)
ax[0].set_yticks(y); ax[0].set_yticklabels([r[0] for r in ROWS], fontsize=11.5)
ax[0].set_xlim(-30, 42); ax[0].set_xlabel("line flux, compact − extended (%)", fontsize=14)
ax[1].set_xlim(0, 40); ax[1].set_xlabel("RMSE (%); black tick: noise alone", fontsize=14)
for a in ax:
    a.tick_params(labelsize=12)
ax[0].plot([], [], "s", color=COL["data"], ms=10, label="the data (no CLEAN of the extended data)")
ax[0].plot([], [], "s", color=COL["images"], ms=10, label="CLEAN images of each array")
ax[0].plot([], [], "s", color="0.88", ms=10, label="±5%: flux calibration")
fig.legend(*ax[0].get_legend_handles_labels(), fontsize=11.5, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("NGC 3110 CO(2−1) at the compact-beam resolution: compact vs extended", fontsize=15, y=0.99)
fig.savefig(os.path.join(OUT, "fig4_summary.png"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig4_summary.pdf"), dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved", os.path.join(OUT, "fig4_summary.png"))
