#!/usr/bin/env python
"""
60: the numbers behind figures 53-59 side by side: compact pdtau k = 5 without a target resolution, with one,
and the extended multiscale CLEAN. One panel per metric (each has its own units), one dot per model with its
value and, in brackets, its difference from the no-target model; the arrow in each title says which direction is better. Reads the JSONs written by
fig53_target_vs_none_extbeam.py (compact-data residuals) and fig56_compact_vs_extended_clean.py (everything on the
extended data), so run those first.

MASK: the mask the inputs were made with (fig53 / fig56 MASK=...); picks their _<mask name> JSONs.

    TAG=t0.1134 CLEAN=ms_clean [MASK=...] python scripts/superres_figures/fig60_metrics_summary.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAG = os.environ.get("TAG", "t0.1134")
CLEAN = os.environ.get("CLEAN", "ms_clean")
MASK = os.environ.get("MASK")
MT = f"_{os.path.basename(MASK).replace('.npy', '')}" if MASK else ""
SUFFIX = (TAG if CLEAN == "ms_clean" else f"{TAG}_{CLEAN}") + MT
D = os.path.join(REPO, "data/experiments/11_target_resolution")
OUT = os.environ.get("OUT", os.path.join(REPO, f"figures/experiments/11_target_resolution/60_metrics_summary_{SUFFIX}.png"))
BLUE, ORANGE, GREEN, MUTED = "#2a6fdb", "#e07b00", "#1a9e77", "#8a8a8a"

cmp = json.load(open(os.path.join(D, f"compare_extbeam_{TAG}{MT}.json")))      # compact data (fig53)
ext = json.load(open(os.path.join(D, f"vs_clean_{SUFFIX}.json")))          # extended data (fig56)
tr = ext["target_res_arcsec"]
NT, TG, CL = "no target", f"target {tr:g}″", "extended CLEAN"
cn = {NT: cmp["no target"], TG: cmp[f"target {tr:g}″"]}
en = {NT: ext["compact, no target"], TG: ext[f"compact, target {tr:g}″"], CL: ext["extended CLEAN"]}
COL = {NT: BLUE, TG: ORANGE, CL: GREEN}
pk = ext["peaks"][0]["peak_mJy_beam"]                                       # the brightest CLEAN knot

# (title, unit, better: "low" | "high" | None, {model: value})
metrics = [
    ("line flux (model)", "Jy km/s", None,
     {NT: en[NT]["flux_line_Jy_kms"], TG: en[TG]["flux_line_Jy_kms"], CL: en[CL]["flux_line_Jy_kms"]}),
    ("COMPACT data: residual rms,\nline channels, whole field", "mJy/beam", "low",
     {NT: cn[NT]["resid_rms_line_mJy_beam"], TG: cn[TG]["resid_rms_line_mJy_beam"]}),
    ("COMPACT data: residual rms,\nline channels, central 10″", "mJy/beam", "low",
     {NT: cn[NT]["resid_rms_line_central10_mJy_beam"], TG: cn[TG]["resid_rms_line_central10_mJy_beam"]}),
    ("COMPACT data: residual mean,\nline channels", "mJy/beam", "zero",
     {NT: cn[NT]["resid_mean_line_mJy_beam"], TG: cn[TG]["resid_mean_line_mJy_beam"]}),
    ("brightest knot, one ext. beam\n" + f"({ext['peaks'][0]['offset_arcsec'][0]:+.2f}″, {ext['peaks'][0]['offset_arcsec'][1]:+.2f}″)",
     "mJy/beam", None,
     {NT: pk["compact, no target"], TG: pk[f"compact, target {tr:g}″"], CL: pk["extended CLEAN"]}),
    ("EXTENDED data: residual rms,\nline channels, whole field", "mJy/beam", "low",
     {m: en[m]["resid_rms_line_mJy_beam"] for m in en}),
    ("EXTENDED data: residual rms,\nline channels, central 10″", "mJy/beam", "low",
     {m: en[m]["resid_rms_line_central10_mJy_beam"] for m in en}),
    ("EXTENDED data: residual rms,\ninside CLEAN's mask", "mJy/beam", "low",
     {m: en[m]["resid_rms_line_in_clean_mask_mJy_beam"] for m in en}),
    ("moment 0 vs CLEAN components:\ncorrelation, central 10″", "Pearson r", "high",
     {NT: en[NT]["mom0_vs_clean_components"]["pearson_central10"], TG: en[TG]["mom0_vs_clean_components"]["pearson_central10"]}),
    ("moment 0 vs CLEAN comps:\nrel. rms diff., central 10″", "‖a − b‖ / ‖b‖", "low",
     {NT: en[NT]["mom0_vs_clean_components"]["rel_rms_diff_central10"], TG: en[TG]["mom0_vs_clean_components"]["rel_rms_diff_central10"]}),
]
ARROW = {"low": "  (↓ better)", "high": "  (↑ better)", "zero": "  (→ 0 better)", None: ""}
FMT = lambda x: f"{x:.0f}" if abs(x) >= 100 else (f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}")

plt.rcParams.update({"font.family": "serif", "font.size": 12, "xtick.direction": "in", "ytick.direction": "in"})
fig, axes = plt.subplots(2, 5, figsize=(25, 8.6), constrained_layout=True)
order = [NT, TG, CL]
for a, (title, unit, better, vals) in zip(axes.ravel(), metrics):
    rows = [m for m in order if m in vals]
    x = np.array([vals[m] for m in rows])
    lo, hi = min(x.min(), 0) if better == "zero" else x.min(), max(x.max(), 0) if better == "zero" else x.max()
    pad = 0.6 * (hi - lo if hi > lo else abs(hi) or 1)
    for y, m in enumerate(rows):
        a.hlines(y, lo - pad, vals[m], color=COL[m], lw=1, alpha=0.35)
        a.plot(vals[m], y, "o", ms=11, color=COL[m], mec="white", mew=2, zorder=3)
        rel = "" if m == NT or vals[NT] == 0 else f"  ({100 * (vals[m] / vals[NT] - 1):+.1f}%)"
        a.annotate(FMT(vals[m]) + rel, (vals[m], y), xytext=(0, 11), textcoords="offset points", ha="center",
                   fontsize=12, color="0.15")
    if better == "zero":
        a.axvline(0, color=MUTED, lw=1)
    a.set_xlim(lo - pad, hi + pad); a.set_ylim(len(rows) - 0.4, -0.6)
    a.set_yticks(range(len(rows))); a.set_yticklabels(rows)
    a.set_title(title + ARROW[better], fontsize=12); a.set_xlabel(unit)
    a.grid(axis="x", alpha=0.25); a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.suptitle(f"NGC 3110: compact pdtau k = 5, no target vs target {tr:g}″, and the extended multiscale CLEAN ({CLEAN}) "
             + (", all models × the dendrogram mask (moment-0 metrics over its footprint)" if MASK else "") + "\nall at the extended clean beam; axes are zoomed (not zero-based): read the % against no target", fontsize=14)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=70)
print("saved", OUT)
