#!/usr/bin/env python
"""
Experiment 22: the 2D dictionary of the 2D-1D deconvolution, starlet vs the ImageNet-trained learnlet filters
(dictionaries/learnlet.npz; learnlet:16 = the 16 filters per scale with the largest share of the reconstruction),
each times the 1D CDF 9/7 along velocity. Same recipe for all (the pipeline's deconvolution, no major cycles):
NGC 3110 compact, 0.04" grid, pdtau, k = 5, 20,000 iterations. Runs (scripts/experiments/run_model.py):

    DICT_2D=<dict> run_model.py --method pdtau --galaxy ngc3110 --cfg compact --k 5 --n-iter 20000 \
        --out data/experiments/22_learnlet/ngc3110_compact_pdtau_k5_<tag>.npy

This script compares whichever have finished, plus the pipeline's original starlet model
(results/NGC3110/compact/deconvolved, 2026-10-01): line moment 0 of the raw models (one colour scale), line moment 0
of the residual d - N x, the integrated spectra, and a table (summary.md / summary.json).

    learnlet_compare.py     -> figures/experiments/22_learnlet/compare.png, data/experiments/22_learnlet/summary.{json,md}
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from common import G, problem, to_native  # noqa: E402

DAT = os.path.join(REPO, "data/experiments/22_learnlet")
FIG = os.path.join(REPO, "figures/experiments/22_learnlet")
os.makedirs(FIG, exist_ok=True)
DV, CELL, C10, EXT = 25.0, 0.04, np.s_[75:325, 75:325], [5, -5, -5, 5]
RUNS = [("starlet (pipeline, 2026-10-01)", os.path.join(REPO, "results/NGC3110/compact/deconvolved/model.npy")),
        ("starlet (re-run)", os.path.join(DAT, "ngc3110_compact_pdtau_k5_starlet.npy")),
        ("learnlet:16", os.path.join(DAT, "ngc3110_compact_pdtau_k5_learnlet16.npy")),
        ("learnlet (64)", os.path.join(DAT, "ngc3110_compact_pdtau_k5_learnlet.npy"))]
RUNS = [(lab, p) for lab, p in RUNS if os.path.exists(p)]

P = problem("ngc3110", "compact", k=5.0)
line = P.line
noise = [c for c in range(32) if c not in line]
d = P.op.d.cpu().numpy()
res, rows = {}, []
for lab, path in RUNS:
    x = to_native(P, np.load(path))
    r = P.op.residual_sky(G.to_t(x)).cpu().numpy()
    meta = json.load(open(path.replace("model.npy", "run.json") if path.endswith("model.npy") else path.replace(".npy", ".json")))
    res[lab] = dict(x=x, r=r)
    rows.append(dict(run=lab, flux=float(x.sum()), flux_line=float(x[line].sum()), flux_noise_channels=float(x[noise].sum()),
                     resid_rms_line=float(r[line].std()), resid_rms_noise=float(r[noise].std()),
                     nonzero_frac_line=float((x[line] > 0).mean()), seconds=meta.get("seconds")))
    if len(res) > 1:
        x0 = res[RUNS[0][0]]["x"]
        rows[-1]["rel_diff_vs_pipeline"] = float(np.linalg.norm(x - x0) / np.linalg.norm(x0))

json.dump(rows, open(os.path.join(DAT, "summary.json"), "w"), indent=1)
with open(os.path.join(DAT, "summary.md"), "w") as f:
    f.write("| run | flux (Jy) | line flux | line-free-channel flux | resid rms line | resid rms line-free | "
            "line voxels > 0 | ||x - x_pipeline|| / ||x_pipeline|| | time |\n|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r['run']} | {r['flux']:.2f} | {r['flux_line']:.2f} | {r['flux_noise_channels']:.3f} | "
                f"{r['resid_rms_line']:.3e} | {r['resid_rms_noise']:.3e} | {100 * r['nonzero_frac_line']:.1f}% | "
                f"{r.get('rel_diff_vs_pipeline', 0):.3g} | {(r['seconds'] or 0) / 3600:.1f} h |\n")
print(open(os.path.join(DAT, "summary.md")).read())

plt.rcParams.update({"font.family": "serif", "font.size": 12})
n = len(res)
fig = plt.figure(figsize=(4.2 * n + 1.2, 13))
gs = fig.add_gridspec(3, n + 1, width_ratios=[1] * n + [0.05], height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.08)
m0 = {lab: v["x"][line].sum(0)[C10] * DV / CELL ** 2 for lab, v in res.items()}
rm0 = {lab: v["r"][line].sum(0)[C10] * DV for lab, v in res.items()}
vm = max(np.percentile(m, 99.9) for m in m0.values())
rv = max(np.percentile(np.abs(m), 99.5) for m in rm0.values())
for i, lab in enumerate(res):
    ax = fig.add_subplot(gs[0, i])
    im = ax.imshow(m0[lab], origin="lower", extent=EXT, cmap="inferno", vmin=0, vmax=vm, interpolation="nearest")
    ax.set_title(lab); ax.set_xlabel(r'$\Delta$RA ["]')
    ax.text(0.97, 0.03, f"{rows[i]['flux_line']:.1f} Jy (line)", transform=ax.transAxes, color="w", ha="right")
    if i == 0:
        ax.set_ylabel("raw model, line moment 0\n" + r'$\Delta$Dec ["]')
    else:
        ax.set_yticklabels([])
    ax2 = fig.add_subplot(gs[1, i])
    im2 = ax2.imshow(rm0[lab], origin="lower", extent=EXT, cmap="RdBu_r", vmin=-rv, vmax=rv, interpolation="nearest")
    ax2.set_xlabel(r'$\Delta$RA ["]')
    ax2.text(0.97, 0.03, f"rms {rows[i]['resid_rms_line']:.2e} Jy/beam", transform=ax2.transAxes, ha="right", fontsize=10)
    if i == 0:
        ax2.set_ylabel("residual d - Nx, line moment 0\n" + r'$\Delta$Dec ["]')
    else:
        ax2.set_yticklabels([])
fig.colorbar(im, cax=fig.add_subplot(gs[0, n])).set_label(r"Jy km s$^{-1}$ arcsec$^{-2}$")
fig.colorbar(im2, cax=fig.add_subplot(gs[1, n])).set_label(r"Jy beam$^{-1}$ km s$^{-1}$")
ax = fig.add_subplot(gs[2, :n])
for lab, v in res.items():
    ax.plot(v["x"].sum((1, 2)), "o-", ms=3, label=f"model: {lab}")
ax.axvspan(line[0] - 0.5, line[-1] + 0.5, color="0.9", zorder=0, label="line channels")
ax.set_xlabel("channel"); ax.set_ylabel("model flux per channel (Jy)"); ax.legend(fontsize=9)
fig.suptitle("NGC 3110 compact, pdtau k = 5, 20,000 iterations: 2D dictionary x 1D CDF 9/7", fontsize=14)
out = os.path.join(FIG, "compare.png")
fig.savefig(out, dpi=110, bbox_inches="tight")
print("wrote", out)
