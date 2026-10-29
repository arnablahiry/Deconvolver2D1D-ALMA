#!/usr/bin/env python
"""
Experiment 23 images (runs: iht_dict.py, scores: iht_score.py): line moment 0 of the compact ground-truth simulation
and of the models, all at the extended clean beam (G * x, the score's resolution, one colour scale), and of the error
G * (x - truth) (one symmetric scale). Columns: truth, the pipeline (starlet pdtau, k = 5), then per arm of
experiment 23 its member with the lowest error.

    iht_images.py  -> figures/experiments/23_iht/images.{png,pdf}
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import beam_ft, conv, ld  # noqa: E402
from common import EXP_DATA, REPO, BEAMS, line_channels  # noqa: E402

D = os.environ.get("EXP23_DIR", os.path.join(EXP_DATA, "23_iht"))      # override: testing
FIG = os.path.join(REPO, "figures/experiments/23_iht"); os.makedirs(FIG, exist_ok=True)
DV, HALF, C10 = 25.0, 5.0, np.s_[75:325, 75:325]
rows = json.load(open(os.path.join(D, "scores.json")))
LINE = line_channels("ngc3110")
BEAM = BEAMS[("ngc3110", "extended")]


def best(d, thr, val=None):
    rr = [r for r in rows if r["dict"] == d and r["thr"] == thr and (val is None or r["val"] == val)]
    return min(rr, key=lambda r: r["err"]) if rr else None


def path(r):
    if r["thr"] == "pdtau":
        return os.path.join(EXP_DATA, "01_simulations", "compact", f"pdtau_k{r['val']:g}.npy")
    return os.path.join(D, "sim", f"{r['dict']}_{r['thr']}_{r['par']}{r['val']:g}.npy")


cols = [("Truth", None, None),
        ("Pipeline", "soft, reweighted, k = 5", best("starlet", "pdtau", 5.0)),
        ("Starlet", "hard, k = {v:g}", best("starlet", "noise")),
        ("Learnlet filters", "hard, k = {v:g}", best("learnlet", "noise")),
        ("Learnlet network", "learnt thresholds × {v:g}", best("learnlet", "learnt")),
        ("Network, S = 0", "starlet, learnt thr. × {v:g}", best("starlet", "learnt"))]
cols = [c for c in cols if c[0] == "Truth" or c[2] is not None]
truth = ld(f"{EXP_DATA}/01_simulations/inputs/compact_sim_truth_0.04arcsec.npy")
kft = beam_ft(truth.shape[1:], BEAM, 0.04)
m0 = lambda x: (conv(x[LINE], kft).sum(0) * DV).cpu().numpy()[C10]
t0 = m0(truth)
imgs = [t0 if r is None else m0(ld(path(r))) for _, _, r in cols]
errs = [None if r is None else im - t0 for im, (_, _, r) in zip(imgs, cols)]
vmax = np.percentile(t0, 99.9); emax = max(np.percentile(np.abs(e), 99.5) for e in errs if e is not None)

plt.rcParams.update({"font.family": "serif", "font.size": 11})
n = len(cols); S = 2.6
fig, ax = plt.subplots(2, n, figsize=(S * n, 2 * S + 0.5), gridspec_kw=dict(wspace=0.03, hspace=0.03))
for i, ((name, sub, r), im, e) in enumerate(zip(cols, imgs, errs)):
    a = ax[0, i]
    hm = a.imshow(im, origin="lower", extent=[HALF, -HALF, -HALF, HALF], cmap="inferno",
                  norm=matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=vmax), interpolation="nearest")
    a.text(0.04, 0.96, name, transform=a.transAxes, color="w", ha="left", va="top", fontsize=11, fontweight="bold")
    if sub:
        a.text(0.04, 0.86, sub.format(v=r["val"]), transform=a.transAxes, color="w", ha="left", va="top", fontsize=8.5)
    pad = 0.06 * HALF; bx, by = HALF - pad - BEAM[0] / 2, -HALF + pad + BEAM[0] / 2
    a.add_patch(Ellipse((bx, by), width=BEAM[0], height=BEAM[1], angle=90 - BEAM[2], facecolor="w", edgecolor="w"))
    if i == 0:                                                    # scale bar, top right (RA increases left)
        xs, ys, cap = -HALF + 0.08 * HALF, HALF - 0.1 * HALF, 0.03 * HALF
        a.plot([xs + 2, xs], [ys, ys], color="w", lw=2.0, solid_capstyle="butt")
        for xe in (xs, xs + 2):
            a.plot([xe, xe], [ys - cap, ys + cap], color="w", lw=1.4, solid_capstyle="butt")
        a.annotate("2″", (xs + 1, ys - cap), xytext=(0, -2.5), textcoords="offset points", color="w", ha="center",
                   va="top", fontsize=10, fontweight="bold")
    b = ax[1, i]
    if e is None:
        b.axis("off")
        b.text(0.5, 0.5, "error =\n‖G ∗ (x − truth)‖ / ‖G ∗ truth‖\n\nG = extended clean beam\nmoment 0 over the line\nchannels, central 10″",
               transform=b.transAxes, ha="center", va="center", fontsize=9.5)
    else:
        em = b.imshow(e, origin="lower", extent=[HALF, -HALF, -HALF, HALF], cmap="RdBu_r", vmin=-emax, vmax=emax,
                      interpolation="nearest")
        b.text(0.04, 0.96, f"error {r['err']:.3f}", transform=b.transAxes, color="k", ha="left", va="top",
               fontsize=10, fontweight="bold")
    for x_ in (a, b):
        x_.set_xticks([]); x_.set_yticks([])
fig.colorbar(hm, ax=ax[0, :], orientation="vertical", fraction=0.012, pad=0.01,
             label="moment 0, Jy beam$^{-1}$ km s$^{-1}$")
fig.colorbar(em, ax=ax[1, :], orientation="vertical", fraction=0.012, pad=0.01, label="model − truth")
fig.suptitle("NGC 3110 compact ground-truth simulation at the extended clean beam: line moment 0 (top), error (bottom)",
             fontsize=12, y=0.93)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"images.{ext}"), dpi=150, bbox_inches="tight")
print("->", os.path.join(FIG, "images.png"))
