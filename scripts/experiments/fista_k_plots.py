#!/usr/bin/env python
"""
FISTA (reweighted, noise-model lambda) at k = 3, 5, 7 on the GUI's synthetic disc and on the compact cube
(runs of scripts/experiments/toy_floor.py, data/experiments/16_floor/toy/).

    figures/experiments/16_floor/16c_fista_k_synthetic.png   truth, dirty, models / model - truth
    figures/experiments/16_floor/16c_fista_k_compact.png     line moment 0 / line-free channels
"""
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "gui"))
T = os.path.join(REPO, "data/experiments/16_floor/toy")
F = os.path.join(REPO, "figures/experiments/16_floor")
KS = (3, 5, 7)
need = [os.path.join(T, f"{d}_fista_noise_k{k}.json") for d in ("toy", "compact") for k in KS]
while not all(os.path.exists(p) for p in need):
    time.sleep(20)
asinh = lambda t: np.arcsinh(t / 0.1) / np.arcsinh(10)
plt.rcParams.update({"font.family": "serif", "font.size": 10})
ld = lambda n: (np.load(os.path.join(T, n + ".npy")).astype(np.float64), json.load(open(os.path.join(T, n + ".json"))))


def show(a, img, hi, title, div=False):
    if div:
        a.imshow(img, origin="lower", cmap="RdBu_r", vmin=-hi, vmax=hi)
    else:
        a.imshow(asinh(np.clip(img / hi, 0, 1)), origin="lower", cmap="inferno", vmin=0, vmax=1)
    a.set_title(title, fontsize=9); a.set_xticks([]); a.set_yticks([])


# ---- synthetic
from deconvgui.toy import make_toy  # noqa: E402
dirty, psf, truth, meta = make_toy()
dv = meta["width_kms"]; tm0 = truth.sum(0) * dv; hi = np.percentile(tm0, 99.5)
fig, ax = plt.subplots(2, 2 + len(KS), figsize=(2.7 * (2 + len(KS)), 5.8), constrained_layout=True)
show(ax[0, 0], tm0, hi, f"true sky\n{truth.sum():.0f} Jy")
dm = dirty.sum(0) * dv
ax[0, 1].imshow(asinh(np.clip((dm - np.percentile(dm, 0.5)) / np.ptp(np.percentile(dm, [0.5, 99.5])), 0, 1)), origin="lower", cmap="inferno")
ax[0, 1].set_title("dirty", fontsize=9); ax[0, 1].set_xticks([]); ax[0, 1].set_yticks([])
ax[1, 0].axis("off"); ax[1, 1].axis("off")
print("synthetic disc (truth 149 Jy):")
for c, k in enumerate(KS, start=2):
    x, s = ld(f"toy_fista_noise_k{k}")
    m0 = x.sum(0) * dv
    show(ax[0, c], m0, hi, f"FISTA, noise-model λ, k = {k}\n{s['flux']:.0f} Jy; outside truth {s['flux_out']:.1f} Jy")
    show(ax[1, c], m0 - tm0, 0.5 * hi, f"model − truth\nerror {s['err']:.2f} (mom0 {s['err_mom0']:.2f})", div=True)
    print(f"  k = {k}: flux {s['flux']:.1f} Jy (inside {s['flux_in']:.1f}, outside {s['flux_out']:.1f}); error {s['err']:.3f}, mom0 {s['err_mom0']:.3f}")
fig.suptitle("GUI synthetic disc: reweighted FISTA, noise-model thresholds, k = 3 / 5 / 7 (moment 0, asinh on the true sky's scale)", fontsize=11)
fig.savefig(os.path.join(F, "16c_fista_k_synthetic.png"), dpi=70); plt.close(fig)

# ---- compact
L = list(range(3, 24)); FREE = [0, 1, 2] + list(range(24, 32))
fig, ax = plt.subplots(2, len(KS), figsize=(3.4 * len(KS), 6.8), constrained_layout=True)
print("compact cube:")
ref = None
for c, k in enumerate(KS):
    x, s = ld(f"compact_fista_noise_k{k}")
    ml, mf = x[L].sum(0) * 25, x[FREE].sum(0) * 25
    if ref is None:
        ref = (np.percentile(ml, 99.5), np.percentile(mf, 99.5))
    show(ax[0, c], ml, ref[0], f"FISTA, noise-model λ, k = {k}\nline channels {x[L].sum():.1f} Jy; residual rms {s['resid_rms']:.1e}")
    show(ax[1, c], mf, ref[1], f"line-free channels (floor + continuum): {x[FREE].sum():.2f} Jy")
    print(f"  k = {k}: flux {s['flux']:.2f} Jy, line {x[L].sum():.2f}, line-free {x[FREE].sum():.3f}, residual rms {s['resid_rms']:.2e}")
fig.suptitle("NGC 3110 compact: reweighted FISTA, noise-model thresholds, k = 3 / 5 / 7\n"
             "top: line moment 0; bottom: line-free channels (asinh, each row on the k = 3 scale)", fontsize=10)
fig.savefig(os.path.join(F, "16c_fista_k_compact.png"), dpi=70); plt.close(fig)
print("wrote 16c_fista_k_synthetic.png, 16c_fista_k_compact.png")
