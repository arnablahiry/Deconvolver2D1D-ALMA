#!/usr/bin/env python
"""
High-resolution evolution GIF of the raw compact models, reweighted FISTA (left) and pdtau (right; primal-dual
with reweighting reference tau*lambda), one row per threshold k, noise-model lambda, 20,000 iterations,
line moment 0, central 10", all panels on one colour scale.

FISTA frames come from data/ngc3110/compact/fista/frames_k<k>_20k.npz (make_fista_run.py). pdtau is solved
here with a frame at the same iterations, cached in data/experiments/04_pd_tau_reweighting/compact/pdtau_k<k>_frames.npz
(reused when present).

    side_by_side_hires.py --ks 3 5   -> figures/experiments/16_floor/fista_vs_pdtau_k3_k5_hires.gif
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser(); ap.add_argument("--ks", type=int, nargs="+", default=[5]); a = ap.parse_args()
CELL, C10 = 0.04, np.s_[75:325, 75:325]

fista, pdtau = {}, {}
for k in a.ks:
    fista[k] = np.load(os.path.join(REPO, f"data/ngc3110/compact/fista/frames_k{k}_20k.npz"))
its = fista[a.ks[0]]["it"].astype(int)
assert all(np.array_equal(fista[k]["it"], its) for k in a.ks), "FISTA frame iterations differ between k"
for k in a.ks:
    cache = os.path.join(REPO, f"data/experiments/04_pd_tau_reweighting/compact/pdtau_k{k}_frames.npz")
    if not os.path.exists(cache):
        from common import G, problem, RW
        P = problem("ngc3110", "compact", k=float(k))
        marks, frames = set(its.tolist()), []

        def cb(it, x, g):
            if it in marks:
                frames.append((it, (x[P.line].sum(0) * 25).cpu().numpy()[C10] / CELL ** 2, float(x.sum())))

        G.solve_pd(P.op, P.D, P.lam, n_iter=20000, beta=P.beta, reweight=dict(RW, ref=P.tau * P.lam), print_every=4000,
                   log=lambda m: print(f"[pdtau k={k}] {m}", flush=True), callback=cb)
        np.savez_compressed(cache, it=np.array([f[0] for f in frames]), model=np.array([f[1] for f in frames], dtype=np.float32),
                            flux=np.array([f[2] for f in frames]))
    pdtau[k] = np.load(cache)
    assert np.array_equal(pdtau[k]["it"], its), "pdtau frame iterations differ"

name = "fista_vs_pdtau_" + "_".join(f"k{k}" for k in a.ks) + "_hires"
OUT = os.path.join(REPO, "figures/experiments/16_floor"); FR = os.path.join(OUT, "frames", name)
os.makedirs(FR, exist_ok=True)
for f in glob.glob(os.path.join(FR, "*.png")):
    os.remove(f)
vm = max(np.percentile(d[k]["model"][-1], 99.9) for d in (fista, pdtau) for k in a.ks)      # one common scale
ext = [5, -5, -5, 5]
nr = len(a.ks)
plt.rcParams.update({"font.family": "serif", "font.size": 15})
paths = []
for i, it in enumerate(its):
    fig = plt.figure(figsize=(17.6, 1.0 + 7.6 * nr))
    H = 1.0 / (1.0 + 7.6 * nr)                                    # one inch, in figure fractions
    top = 1 - 0.95 * H
    cax = fig.add_axes([0.925, H * 0.6, 0.015, top - 0.6 * H - 0.25 * H])
    for r, k in enumerate(a.ks):
        y0 = top - (r + 1) * 7.6 * H + 0.55 * H
        axes = [fig.add_axes([0.05, y0, 0.41, 6.9 * H]), fig.add_axes([0.5, y0, 0.41, 6.9 * H])]
        for ax, (lab, d) in zip(axes, (("reweighted FISTA", fista[k]), ("pdtau (primal-dual)", pdtau[k]))):
            im = ax.imshow(d["model"][i], origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=vm, interpolation="bicubic")
            ax.set_xlabel(r'$\Delta$RA ["]'); ax.set_ylabel(r'$\Delta$Dec ["]')
            ax.text(0.03, 0.97, lab, transform=ax.transAxes, color="w", fontsize=20, fontweight="bold", va="top")
            ax.text(0.03, 0.90, f"k = {k}, noise-model λ" + (", reweighted" if it > 6000 else ""), transform=ax.transAxes,
                    color="w", fontsize=14, fontweight="bold", va="top")
            ax.text(0.97, 0.03, f"model flux {d['flux'][i]:.1f} Jy", transform=ax.transAxes, color="w", fontsize=14,
                    fontweight="bold", ha="right")
    cb = fig.colorbar(im, cax=cax); cb.set_label(r"Jy km s$^{-1}$ arcsec$^{-2}$")
    fig.suptitle(f"NGC 3110 compact: raw 2D-1D models (line moment 0, no convolution)  --  iteration {int(it):,}",
                 fontsize=19, y=1 - 0.3 * H)
    p = os.path.join(FR, f"frame_{i:03d}.png"); fig.savefig(p, dpi=80); plt.close(fig); paths.append(p)
imgs = [Image.open(p).convert("RGB") for p in paths]
durs = [120] * len(imgs); durs[-1] = 2500
gif = os.path.join(OUT, f"{name}.gif")
imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=durs, loop=0, optimize=True)
print("wrote", gif, flush=True)
