#!/usr/bin/env python
"""
Re-render the compact reweighted-FISTA evolution in the layout of scripts/experiments/toy_gif.py
(dirty | model | residual, line moment 0, central 10" | convergence) from an existing frame cache
(make_fista_run.py: it, model, resid [central 10", 0.04"], flux, trace), without re-solving.

    compact_gif_from_cache.py [--k 5]   (cache data/ngc3110/compact/fista/frames_k<k>_20k.npz)
    -> figures/experiments/16_floor/compact_fista_k<k>.gif (+ frames/compact_fista_k<k>/)
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=5); a = ap.parse_args()
c = np.load(os.path.join(REPO, f"data/ngc3110/compact/fista/frames_k{a.k}_20k.npz"))
name = f"compact_fista_k{a.k}"
OUT = os.path.join(REPO, "figures/experiments/16_floor"); FR = os.path.join(OUT, "frames", name)
os.makedirs(FR, exist_ok=True)
for f in glob.glob(os.path.join(FR, "*.png")):
    os.remove(f)

vel = 5073.9 + 25.0 * (np.arange(32) - 15.5)
L = [i for i, v in enumerate(vel) if 4750 <= v <= 5280]
d = np.nan_to_num(fits.getdata(os.path.join(REPO, "data/ngc3110/compact/cube/dirty_cube.fits")).astype(np.float64)).squeeze()
d = d.reshape(32, 400, 2, 400, 2).mean((2, 4))                 # the 0.04" grid (block mean, as the solve)
dm0 = d[L].sum(0)[75:325, 75:325] * 25                          # central 10", Jy/beam km/s
lo_d, hi_d = np.percentile(dm0, [0.5, 99.5])
it, model, resid, flux = c["it"], c["model"], c["resid"], c["flux"]
ti, tc = c["trace"].T
hi = np.percentile(model[-1], 99.5); vr = 5 * resid[-1].std()
asinh = lambda q: np.arcsinh(q / 0.1) / np.arcsinh(10)
ext = [5, -5, -5, 5]
plt.rcParams.update({"font.family": "serif", "font.size": 11})
paths = []
for i in range(len(it)):
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.4), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.15]), constrained_layout=True)
    ax[0].imshow(asinh(np.clip((dm0 - lo_d) / (hi_d - lo_d), 0, 1)), origin="lower", cmap="inferno", extent=ext)
    ax[0].set_title("dirty, line moment 0")
    ax[1].imshow(asinh(np.clip(model[i] / hi, 0, 1)), origin="lower", cmap="inferno", extent=ext)
    ax[1].set_title(f"model, line moment 0 (model flux {flux[i]:.1f} Jy)", fontsize=10)
    ax[2].imshow(resid[i], origin="lower", cmap="RdBu_r", vmin=-vr, vmax=vr, extent=ext)
    ax[2].set_title("residual d − N x, line moment 0", fontsize=10)
    for q in ax[:3]:
        q.set_xlabel(r'$\Delta$RA ["]'); q.set_ylabel(r'$\Delta$Dec ["]')
    a2 = ax[3]; a2b = a2.twinx()
    a2.loglog(ti, tc, color="0.8", lw=1); done = ti <= it[i]
    a2.loglog(ti[done], tc[done], color="#2a78d6", lw=1.6)
    a2b.semilogx(it, flux, color="0.85", lw=1, ls="--"); a2b.semilogx(it[: i + 1], flux[: i + 1], color="#eb6834", lw=1.4)
    for r in range(6000, 20000, 2000):
        a2.axvline(r, color="0.9", lw=0.6, zorder=0)
    a2.axvline(it[i], color="k", lw=1)
    a2.set_xlabel("iteration"); a2.set_ylabel("relative change per iteration", color="#2a78d6")
    a2b.set_ylabel("model flux (Jy)", color="#eb6834"); a2.set_xlim(20, 20000)
    fig.suptitle(f"NGC 3110 compact (0.04″, central 10″): reweighted FISTA, noise-model thresholds, k = {a.k}"
                 f"  --  iteration {int(it[i]):,}", fontsize=13)
    p = os.path.join(FR, f"frame_{i:03d}.png"); fig.savefig(p, dpi=65); plt.close(fig); paths.append(p)
imgs = [Image.open(p).convert("RGB") for p in paths]
durs = [110] * len(imgs); durs[-1] = 2500
gif = os.path.join(OUT, f"{name}.gif")
imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=durs, loop=0, optimize=True)
print("wrote", gif)
