#!/usr/bin/env python
"""
Side-by-side GIF of the compact moment-0 model, FISTA (left) vs primal-dual (right),
at the same iterations up to 20,000. Both solve the SAME problem -- same data
(0.04" grid), operator, per-sub-band lambda (k = 2), step 1/beta and reweighting
schedule (every 2000 from 6000) -- so the only difference is the solver:

  FISTA (the loop of simple/deconvolve.py):  x = max(0, W^-1 soft(W(z - tau grad), tau lambda)),
                                            z = x + momentum
  primal-dual (simple/gpu_pd.py):            see that module

The primal-dual frames come from scripts/make_pd_evolution_gif.py's cache
(data/ngc3110/compact/pd/pd_evolution_frames.npz); run that first.

    CUDA_VISIBLE_DEVICES=0 /scratch/alahiry/conda/envs/denoise2d1d/bin/python scripts/make_fista_vs_pd_gif.py
Writes figures/fista/fista_vs_pd.gif (+ frames in figures/fista/frames/fista_vs_pd_NNN.png) and
data/ngc3110/compact/pd/fista_evolution_frames.npz.
"""
import os, sys, time, glob
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from astropy.io import fits
from PIL import Image
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "simple"))
import gpu_pd as G

OUT = os.path.join(REPO, "figures/fista"); FRAMES = os.path.join(OUT, "frames")
CACHE = os.path.join(REPO, "data/ngc3110/compact/pd")
N_ITER, K = 20000, 2.0
REWEIGHT = dict(start=6000, every=2000, eps=1.0)
vel = 5073.9 + 25.0 * (np.arange(32) - 15.5)
LINE = [i for i, v in enumerate(vel) if 4750 <= v <= 5280]
NOISE = [i for i in range(32) if i not in LINE]
CELL = 0.04
C10 = np.s_[75:325, 75:325]; EXT = [5, -5, -5, 5]

pd = np.load(os.path.join(CACHE, "pd_evolution_frames.npz"))
pd_it, pd_m, pd_trace = pd["it"], pd["model"], pd["trace"]
marks = set(int(i) for i in pd_it)

read = lambda f: np.nan_to_num(fits.getdata(os.path.join(REPO, "data/ngc3110/compact/cube", f)).astype(np.float64)).squeeze()
dirty, psf = G.rebin(read("dirty_cube.fits"), read("dirty_beam_cube.fits"), 2)
op = G.Op(psf, dirty); D = G.Dict2D1D(dirty.shape, 7, 5)
lam, _, _, _ = G.noise_lambdas(D, op.d, NOISE, k=K)
tau = 1.0 / G.field_lipschitz(op)
T0 = tau * lam                                            # thresholds, in the units of the coefficients of v
soft = lambda c, t: torch.sign(c) * torch.clamp(c.abs() - t, min=0)

# ---- FISTA, exactly as in simple/deconvolve.py but with the primal-dual problem's lambda / step / reweighting
x = torch.zeros_like(op.d); z = x.clone(); t = 1.0; T = T0
frames, trace = {}, []
t0 = time.time()
for it in range(1, N_ITER + 1):
    if it > REWEIGHT["start"] and (it - 1 - REWEIGHT["start"]) % REWEIGHT["every"] == 0:
        T = T0 / (D.W(x).abs() / T0 + REWEIGHT["eps"])
    xn = D.Winv(soft(D.W(z - tau * op.grad(z)), T)).clamp_(min=0)
    tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
    z = xn + ((t - 1) / tn) * (xn - x)
    if it % 25 == 0:
        trace.append((it, float((xn - x).norm() / max(float(xn.norm()), 1e-30))))
    x, t = xn, tn
    if it in marks:
        frames[it] = ((x[LINE].sum(0) * 25).cpu().numpy() / CELL ** 2, float(x.sum()))
    if it % 4000 == 0:
        print(f"FISTA iter {it:6d}  flux {float(x.sum()):8.3f}  change {trace[-1][1]:.2e}  ({time.time() - t0:.0f} s)", flush=True)
print(f"FISTA: {N_ITER} iterations in {time.time() - t0:.0f} s")
np.savez_compressed(os.path.join(CACHE, "fista_evolution_frames.npz"), it=np.array(sorted(frames)),
                    model=np.array([frames[i][0] for i in sorted(frames)], dtype=np.float32),
                    flux=np.array([frames[i][1] for i in sorted(frames)]), trace=np.array(trace))

# ---- render
os.makedirs(FRAMES, exist_ok=True)
for f in glob.glob(os.path.join(FRAMES, "fista_vs_pd_*.png")):
    os.remove(f)
ft, fc = np.array(trace).T; pt, pc = pd_trace.T
change_at = lambda ti, tc, it: tc[np.searchsorted(ti, it)] if it >= ti[0] else np.nan
vm = np.percentile(pd_m[-1][C10], 99.9)
plt.rcParams.update({"font.family": "serif", "font.size": 15})
paths = []
for k, it in enumerate(pd_it):
    it = int(it)
    fig = plt.figure(figsize=(16.6, 8.4))
    axs = [fig.add_axes([0.06, 0.1, 0.4, 0.78]), fig.add_axes([0.5, 0.1, 0.4, 0.78])]
    cax = fig.add_axes([0.915, 0.1, 0.018, 0.78])
    for a, img, name, ti, tc in ((axs[0], frames[it][0], "FISTA (soft thresholding)", ft, fc),
                                 (axs[1], pd_m[k], "Primal-dual (this method)", pt, pc)):
        im = a.imshow(img[C10], origin="lower", extent=EXT, cmap="inferno", vmin=0, vmax=vm, interpolation="bicubic")
        for sp in a.spines.values(): sp.set_visible(True)
        a.set_xlabel(r'$\Delta$RA ["]')
        a.text(0.03, 0.97, name, transform=a.transAxes, color="w", fontsize=18, fontweight="bold", va="top")
        ch = change_at(ti, tc, it)
        a.text(0.03, 0.9, "change per iteration: " + ("--" if np.isnan(ch) else f"{ch:.1e}"),
               transform=a.transAxes, color="w", fontsize=14, fontweight="bold", va="top")
    axs[0].set_ylabel(r'$\Delta$Dec ["]'); plt.setp(axs[1].get_yticklabels(), visible=False)
    cb = fig.colorbar(im, cax=cax); cb.set_label(r"Jy km s$^{-1}$ arcsec$^{-2}$")
    fig.suptitle(f"NGC 3110 compact, raw model moment 0 -- same problem, two solvers -- iteration {it:,}", fontsize=18, y=0.965)
    p = os.path.join(FRAMES, f"fista_vs_pd_{k:03d}.png"); fig.savefig(p, dpi=90); plt.close(fig); paths.append(p)
imgs = [Image.open(p).convert("RGB") for p in paths]
durs = [120] * len(imgs); durs[-1] = 3000
imgs[0].save(os.path.join(OUT, "fista_vs_pd.gif"), save_all=True, append_images=imgs[1:], duration=durs, loop=0, optimize=True)
print("wrote", os.path.join(OUT, "fista_vs_pd.gif"))
