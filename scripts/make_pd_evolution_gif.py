#!/usr/bin/env python
"""
Runs the compact 2D-1D primal-dual deconvolution (simple/gpu_pd.py, the recipe of
notebooks/pd_deconvolution_explained.ipynb): k = 2, reweighting (eps = 1) every
2000 iterations from 6000, 20,000 iterations -- and saves a frame of the model
at log-spaced iterations, then two GIFs:

    figures/pd/pd_evolution.gif               model | residual | convergence
    figures/pd/pd_evolution_model_hires.gif   model panel alone, high resolution
    figures/pd/frames/                        the PNG frames of both GIFs
    data/ngc3110/compact/pd/model_20k_iterations.npy     final model, Jy/pixel, 0.02" (git-ignored)
    data/ngc3110/compact/pd/pd_evolution_frames.npz      frame cache, to re-render without re-running

    CUDA_VISIBLE_DEVICES=0 /scratch/alahiry/conda/envs/denoise2d1d/bin/python scripts/make_pd_evolution_gif.py
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

OUT = os.path.join(REPO, "figures/pd")                                      # frames + GIFs (tracked)
MODEL_OUT = os.path.join(REPO, "data/ngc3110/compact/pd/model_20k_iterations.npy")   # git-ignored
FRAMES = os.path.join(OUT, "frames")
os.makedirs(FRAMES, exist_ok=True); os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
for f in glob.glob(os.path.join(FRAMES, "*.png")):                                        # frames of earlier runs
    os.remove(f)
N_ITER, K = 20000, 2.0
REWEIGHT = dict(start=6000, every=2000, eps=1.0)
vel = 5073.9 + 25.0 * (np.arange(32) - 15.5)
LINE = [i for i, v in enumerate(vel) if 4750 <= v <= 5280]
NOISE = [i for i in range(32) if i not in LINE]
CELL = 0.04                                              # after 2 x 2 rebinning
C10 = np.s_[75:325, 75:325]; EXT = [5, -5, -5, 5]

read = lambda f: np.nan_to_num(fits.getdata(os.path.join(REPO, "data/ngc3110/compact/cube", f)).astype(np.float64)).squeeze()
dirty, psf = G.rebin(read("dirty_cube.fits"), read("dirty_beam_cube.fits"), 2)
op = G.Op(psf, dirty); D = G.Dict2D1D(dirty.shape, 7, 5)
lam, _, _, s_pix = G.noise_lambdas(D, op.d, NOISE, k=K)
beta = G.field_lipschitz(op)

marks = set(np.unique(np.geomspace(1, N_ITER, 140).astype(int)).tolist())   # log-spaced: the image changes fastest early on
frames, trace = [], []                                   # frames: (iteration, model mom0, residual mom0, flux)
prev = {"x": None}

def grab(it, x, g):
    if it % 25 == 24:                                    # keep x_it to measure the change at it + 1
        prev["x"] = x.clone()
    elif it % 25 == 0 and prev["x"] is not None:         # true per-iteration relative change, every 25 iterations
        trace.append((it, float((x - prev["x"]).norm() / max(float(x.norm()), 1e-30))))
    if it in marks:
        m = (x[LINE].sum(0) * 25).cpu().numpy() / CELL ** 2          # Jy km/s arcsec^-2
        r = (-g[LINE].sum(0) * 25).cpu().numpy()                      # Jy/beam km/s
        frames.append((it, m, r, float(x.sum())))

t0 = time.time()
x, _, _, _ = G.solve_pd(op, D, lam, n_iter=N_ITER, beta=beta, reweight=REWEIGHT, print_every=4000, callback=grab)
print(f"{N_ITER} iterations in {time.time() - t0:.0f} s, {len(frames)} frames")

# frame cache and model first, so they survive anything that fails below
np.savez_compressed(os.path.join(os.path.dirname(MODEL_OUT), "pd_evolution_frames.npz"),
                    it=np.array([f[0] for f in frames]), model=np.array([f[1] for f in frames], dtype=np.float32),
                    resid=np.array([f[2] for f in frames], dtype=np.float32), flux=np.array([f[3] for f in frames]),
                    trace=np.array(trace))
np.save(MODEL_OUT, G.upsample(x.cpu().numpy().astype(np.float64), 2).astype(np.float32))

# ---- what the residual says: compare it with pure noise of the same kind
bpx = np.pi * 1.418 * 0.976 / (4 * np.log(2)) / CELL ** 2                 # compact clean beam, in 0.04" pixels
nz = G.noise_planes(op.d, NOISE)
sgn = torch.tensor([(-1) ** i for i in range(len(NOISE) // 2 * 2)], device=G.DEV, dtype=G.DT)[:, None, None]
noise_m0 = (nz[: len(sgn)] * sgn).sum(0) * 25 * np.sqrt(len(LINE) / len(sgn))  # +- sum: noise-only moment 0, same no. of channels
noise_rms = float(noise_m0[C10].std())
rm0 = -op.grad(x)[LINE].sum(0) * 25
print(f"final residual: moment-0 rms {float(rm0[C10].std()):.3f} Jy/beam km/s = {float(rm0[C10].std()) / noise_rms:.2f} x noise; "
      f"net residual flux in central 10\" {float(rm0[C10].sum()) / bpx:+.2f} Jy km/s (model flux {float(x[LINE].sum()) * 25:.1f} Jy km/s)")
print(f"pure-noise moment 0 (line-free channels, same number of channels): rms {noise_rms:.3f} Jy/beam km/s")

# ---- render: fixed colour scales across all frames
vm = np.percentile(frames[-1][1][C10], 99.9)
vr = 5 * frames[-1][2][C10].std()
ti, tc = np.array(trace).T
RW = list(range(REWEIGHT["start"], N_ITER, REWEIGHT["every"]))
label = lambda it: f"k = 2" + (", reweighted" if it > REWEIGHT["start"] else "")
plt.rcParams.update({"font.family": "serif", "font.size": 13, "xtick.direction": "in", "ytick.direction": "in"})
paths = []
for k, (it, m, r, flux) in enumerate(frames):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw=dict(width_ratios=[1, 1, 1.1]), constrained_layout=True)
    im = ax[0].imshow(m[C10], origin="lower", extent=EXT, cmap="inferno", vmin=0, vmax=vm)
    ax[0].set_title("raw model, moment 0 (no convolution)"); plt.colorbar(im, ax=ax[0], shrink=0.85, label=r"Jy km s$^{-1}$ arcsec$^{-2}$")
    im = ax[1].imshow(r[C10], origin="lower", extent=EXT, cmap="RdBu_r", vmin=-vr, vmax=vr)
    ax[1].set_title("residual d $-$ N x, moment 0"); plt.colorbar(im, ax=ax[1], shrink=0.85, label=r"Jy beam$^{-1}$ km s$^{-1}$")
    for a in ax[:2]:
        a.set_xlabel(r'$\Delta$RA ["]'); a.set_ylabel(r'$\Delta$Dec ["]')
    ax[2].loglog(ti, tc, color="0.8", lw=1.2)
    done = ti <= it
    ax[2].loglog(ti[done], tc[done], color="#2a78d6", lw=1.8)
    for rw in RW:
        ax[2].axvline(rw, color="0.85", lw=0.8, zorder=0)
    ax[2].axvline(it, color="#eb6834", lw=1.5)
    ax[2].set_xlabel("iteration"); ax[2].set_ylabel("relative change per iteration"); ax[2].set_xlim(20, N_ITER)
    ax[2].set_title("convergence (thin grey lines: reweighting)")
    fig.suptitle(f"NGC 3110 compact, 2D-1D primal-dual deconvolution  --  iteration {it:,}  |  {label(it)}  --  model flux {flux:.1f} Jy", fontsize=14)
    p = os.path.join(FRAMES, f"frame_{k:03d}.png"); fig.savefig(p, dpi=70); plt.close(fig); paths.append(p)

def gif(files, name):
    imgs = [Image.open(p).convert("RGB") for p in files]
    durs = [120] * len(imgs); durs[-1] = 2500
    imgs[0].save(os.path.join(OUT, name), save_all=True, append_images=imgs[1:], duration=durs, loop=0, optimize=True)
    print("wrote", os.path.join(OUT, name))
gif(paths, "pd_evolution.gif")

# ---- high-resolution GIF of the model panel alone
plt.rcParams.update({"font.size": 15})
hi = []
for k, (it, m, r, flux) in enumerate(frames):
    fig = plt.figure(figsize=(9.2, 8.6)); ax = fig.add_axes([0.1, 0.08, 0.72, 0.8]); cax = fig.add_axes([0.84, 0.08, 0.03, 0.8])
    im = ax.imshow(m[C10], origin="lower", extent=EXT, cmap="inferno", vmin=0, vmax=vm, interpolation="bicubic")
    cb = fig.colorbar(im, cax=cax); cb.set_label(r"Jy km s$^{-1}$ arcsec$^{-2}$")
    for sp in ax.spines.values(): sp.set_visible(True)
    ax.set_xlabel(r'$\Delta$RA ["]'); ax.set_ylabel(r'$\Delta$Dec ["]')
    ax.text(0.03, 0.97, f"iteration {it:,}", transform=ax.transAxes, color="w", fontsize=20, fontweight="bold", va="top")
    ax.text(0.03, 0.905, label(it), transform=ax.transAxes, color="w", fontsize=14, fontweight="bold", va="top")
    ax.text(0.97, 0.03, f"model flux {flux:.1f} Jy", transform=ax.transAxes, color="w", fontsize=14, fontweight="bold", ha="right")
    fig.suptitle("NGC 3110 compact: raw 2D-1D model (moment 0, no convolution)", fontsize=16, y=0.955)
    p = os.path.join(FRAMES, f"model_hires_{k:03d}.png"); fig.savefig(p, dpi=130); plt.close(fig); hi.append(p)
gif(hi, "pd_evolution_model_hires.gif")
