#!/usr/bin/env python
"""
FISTA counterpart of scripts/make_pd_evolution_gif.py: the loop of simple/deconvolve.py
(x = max(0, W^-1 soft(W(z - tau grad), tau lambda)), FISTA momentum) run on the SAME
problem as the primal-dual solve -- compact cube on the 0.04" grid, lambda_b =
k x max(real, white) noise MAD, step 1/beta, reweighting (eps = 1) every 2000 from 6000 --
for 20,000 iterations.

    CUDA_VISIBLE_DEVICES=0 /scratch/alahiry/conda/envs/denoise2d1d/bin/python scripts/make_fista_run.py [compact|extended] [k]
    (defaults: compact, k = 2; extended defaults to k = 3)

compact : as above.
extended: the problem of the extended primal-dual model (data/ngc3110/extended/pd/README.txt) --
          full 0.02" grid, 8 starlet scales, noise channels v < 4780 or v > 5180 km/s, no reweighting,
          and x = 0 outside that model's support mask.

Writes (<cfg> = compact | extended)
    data/ngc3110/<cfg>/fista/model_k<k>_20k.npy             final model, Jy/pixel, 0.02" (git-ignored)
    data/ngc3110/<cfg>/fista/fista_k<k>_frames.npz           frame cache (model, residual, change trace)
    figures/fista/ (compact) or figures/fista/extended/<k>-sigma/ (extended):
        fista_evolution.gif, fista_evolution_model_hires.gif, frames/
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

args = sys.argv[1:]
CFG = args.pop(0) if args and args[0] in ("compact", "extended") else "compact"
K = float(args[0]) if args else (2.0 if CFG == "compact" else 3.0)
TAG = f"{K:g}"
FIG = os.path.join(REPO, "figures/fista" if CFG == "compact" else f"figures/fista/extended/{TAG}-sigma")
FRAMES = os.path.join(FIG, "frames")
DAT = os.path.join(REPO, f"data/ngc3110/{CFG}/fista")
for d in (FIG, FRAMES, DAT):
    os.makedirs(d, exist_ok=True)
for f in glob.glob(os.path.join(FRAMES, "frame_*.png")) + glob.glob(os.path.join(FRAMES, "model_hires_*.png")):   # only this script's frames
    os.remove(f)
N_ITER = 20000
vel = 5073.9 + 25.0 * (np.arange(32) - 15.5)
LINE = [i for i, v in enumerate(vel) if 4750 <= v <= 5280]
EXT = [5, -5, -5, 5]
read = lambda f: np.nan_to_num(fits.getdata(os.path.join(REPO, f"data/ngc3110/{CFG}/cube", f)).astype(np.float64)).squeeze()
if CFG == "compact":
    REWEIGHT = dict(start=6000, every=2000, eps=1.0)
    NOISE = [i for i in range(32) if i not in LINE]
    CELL, BIN, J = 0.04, 2, 7
    C10 = np.s_[75:325, 75:325]; BEAM = (1.418, 0.976)
    dirty, psf = G.rebin(read("dirty_cube.fits"), read("dirty_beam_cube.fits"), 2)
    SUPPORT = None
else:
    REWEIGHT = None
    NOISE = [i for i, v in enumerate(vel) if v < 4780 or v > 5180]     # as the extended primal-dual model
    CELL, BIN, J = 0.02, 1, 8
    C10 = np.s_[150:650, 150:650]; BEAM = (0.220, 0.172)
    dirty, psf = read("dirty_cube.fits"), read("dirty_beam_cube.fits")
    SUPPORT = torch.tensor(np.load(os.path.join(REPO, "data/ngc3110/extended/pd/support_mask.npy")), device=G.DEV).to(G.DT)
op = G.Op(psf, dirty); D = G.Dict2D1D(dirty.shape, J, 5)
lam, _, _, _ = G.noise_lambdas(D, op.d, NOISE, k=K)
tau = 1.0 / G.field_lipschitz(op)
T0 = tau * lam
soft = lambda c, t: torch.sign(c) * torch.clamp(c.abs() - t, min=0)
marks = set(np.unique(np.geomspace(1, N_ITER, 140).astype(int)).tolist())

x = torch.zeros_like(op.d); z = x.clone(); t = 1.0; T = T0
frames, trace = [], []
t0 = time.time()
for it in range(1, N_ITER + 1):
    if REWEIGHT and it > REWEIGHT["start"] and (it - 1 - REWEIGHT["start"]) % REWEIGHT["every"] == 0:
        T = T0 / (D.W(x).abs() / T0 + REWEIGHT["eps"])
    xn = D.Winv(soft(D.W(z - tau * op.grad(z)), T)).clamp_(min=0)
    if SUPPORT is not None:
        xn.mul_(SUPPORT)
    tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
    z = xn + ((t - 1) / tn) * (xn - x)
    if it % 25 == 0:
        trace.append((it, float((xn - x).norm() / max(float(xn.norm()), 1e-30))))
    x, t = xn, tn
    if it in marks:
        r = -op.grad(x)
        frames.append((it, (x[LINE].sum(0) * 25).cpu().numpy() / CELL ** 2, (r[LINE].sum(0) * 25).cpu().numpy(), float(x.sum())))
    if it % 4000 == 0:
        print(f"FISTA iter {it:6d}  flux {float(x.sum()):8.3f}  change {trace[-1][1]:.2e}  ({time.time() - t0:.0f} s)", flush=True)
print(f"FISTA {CFG} k = {TAG}: {N_ITER} iterations in {time.time() - t0:.0f} s, {len(frames)} frames", flush=True)

xm = x.cpu().numpy().astype(np.float64)
np.save(os.path.join(DAT, f"model_k{TAG}_20k.npy"), (G.upsample(xm, BIN) if BIN > 1 else xm).astype(np.float32))
np.savez_compressed(os.path.join(DAT, f"fista_k{TAG}_frames.npz"), it=np.array([f[0] for f in frames]),
                    model=np.array([f[1] for f in frames], dtype=np.float32), resid=np.array([f[2] for f in frames], dtype=np.float32),
                    flux=np.array([f[3] for f in frames]), trace=np.array(trace))

# residual vs pure noise (same diagnostic as the primal-dual run)
bpx = np.pi * BEAM[0] * BEAM[1] / (4 * np.log(2)) / CELL ** 2
nz = G.noise_planes(op.d, NOISE)
sgn = torch.tensor([(-1) ** i for i in range(len(NOISE) // 2 * 2)], device=G.DEV, dtype=G.DT)[:, None, None]
noise_rms = float(((nz[: len(sgn)] * sgn).sum(0) * 25 * np.sqrt(len(LINE) / len(sgn)))[C10].std())
rm0 = -op.grad(x)[LINE].sum(0) * 25
print(f"final residual: moment-0 rms {float(rm0[C10].std()):.3f} Jy/beam km/s = {float(rm0[C10].std()) / noise_rms:.2f} x noise; "
      f"net residual flux in central 10\" {float(rm0[C10].sum()) / bpx:+.2f} Jy km/s (model flux {float(x[LINE].sum()) * 25:.1f} Jy km/s)", flush=True)

# ---- render (fixed colour scales; same scales as the primal-dual GIFs when their cache exists)
pdc = os.path.join(REPO, "data/ngc3110/compact/pd/pd_evolution_frames.npz")
if CFG == "compact" and os.path.exists(pdc):
    p = np.load(pdc); vm = np.percentile(p["model"][-1][C10], 99.9); vr = 5 * p["resid"][-1][C10].std()
else:
    vm = np.percentile(frames[-1][1][C10], 99.9); vr = 5 * frames[-1][2][C10].std()
ti, tc = np.array(trace).T
RW = list(range(REWEIGHT["start"], N_ITER, REWEIGHT["every"])) if REWEIGHT else []
label = lambda it: f"k = {TAG}" + (", reweighted" if REWEIGHT and it > REWEIGHT["start"] else "") + ("" if SUPPORT is None else ", support mask")

def gif(files, name):
    imgs = [Image.open(q).convert("RGB") for q in files]
    durs = [120] * len(imgs); durs[-1] = 2500
    imgs[0].save(os.path.join(FIG, name), save_all=True, append_images=imgs[1:], duration=durs, loop=0, optimize=True)
    print("wrote", os.path.join(FIG, name), flush=True)

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
    ax[2].loglog(ti, tc, color="0.8", lw=1.2); done = ti <= it
    ax[2].loglog(ti[done], tc[done], color="#2a78d6", lw=1.8)
    for rw in RW:
        ax[2].axvline(rw, color="0.85", lw=0.8, zorder=0)
    ax[2].axvline(it, color="#eb6834", lw=1.5)
    ax[2].set_xlabel("iteration"); ax[2].set_ylabel("relative change per iteration"); ax[2].set_xlim(20, N_ITER)
    ax[2].set_title("convergence" + (" (thin grey lines: reweighting)" if RW else ""))
    fig.suptitle(f"NGC 3110 {CFG}, 2D-1D FISTA deconvolution  --  iteration {it:,}  |  {label(it)}  --  model flux {flux:.1f} Jy", fontsize=14)
    q = os.path.join(FRAMES, f"frame_{k:03d}.png"); fig.savefig(q, dpi=70); plt.close(fig); paths.append(q)
gif(paths, "fista_evolution.gif")

plt.rcParams.update({"font.size": 15})
hi = []
for k, (it, m, r, flux) in enumerate(frames):
    fig = plt.figure(figsize=(9.2, 8.6)); ax = fig.add_axes([0.1, 0.08, 0.72, 0.8]); cax = fig.add_axes([0.84, 0.08, 0.03, 0.8])
    im = ax.imshow(m[C10], origin="lower", extent=EXT, cmap="inferno", vmin=0, vmax=vm, interpolation="bicubic")
    cb = fig.colorbar(im, cax=cax); cb.set_label(r"Jy km s$^{-1}$ arcsec$^{-2}$")
    for sp in ax.spines.values(): sp.set_visible(True)
    ax.set_xlabel(r'$\Delta$RA ["]'); ax.set_ylabel(r'$\Delta$Dec ["]')
    ax.text(0.03, 0.97, f"iteration {it:,}", transform=ax.transAxes, color="w", fontsize=20, fontweight="bold", va="top")
    ax.text(0.03, 0.905, "FISTA, " + label(it), transform=ax.transAxes, color="w", fontsize=14, fontweight="bold", va="top")
    ax.text(0.97, 0.03, f"model flux {flux:.1f} Jy", transform=ax.transAxes, color="w", fontsize=14, fontweight="bold", ha="right")
    fig.suptitle(f"NGC 3110 {CFG}: raw 2D-1D FISTA model (moment 0, no convolution)", fontsize=16, y=0.955)
    q = os.path.join(FRAMES, f"model_hires_{k:03d}.png"); fig.savefig(q, dpi=130); plt.close(fig); hi.append(q)
gif(hi, "fista_evolution_model_hires.gif")
