#!/usr/bin/env python
"""
Evolution GIF of a deconvolution (reweighted FISTA or pdtau, noise-model thresholds):

    toy_gif.py [--dataset toy|compact] [--solver fista|pdtau] [--k 5] [--n-iter 20000]
    -> figures/experiments/16_floor/<dataset>_<solver>_k<k>.gif  (+ frames/<name>/)

toy      the GUI's synthetic disc (truth known): true sky | model moment 0 (asinh, on the true sky's scale) |
         model - truth | convergence; line-free channels auto-detected as in the GUI
compact  NGC 3110 compact cube (0.04"): dirty | model | residual (line moment 0) | convergence
Recipe as scripts/experiments/toy_floor.py: reweighting every 2000 iterations from 6000 (eps 1); FISTA
soft-thresholds W(z - tau grad) at T (T0 = tau lambda), pdtau uses the reweighting reference tau*lambda.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(REPO, "gui"))
from common import G, RW, problem  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="toy", choices=("toy", "compact"))
ap.add_argument("--solver", default="fista", choices=("fista", "pdtau"))
ap.add_argument("--k", type=float, default=5.0)
ap.add_argument("--n-iter", type=int, default=20000)
a = ap.parse_args()
name = f"{a.dataset}_{a.solver}_k{a.k:g}"
OUT = os.path.join(REPO, "figures/experiments/16_floor")
FR = os.path.join(OUT, "frames", name)
os.makedirs(FR, exist_ok=True)
for f in glob.glob(os.path.join(FR, "*.png")):
    os.remove(f)

if a.dataset == "toy":
    from deconvgui.toy import make_toy
    from deconvgui.worker import _auto_scales, _auto_line_free
    dirty, psf, truth, meta = make_toy()
    dirty, psf, truth = dirty.astype(np.float64), psf.astype(np.float64), truth.astype(np.float64)
    dv, cell = meta["width_kms"], meta["cell_arcsec"]
    nz, ny, nx = dirty.shape
    op = G.Op(psf, dirty)
    J, L = _auto_scales(ny, nx, nz, None, None)
    D = G.Dict2D1D(dirty.shape, J, L)
    lam = G.noise_lambdas(D, op.d, _auto_line_free(G, op.d, J), k=a.k)[0]
    beta = G.field_lipschitz(op)
    line = list(range(nz))
else:
    P = problem("ngc3110", "compact", k=a.k)
    op, D, lam, beta, line = P.op, P.D, P.lam, P.beta, P.line
    dv, cell, truth = 25.0, P.cell, None
    nz, ny, nx = op.d.shape
    dirty = op.d.cpu().numpy().astype(np.float64)
tau = 1.0 / beta
marks = set(np.unique(np.geomspace(1, a.n_iter, 120).astype(int)).tolist())
frames, trace, prev = [], [], {"x": None}


def grab(it, x):
    if it in marks:
        resid = None if truth is not None else ((-op.grad(x))[line].sum(0) * dv).cpu().numpy()
        frames.append((it, (x[line].sum(0) * dv).cpu().numpy(), float(x.sum()), resid))
    if it % 25 == 24:
        prev["x"] = x.clone()
    elif it % 25 == 0 and prev["x"] is not None:
        trace.append((it, float((x - prev["x"]).norm() / max(float(x.norm()), 1e-30)), float(x.sum())))


if a.solver == "fista":
    T0 = tau * lam; T = T0
    soft = lambda c, t: torch.sign(c) * torch.clamp(c.abs() - t, min=0)
    x = torch.zeros_like(op.d); z = x.clone(); t = 1.0
    for it in range(1, a.n_iter + 1):
        if it > RW["start"] and (it - 1 - RW["start"]) % RW["every"] == 0:
            T = T0 / (D.W(x).abs() / T0 + RW["eps"])
        xn = D.Winv(soft(D.W(z - tau * op.grad(z)), T)).clamp_(min=0)
        tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t)); z = xn + ((t - 1) / tn) * (xn - x); x, t = xn, tn
        grab(it, x)
else:
    x, _, _, _ = G.solve_pd(op, D, lam, n_iter=a.n_iter, beta=beta, reweight=dict(RW, ref=tau * lam), print_every=10 ** 9,
                            log=lambda m: None, callback=lambda it, xx, g: grab(it, xx))
print(f"{name}: {a.n_iter} iterations, model {float(x.sum()):.2f} Jy"
      + (f" (truth {truth.sum():.1f})" if truth is not None else "") + f", {len(frames)} frames", flush=True)

# ---- render
asinh = lambda q: np.arcsinh(q / 0.1) / np.arcsinh(10)
ext = [nx * cell / 2, -nx * cell / 2, -ny * cell / 2, ny * cell / 2]
ti, tc, tf = np.array(trace).T
if truth is not None:
    tm0 = truth.sum(0) * dv
    hi = np.percentile(tm0, 99.5)
    supp = truth.sum(0) > 1e-3 * truth.sum(0).max()
else:
    dm0 = dirty[line].sum(0) * dv
    lo_d, hi_d = np.percentile(dm0, [0.5, 99.5])
    hi = np.percentile(frames[-1][1], 99.5)
    vr = 5 * frames[-1][3].std()
title = "GUI synthetic disc" if truth is not None else "NGC 3110 compact (0.04″)"
plt.rcParams.update({"font.family": "serif", "font.size": 11})
paths = []
for i, (it, m0, flux, rm0) in enumerate(frames):
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.4), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.15]), constrained_layout=True)
    ax[1].imshow(asinh(np.clip(m0 / hi, 0, 1)), origin="lower", cmap="inferno", extent=ext)
    if truth is not None:
        ax[0].imshow(asinh(np.clip(tm0 / hi, 0, 1)), origin="lower", cmap="inferno", extent=ext)
        ax[0].set_title(f"true sky ({truth.sum():.0f} Jy)")
        ax[1].set_title(f"model moment 0: {flux:.0f} Jy\n(outside the true emission {m0[~supp].sum() / dv:.1f} Jy)", fontsize=10)
        d = m0 - tm0
        ax[2].imshow(d, origin="lower", cmap="RdBu_r", vmin=-0.5 * hi, vmax=0.5 * hi, extent=ext)
        ax[2].set_title(f"model − truth (moment-0 error {np.linalg.norm(d) / np.linalg.norm(tm0):.2f})", fontsize=10)
    else:
        ax[0].imshow(asinh(np.clip((dm0 - lo_d) / (hi_d - lo_d), 0, 1)), origin="lower", cmap="inferno", extent=ext)
        ax[0].set_title("dirty, line moment 0")
        ax[1].set_title(f"model, line moment 0 (model flux {flux:.1f} Jy)", fontsize=10)
        ax[2].imshow(rm0, origin="lower", cmap="RdBu_r", vmin=-vr, vmax=vr, extent=ext)
        ax[2].set_title("residual d − N x, line moment 0", fontsize=10)
    for q in ax[:3]:
        q.set_xlabel(r'$\Delta$RA ["]'); q.set_ylabel(r'$\Delta$Dec ["]')
    a2 = ax[3]; a2b = a2.twinx()
    a2.loglog(ti, tc, color="0.8", lw=1); done = ti <= it
    a2.loglog(ti[done], tc[done], color="#2a78d6", lw=1.6)
    a2b.semilogx(ti, tf, color="0.85", lw=1, ls="--"); a2b.semilogx(ti[done], tf[done], color="#eb6834", lw=1.4)
    if truth is not None:
        a2b.axhline(truth.sum(), color="#1baf7a", lw=1, ls=":")
    for r in range(RW["start"], a.n_iter, RW["every"]):
        a2.axvline(r, color="0.9", lw=0.6, zorder=0)
    a2.axvline(it, color="k", lw=1)
    a2.set_xlabel("iteration"); a2.set_ylabel("relative change per iteration", color="#2a78d6")
    a2b.set_ylabel("model flux (Jy)" + ("; dotted: truth" if truth is not None else ""), color="#eb6834")
    a2.set_xlim(20, a.n_iter)
    fig.suptitle(f"{title}: {'reweighted FISTA' if a.solver == 'fista' else 'pdtau'}, noise-model thresholds, "
                 f"k = {a.k:g}  --  iteration {it:,}", fontsize=13)
    p = os.path.join(FR, f"frame_{i:03d}.png"); fig.savefig(p, dpi=65); plt.close(fig); paths.append(p)
imgs = [Image.open(p).convert("RGB") for p in paths]
durs = [110] * len(imgs); durs[-1] = 2500
gif = os.path.join(OUT, f"{name}.gif")
imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=durs, loop=0, optimize=True)
print("wrote", gif, flush=True)
