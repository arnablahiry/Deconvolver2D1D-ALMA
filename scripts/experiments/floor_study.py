#!/usr/bin/env python
"""
Experiment 16: where does the k-independent low-level floor of the compact pdtau models come from?
(~0.7 Jy in the line-free channels at every k = 2..7, blobs and north-south streaks outside the galaxy)

One pdtau-type primal-dual solve of the compact NGC 3110 cube (0.04" grid), with lambda_b always from the
REAL line-free noise (as every solve), on a chosen data cube, with switches:

  --data real              the dirty cube
         noise_synth       null test: Gaussian noise with the measured 2D power spectrum, channels independent
         noise_linefree    null test: the 11 real line-free channels, tiled to 32 (keeps continuum, spectral
                           correlation within each copy, real non-Gaussian / non-stationary noise)
         noise_linefree_nocont   the same with each pixel's mean over the line-free channels removed
  --no-reweight            plain L1 (no reweighting)
  --k, --n-iter

Writes <out>.npy (model, 0.04" grid) and <out>.json (settings, final numbers, and a trace every 500
iterations: total flux, line-free-channel flux, positive and negative parts, relative change).

    floor_study.py --data noise_synth --k 5 --out data/experiments/16_floor/null_synth_k5.npy
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import G, problem, read, RW  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="real", choices=("real", "noise_synth", "noise_linefree", "noise_linefree_nocont"))
ap.add_argument("--k", type=float, default=5.0)
ap.add_argument("--n-iter", type=int, default=20000)
ap.add_argument("--no-reweight", action="store_true")
ap.add_argument("--lam", default="noise", choices=("noise", "data_mad"),
                help="noise: the noise model (default); data_mad: k x MAD of the dirty cube's own coefficients per sub-band")
ap.add_argument("--out", required=True)
a = ap.parse_args()
t0 = time.time()

P0 = problem("ngc3110", "compact", k=a.k)
free = P0.noise
d_real = P0.op.d.cpu().numpy().astype(np.float64)                 # rebinned dirty cube (0.04")
if a.data == "real":
    data = None
elif a.data == "noise_synth":
    ps = G.measured_ps(G.noise_planes(P0.op.d, free))
    data = G.noise_with_ps(ps, tuple(P0.op.d.shape), seed=4242).cpu().numpy().astype(np.float64)
else:
    lf = d_real[free]
    if a.data == "noise_linefree_nocont":
        lf = lf - lf.mean(0, keepdims=True)
    data = lf[[i % len(free) for i in range(d_real.shape[0])]]
# `problem(data=...)` replaces the cube on the problem's (rebinned) grid; lambda stays from the real cube
P = problem("ngc3110", "compact", k=a.k, data=data, lam_mode=a.lam) if (data is not None or a.lam != "noise") else P0
if a.lam == "data_mad":
    ratio = (P.lam / P.lam_noise).squeeze()
    print(f"lambda (data MAD) / lambda (noise model) per spatial plane, median over spectral bands: "
          + " ".join(f"{float(ratio[j].median()):.2f}" for j in range(ratio.shape[0])), flush=True)
line = P.line
rw = None if a.no_reweight else dict(RW, ref=P.tau * P.lam)
trace, prev = [], {"x": None}


def cb(it, x, g):
    if it % 500 == 499:
        prev["x"] = x.clone()
    if it % 500 == 0:
        rel = float((x - prev["x"]).norm() / max(float(x.norm()), 1e-30)) if prev["x"] is not None else float("nan")
        trace.append(dict(iter=it, flux=float(x.sum()), flux_linefree=float(x[free].sum()),
                          pos=float(x.clamp(min=0).sum()), neg=float(x.clamp(max=0).sum()), rel_change=rel))


x, _, _, _ = G.solve_pd(P.op, P.D, P.lam, n_iter=a.n_iter, beta=P.beta, reweight=rw, print_every=10 ** 9,
                        log=lambda m: None, callback=cb)
xm = x.cpu().numpy()
r = -P.op.grad(x)
# the objective with the FINAL thresholds (reweighting changes them): F(x) = 1/2 x'Nx - d'x + sum lam |Wx|; F(0) = 0
lam_final = P.lam if rw is None else None
def objective(z, lam):
    data = float(0.5 * (z * P.op.apply(z)).sum() - (P.op.d * z).sum())
    prior = float((lam * P.D.W(z).abs()).sum())
    return data, prior
F_data, F_prior = objective(x, P.lam) if rw is None else (float("nan"), float("nan"))
summary = dict(data=a.data, lam=a.lam, k=a.k, n_iter=a.n_iter, positivity=True, reweight=not a.no_reweight,
               flux=float(xm.sum()), flux_line=float(xm[line].sum()), flux_linefree=float(xm[free].sum()),
               pos=float(np.clip(xm, 0, None).sum()), neg=float(np.clip(xm, None, 0).sum()),
               resid_rms=float(r.std()), data_rms=float(P.op.d.std()), seconds=round(time.time() - t0),
               objective_data=F_data, objective_prior=F_prior, objective=F_data + F_prior, trace=trace)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
np.save(a.out, xm.astype(np.float32))
json.dump(summary, open(a.out.replace(".npy", ".json"), "w"), indent=1)
print(f"[{os.path.basename(a.out)}] data={a.data} lam={a.lam} k={a.k:g} reweight={not a.no_reweight}: "
      f"flux {summary['flux']:.3f} Jy (line {summary['flux_line']:.3f}, line-free {summary['flux_linefree']:.3f}; "
      f"+{summary['pos']:.3f} / {summary['neg']:.3f}); residual rms {summary['resid_rms']:.3e} "
      f"(data {summary['data_rms']:.3e}); objective F(x) = {F_data:+.4e} (data) + {F_prior:.4e} (prior) = {F_data + F_prior:+.4e} "
      f"vs F(0) = 0; {summary['seconds']} s", flush=True)
