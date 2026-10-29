#!/usr/bin/env python
"""
Experiment 16b: the low-level floor on the GUI's synthetic disc (truth known) and on the compact cube,
for both solvers and three threshold calibrations.

    toy_floor.py --dataset toy|compact --solver pdtau|fista --lam noise|data_mad|true_noise --k 5 --out X.npy

dataset  toy      deconvgui.toy.make_toy() with the GUI's defaults (128 x 128 x 40, 0.1"), truth known
         compact  NGC 3110 compact cube on the 0.04" grid (scripts/experiments/common.problem)
solver   pdtau    gpu_pd.solve_pd with reweighting reference tau*lambda
         fista    x = max(0, W^-1 soft(W(z - tau grad), T)), FISTA momentum, T0 = tau*lambda,
                  reweighting T = T0 / (|W x| / T0 + 1)  (scripts/make_fista_run.py)
         both: 20,000 iterations, reweighting every 2000 from 6000, eps 1 (--no-reweight: off)
lam      noise       gpu_pd.noise_lambdas from line-free channels (toy: detected as in the GUI; if none,
                     white noise of the robust pixel sigma, as the GUI)
         data_mad    lambda_b = k x MAD of the DIRTY cube's own 2D-1D coefficients in sub-band b
         true_noise  (toy only) lambda_b = k x MAD of W(noise), the noise realisation actually in the data
         certificate lambda_b = k x MAD_b(u), u = W (W^T W)^-1 n the minimum-norm dual certificate of a noise cube
                     n (toy: the true noise; compact: a noise realisation with the measured power spectrum), so that
                     W^T u = n exactly: the scale the dual variable must reach for x = 0 to be optimal on noise

Writes the model (.npy) and a .json: flux, flux outside the true emission (toy) / in line-free channels,
error vs truth (toy), residual rms, and lambda ratios.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "gui"))
from common import G, problem, RW  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="toy", choices=("toy", "compact"))
ap.add_argument("--solver", default="pdtau", choices=("pdtau", "fista"))
ap.add_argument("--lam", default="noise", choices=("noise", "data_mad", "true_noise", "certificate"))
ap.add_argument("--k", type=float, default=5.0)
ap.add_argument("--n-iter", type=int, default=20000)
ap.add_argument("--no-reweight", action="store_true")
ap.add_argument("--null", action="store_true", help="toy: replace the data by its noise realisation alone")
ap.add_argument("--out", required=True)
a = ap.parse_args()
t0 = time.time()

truth = noise_cube = None


def certificate_lambdas(D, n, k, max_cg=3000, tol=1e-6):
    """k x per-sub-band MAD of u = W (W^T W)^-1 n, by conjugate gradient on W^T W v = n"""
    A = lambda v: D.WT(D.W(v))
    v = torch.zeros_like(n); r = n.clone(); p = r.clone(); rs = float((r * r).sum()); nn = float((n * n).sum())
    for i in range(max_cg):
        Ap = A(p); al = rs / float((p * Ap).sum()); v += al * p; r -= al * Ap
        rn = float((r * r).sum())
        if rn < tol ** 2 * nn:
            break
        p = r + (rn / rs) * p; rs = rn
    u = D.W(v)
    print(f"[certificate] CG {i + 1} steps, relative residual {np.sqrt(rn / nn):.1e}; ||W^T u - n|| / ||n|| = "
          f"{float((D.WT(u) - n).norm() / n.norm()):.1e}", flush=True)
    return k * D.band_mad(u)

if a.dataset == "toy":
    from deconvgui.toy import make_toy
    from deconvgui.worker import _auto_scales, _auto_line_free
    dirty, psf, truth, meta = make_toy()
    dirty, psf, truth = dirty.astype(np.float64), psf.astype(np.float64), truth.astype(np.float64)
    op0 = G.Op(psf, np.zeros_like(dirty))
    noise_cube = dirty - op0.apply(G.to_t(truth)).cpu().numpy()
    if a.null:
        dirty = noise_cube.copy()
    nz, ny, nx = dirty.shape
    J, L = _auto_scales(ny, nx, nz, None, None)
    op = G.Op(psf, dirty)
    D = G.Dict2D1D(dirty.shape, J, L)
    free = _auto_line_free(G, op.d, J)
    if a.lam == "noise":
        if len(free) >= 2:
            lam = G.noise_lambdas(D, op.d, free, k=a.k)[0]
        else:
            s_pix = 1.4826 * float((op.d - op.d.median()).abs().median())
            white = torch.randn(tuple(op.d.shape), device=G.DEV, dtype=G.DT, generator=torch.Generator(G.DEV).manual_seed(101)) * s_pix
            lam = a.k * D.band_mad(D.W(white))
    elif a.lam == "true_noise":
        lam = a.k * D.band_mad(D.W(G.to_t(noise_cube)))
    elif a.lam == "certificate":
        lam_tn = a.k * D.band_mad(D.W(G.to_t(noise_cube)))
        lam = certificate_lambdas(D, G.to_t(noise_cube), a.k)
        rat = (lam / lam_tn).squeeze()
        print("[certificate] lambda / true-noise lambda per spatial plane (median over bands): "
              + " ".join(f"{float(rat[j].median()):.2f}" for j in range(rat.shape[0])), flush=True)
    else:
        lam = a.k * D.band_mad(D.W(op.d))
    beta = G.field_lipschitz(op)
    line = [c for c in range(nz) if c not in free]
else:
    P = problem("ngc3110", "compact", k=a.k, lam_mode="data_mad" if a.lam == "data_mad" else "noise")
    op, D, lam, beta, free, line = P.op, P.D, P.lam, P.beta, P.noise, P.line
    if a.lam == "certificate":
        n = G.noise_with_ps(G.measured_ps(G.noise_planes(op.d, free)), tuple(op.d.shape), seed=4242)
        lam_nm = lam
        lam = torch.maximum(certificate_lambdas(D, n, a.k), lam_nm)
        rat = (lam / lam_nm).squeeze()
        print("[certificate] lambda / noise-model lambda per spatial plane (median over bands): "
              + " ".join(f"{float(rat[j].median()):.2f}" for j in range(rat.shape[0])), flush=True)
tau = 1.0 / beta
print(f"[setup] {a.dataset} {tuple(op.d.shape)}, {a.solver}, lam={a.lam}, k={a.k:g}, line-free channels "
      f"{len(free)}, beta {beta:.1f}", flush=True)

rw = None if a.no_reweight else dict(RW)
if a.solver == "pdtau":
    x, _, _, _ = G.solve_pd(op, D, lam, n_iter=a.n_iter, beta=beta, reweight=None if rw is None else dict(rw, ref=tau * lam),
                            print_every=10 ** 9, log=lambda m: None)
else:
    T0 = tau * lam; T = T0
    soft = lambda c, t: torch.sign(c) * torch.clamp(c.abs() - t, min=0)
    x = torch.zeros_like(op.d); z = x.clone(); t = 1.0
    for it in range(1, a.n_iter + 1):
        if rw and it > rw["start"] and (it - 1 - rw["start"]) % rw["every"] == 0:
            T = T0 / (D.W(x).abs() / T0 + rw["eps"])
        xn = D.Winv(soft(D.W(z - tau * op.grad(z)), T)).clamp_(min=0)
        tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t)); z = xn + ((t - 1) / tn) * (xn - x); x, t = xn, tn

xm = x.cpu().numpy().astype(np.float64)
r = (-op.grad(x)).cpu().numpy()
out = dict(dataset=a.dataset, solver=a.solver, lam=a.lam, k=a.k, reweight=not a.no_reweight, null=a.null,
           flux=float(xm.sum()), resid_rms=float(r.std()), data_rms=float(op.d.std()), seconds=round(time.time() - t0),
           flux_linefree=float(xm[free].sum()) if free else None)
if truth is not None:
    supp = truth.sum(0) > 1e-3 * truth.sum(0).max()                     # where the true sky has emission (any channel)
    out.update(truth_flux=float(truth.sum()), flux_in=float(xm[:, supp].sum()), flux_out=float(xm[:, ~supp].sum()),
               err=float(np.linalg.norm(xm - truth) / np.linalg.norm(truth)),
               err_mom0=float(np.linalg.norm(xm.sum(0) - truth.sum(0)) / np.linalg.norm(truth.sum(0))),
               frac_outside=float(xm[:, ~supp].sum() / max(xm.sum(), 1e-30)), support_frac=float(supp.mean()))
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
np.save(a.out, xm.astype(np.float32))
json.dump(out, open(a.out.replace(".npy", ".json"), "w"), indent=1)
print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()}), flush=True)
