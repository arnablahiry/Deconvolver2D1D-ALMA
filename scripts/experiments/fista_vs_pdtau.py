#!/usr/bin/env python
"""
Does FISTA reproduce pdtau? Four solvers of the pipeline's deconvolution on the REAL NGC 3110 compact data, all with the
starlet, the same lambda_b = k x noise (k = 5), the same reweighting schedule (or none) and the same iteration count:

  pdtau          the production solver (gpu_pd.solve_pd, Condat-Vu): the ANALYSIS problem
                   min_x>=0  f(x) + sum_b lambda_b |(W x)_b|,   f(x) = 1/2 x'Nx - d'x  (gradient N x - d)
  fista_exact    FISTA on the same analysis problem, with its exact proximal step: prox of tau (sum lambda|W.| + i_{x>=0})
                 has no closed form for a redundant W, so it is solved by a warm-started inner dual projected gradient,
                 x(u) = P+(v - W'u),  u <- clip(u + eta W x(u), +-tau lambda),  eta = 1 / ||W||^2  (it uses W and W')
  fista_current  the pipeline's FISTA (common.fista): x <- P+( W^-1 soft_{tau lambda}( W (z - tau grad f(z)) ) ), the
                 starlet's synthesis (sum of planes) after thresholding; not the proximal step of any penalty for a
                 redundant W, so it has its own fixed point
  fista_synth    FISTA on the SYNTHESIS problem  min_alpha f(R alpha) + sum_b lambda_b |alpha_b|, x = R alpha, R = W^-1
                 (the starlet's sum of planes), gradient R'(N R alpha - d), WITHOUT positivity (a coefficient-wise prox
                 cannot impose R alpha >= 0): ill-posed for these data -- the compact array does not measure the shortest
                 spacings, so a negative bowl in x costs almost nothing; x drifts negative and diverges (NaN by 20,000)
  synth_pd       the synthesis problem WITH positivity, min_alpha f(R alpha) + sum lambda|alpha| + i(R alpha >= 0), by
                 Condat-Vu on alpha: alpha <- soft(alpha - t (R'grad f + R'v), t lambda), v <- min(v + s R(2 alpha' - alpha), 0)
                 (the conjugate prox of the positivity indicator), t = 1 / (beta ||R||^2), s = 1 / (2 t ||R||^2 x 1.1)

Expectation: without reweighting (pure convex problems) fista_exact converges to pdtau's minimiser (same objective, same
x up to the iterations); fista_current and fista_synth do not, because they solve something else. With reweighting the
weights follow each solver's own iterates, so exact agreement needs the same trajectory; fista_exact should still
track pdtau closely.

Measured, every 250 iterations: the analysis objective F(x) (the base lambda) and ||x - x_pdtau|| / ||x_pdtau||
(x_pdtau: pdtau's final model of the same mode); at the end, the cube RMSE vs the joint CLEAN at the compact clean
beam, the compact lambda/B_max and the extended clean beam (results/NGC3110/ladder_joint b00, b05, b10).

    CUDA_VISIBLE_DEVICES=2 python scripts/experiments/fista_vs_pdtau.py --rw off   (and --rw on)
    -> data/experiments/24_fista_vs_pdtau/<mode>/*.npy, results_<mode>.json
    python scripts/experiments/fista_vs_pdtau.py --plot  -> figures/experiments/24_fista_vs_pdtau/fista_vs_pdtau.png
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import G, EXP_DATA, REPO, RW, problem, line_channels, fista  # noqa: E402
from analyze import beam_ft, conv  # noqa: E402
from exp23_style import HUE, INK, MUTED, rc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rw", choices=["off", "on"], default="off"); ap.add_argument("--k", type=float, default=5.0)
ap.add_argument("--n-iter", type=int, default=20000); ap.add_argument("--inner", type=int, default=10)
ap.add_argument("--plot", action="store_true")
ap.add_argument("--only", nargs="+", help="run only these solvers (pdtau is re-read as the reference), merged into the results")
a = ap.parse_args()
D_OUT = os.path.join(EXP_DATA, "24_fista_vs_pdtau"); FIG = os.path.join(REPO, "figures", "experiments", "24_fista_vs_pdtau")
SOLVERS = ["pdtau", "fista_exact", "fista_current", "fista_synth", "synth_pd"]
NAME = {"pdtau": "pdtau (production, analysis)", "fista_exact": "FISTA, exact analysis prox (inner dual loop)",
        "fista_current": "FISTA as in the pipeline (W$^{-1}$ ∘ soft ∘ W)", "fista_synth": "FISTA, synthesis problem, no positivity",
        "synth_pd": "synthesis problem with positivity (x = R α ≥ 0)"}
COL = {"pdtau": HUE["pipeline"], "fista_exact": HUE["starlet"], "fista_current": HUE["network"], "fista_synth": HUE["imagenet"],
       "synth_pd": HUE["fire400"]}
LS = {"pdtau": "-", "fista_exact": "--", "fista_current": "-", "fista_synth": ":", "synth_pd": "-"}
LINE = line_channels("ngc3110")
EVERY = 250


def truth():
    out = {}
    for i, nm in ((0, "compact beam"), (5, "λ/B_max"), (10, "extended beam")):
        p = os.path.join(REPO, f"results/NGC3110/ladder_joint/b{i:02d}/clean_finished/image.fits")
        h = fits.getheader(p)
        J = np.nan_to_num(fits.getdata(p)).squeeze()[:, 440:840, 440:840].astype(np.float64)
        out[nm] = (J[LINE], beam_ft((400, 400), (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"]), 0.04))
    return out


def scores(x, T):
    xl = x[LINE]
    return {nm: float(100 * np.linalg.norm(conv(xl, k).cpu().numpy() - J) / np.linalg.norm(J)) for nm, (J, k) in T.items()}


if not a.plot:
    out = os.path.join(D_OUT, a.rw); os.makedirs(out, exist_ok=True)
    P = problem("ngc3110", "compact", k=a.k)                              # the starlet (gpu_pd default), real data
    op, D, lam, tau = P.op, P.D, P.lam, P.tau
    rw = (dict(RW, ref=tau * lam) if a.rw == "on" else None)
    F = lambda x: float(0.5 * (x * op.apply(x)).sum() - (op.d * x).sum() + (lam * D.W(x).abs()).sum())
    T = truth()
    res = {}
    hist = {s: [] for s in SOLVERS}
    xref = None

    def recorder(name):
        def cb(it, x, g=None):
            if it % EVERY == 0:
                hist[name].append((it, F(x), None if xref is None else float((x - xref).norm() / xref.norm())))
        return cb

    def finish(name, x, t0):
        np.save(os.path.join(out, f"{name}.npy"), x.cpu().numpy().astype(np.float32))
        res[name] = dict(objective=F(x), seconds=round(time.time() - t0), flux=float(x.sum()), min=float(x.min()),
                         rel_diff_to_pdtau=None if xref is None else float((x - xref).norm() / xref.norm()),
                         rmse=scores(x.cpu().numpy().astype(np.float64), T), history=hist[name])
        print(name, {k_: v for k_, v in res[name].items() if k_ != "history"}, flush=True)

    prev = json.load(open(os.path.join(D_OUT, f"results_{a.rw}.json")))["results"] if a.only else {}
    run = lambda s_: a.only is None or s_ in a.only

    # 1. pdtau (the reference; re-read when not re-run)
    if run("pdtau"):
        t0 = time.time()
        x_pd, _, _, _ = G.solve_pd(op, D, lam, n_iter=a.n_iter, beta=P.beta, reweight=rw, print_every=10 ** 9,
                                   log=lambda m: None, callback=recorder("pdtau"))
        xref = x_pd
        finish("pdtau", x_pd, t0)
    else:
        xref = G.to_t(np.load(os.path.join(out, "pdtau.npy")))

    # 2. FISTA, exact analysis prox (warm-started inner dual projected gradient)
    if run("fista_exact"):
        t0 = time.time()
        eta = 1.0 / D.norm_sq(op.d.shape)
        x = torch.zeros_like(op.d); z = x.clone(); t = 1.0; u = torch.zeros_like(D.W(x)); bound = tau * lam
        cb = recorder("fista_exact")
        for it in range(1, a.n_iter + 1):
            if rw and it > rw["start"] and (it - 1 - rw["start"]) % rw["every"] == 0:
                bound = tau * lam / (D.W(x).abs() / (tau * lam) + rw["eps"])
            v = z - tau * op.grad(z)
            for _ in range(a.inner):
                u = torch.maximum(torch.minimum(u + eta * D.W((v - D.WT(u)).clamp(min=0)), bound), -bound)
            xn = (v - D.WT(u)).clamp(min=0)
            tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t)); z = xn + ((t - 1) / tn) * (xn - x); x, t = xn, tn
            cb(it, x)
        finish("fista_exact", x, t0)

    # 3. FISTA as in the pipeline
    if run("fista_current"):
        t0 = time.time()
        x = fista(P, a.n_iter, reweight=rw if rw else None, callback=recorder("fista_current"), log=lambda m: None,
                  every=10 ** 9)
        finish("fista_current", x, t0)

    # 4-5. the synthesis problem: alpha, x = R alpha (R = W^-1, the starlet's sum of planes)
    if run("fista_synth") or run("synth_pd"):
        a0 = D.W(torch.zeros_like(op.d))
        _, RT = torch.func.vjp(D.Winv, a0)                                 # R' (R is linear)
        v_ = torch.randn_like(a0); v_ /= v_.norm(); nR = 0.0
        for _ in range(40):                                                # ||R||^2 by power iteration
            w_ = RT(D.Winv(v_))[0]; nR = float(w_.norm()); v_ = w_ / nR
        tau_a = 1.0 / (P.beta * nR * 1.1)
        soft = lambda c, th: torch.sign(c) * torch.clamp(c.abs() - th, min=0)
        T0 = tau_a * lam
    if run("fista_synth"):                                                 # FISTA, no positivity (diverges here)
        t0 = time.time()
        al = torch.zeros_like(a0); za = al.clone(); t = 1.0; Th = T0
        cb = recorder("fista_synth")
        for it in range(1, a.n_iter + 1):
            if rw and it > rw["start"] and (it - 1 - rw["start"]) % rw["every"] == 0:
                Th = T0 / (al.abs() / T0 + rw["eps"])
            an = soft(za - tau_a * RT(op.grad(D.Winv(za)))[0], Th)
            tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t)); za = an + ((t - 1) / tn) * (an - al); al, t = an, tn
            cb(it, D.Winv(al))
        finish("fista_synth", D.Winv(al), t0)
    if run("synth_pd"):                                                    # primal-dual, x = R alpha >= 0 imposed
        t0 = time.time()
        ta = 1.0 / (P.beta * nR); sa = 1.0 / (2 * ta * nR * 1.1); Th = ta * lam
        al = torch.zeros_like(a0); vd = torch.zeros_like(op.d)
        cb = recorder("synth_pd")
        for it in range(1, a.n_iter + 1):
            if rw and it > rw["start"] and (it - 1 - rw["start"]) % rw["every"] == 0:
                Th = ta * lam / (al.abs() / (ta * lam) + rw["eps"])
            an = soft(al - ta * RT(op.grad(D.Winv(al)) + vd)[0], Th)
            vd = torch.clamp(vd + sa * D.Winv(2 * an - al), max=0)        # prox of the conjugate of i(x >= 0)
            al = an
            cb(it, D.Winv(al))
        finish("synth_pd", D.Winv(al), t0)
    merged = dict(prev, **res)
    json.dump(dict(k=a.k, rw=a.rw, n_iter=a.n_iter, inner=a.inner, results=merged),
              open(os.path.join(D_OUT, f"results_{a.rw}.json"), "w"), indent=1)
    sys.exit()

# ------------------------------------------------------------------ figure
rc(); os.makedirs(FIG, exist_ok=True)
modes = [m for m in ("off", "on") if os.path.exists(os.path.join(D_OUT, f"results_{m}.json"))]
fig, ax = plt.subplots(len(modes), 3, figsize=(19, 5 * len(modes)), constrained_layout=True, squeeze=False)
for r, m in enumerate(modes):
    R = json.load(open(os.path.join(D_OUT, f"results_{m}.json")))["results"]
    Fmin = min(min(f for _, f, _ in R[s]["history"]) for s in R if np.isfinite(R[s]["objective"]))
    for s in SOLVERS:
        if s not in R or not np.isfinite(R[s]["objective"]):
            continue
        h = R[s]["history"]
        it = [e[0] for e in h]
        ax[r, 0].semilogy(it, [e[1] - Fmin + 1e-12 for e in h], color=COL[s], ls=LS[s], lw=1.8, label=NAME[s])
        if s != "pdtau":
            ax[r, 1].semilogy(it, [e[2] for e in h], color=COL[s], ls=LS[s], lw=1.8, label=NAME[s])
    ax[r, 0].set_title(f"analysis objective F(x) − min (reweighting {m})", fontsize=10.5)
    ax[r, 1].set_title(f"‖x − x$_{{pdtau}}$‖ / ‖x$_{{pdtau}}$‖ (reweighting {m})", fontsize=10.5)
    for c in (0, 1):
        ax[r, c].set_xlabel("iteration")
    ax[r, 0].legend(fontsize=8)
    names = list(R["pdtau"]["rmse"])
    ok = [s for s in SOLVERS if s in R and np.isfinite(R[s]["objective"])]  # fista_synth diverges: no bar
    w = 0.8 / len(ok)
    for j, s in enumerate(ok):
        vals = [R[s]["rmse"][n] - R["pdtau"]["rmse"][n] for n in names]
        ax[r, 2].bar(np.arange(len(names)) + (j - (len(ok) - 1) / 2) * w, vals, w * 0.92, color=COL[s], label=NAME[s],
                     hatch="//" if LS[s] != "-" else None, edgecolor="white")
        for xi, v in zip(np.arange(len(names)) + (j - (len(ok) - 1) / 2) * w, vals):
            ax[r, 2].text(xi, v, f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7, color=MUTED)
    ax[r, 2].axhline(0, color=INK, lw=0.8)
    ax[r, 2].set_xticks(range(len(names))); ax[r, 2].set_xticklabels(names)
    ax[r, 2].set_ylabel("cube RMSE − pdtau's [percentage points]")
    ax[r, 2].set_title(f"vs the joint CLEAN (reweighting {m}); pdtau: " +
                       ", ".join(f"{R['pdtau']['rmse'][n]:.2f}%" for n in names), fontsize=10)
fig.suptitle("NGC 3110, real data, starlet, k = 5: does FISTA reproduce pdtau?", fontsize=12)
fig.savefig(os.path.join(FIG, "fista_vs_pdtau.png"), dpi=140, bbox_inches="tight"); plt.close(fig)
print("->", os.path.join(FIG, "fista_vs_pdtau.png"))
