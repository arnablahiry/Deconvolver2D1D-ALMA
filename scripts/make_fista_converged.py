#!/usr/bin/env python
"""
FISTA run of scripts/make_fista_run.py (compact NGC 3110, 0.04" grid, 7 starlet scales x 5 CDF 9/7
levels, lambda_b = k x max(real, white) noise MAD, step 1/beta, reweighting eps = 1) -- but run to
CONVERGENCE instead of a fixed 20,000 iterations.

Why a different iteration: x+ = max(0, W^-1 soft(W(z - tau grad), tau lambda)) is not the proximal
operator of anything for the redundant starlet frame (see simple/gpu_pd.py), and plain FISTA momentum
on it never settles: from iteration ~1000 on, the model keeps changing by ~3e-4 per iteration (flux
fixed to 1e-5), the change sitting on ~0.1% of the voxels, in the line channels. Without momentum the
same map converges, but slowly. Speed restart (Su, Boyd & Candes 2016: t -> 1 whenever
||x_k+1 - x_k|| grows) keeps the acceleration while the model is still moving and drops it once it
starts to oscillate; measured on this problem it is the only variant that keeps converging
(fixed-point residual 3.6e-6 at 12k iterations vs a flat 3e-5 for plain FISTA and Chambolle-Dossal).

Convergence criterion: the standard deviation of the residual, sigma_k = std(d - N x_k), stops
changing between iterations:
    |sigma_k - sigma_k-1| / sigma_k < TOL     for PATIENCE consecutive iterations
(PATIENCE guards against a single chance crossing while the iterate still oscillates; PATIENCE = 1
is the literal two-iteration test).
    stage 0           plain thresholds T0 = tau lambda, until converged as above
    stage s >= 1      reweighted T = T0 / (|W x| / T0 + eps) from the previous stage's model, momentum
                      reset, until converged as above
    stop reweighting  when a stage changes sigma by < RW_TOL (relative), or after MAX_RW stages
STAGE_MAX / TOTAL_MAX cap a stage / the run; the summary says whether a cap was hit. The fixed-point
residual r(x) = ||T(x) - x|| / ||x|| of the forward-backward map T (zero exactly at a fixed point) is
recorded every CHECK iterations as a diagnostic only.

    CUDA_VISIBLE_DEVICES=3 /scratch/alahiry/conda/envs/denoise2d1d/bin/python scripts/make_fista_converged.py [k]
    ... make_fista_converged.py [k] --render      re-render the GIFs from the frame cache only

Writes (k = 2 by default)
    data/ngc3110/compact/fista_converged/model_k<k>.npy        final model, Jy/pixel, 0.02" (git-ignored)
    data/ngc3110/compact/fista_converged/stage_<s>_k<k>.npy     model at the end of each stage, 0.04"
    data/ngc3110/compact/fista_converged/frames_k<k>.npz        frame cache (model, residual, traces)
    data/ngc3110/compact/fista_converged/summary_k<k>.json      stages, iterations, residuals, flux
    figures/fista/converged/<k>_sigma/
        fista_converged.gif, fista_converged_model_hires.gif, frames/
        vs_fista_20k.png      moment 0: earlier fixed 20k-iteration model | converged | difference
"""
import glob
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from PIL import Image
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "simple"))
import gpu_pd as G

args = sys.argv[1:]
RENDER_ONLY = "--render" in args
args = [a for a in args if a != "--render"]
K = float(args[0]) if args else 2.0
TAG = f"{K:g}"

TOL = 1e-7             # relative change of the residual std between two iterations
PATIENCE = 100         # ... below TOL for this many consecutive iterations
CHECK = 100            # diagnostic: fixed-point residual every CHECK iterations (1% overhead)
RW_EPS = 1.0
RW_TOL = 1e-3          # stop reweighting once a stage changes the residual std by less than this (relative)
MAX_RW = 10
STAGE_MAX = 200_000
TOTAL_MAX = 800_000

DAT = os.path.join(REPO, "data/ngc3110/compact/fista_converged")
FIG = os.path.join(REPO, f"figures/fista/converged/{TAG}_sigma")
FRAMES = os.path.join(FIG, "frames")
for d in (DAT, FIG, FRAMES):
    os.makedirs(d, exist_ok=True)
CACHE = os.path.join(DAT, f"frames_k{TAG}.npz")
SUMMARY = os.path.join(DAT, f"summary_k{TAG}.json")

vel = 5073.9 + 25.0 * (np.arange(32) - 15.5)
LINE = [i for i, v in enumerate(vel) if 4750 <= v <= 5280]
NOISE = [i for i in range(32) if i not in LINE]
CELL, BIN, J, L = 0.04, 2, 7, 5
C10 = np.s_[75:325, 75:325]; EXT = [5, -5, -5, 5]
BEAM = (1.418, 0.976)

read = lambda f: np.nan_to_num(fits.getdata(os.path.join(REPO, "data/ngc3110/compact/cube", f)).astype(np.float64)).squeeze()
mom0 = lambda x: (x[LINE].sum(0) * 25).cpu().numpy()           # Jy/pixel km/s (model) or Jy/beam km/s (residual)


def run():
    dirty, psf = G.rebin(read("dirty_cube.fits"), read("dirty_beam_cube.fits"), BIN)
    op = G.Op(psf, dirty); D = G.Dict2D1D(dirty.shape, J, L)
    lam, _, _, _ = G.noise_lambdas(D, op.d, NOISE, k=K)
    tau = 1.0 / G.field_lipschitz(op)
    T0 = tau * lam
    soft = lambda c, t: torch.sign(c) * torch.clamp(c.abs() - t, min=0)
    step = lambda z, T: D.Winv(soft(D.W(z - tau * op.grad(z)), T)).clamp_(min=0)
    rel = lambda a, b: float((a - b).norm() / max(float(b.norm()), 1e-30))

    marks = set(np.unique(np.geomspace(1, TOTAL_MAX, 300).astype(int)).tolist())
    frames, change, fpres, sig_trace, stages = [], [], [], [], []
    rstd = lambda Ax: float((op.d - Ax).double().std())

    def grab(it, x):
        r = -op.grad(x)
        frames.append((it, mom0(x) / CELL ** 2, mom0(r), float(x.sum())))

    print(f"[setup] k = {TAG}, tau = {tau:.3e}, {dirty.shape} cube; residual-std TOL = {TOL:g} x {PATIENCE} iterations, RW_TOL = {RW_TOL:g}, "
          f"MAX_RW = {MAX_RW}, caps {STAGE_MAX:,} / stage, {TOTAL_MAX:,} total", flush=True)
    x = torch.zeros_like(op.d)
    Ax = op.apply(x)
    it, T, t0 = 0, T0, time.time()
    sig_prev_stage = None
    for s in range(MAX_RW + 1):
        if s > 0:
            T = T0 / (D.W(x).abs() / T0 + RW_EPS)
        x_start = x.clone()
        z, Az, t, prev, n_restart, start_it = x.clone(), Ax.clone(), 1.0, np.inf, 0, it
        sig, calm, converged = rstd(Ax), 0, False
        while it - start_it < STAGE_MAX and it < TOTAL_MAX:
            xn = D.Winv(soft(D.W(z - tau * (Az - op.d)), T)).clamp_(min=0)
            Axn = op.apply(xn)
            it += 1
            sig_n = rstd(Axn)
            dsig = abs(sig_n - sig) / sig_n
            calm = calm + 1 if dsig < TOL else 0
            dn = float((xn - x).norm())
            if dn > prev:                                   # speed restart
                t = 1.0; n_restart += 1
            prev = dn
            tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            mom = (t - 1) / tn
            z = xn + mom * (xn - x)
            Az = Axn + mom * (Axn - Ax)                     # N is linear: N z without another FFT
            if it % 25 == 0:
                change.append((it, dn / max(float(xn.norm()), 1e-30)))
                sig_trace.append((it, sig_n, dsig))
            x, Ax, t, sig = xn, Axn, tn, sig_n
            if it in marks:
                grab(it, x)
            if it % CHECK == 0:
                fpres.append((it, rel(step(x, T), x)))
                if (it - start_it) % 5000 == 0:
                    print(f"stage {s}  iter {it:7,d}  residual std {sig:.6e}  change {dsig:.1e}  "
                          f"(fixed-point residual {fpres[-1][1]:.1e})  flux {float(x.sum()):.5f}  "
                          f"restarts {n_restart}  ({time.time() - t0:.0f} s)", flush=True)
            if calm >= PATIENCE:
                converged = True
                break
        moved = rel(x, x_start) if s > 0 else float("nan")
        sig_moved = abs(sig - sig_prev_stage) / sig if sig_prev_stage is not None else float("nan")
        sig_prev_stage = sig
        if not fpres or fpres[-1][0] != it:
            fpres.append((it, rel(step(x, T), x)))
        stages.append(dict(stage=s, reweighted=s > 0, start=start_it, end=it, iterations=it - start_it,
                           converged=converged, residual_std=sig, residual_std_change_last=dsig,
                           residual_std_change_vs_previous_stage=sig_moved, fixed_point_residual=fpres[-1][1],
                           restarts=n_restart, flux=float(x.sum()), moved_vs_previous_stage=moved))
        if not frames or frames[-1][0] != it:
            grab(it, x)
        np.save(os.path.join(DAT, f"stage_{s}_k{TAG}.npy"), x.cpu().numpy().astype(np.float32))
        print(f"== stage {s} {'(reweighted) ' if s else ''}{'converged' if converged else 'HIT CAP'} after "
              f"{it - start_it:,} iterations (total {it:,}): residual std {sig:.6e}, flux {float(x.sum()):.4f} Jy, "
              f"fixed-point residual {fpres[-1][1]:.1e}"
              + (f"; vs previous stage: residual std changed {sig_moved:.2e}, model moved {moved:.2e}" if s else ""), flush=True)
        if it >= TOTAL_MAX or (s > 0 and sig_moved < RW_TOL):
            break
    elapsed = time.time() - t0

    # residual vs pure noise (same diagnostic as make_fista_run.py)
    bpx = np.pi * BEAM[0] * BEAM[1] / (4 * np.log(2)) / CELL ** 2
    nz = G.noise_planes(op.d, NOISE)
    sgn = torch.tensor([(-1) ** i for i in range(len(NOISE) // 2 * 2)], device=G.DEV, dtype=G.DT)[:, None, None]
    noise_rms = float(((nz[: len(sgn)] * sgn).sum(0) * 25 * np.sqrt(len(LINE) / len(sgn)))[C10].std())
    rm0 = -op.grad(x)[LINE].sum(0) * 25
    summary = dict(k=K, criterion="relative change of std(d - N x) between iterations", tol=TOL, patience=PATIENCE,
                   rw_tol=RW_TOL, rw_eps=RW_EPS, total_iterations=it, seconds=round(elapsed),
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                   converged=all(st["converged"] for st in stages) and (len(stages) == 1 or stages[-1]["residual_std_change_vs_previous_stage"] < RW_TOL),
                   stages=stages, model_flux_jy=float(x.sum()), line_flux_jy_kms=float(x[LINE].sum()) * 25,
                   residual_mom0_rms_over_noise=float(rm0[C10].std()) / noise_rms,
                   residual_net_flux_central10_jy_kms=float(rm0[C10].sum()) / bpx)
    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"done: {it:,} iterations in {elapsed:.0f} s; residual moment-0 rms = "
          f"{summary['residual_mom0_rms_over_noise']:.2f} x noise; net residual flux in central 10\" "
          f"{summary['residual_net_flux_central10_jy_kms']:+.2f} Jy km/s; converged = {summary['converged']}", flush=True)

    xm = x.cpu().numpy().astype(np.float64)
    np.save(os.path.join(DAT, f"model_k{TAG}.npy"), G.upsample(xm, BIN).astype(np.float32))
    np.savez_compressed(CACHE, it=np.array([f[0] for f in frames]),
                        model=np.array([f[1] for f in frames], dtype=np.float32),
                        resid=np.array([f[2] for f in frames], dtype=np.float32),
                        flux=np.array([f[3] for f in frames]), change=np.array(change), fpres=np.array(fpres),
                        sig=np.array(sig_trace))


def render():
    c = np.load(CACHE)
    with open(SUMMARY) as f:
        summ = json.load(f)
    frames = list(zip(c["it"].tolist(), c["model"], c["resid"], c["flux"].tolist()))
    n_iter = summ["total_iterations"]
    rw = [st["start"] for st in summ["stages"] if st["reweighted"]]
    fi, fr = c["fpres"].T
    si, _, sd = c["sig"].T
    vm = np.percentile(frames[-1][1][C10], 99.9)
    vr = 5 * frames[-1][2][C10].std()
    stage_of = lambda it: sum(1 for r in rw if it > r)
    label = lambda it: f"k = {TAG}" + (f", reweighting {stage_of(it)}" if stage_of(it) else ", plain L1")
    for f in glob.glob(os.path.join(FRAMES, "*.png")):
        os.remove(f)

    def gif(files, name):
        imgs = [Image.open(q).convert("RGB") for q in files]
        durs = [120] * len(imgs); durs[-1] = 3000
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
        ax[2].loglog(fi, fr, color="0.85", lw=1.0, label=r"fixed-point residual $\|T(x)-x\|/\|x\|$ (diagnostic)")
        ax[2].loglog(si, sd, color="0.75", lw=1.0)
        done = si <= it
        ax[2].loglog(si[done], sd[done], color="#2a78d6", lw=1.4, label=r"$|\Delta\sigma_{\rm res}| / \sigma_{\rm res}$ per iteration")
        ax[2].axhline(TOL, color="#2a9d55", lw=1.2, ls="--", label=f"tolerance {TOL:g}")
        for q in rw:
            ax[2].axvline(q, color="0.85", lw=0.8, zorder=0)
        ax[2].axvline(it, color="#eb6834", lw=1.5)
        ax[2].set_xlabel("iteration"); ax[2].set_ylabel("relative"); ax[2].set_xlim(100, n_iter * 1.05)
        ax[2].set_title("convergence" + (" (thin grey lines: reweighting)" if rw else ""))
        ax[2].legend(loc="upper right", fontsize=10, framealpha=0.9)
        fig.suptitle(f"NGC 3110 compact, 2D-1D FISTA (speed restart) to convergence  --  iteration {it:,}  |  {label(it)}  --  model flux {flux:.1f} Jy", fontsize=14)
        q = os.path.join(FRAMES, f"frame_{k:03d}.png"); fig.savefig(q, dpi=70); plt.close(fig); paths.append(q)
    gif(paths, "fista_converged.gif")

    plt.rcParams.update({"font.size": 15})
    hi = []
    for k, (it, m, r, flux) in enumerate(frames):
        fig = plt.figure(figsize=(9.2, 8.6)); ax = fig.add_axes([0.1, 0.08, 0.72, 0.8]); cax = fig.add_axes([0.84, 0.08, 0.03, 0.8])
        im = ax.imshow(m[C10], origin="lower", extent=EXT, cmap="inferno", vmin=0, vmax=vm, interpolation="bicubic")
        cb = fig.colorbar(im, cax=cax); cb.set_label(r"Jy km s$^{-1}$ arcsec$^{-2}$")
        ax.set_xlabel(r'$\Delta$RA ["]'); ax.set_ylabel(r'$\Delta$Dec ["]')
        ax.text(0.03, 0.97, f"iteration {it:,}", transform=ax.transAxes, color="w", fontsize=20, fontweight="bold", va="top")
        ax.text(0.03, 0.905, "FISTA, " + label(it), transform=ax.transAxes, color="w", fontsize=14, fontweight="bold", va="top")
        ax.text(0.97, 0.03, f"model flux {flux:.1f} Jy", transform=ax.transAxes, color="w", fontsize=14, fontweight="bold", ha="right")
        fig.suptitle("NGC 3110 compact: raw 2D-1D FISTA model, run to convergence", fontsize=16, y=0.955)
        q = os.path.join(FRAMES, f"model_hires_{k:03d}.png"); fig.savefig(q, dpi=130); plt.close(fig); hi.append(q)
    gif(hi, "fista_converged_model_hires.gif")

    # against the earlier fixed-length run (make_fista_run.py compact, same k), when it exists
    old = os.path.join(REPO, f"data/ngc3110/compact/fista/model_k{TAG}_20k.npy")
    if os.path.exists(old):
        a = np.load(old).astype(np.float64)
        b = np.load(os.path.join(DAT, f"model_k{TAG}.npy")).astype(np.float64)
        crop = np.s_[150:650, 150:650]                       # central 10" on the 0.02" grid
        ma = a[LINE].sum(0)[crop] * 25 / 0.02 ** 2; mb = b[LINE].sum(0)[crop] * 25 / 0.02 ** 2
        d = mb - ma
        relcube = np.linalg.norm(b - a) / np.linalg.norm(b)
        print(f"vs fixed 20k-iteration model: cube relative difference {relcube:.3e}; moment-0 relative "
              f"difference {np.linalg.norm(d) / np.linalg.norm(mb):.3e}; line flux {a[LINE].sum() * 25:.2f} -> "
              f"{b[LINE].sum() * 25:.2f} Jy km/s", flush=True)
        summ["vs_fista_20k"] = dict(cube_rel_diff=float(relcube), mom0_rel_diff=float(np.linalg.norm(d) / np.linalg.norm(mb)),
                                    line_flux_20k=float(a[LINE].sum() * 25), line_flux_converged=float(b[LINE].sum() * 25))
        with open(SUMMARY, "w") as f:
            json.dump(summ, f, indent=1)
        plt.rcParams.update({"font.size": 13})
        vmx = np.percentile(mb, 99.9); dv = np.percentile(np.abs(d), 99.9)
        fig, ax = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
        for a_, img, ttl, cm, lo, hi_ in ((ax[0], ma, "fixed 20,000 iterations (earlier run)", "inferno", 0, vmx),
                                          (ax[1], mb, f"converged ({n_iter:,} iterations)", "inferno", 0, vmx),
                                          (ax[2], d, "converged $-$ 20k", "RdBu_r", -dv, dv)):
            im = a_.imshow(img, origin="lower", extent=EXT, cmap=cm, vmin=lo, vmax=hi_, interpolation="nearest")
            a_.set_title(ttl); a_.set_xlabel(r'$\Delta$RA ["]'); a_.set_ylabel(r'$\Delta$Dec ["]')
            plt.colorbar(im, ax=a_, shrink=0.85, label=r"Jy km s$^{-1}$ arcsec$^{-2}$")
        fig.suptitle(f"NGC 3110 compact, raw FISTA model moment 0, k = {TAG}: relative difference "
                     f"{np.linalg.norm(d) / np.linalg.norm(mb):.1%} (moment 0), {relcube:.1%} (cube)", fontsize=14)
        q = os.path.join(FIG, "vs_fista_20k.png"); fig.savefig(q, dpi=110); plt.close(fig)
        print("wrote", q, flush=True)


if __name__ == "__main__":
    if not RENDER_ONLY:
        run()
    render()
