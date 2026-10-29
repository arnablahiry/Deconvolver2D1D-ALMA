#!/usr/bin/env python
"""
Convergence history of the compact 2D-1D deconvolution with outer major cycles, recorded every EVERY iterations:
the initial solve from x = 0 (pdtau, k = 5, N0 iterations, reweighting at 6000, 8000, ...), then CYCLES major cycles
(scripts/experiments/major_cycle_loop.py: CASA true residual, corrected data term d' = N x + Rt, re-solve warm-started
for N1 iterations with the same reweighting schedule restarted). Recorded: the line flux of the model and the rms of the
solver's residual (its data term minus N x) in the line channels; at every major cycle, the TRUE residual rms from CASA.

Writes results/<T>/compact/major_cycle_convergence/{history.json, convergence.png}.

    CUDA_VISIBLE_DEVICES=1 /scratch/alahiry/conda/envs/deconvolver2d1d/bin/python scripts/experiments/major_cycle_convergence.py \
        [NGC3110] [--n0 20000] [--n1 8000] [--cycles 3] [--every 20]
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("simple", "scripts/experiments"):
    sys.path.insert(0, os.path.join(REPO, p))
RES = os.path.join(REPO, "results")
ap = argparse.ArgumentParser()
ap.add_argument("target", nargs="?", default="NGC3110")
ap.add_argument("--n0", type=int, default=20000); ap.add_argument("--n1", type=int, default=8000)
ap.add_argument("--cycles", type=int, default=3); ap.add_argument("--every", type=int, default=20)
ap.add_argument("--plot-only", action="store_true")
ap.add_argument("--carry", action="store_true", help="carry the reweighted thresholds and dual variables from one cycle "
                "into the next (reweighting then every 2000 iterations from 2000) instead of restarting from lambda")
ap.add_argument("--eps", type=float, default=None, help="reweighting floor epsilon (default: common.RW's, 1)")
ap.add_argument("--rw-cycles", type=int, default=None, help="with --carry: keep reweighting only in the first N major "
                "cycles, then freeze the carried thresholds")
ap.add_argument("--tag", default="", help="suffix of the output folder")
a = ap.parse_args()
T = a.target; R = os.path.join(RES, T)
OUT = os.path.join(R, "compact", "major_cycle_convergence" + ("_carry" if a.carry else "") + a.tag); os.makedirs(OUT, exist_ok=True)
GC = json.load(open(os.path.join(R, "config.json")))
DV = GC["width_kms"]
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

if not a.plot_only:
    import casatools
    from casatasks import tclean
    from casa_logs import redirect_casa_logs
    redirect_casa_logs(f"major_cycle_convergence_{T}{a.tag}.log")
    cc = GC["configs"]["compact"]
    KW = dict(vis=cc["ms"], field=cc["field"] if len(set(cc["field"])) > 1 else cc["field"][0], spw=cc["spw"],
              datacolumn="data", imsize=GC["imsize"], cell=f"{GC['cell_arcsec']}arcsec", phasecenter=GC["phasecenter"],
              specmode="cube", start=f"{GC['start_kms']}km/s", width=f"{GC['width_kms']}km/s", nchan=GC["nchan"],
              restfreq=f"{GC['restfreq_ghz']}GHz", outframe="LSRK", veltype="radio", gridder="standard",
              weighting="natural", niter=0, restoration=False, pbcor=False)
    ia = casatools.image(); CS = None

    def casa_residual(x):
        global CS
        get = lambda p: np.asarray((ia.open(p), ia.getchunk(), ia.close())[1], dtype=np.float64)[:, :, 0, :].transpose(2, 1, 0)
        if CS is None:
            base = os.path.join(OUT, "_grid")
            for p in glob.glob(base + ".*"):
                shutil.rmtree(p)
            tclean(imagename=base, **KW); ia.open(base + ".residual"); CS = ia.coordsys().torecord(); ia.close()
        mpath = os.path.join(OUT, "_model.image")
        if os.path.exists(mpath):
            shutil.rmtree(mpath)
        ia.fromarray(outfile=mpath, pixels=x.transpose(2, 1, 0)[:, :, None, :].astype(np.float32), csys=CS, overwrite=True)
        ia.close()
        base = os.path.join(OUT, "_resid")
        for p in glob.glob(base + ".*"):
            shutil.rmtree(p)
        tclean(imagename=base, startmodel=mpath, **KW)
        return get(base + ".residual")

    import common
    common.DATA = RES
    for c, b in GC["beams_natural"].items():
        common.BEAMS[(T, c)] = tuple(b)
    G = common.G
    P = common.problem(T, "compact", k=5.0)
    L, NC = P.line, P.noise
    M = np.load(os.path.join(R, "extended/msclean_finished/mask_final.npy")).astype(bool)
    H = dict(iter=[], flux=[], resid_rms=[], reweights=[], major_cycles=[], true=[], segments=[])
    offset = 0

    def recorder(off):
        def cb(it, x, g):
            if it % a.every == 0:
                H["iter"].append(off + it)
                H["flux"].append(float(x[L].sum()) * DV)
                H["resid_rms"].append(1e3 * float(g[L].std()))
        return cb

    def true_point(x_fine, at):
        Rt = casa_residual(x_fine)
        sig = float(np.std(Rt[NC]))
        t = dict(iter=at, field_sigma=float(np.sqrt(np.mean(Rt[L] ** 2))) / sig,
                 mask_sigma=float(np.sqrt(np.mean(Rt[L][M[L]] ** 2))) / sig, rms_mJy=1e3 * float(np.sqrt(np.mean(Rt[L] ** 2))))
        H["true"].append(t); log(f"  true residual at {at}: field {t['field_sigma']:.2f} sigma, bright {t['mask_sigma']:.2f} sigma")
        return Rt

    rw = lambda off, n: [off + i for i in range(common.RW["start"], n, common.RW["every"])]
    # initial solve from zero
    log(f"initial solve, {a.n0} iterations"); H["segments"].append(dict(start=0, n=a.n0, kind="initial"))
    ST = {}

    if a.eps is not None:
        common.RW["eps"] = a.eps
    H["config"] = dict(eps=common.RW["eps"], carry=a.carry, rw_cycles=a.rw_cycles, n0=a.n0, n1=a.n1)

    def pdtau(Pp, n, x0=None, cb=None, carry=None, reweight=True):
        """common.solve's pdtau, optionally starting from carried thresholds / duals (reweighting then from 2000)"""
        rwd = dict(common.RW, ref=Pp.tau * Pp.lam)
        if carry:
            rwd["start"] = common.RW["every"]
        if not reweight:
            rwd = None
        return G.solve_pd(Pp.op, Pp.D, Pp.lam, n_iter=n, beta=Pp.beta, x0=x0, support=Pp.support, reweight=rwd,
                          print_every=4000, log=lambda m: log("  " + m), callback=cb,
                          bound0=carry["bound"] if carry else None, u0=carry["u"] if carry else None, state=ST)[0]

    x = pdtau(P, a.n0, cb=recorder(0))
    H["reweights"] += rw(0, a.n0); offset = a.n0
    x_fine = common.to_saved(P, x).astype(np.float64)
    for k in range(1, a.cycles + 1):
        Rt = true_point(x_fine, offset)
        H["major_cycles"].append(offset)
        b_ = P.bin; ny_, nx_ = Rt.shape[1] // b_ * b_, Rt.shape[2] // b_ * b_
        Rt_c = Rt[:, :ny_, :nx_].reshape(Rt.shape[0], ny_ // b_, b_, nx_ // b_, b_).mean((2, 4))
        xc = G.to_t(common.to_native(P, x_fine))
        Pk = common.problem(T, "compact", k=5.0, data=P.op.apply(xc).cpu().numpy().astype(np.float64) + Rt_c)
        log(f"major cycle {k}: re-solve {a.n1} iterations from iteration {offset}")
        H["segments"].append(dict(start=offset, n=a.n1, kind=f"major cycle {k}"))
        rwk = a.rw_cycles is None or k <= a.rw_cycles
        x = pdtau(Pk, a.n1, x0=xc, cb=recorder(offset), carry=dict(ST) if a.carry else None, reweight=rwk)
        if rwk:
            H["reweights"] += ([offset + i for i in range(common.RW["every"], a.n1, common.RW["every"])] if a.carry
                               else rw(offset, a.n1))
        offset += a.n1
        x_fine = common.to_saved(Pk, x).astype(np.float64)
        np.save(os.path.join(OUT, f"model_cycle{k}.npy"), x_fine.astype(np.float32))
        json.dump(H, open(os.path.join(OUT, "history.json"), "w"))
        del Pk
    true_point(x_fine, offset)
    np.save(os.path.join(OUT, "model.npy"), x_fine.astype(np.float32))
    json.dump(H, open(os.path.join(OUT, "history.json"), "w"))

# ======================================================================== the plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = json.load(open(os.path.join(OUT, "history.json")))
BLUE, RED, MUTED = "#2a6fdb", "#c0392b", "#8a8a8a"
plt.rcParams.update({"font.family": "serif", "font.size": 14, "xtick.direction": "in", "ytick.direction": "in"})
it = np.array(H["iter"])
fig, ax = plt.subplots(2, 1, figsize=(26, 10), sharex=True, constrained_layout=True)
ax[0].plot(it, H["resid_rms"], color=BLUE, lw=1.4, label="solver residual rms (its data term − N x), line channels")
ax[0].plot([t["iter"] for t in H["true"]], [t["rms_mJy"] for t in H["true"]], "o", color=RED, ms=10, mec="white", mew=1.5,
           label="TRUE residual rms (CASA, from the visibilities)", zorder=5)
ax[0].set_yscale("log"); ax[0].set_ylabel("residual rms [mJy beam$^{-1}$]", fontsize=16)
ax[1].plot(it, H["flux"], color=BLUE, lw=1.4)
ax[1].set_ylabel(r"model line flux [Jy km s$^{-1}$]", fontsize=16)
ax[1].set_xlabel("iteration (cumulative)", fontsize=16)
for a_ in ax:
    for r in H["reweights"]:
        a_.axvline(r, color=MUTED, lw=1, ls=":")
    for m in H["major_cycles"]:
        a_.axvline(m, color=RED, lw=2.2)
    a_.tick_params(which="both", top=True, right=True, labelsize=14)
    a_.set_xlim(0, it.max())
ax[0].plot([], [], color=MUTED, ls=":", label="reweighting"); ax[0].plot([], [], color=RED, lw=2.2, label="major cycle")
ax[0].legend(frameon=False, fontsize=13, loc="upper right", ncol=2)
for t in H["true"]:
    ax[0].annotate(f"{t['mask_sigma']:.1f}σ bright\n{t['field_sigma']:.1f}σ field", (t["iter"], t["rms_mJy"]), xytext=(8, 10),
                   textcoords="offset points", fontsize=12, color=RED)
cfg = H.get("config", {})
sub = ("thresholds carried across cycles" if a.carry else "thresholds reset each cycle") + f", ε = {cfg.get('eps', 1):g}" + \
      (f", reweighting frozen after major cycle {a.rw_cycles}" if a.rw_cycles is not None else "")
fig.suptitle(f"{GC['label']} compact: convergence with reweighting and major cycles ({sub})", fontsize=20, fontweight="bold")
fig.savefig(os.path.join(OUT, "convergence.png"), dpi=70)
print("saved", os.path.join(OUT, "convergence.png"))
