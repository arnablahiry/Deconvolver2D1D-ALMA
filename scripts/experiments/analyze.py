#!/usr/bin/env python
"""
Analysis of all experiments (scripts/experiments/run_queue.py) -> figures/experiments/<nn>_*/ and
data/experiments/results.json. Every metric is computed here, the same way for every model:

  sim error      ||G*(x - truth)|| / ||G*truth||, G = extended clean beam, total and per starlet scale;
                 flux inside / outside the true emission (G*truth > 1% of its peak)
  xval           compact model -> extended PSF vs the extended dirty cube (line channels), after a
                 least-squares flux scale a: fraction of the extended signal energy NOT explained,
                 per starlet scale (lower = better)
  bias           residual (d - N x) summed over the emission region (model * own clean beam > 10% of
                 peak), as a fraction of the predicted emission there: > 0 = model under-predicts
                 (L1 shrinkage), < 0 = over-predicts
  resid/noise    std of the residual moment 0 (line channels) / std of a pure-noise moment 0 with the
                 same number of channels (+- sum of line-free channels)
  lf flux        model flux in the line-free channels (fitted noise / pedestal)

    CUDA_VISIBLE_DEVICES=0 /scratch/alahiry/conda/envs/denoise2d1d/bin/python scripts/experiments/analyze.py
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal.windows import tukey

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import G, REPO, DATA, EXP_DATA, EXP_FIG, BEAMS, read, problem, to_native, line_channels, noise_channels, vel  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 12, "axes.titlesize": 12, "xtick.direction": "in", "ytick.direction": "in"})
COL = {"pd": "#2a78d6", "pdtau": "#1baf7a", "fista": "#eb6834"}
NAME = {"pd": "primal-dual", "pdtau": "primal-dual, tau-lambda reweighting", "fista": "FISTA"}
R = {}                                               # all results -> results.json
fig_dir = lambda n: os.makedirs(os.path.join(EXP_FIG, n), exist_ok=True) or os.path.join(EXP_FIG, n)
ld = lambda p: np.load(os.path.join(REPO, p) if not os.path.isabs(p) else p).astype(np.float64)


# ---------------------------------------------------------------- shared tools
def beam_ft(shape, beam, cell):
    """peak-1 elliptical Gaussian kernel FFT on a 2x padded grid (torch)"""
    from restore import gaussian_beam
    ny, nx = shape
    k = gaussian_beam((2 * ny, 2 * nx), cell, *beam, center=(ny, nx))
    return torch.fft.rfft2(G.to_t(np.roll(k, (-ny, -nx), axis=(0, 1))))


def conv(x, kft):
    x = G.to_t(x) if not torch.is_tensor(x) else x
    nz, ny, nx = x.shape
    big = torch.zeros((nz, 2 * ny, 2 * nx), device=G.DEV, dtype=G.DT); big[:, :ny, :nx] = x
    return torch.fft.irfft2(torch.fft.rfft2(big) * kft, s=(2 * ny, 2 * nx))[:, :ny, :nx]


def mom0(c, line):
    return c[line].sum(0) * 25.0


class Cfg:
    """real-data problem of one configuration + the noise moment 0, for bias / residual metrics"""
    def __init__(self, galaxy, cfg, grid=None, mask=None):
        self.P = P = problem(galaxy, cfg, grid, k=2.0, mask=mask)
        self.galaxy, self.cfg = galaxy, cfg
        self.line = P.line
        npl = G.noise_planes(P.op.d, P.noise)
        n = len(npl) // 2 * 2
        sg = torch.tensor([(-1) ** i for i in range(n)], device=G.DEV, dtype=G.DT)[:, None, None]
        self.noise_m0 = ((npl[:n] * sg).sum(0) * 25 * np.sqrt(len(self.line) / n))
        self.kft = beam_ft(P.op.d.shape[1:], BEAMS[(galaxy, cfg)], P.cell)
        self.bpx = np.pi * BEAMS[(galaxy, cfg)][0] * BEAMS[(galaxy, cfg)][1] / (4 * np.log(2)) / P.cell ** 2

    def metrics(self, x_saved, native=False):
        P = self.P
        x = G.to_t(x_saved if native else to_native(P, x_saved))
        pred = P.op.apply(x); r = P.op.d - pred
        rm = mom0(r, self.line); pm = mom0(pred, self.line)
        region = mom0(conv(x, self.kft), self.line)
        region = region > 0.1 * region.max()
        lf = [c for c in range(32) if c not in self.line]
        return dict(bias=float(rm[region].sum() / pm[region].sum()),
                    resid_over_noise=float(rm.std() / self.noise_m0.std()),
                    flux=float(x.sum()), lf_flux=float(x[lf].sum()), lf_frac=float(x[lf].sum() / x.sum())), (x, r)


class XVal:
    """compact model (0.02") predicting the independent extended dirty cube of the same galaxy"""
    def __init__(self, galaxy="ngc3110", J=8):
        self.op = G.Op(read(galaxy, "extended", "dirty_beam_cube.fits"), read(galaxy, "extended", "dirty_cube.fits"))
        self.line, self.J = line_channels(galaxy), J
        self.d = self.op.d[self.line]; self.dp = G.starlet(self.d, J)
        self.den = [float((self.dp[j] ** 2).sum()) for j in range(J + 1)]

    def __call__(self, x_fine):
        p = self.op.apply(G.to_t(x_fine))[self.line]
        a = float((p * self.d).sum() / (p * p).sum())
        rp = G.starlet(self.d - a * p, self.J)
        return dict(a=a, per_scale=[float((rp[j] ** 2).sum()) / self.den[j] for j in range(self.J + 1)],
                    total=float(((self.d - a * p) ** 2).sum() / (self.d ** 2).sum()))


def scale_labels(cell, J):
    return [f"{2**j * cell:.2f}-{2**(j+1) * cell:.2f}\"" for j in range(J)] + ["coarse"]


# ---------------------------------------------------------------- 01 simulations
def exp01():
    out = fig_dir("01_simulations"); res = {}
    for cfg, cell, J, ks in (("compact", 0.04, 7, (2, 3, 4, 5)), ("extended", 0.02, 8, (3,))):
        truth = ld(f"{EXP_DATA}/01_simulations/inputs/{cfg}_sim_truth{'_0.04arcsec' if cfg == 'compact' else ''}.npy")
        kft = beam_ft(truth.shape[1:], BEAMS[("ngc3110", "extended")], cell)
        tb = conv(truth, kft); supp = tb > 0.01 * tb.max(); tp = G.starlet(tb, J)
        res[cfg] = dict(truth_flux=float(truth.sum()), truth_flux_in=float(G.to_t(truth)[supp].sum()))
        for m in ("pd", "pdtau", "fista"):
            for k in ks:
                for kind, path in (("solve", f"{EXP_DATA}/01_simulations/{cfg}/{m}_k{k}.npy"),
                                   ("debiased", f"{EXP_DATA}/03_debiasing/sim/{cfg}/{m}_k{k}_debiased.npy")):
                    x = G.to_t(ld(path)); e = conv(x, kft) - tb; ep = G.starlet(e, J)
                    res[cfg][f"{m}_k{k}_{kind}"] = dict(
                        err=float(e.norm() / tb.norm()), per_scale=[float(ep[j].norm() / tp[j].norm()) for j in range(J + 1)],
                        flux=float(x.sum()), flux_in=float(x[supp].sum()), flux_out=float(x[~supp].sum()))
    R["01_simulations"] = res
    # figure: error vs k (compact), bars (extended), per-scale (compact k = 2)
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.2), constrained_layout=True)
    c = res["compact"]
    for m in ("pd", "pdtau", "fista"):
        ax[0].plot([2, 3, 4, 5], [c[f"{m}_k{k}_solve"]["err"] for k in (2, 3, 4, 5)], "o-", color=COL[m], lw=2, label=NAME[m])
        ax[0].plot([2, 3, 4, 5], [c[f"{m}_k{k}_debiased"]["err"] for k in (2, 3, 4, 5)], "s--", color=COL[m], lw=1.4, label=f"{NAME[m]}, debiased")
    ax[0].set_xlabel("k (threshold in noise sigmas)"); ax[0].set_ylabel("error vs truth at the extended beam"); ax[0].set_xticks([2, 3, 4, 5])
    ax[0].set_title("compact simulation: error vs k (lower = better)"); ax[0].legend(frameon=False, fontsize=8.5)
    e = res["extended"]; labels = [f"{NAME[m]}" for m in ("pd", "pdtau", "fista")]; xx = np.arange(3)
    ax[1].bar(xx - 0.2, [e[f"{m}_k3_solve"]["err"] for m in ("pd", "pdtau", "fista")], 0.38, color=[COL[m] for m in ("pd", "pdtau", "fista")], label="solve")
    ax[1].bar(xx + 0.2, [e[f"{m}_k3_debiased"]["err"] for m in ("pd", "pdtau", "fista")], 0.38, color=[COL[m] for m in ("pd", "pdtau", "fista")], alpha=0.45, hatch="//", label="debiased")
    for i, m in enumerate(("pd", "pdtau", "fista")):
        ax[1].text(i - 0.2, e[f"{m}_k3_solve"]["err"], f"{e[f'{m}_k3_solve']['err']:.3f}", ha="center", va="bottom", fontsize=9)
        ax[1].text(i + 0.2, e[f"{m}_k3_debiased"]["err"], f"{e[f'{m}_k3_debiased']['err']:.3f}", ha="center", va="bottom", fontsize=9)
    ax[1].set_xticks(xx); ax[1].set_xticklabels(labels, fontsize=9); ax[1].set_ylabel("error vs truth at the extended beam")
    ax[1].set_title("extended simulation, k = 3 (support mask)"); ax[1].legend(frameon=False)
    for m in ("pd", "pdtau", "fista"):
        ax[2].plot(range(8), c[f"{m}_k2_solve"]["per_scale"], "o-", color=COL[m], lw=2, label=NAME[m])
    ax[2].set_xticks(range(8)); ax[2].set_xticklabels(scale_labels(0.04, 7), rotation=35, fontsize=9)
    ax[2].set_ylabel("error / truth, per scale"); ax[2].set_title("compact simulation, k = 2: error per spatial scale"); ax[2].legend(frameon=False)
    for a in ax:
        a.grid(color="0.9")
    fig.savefig(f"{out}/01_sim_errors.png", dpi=80); plt.close(fig)
    # maps: truth and the three methods (compact k = 2, extended k = 3) at the extended beam
    fig, ax = plt.subplots(2, 4, figsize=(21, 10.5), constrained_layout=True)
    for row, (cfg, cell, k, sl) in enumerate((("compact", 0.04, 2, np.s_[75:325, 75:325]), ("extended", 0.02, 3, np.s_[150:650, 150:650]))):
        truth = ld(f"{EXP_DATA}/01_simulations/inputs/{cfg}_sim_truth{'_0.04arcsec' if cfg == 'compact' else ''}.npy")
        kft = beam_ft(truth.shape[1:], BEAMS[("ngc3110", "extended")], cell)
        imgs = [("truth", truth)] + [(f"{NAME[m]} (err {res[cfg][f'{m}_k{k}_solve']['err']:.3f})", ld(f"{EXP_DATA}/01_simulations/{cfg}/{m}_k{k}.npy")) for m in ("pd", "pdtau", "fista")]
        vmax = None
        for a, (t, x) in zip(ax[row], imgs):
            m0 = conv(x, kft).sum(0).cpu().numpy()[sl]; vmax = vmax or np.percentile(m0, 99.9)
            a.imshow(m0, origin="lower", cmap="inferno", vmin=0, vmax=vmax, extent=[5, -5, -5, 5]); a.set_title(f"{cfg} sim, k = {k}: {t}")
    fig.suptitle("Simulations: moment 0 at the extended clean beam"); fig.savefig(f"{out}/01_sim_maps.png", dpi=70); plt.close(fig)


# ---------------------------------------------------------------- real-model registry
def real_models():
    M = []                                           # (label, method, k, kind, grid, path)
    add = lambda *t: M.append(t) if os.path.exists(os.path.join(REPO, t[-1])) else print("missing", t[-1])
    add("pd k2", "pd", 2, "solve", "coarse", "data/ngc3110/compact/pd/model.npy")
    add("fista k2", "fista", 2, "solve", "coarse", "data/ngc3110/compact/fista/model_k2_20k.npy")
    for k in (3, 4, 5):
        for m in ("pd", "fista"):
            add(f"{m} k{k}", m, k, "solve", "coarse", f"data/ngc3110/compact/{m}/model_k{k}_20k.npy")
            add(f"{m} k{k} 0.02\"", m, k, "solve", "fine", f"data/ngc3110/compact/{m}/0.02arcsec/model_k{k}_20k.npy")
    for k in (2, 3, 4, 5):
        add(f"pdtau k{k}", "pdtau", k, "solve", "coarse", f"data/experiments/04_pd_tau_reweighting/compact/pdtau_k{k}.npy")
        for m in ("pd", "pdtau", "fista"):
            add(f"{m} k{k} debiased", m, k, "debiased", "coarse", f"data/experiments/03_debiasing/real/compact/{m}_k{k}_debiased.npy")
    for k in (2, 3):
        add(f"pd k{k} 32\" field", "pd", k, "large", "large", f"data/experiments/05_large_field/pd_k{k}.npy")
    return M


def large_to_fine(x):
    """32" field model at 0.04" -> its central 16" at 0.02" (flux-conserving)"""
    return G.upsample(x[:, 200:600, 200:600], 2)


# ---------------------------------------------------------------- 02 cross-validation + compact bias table
def exp02():
    out = fig_dir("02_crossvalidation"); xv = XVal(); cc = Cfg("ngc3110", "compact"); res = {}
    for label, m, k, kind, grid, path in real_models():
        x = ld(path); xf = large_to_fine(x) if grid == "large" else x
        r = dict(method=m, k=k, kind=kind, grid=grid, xval=xv(xf))
        r.update(cc.metrics(xf)[0])
        res[label] = r
        print(f"02 {label:<24} xval 0.16-0.32\" {r['xval']['per_scale'][3]:.3f}  0.08-0.16\" {r['xval']['per_scale'][2]:.3f}  "
              f"total {r['xval']['total']:.3f}  bias {r['bias']:+.4f}  resid/noise {r['resid_over_noise']:.2f}  flux {r['flux']:.2f}", flush=True)
    R["02_crossvalidation"] = res
    with open(f"{out}/xval_table.md", "w") as f:
        f.write("| model | 0.02-0.04\" | 0.04-0.08\" | 0.08-0.16\" | **0.16-0.32\"** | 0.32-0.64\" | 0.64-1.28\" | total | bias | resid/noise | flux [Jy] | line-free flux [Jy] |\n|" + "---|" * 12 + "\n")
        for label, r in res.items():
            p = r["xval"]["per_scale"]
            f.write(f"| {label} | {p[0]:.3f} | {p[1]:.3f} | {p[2]:.3f} | **{p[3]:.3f}** | {p[4]:.3f} | {p[5]:.3f} | {r['xval']['total']:.3f} | "
                    f"{r['bias']:+.4f} | {r['resid_over_noise']:.2f} | {r['flux']:.2f} | {r['lf_flux']:.3f} |\n")
    # figure: xval at the two key scales, and bias, per model
    sel = [l for l in res if res[l]["kind"] in ("solve", "large") and res[l]["grid"] in ("coarse", "large")]
    fig, ax = plt.subplots(1, 3, figsize=(21, 5.6), constrained_layout=True)
    for a, j, t in ((ax[0], 3, "0.16-0.32\" (extended-beam scale)"), (ax[1], 2, "0.08-0.16\"")):
        for m in ("pd", "pdtau", "fista"):
            ks = [res[l]["k"] for l in res if res[l]["method"] == m and res[l]["kind"] == "solve" and res[l]["grid"] == "coarse"]
            ys = [res[l]["xval"]["per_scale"][j] for l in res if res[l]["method"] == m and res[l]["kind"] == "solve" and res[l]["grid"] == "coarse"]
            yd = [res[l]["xval"]["per_scale"][j] for l in res if res[l]["method"] == m and res[l]["kind"] == "debiased"]
            a.plot(ks, ys, "o-", color=COL[m], lw=2, label=NAME[m])
            if yd:
                a.plot(sorted(res[l]["k"] for l in res if res[l]["method"] == m and res[l]["kind"] == "debiased"), yd, "s--", color=COL[m], lw=1.3, label=f"{NAME[m]}, debiased")
            yf = [(res[l]["k"], res[l]["xval"]["per_scale"][j]) for l in res if res[l]["method"] == m and res[l]["grid"] == "fine"]
            if yf:
                a.plot(*zip(*sorted(yf)), "^:", color=COL[m], lw=1.3, label=f"{NAME[m]}, 0.02\" grid")
        for l in res:
            if res[l]["grid"] == "large":
                a.plot(res[l]["k"], res[l]["xval"]["per_scale"][j], "k*", ms=13, label="pd, 32\" field" if res[l]["k"] == 2 else None)
        a.set_xlabel("k"); a.set_xticks([2, 3, 4, 5]); a.set_ylabel("unexplained extended signal energy"); a.set_title(f"cross-validation at {t} (lower = better)")
        a.grid(color="0.9")
    ax[0].legend(frameon=False, fontsize=8)
    for m in ("pd", "pdtau", "fista"):
        ks = sorted(res[l]["k"] for l in res if res[l]["method"] == m and res[l]["kind"] == "solve" and res[l]["grid"] == "coarse")
        ax[2].plot(ks, [res[f"{m} k{k}"]["bias"] * 100 for k in ks], "o-", color=COL[m], lw=2, label=NAME[m])
        kd = sorted(res[l]["k"] for l in res if res[l]["method"] == m and res[l]["kind"] == "debiased")
        if kd:
            ax[2].plot(kd, [res[f"{m} k{k} debiased"]["bias"] * 100 for k in kd], "s--", color=COL[m], lw=1.3, label=f"{NAME[m]}, debiased")
    ax[2].axhline(0, color="0.5", lw=1); ax[2].set_xlabel("k"); ax[2].set_xticks([2, 3, 4, 5])
    ax[2].set_ylabel("residual in emission / predicted emission [%]"); ax[2].set_title("compact L1 bias (> 0 = model under-predicts)"); ax[2].legend(frameon=False, fontsize=8); ax[2].grid(color="0.9")
    fig.savefig(f"{out}/02_xval_and_bias.png", dpi=80); plt.close(fig)


# ---------------------------------------------------------------- 03 debiasing + 04 pd / pdtau / fista on real extended
def exp03_04():
    out3, out4 = fig_dir("03_debiasing"), fig_dir("04_pd_tau_reweighting")
    ce = Cfg("ngc3110", "extended"); cc = Cfg("ngc3110", "compact"); res = {}
    ext = {}
    for m in ("pd", "fista", "pdtau"):
        for k in (3, 4, 5):
            base = f"data/ngc3110/extended/{m}/model_k{k}_20k.npy" if m != "pdtau" else f"data/experiments/04_pd_tau_reweighting/extended/pdtau_k{k}.npy"
            for kind, p in (("solve", base), ("debiased", f"data/experiments/03_debiasing/real/extended/{m}_k{k}_debiased.npy")):
                met, (x, r) = ce.metrics(ld(p))
                res[f"extended {m} k{k} {kind}"] = met
                if k == 3:
                    ext[(m, kind)] = (mom0(x, ce.line).cpu().numpy() / 0.02 ** 2, mom0(r, ce.line).cpu().numpy())
                print(f"03/04 extended {m} k{k} {kind}: bias {met['bias']:+.4f} resid/noise {met['resid_over_noise']:.2f} flux {met['flux']:.2f}", flush=True)
    R["03_04_extended"] = res
    sl = np.s_[150:650, 150:650]
    # 04: extended k = 3, three methods: model and residual
    fig, ax = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    vm = np.percentile(ext[("pd", "solve")][0][sl], 99.9); vr = 4 * ext[("pd", "solve")][1][sl].std()
    for i, m in enumerate(("pd", "pdtau", "fista")):
        mm, rr = ext[(m, "solve")]; met = res[f"extended {m} k3 solve"]
        ax[0, i].imshow(mm[sl], origin="lower", cmap="inferno", vmin=0, vmax=vm, extent=[5, -5, -5, 5]); ax[0, i].set_title(f"extended, k = 3: {NAME[m]}\nraw model moment 0; flux {met['flux']:.1f} Jy")
        im = ax[1, i].imshow(rr[sl], origin="lower", cmap="RdBu_r", vmin=-vr, vmax=vr, extent=[5, -5, -5, 5])
        ax[1, i].set_title(f"residual moment 0: bias {met['bias'] * 100:+.1f}%, rms {met['resid_over_noise']:.2f} x noise")
    plt.colorbar(im, ax=ax[1, :], shrink=0.8, label="Jy/beam km/s")
    fig.savefig(f"{out4}/04_extended_k3_methods.png", dpi=70); plt.close(fig)
    # 03: before / after debiasing (extended k = 3, all methods) -- residual maps
    fig, ax = plt.subplots(3, 4, figsize=(23, 16), constrained_layout=True)
    for i, m in enumerate(("pd", "pdtau", "fista")):
        for j, kind in enumerate(("solve", "debiased")):
            mm, rr = ext[(m, kind)]; met = res[f"extended {m} k3 {kind}"]
            ax[i, 2 * j].imshow(mm[sl], origin="lower", cmap="inferno", vmin=0, vmax=vm, extent=[5, -5, -5, 5])
            ax[i, 2 * j].set_title(f"{NAME[m]}, k = 3, {kind}: model (flux {met['flux']:.1f} Jy)")
            ax[i, 2 * j + 1].imshow(rr[sl], origin="lower", cmap="RdBu_r", vmin=-vr, vmax=vr, extent=[5, -5, -5, 5])
            ax[i, 2 * j + 1].set_title(f"residual: bias {met['bias'] * 100:+.1f}%, rms {met['resid_over_noise']:.2f} x noise")
    fig.suptitle("Extended (k = 3): before and after debiasing -- the arm-shaped residual is the L1 bias", fontsize=14)
    fig.savefig(f"{out3}/03_extended_before_after.png", dpi=60); plt.close(fig)
    # 03: compact pd k2 before / after
    fig, ax = plt.subplots(1, 4, figsize=(23, 5.6), constrained_layout=True)
    csl = np.s_[75:325, 75:325]
    for j, (kind, p) in enumerate((("solve", "data/ngc3110/compact/pd/model.npy"), ("debiased", "data/experiments/03_debiasing/real/compact/pd_k2_debiased.npy"))):
        met, (x, r) = cc.metrics(ld(p)); mm = mom0(x, cc.line).cpu().numpy() / 0.04 ** 2; rr = mom0(r, cc.line).cpu().numpy()
        if j == 0:
            cvm, cvr = np.percentile(mm[csl], 99.9), 4 * rr[csl].std()
        ax[2 * j].imshow(mm[csl], origin="lower", cmap="inferno", vmin=0, vmax=cvm, extent=[5, -5, -5, 5]); ax[2 * j].set_title(f"compact pd k = 2, {kind}: model ({met['flux']:.1f} Jy)")
        ax[2 * j + 1].imshow(rr[csl], origin="lower", cmap="RdBu_r", vmin=-cvr, vmax=cvr, extent=[5, -5, -5, 5]); ax[2 * j + 1].set_title(f"residual: bias {met['bias'] * 100:+.2f}%, rms {met['resid_over_noise']:.2f} x noise")
    fig.savefig(f"{out3}/03_compact_before_after.png", dpi=70); plt.close(fig)
    # 03: bias before / after for every model
    c2 = R["02_crossvalidation"]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for a, src, cfg, ks in ((ax[0], c2, "compact", (2, 3, 4, 5)), (ax[1], res, "extended", (3, 4, 5))):
        for m in ("pd", "pdtau", "fista"):
            if cfg == "compact":
                b = [src[f"{m} k{k}"]["bias"] * 100 for k in ks]; d = [src[f"{m} k{k} debiased"]["bias"] * 100 for k in ks]
            else:
                b = [src[f"extended {m} k{k} solve"]["bias"] * 100 for k in ks]; d = [src[f"extended {m} k{k} debiased"]["bias"] * 100 for k in ks]
            a.plot(ks, b, "o-", color=COL[m], lw=2, label=f"{NAME[m]}"); a.plot(ks, d, "s--", color=COL[m], lw=1.4, label=f"{NAME[m]}, debiased")
        a.axhline(0, color="0.5"); a.set_xticks(ks); a.set_xlabel("k"); a.set_ylabel("residual / predicted emission in the emission region [%]")
        a.set_title(f"{cfg}: L1 bias before (solid) and after (dashed) debiasing"); a.grid(color="0.9"); a.legend(frameon=False, fontsize=8)
    fig.savefig(f"{out3}/03_bias_summary.png", dpi=80); plt.close(fig)


# ---------------------------------------------------------------- 05 large field
def exp05():
    out = fig_dir("05_large_field"); res = {}
    cl = Cfg("ngc3110", "compact", "large"); cc = Cfg("ngc3110", "compact")
    yy, xx = np.indices((800, 800)); rad = np.hypot(yy - 400, xx - 400) * 0.04
    std = cc.metrics(ld("data/ngc3110/compact/pd/model.npy"))
    rstd = mom0(std[1][1], cc.line).cpu().numpy()                   # 16" run residual, 0.04", 400 x 400
    yy2, xx2 = np.indices((400, 400)); rad2 = np.hypot(yy2 - 200, xx2 - 200) * 0.04
    fig, ax = plt.subplots(1, 4, figsize=(25, 5.8), constrained_layout=True)
    for k in (2, 3):
        x = ld(f"data/experiments/05_large_field/pd_k{k}.npy")
        met, (xt, r) = cl.metrics(x, native=True); rm = mom0(r, cl.line).cpu().numpy(); mm = mom0(xt, cl.line).cpu().numpy() / 0.04 ** 2
        inner = xt[:, 200:600, 200:600]
        res[f"pd k{k}"] = dict(met, flux_inner16=float(inner.sum()), flux_outer=float(xt.sum() - inner.sum()),
                               lf_flux_inner16=float(inner[[c for c in range(32) if c not in cl.line]].sum()))
        if k == 2:
            ax[0].imshow(np.arcsinh(mm / (0.01 * mm.max())), origin="lower", cmap="inferno", extent=[16, -16, -16, 16])
            ax[0].add_patch(Rectangle((-8, -8), 16, 16, fc="none", ec="c", lw=1.5, ls="--")); ax[0].set_title("32\" field, pd k = 2: raw model mom 0 (asinh)\ndashed: the 16\" field")
            v = 4 * rm.std(); ax[1].imshow(rm, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, extent=[16, -16, -16, 16]); ax[1].set_title("32\" field: residual moment 0")
            prof_l = [rm[(rad >= a) & (rad < a + 0.5)].std() for a in np.arange(0, 15.5, 0.5)]
            prof_s = [rstd[(rad2 >= a) & (rad2 < a + 0.5)].std() for a in np.arange(0, 11, 0.5)]
            ax[2].plot(np.arange(0, 15.5, 0.5) + 0.25, prof_l, "o-", color=COL["pd"], label="32\" field"); ax[2].plot(np.arange(0, 11, 0.5) + 0.25, prof_s, "s-", color=COL["fista"], label="16\" field")
            ax[2].axhline(float(cl.noise_m0.std()), color="0.5", ls="--", label="noise"); ax[2].axvline(8, color="0.7", ls=":")
            ax[2].set_xlabel("radius [\"]"); ax[2].set_ylabel("residual mom-0 rms in annulus [Jy/beam km/s]"); ax[2].legend(frameon=False); ax[2].set_title("residual vs radius")
            sp_out = [float(xt[c][rad > 8].sum()) for c in range(32)]; sp_in = [float(xt[c][rad <= 8].sum()) for c in range(32)]
            ax[3].step(vel("ngc3110"), sp_in, where="mid", color=COL["pd"], label="model within r < 8\""); ax[3].step(vel("ngc3110"), sp_out, where="mid", color=COL["fista"], label="model at r > 8\"")
            ax[3].set_xlabel("v [km/s]"); ax[3].set_ylabel("model flux per channel [Jy]"); ax[3].legend(frameon=False); ax[3].set_title("is the outer flux CO (line-shaped) or noise (flat)?")
        print(f"05 pd k{k}: {res[f'pd k{k}']}", flush=True)
    res["16arcsec pd k2"] = std[0]
    R["05_large_field"] = res
    fig.savefig(f"{out}/05_large_field.png", dpi=70); plt.close(fig)


# ---------------------------------------------------------------- 07 NGC 2369
def exp07():
    out = fig_dir("07_ngc2369"); g = "ngc2369"
    xc = ld(f"{EXP_DATA}/07_ngc2369/compact_pd_k2.npy"); xe = ld(f"{EXP_DATA}/07_ngc2369/extended_pd_k3.npy")
    line = line_channels(g); BE, BC = BEAMS[(g, "extended")], BEAMS[(g, "compact")]
    kE = beam_ft((800, 800), BE, 0.02); kC = beam_ft((800, 800), BC, 0.02)
    om = lambda b: np.pi * b[0] * b[1] / (4 * np.log(2))
    C = mom0(conv(xc, kE), line).cpu().numpy() / om(BE); E = mom0(conv(xe, kE), line).cpu().numpy() / om(BE); Rc = mom0(conv(xc, kC), line).cpu().numpy() / om(BC)
    xv = XVal(g)(xc); ce, cc = Cfg(g, "extended", mask=os.path.join(DATA, g, "extended/pd/support_mask.npy")), Cfg(g, "compact")
    me = ce.metrics(xe)[0]; mc = cc.metrics(xc)[0]
    R["07_ngc2369"] = dict(line_channels=line, xval=xv, compact=mc, extended=me,
                           flux_compact=float(xc.sum()), flux_extended=float(xe.sum()),
                           flux_box10_compact=float(C[150:650, 150:650].sum() * 0.02 ** 2), flux_box10_extended=float(E[150:650, 150:650].sum() * 0.02 ** 2))
    sl = np.s_[150:650, 150:650]
    fig, ax = plt.subplots(2, 3, figsize=(20, 12.5), constrained_layout=True)
    vm = max(C[sl].max(), E[sl].max())
    for a, img, t, vv in ((ax[0, 0], Rc, "compact (k = 2) * compact beam", Rc[sl].max()), (ax[0, 1], C, "compact (k = 2) * extended beam", vm),
                          (ax[0, 2], E, "extended (k = 3, mask) * extended beam", vm)):
        im = a.imshow(img[sl], origin="lower", cmap="inferno", vmin=0, vmax=vv, extent=[5, -5, -5, 5]); a.set_title(f"NGC 2369: {t}"); plt.colorbar(im, ax=a, shrink=0.8, label="Jy km/s arcsec$^{-2}$")
    d = (C - E)[sl]; im = ax[1, 0].imshow(d, origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm, extent=[5, -5, -5, 5])
    ax[1, 0].set_title(f"compact - extended (+- peak): rms {d.std() / vm:.1%} of peak"); plt.colorbar(im, ax=ax[1, 0], shrink=0.8)
    ax[1, 1].plot(range(9), xv["per_scale"], "o-", color=COL["pd"], lw=2, label="NGC 2369 compact model")
    if "02_crossvalidation" in R:
        ax[1, 1].plot(range(9), R["02_crossvalidation"]["pd k2"]["xval"]["per_scale"], "s--", color="0.5", lw=1.5, label="NGC 3110 compact model (same recipe)")
    ax[1, 1].axhline(1, color="0.6", ls=":"); ax[1, 1].set_xticks(range(9)); ax[1, 1].set_xticklabels(scale_labels(0.02, 8), rotation=35, fontsize=9)
    ax[1, 1].set_ylabel("unexplained extended signal energy"); ax[1, 1].set_title("blind cross-validation: compact predicting the independent extended data"); ax[1, 1].legend(frameon=False)
    v = vel(g); bx = np.s_[:, 150:650, 150:650]
    for x, b, col, lab in ((xc, BC, COL["pd"], "compact"), (xe, BE, COL["fista"], "extended")):
        ax[1, 2].step(v, x[bx].sum((1, 2)), where="mid", color=col, lw=2, label=f"{lab} model ({x[bx][line].sum() * 25:.0f} Jy km/s)")
    ax[1, 2].axvspan(v[line[0]] - 12.5, v[line[-1]] + 12.5, color="0.92", zorder=0, label="line channels (auto)")
    ax[1, 2].set_xlabel("v [km/s]"); ax[1, 2].set_ylabel("model flux per channel, central 10\" [Jy]"); ax[1, 2].legend(frameon=False, fontsize=9)
    fig.savefig(f"{out}/07_ngc2369_blind_test.png", dpi=70); plt.close(fig)
    print("07", json.dumps(R["07_ngc2369"])[:600], flush=True)


# ---------------------------------------------------------------- 08 error maps
def exp08():
    out = fig_dir("08_error_maps"); cc = Cfg("ngc3110", "compact")
    line = cc.line; ref = ld("data/ngc3110/compact/pd/model.npy")
    kE = beam_ft((800, 800), BEAMS[("ngc3110", "extended")], 0.02); om = np.pi * 0.220 * 0.172 / (4 * np.log(2))
    B = [ld(f) for f in sorted(glob.glob(f"{EXP_DATA}/08_error_maps/bootstrap_*.npy"))]
    m_raw = np.array([x[line].sum(0) * 25 / 0.02 ** 2 for x in B]); m_ext = np.array([mom0(conv(x, kE), line).cpu().numpy() / om for x in B])
    r_raw = ref[line].sum(0) * 25 / 0.02 ** 2; r_ext = mom0(conv(ref, kE), line).cpu().numpy() / om
    sl = np.s_[150:650, 150:650]
    boxes = {"A: central ridge": (-0.4, -0.3, 2.8), "B: northern knot": (-0.4, 1.8, 2.2), "C: northern arm": (2.4, 1.0, 3.0), "D: southern arm": (-1.9, -3.9, 3.2)}
    res = dict(n=len(B), flux_ref=float(ref.sum()), flux_boot_mean=float(np.mean([x.sum() for x in B])), flux_boot_std=float(np.std([x.sum() for x in B])))
    for nm, (ra, dec, sz) in boxes.items():
        r0, r1 = int(400 + dec / 0.02 - sz / 0.04), int(400 + dec / 0.02 + sz / 0.04); c0, c1 = int(400 - ra / 0.02 - sz / 0.04), int(400 - ra / 0.02 + sz / 0.04)
        f = [float(x[line][:, r0:r1, c0:c1].sum() * 25) for x in B]
        res[nm] = dict(ref=float(ref[line][:, r0:r1, c0:c1].sum() * 25), mean=float(np.mean(f)), std=float(np.std(f)))
    pk = (r_ext == np.maximum.reduce([np.roll(np.roll(r_ext, i, 0), j, 1) for i in (-12, 0, 12) for j in (-12, 0, 12)])) & (r_ext > 0.3 * r_ext.max())
    res["knot_peak_snr_median"] = float(np.median((m_ext.mean(0) / m_ext.std(0))[pk]))
    R["08_error_maps"] = res
    fig, ax = plt.subplots(2, 4, figsize=(25, 12), constrained_layout=True)
    for row, (mm, rr, t) in enumerate(((m_raw, r_raw, "raw model"), (m_ext, r_ext, "model * extended beam"))):
        for a, img, tt, cm, vv in ((ax[row, 0], mm.mean(0), f"{t}: bootstrap mean", "inferno", None), (ax[row, 1], mm.std(0), f"{t}: bootstrap std (error map)", "viridis", None),
                                   (ax[row, 2], mm.mean(0) / np.maximum(mm.std(0), 1e-9), f"{t}: mean / std", "magma", None),
                                   (ax[row, 3], mm.mean(0) - rr, f"{t}: bootstrap mean - input model (bias)", "RdBu_r", "sym")):
            if vv == "sym":
                v = np.percentile(np.abs(img[sl]), 99.5); im = a.imshow(img[sl], origin="lower", cmap=cm, vmin=-v, vmax=v, extent=[5, -5, -5, 5])
            else:
                im = a.imshow(img[sl], origin="lower", cmap=cm, vmin=0, vmax=np.percentile(img[sl], 99.5), extent=[5, -5, -5, 5])
            a.set_title(tt); plt.colorbar(im, ax=a, shrink=0.8)
    fig.suptitle(f"Error maps: {len(B)} parametric-bootstrap realizations of the compact recipe (data = N x_model + new noise)", fontsize=14)
    fig.savefig(f"{out}/08_error_maps.png", dpi=60); plt.close(fig)
    print("08", json.dumps(res)[:800], flush=True)


if __name__ == "__main__":
    steps = sys.argv[1:] or ["01", "02", "03", "05", "07", "08"]
    if os.path.exists(os.path.join(EXP_DATA, "results.json")):
        R.update(json.load(open(os.path.join(EXP_DATA, "results.json"))))
    for s in steps:
        {"01": exp01, "02": exp02, "03": exp03_04, "05": exp05, "07": exp07, "08": exp08}[s]()
        json.dump(R, open(os.path.join(EXP_DATA, "results.json"), "w"), indent=1)
    print("ANALYSIS DONE", flush=True)
