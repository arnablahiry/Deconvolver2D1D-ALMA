#!/usr/bin/env python
"""
Experiment 23 report figures (runs: iht_dict.py, scores: iht_score.py -> data/experiments/23_iht/scores.json): every
dictionary and threshold rule on the NGC 3110 compact ground-truth simulation, per family of exp23_style.FAMILIES
(imagenet, fire, learnt), each arm at its best member (lowest error) unless a figure says otherwise.

  tradeoff_<family>   error vs spurious flux on pure noise (null test) and vs flux outside the true emission, over k / s
  summary             best error per arm (with its null flux), and per-scale error relative to starlet's (heatmap)
  maps_<family>       moment 0 at the extended clean beam: model, error G*(x - truth), data residual d - N x (moment
                      0 of the line channels, Jy/beam km/s, at the compact dirty beam), null-test model (all channels)
  spectra_<family>    integrated spectrum vs truth; model - truth per channel
  channels_<family>   four line channels at the extended clean beam, truth and each arm
  power_<family>      azimuthally averaged power of the raw moment 0 vs truth (transfer) and of the error, vs spatial
                      frequency; compact / extended beams marked
  convergence         relative change and flux vs iteration (best member per arm), 10k vs 20k iteration checks

    iht_report.py  -> figures/experiments/23_iht/report/*.png
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import beam_ft, conv, ld  # noqa: E402
from common import G, EXP_DATA, REPO, BEAMS, SIM, line_channels, read  # noqa: E402
from exp23_style import STYLE, SHORT, FAMILIES, INK, MUTED, rc  # noqa: E402

D = os.path.join(EXP_DATA, "23_iht")
FIG = os.path.join(REPO, "figures/experiments/23_iht/report"); os.makedirs(FIG, exist_ok=True)
DV, CELL, HALF, C10 = 25.0, 0.04, 5.0, np.s_[75:325, 75:325]
LINE = line_channels("ngc3110")
BE, BC = BEAMS[("ngc3110", "extended")], BEAMS[("ngc3110", "compact")]
rc()
rows = json.load(open(os.path.join(D, "scores.json")))
key = lambda r: (r["dict"], r["thr"])


def best(k):
    rr = [r for r in rows if key(r) == k]
    return min(rr, key=lambda r: r["err"]) if rr else None


def path(r, kind="sim"):
    if r["thr"] == "pdtau" and r["dict"] == "starlet":                    # the pipeline's own runs (exp 01 / 22)
        return (os.path.join(EXP_DATA, "01_simulations", "compact", f"pdtau_k{r['val']:g}.npy") if kind == "sim" else
                os.path.join(EXP_DATA, "22_learnlet", "null", f"starlet_k{r['val']:g}.npy"))
    return os.path.join(D, kind, f"{r['dict']}_{r['thr']}_{r['par']}{r['val']:g}.npy")


def short(r):
    return f"{r['par']} = {r['val']:g}"


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=140, bbox_inches="tight"); plt.close(fig)
    print("->", os.path.join(FIG, name + ".png"))


truth = ld(f"{SIM}/compact_sim_truth_0.04arcsec.npy")
kft = beam_ft(truth.shape[1:], BE, CELL)
dirty_r, psf = G.rebin(read("ngc3110", "compact", "dirty_cube.fits"), read("ngc3110", "compact", "dirty_beam_cube.fits"), 2)
op = G.Op(psf, np.load(f"{SIM}/compact_sim_dirty_0.04arcsec.npy").astype(np.float64))
del dirty_r
m0b = lambda x, ch=LINE: (conv(x[ch], kft).sum(0) * DV).cpu().numpy()      # moment 0 at the extended beam
t0 = m0b(truth)
LB = float(json.load(open(os.path.join(REPO, "results/NGC3110/config.json")))["lbmax_compact_arcsec"])
BLB = (LB, LB, 0.0)                                                         # the compact array's lambda/B_max
kft_lb = beam_ft(truth.shape[1:], BLB, CELL)
m0lb = lambda x, ch=LINE: (conv(x[ch], kft_lb).sum(0) * DV).cpu().numpy()
t0lb = m0lb(truth)
cache = {}


def model(r, kind="sim"):
    p = path(r, kind)
    if (p, kind) not in cache:
        x = np.load(p) if os.path.exists(p) else None
        if x is not None and x.shape[-1] == 2 * truth.shape[-1]:       # null runs are real-data solves, saved at 0.02"
            x = x.reshape(x.shape[0], x.shape[1] // 2, 2, x.shape[2] // 2, 2).sum((2, 4))
        cache[(p, kind)] = x
    return cache[(p, kind)]


def arms(fam, pipeline=True):
    out = [(("starlet", "pdtau"), best(("starlet", "pdtau")))] if pipeline else []
    return out + [(k, best(k)) for k in FAMILIES[fam] if best(k) is not None]


# ------------------------------------------------------------------ trade-off per family
for fam in FAMILIES:
    present = [k for k in FAMILIES[fam] if best(k)]
    if len(present) < 2:
        continue
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6), constrained_layout=True)
    for k in [("starlet", "pdtau")] + present:
        lab, c, ls, mk = STYLE[k]
        rr = sorted([r for r in rows if key(r) == k], key=lambda r: r["val"])
        pts = [r for r in rr if r["null_flux"] is not None]
        x0 = [max(r["null_flux"], 1e-4) for r in pts]
        if pts:
            ax[0].plot(x0, [r["err"] for r in pts], ls=ls, marker=mk, color=c, label=lab,
                       mfc="white" if ls != "-" else c, mew=1.6)
            e = min(pts, key=lambda r: r["err"])                         # direct label at the best member
            ax[0].annotate(short(e), (max(e["null_flux"], 1e-4), e["err"]), xytext=(4, -10), textcoords="offset points",
                           fontsize=7.5, color=MUTED)
            ax[1].plot(x0, [r["err_lb"] for r in pts], ls=ls, marker=mk, color=c, mfc="white" if ls != "-" else c, mew=1.6)
            e = min(pts, key=lambda r: r["err_lb"])
            ax[1].annotate(short(e), (max(e["null_flux"], 1e-4), e["err_lb"]), xytext=(4, -10),
                           textcoords="offset points", fontsize=7.5, color=MUTED)
        ax[2].plot([r["flux_out"] for r in rr], [r["err"] for r in rr], ls=ls, marker=mk, color=c,
                   mfc="white" if ls != "-" else c, mew=1.6, label=lab)
    for a in ax[:2]:
        a.set_xscale("log"); a.set_xlabel("spurious flux on pure noise (null test), Jy  [0 drawn at 10$^{-4}$]")
    ax[2].set_xlabel("flux outside the true emission, Jy")
    ax[0].set_ylabel(f"error vs truth, extended clean beam ({BE[0]:.2f}″ × {BE[1]:.2f}″)")
    ax[1].set_ylabel(f"error vs truth, compact λ/B$_{{max}}$ ({LB:.2f}″)")
    ax[2].set_ylabel("error vs truth, extended clean beam")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8.5)
    fig.suptitle(f"NGC 3110 compact simulation, iterative hard thresholding: {fam} family "
                 "(lower = better; labels: best member)", fontsize=11.5)
    save(fig, f"tradeoff_{fam}")

# ------------------------------------------------------------------ summary: best error, per-scale error vs starlet
allk = []
for fam in FAMILIES:
    allk += [k for k in FAMILIES[fam] if best(k) and k not in allk]
allk = [("starlet", "pdtau")] + allk
B = [(k, best(k)) for k in allk]
B.sort(key=lambda kb: kb[1]["err"])
ref = best(("starlet", "noise"))
SC = ["0.04–0.08″", "0.08–0.16″", "0.16–0.32″", "0.32–0.64″", "0.64–1.28″", "1.28–2.56″", "2.56–5.12″", "coarse"]
fig, ax = plt.subplots(1, 2, figsize=(16, 0.42 * len(B) + 1.8), gridspec_kw=dict(width_ratios=[1, 1.35]),
                       constrained_layout=True)
y = np.arange(len(B))
for i, (k, r) in enumerate(B):
    lab, c, ls, mk = STYLE[k]
    ax[0].plot(r["err"], i, marker=mk, color=c, ms=9, mfc="white" if ls != "-" else c, mew=1.8, ls="none")
    ax[0].plot(r["err_lb"], i, marker="|", color=c, ms=14, mew=2.2, ls="none")
    ax[0].plot([r["err_lb"], r["err"]], [i, i], color=c, lw=0.8, alpha=0.5)
    nf = "–" if r["null_flux"] is None else f"{r['null_flux']:.3f} Jy"
    ax[0].annotate(f"{r['err']:.3f} / {r['err_lb']:.3f}  ({short(r)}, null {nf})", (r["err"], i), xytext=(8, -3.5),
                   textcoords="offset points", fontsize=8, color=MUTED)
ax[0].set_yticks(y); ax[0].set_yticklabels([STYLE[k][0] for k, _ in B], fontsize=8.5); ax[0].invert_yaxis()
ax[0].set_xlabel("best error vs truth: marker = extended clean beam, bar = compact λ/B$_{max}$")
ax[0].set_xlim(min(r["err_lb"] for _, r in B) - 0.02, max(r["err"] for _, r in B) + 0.07)
ax[0].set_title("best member of each arm (ranked at the extended beam)", fontsize=11)
R = np.array([[r["per_scale"][j] / ref["per_scale"][j] for j in range(8)] for _, r in B])
lim = min(1.0, max(0.05, float(np.abs(np.log2(R)).max())))        # 0.5x - 2x; cells print the true ratio
im = ax[1].imshow(np.log2(R), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
for i in range(R.shape[0]):
    for j in range(8):
        ax[1].text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                   color="white" if abs(np.log2(R[i, j])) > 0.6 * lim else INK)
ax[1].set_xticks(range(8)); ax[1].set_xticklabels(SC, rotation=30, fontsize=8.5); ax[1].set_yticks(y)
ax[1].set_yticklabels([]); ax[1].grid(False)
ax[1].set_title(f"per-scale error / starlet's (best starlet: {short(ref)}); blue = better", fontsize=11)
cb = fig.colorbar(im, ax=ax[1], fraction=0.03, pad=0.01); cb.set_label("log$_2$ ratio")
save(fig, "summary")


# ------------------------------------------------------------------ maps per family
def beam(a, b, half=HALF):
    pad = 0.06 * half; bx, by = half - pad - b[0] / 2, -half + pad + b[0] / 2
    a.add_patch(Ellipse((bx, by), width=b[0], height=b[1], angle=90 - b[2], facecolor="w", edgecolor="w"))


def scalebar(a, half=HALF, color="w"):
    xs, ys, cap = -half + 0.08 * half, half - 0.1 * half, 0.03 * half
    a.plot([xs + 2, xs], [ys, ys], color=color, lw=2.0, solid_capstyle="butt")
    for xe in (xs, xs + 2):
        a.plot([xe, xe], [ys - cap, ys + cap], color=color, lw=1.4, solid_capstyle="butt")
    a.annotate("2″", (xs + 1, ys - cap), xytext=(0, -2.5), textcoords="offset points", color=color, ha="center",
               va="top", fontsize=9, fontweight="bold")


EXT = [HALF, -HALF, -HALF, HALF]
for fam in FAMILIES:
    A = arms(fam)
    if len(A) < 3:
        continue
    cols = [("Truth", None, None)] + [(SHORT[k], k, r) for k, r in A]
    imgs = [t0[C10]] + [m0b(G.to_t(model(r)))[C10] for _, _, r in cols[1:]]
    elb = [None] + [m0lb(G.to_t(model(r)))[C10] - t0lb[C10] for _, _, r in cols[1:]]
    lmax = max(np.percentile(np.abs(x), 99.5) for x in elb[1:])
    res = [None] + [(-op.grad(G.to_t(model(r)))[LINE].sum(0) * DV).cpu().numpy()[C10] for _, _, r in cols[1:]]
    nul = [None] + [None if model(r, "null") is None else m0b(G.to_t(model(r, "null")), slice(None))[C10]
                    for _, _, r in cols[1:]]
    vmax = np.percentile(t0, 99.9)
    emax = max(np.percentile(np.abs(im - t0[C10]), 99.5) for im in imgs[1:])
    rmax = max(np.percentile(np.abs(x), 99.5) for x in res[1:])
    nmax = max([np.percentile(np.abs(x), 99.9) for x in nul[1:] if x is not None] + [1e-6])
    n = len(cols); S = 2.15
    fig, ax = plt.subplots(5, n, figsize=(S * n + 0.9, 5 * S + 0.8), gridspec_kw=dict(wspace=0.03, hspace=0.05))
    for i, ((name, k, r), im) in enumerate(zip(cols, imgs)):
        a = ax[0, i]
        h0 = a.imshow(im, origin="lower", extent=EXT, cmap="inferno", interpolation="nearest",
                      norm=matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=vmax))
        a.text(0.04, 0.96, name, transform=a.transAxes, color="w", ha="left", va="top", fontsize=8, fontweight="bold")
        if r is not None:
            a.text(0.04, 0.84, short(r), transform=a.transAxes, color="w", ha="left", va="top", fontsize=7.5)
        beam(a, BE)
        if i == 0:
            scalebar(a)
        for row, data, cm, lim, lab in ((1, None if k is None else im - t0[C10], "RdBu_r", emax, f"error {r['err']:.3f}" if r else ""),
                                        (2, elb[i], "RdBu_r", lmax, f"error {r['err_lb']:.3f}" if r else ""),
                                        (3, res[i], "PuOr_r", rmax, "data residual"),
                                        (4, nul[i], "inferno", nmax, None if r is None or r["null_flux"] is None else f"null {r['null_flux']:.3f} Jy")):
            b = ax[row, i]
            if data is None:
                b.axis("off")
                if i == 0:
                    b.text(0.5, 0.5, {1: "model − truth\n(extended clean\nbeam)", 2: f"model − truth\n(compact λ/B$_{{max}}$,\n{LB:.2f}″)",
                                      3: "d − N x, line\nmoment 0\n(compact dirty\nbeam, Jy/beam)",
                                      4: "null test: model\nof pure noise,\nall channels"}[row],
                           transform=b.transAxes, ha="center", va="center", fontsize=9, color=INK)
                continue
            hh = b.imshow(data, origin="lower", extent=EXT, cmap=cm, interpolation="nearest",
                          **(dict(vmin=-lim, vmax=lim) if row < 4 else dict(norm=matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=lim))))
            if lab:
                b.text(0.04, 0.96, lab, transform=b.transAxes, color="w" if row == 4 else INK, ha="left", va="top",
                       fontsize=8, fontweight="bold")
            if row == 2:
                beam(b, BLB)
            if row == 3:
                beam(b, BC)
            ax[row, n - 1].figure._last = hh
            if i == n - 1:
                fig.colorbar(hh, ax=ax[row, :], fraction=0.012, pad=0.01,
                             label="Jy beam$^{-1}$ km s$^{-1}$")
    for a in ax.flat:
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.colorbar(h0, ax=ax[0, :], fraction=0.012, pad=0.01, label="moment 0, Jy beam$^{-1}$ km s$^{-1}$")
    fig.suptitle(f"{fam} family, best member of each arm: moment 0, error at the extended beam and at the compact "
                 "λ/B$_{max}$, data residual, null test", fontsize=11.5, y=0.905)
    save(fig, f"maps_{fam}")

    # ---------------------------------------------------------------- spectra
    fig, ax = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True, gridspec_kw=dict(height_ratios=[2, 1.2]),
                           constrained_layout=True)
    v = 5073.9 + 25.0 * (np.arange(32) - 15.5)
    ts = truth.sum((1, 2))
    ax[0].step(v, ts, where="mid", color=INK, lw=2.4, label="truth")
    for k, r in A:
        lab, c, ls, mk = STYLE[k]
        s = model(r).sum((1, 2))
        ax[0].step(v, s, where="mid", color=c, ls=ls, lw=1.6, label=f"{lab} ({short(r)})")
        ax[1].plot(v, s - ts, color=c, ls=ls, marker=mk, ms=4.5, mfc="white" if ls != "-" else c, lw=1.4)
    ax[1].axhline(0, color=MUTED, lw=0.8)
    ax[0].set_ylabel("flux density, Jy"); ax[1].set_ylabel("model − truth, Jy"); ax[1].set_xlabel("velocity, km s$^{-1}$")
    ax[0].legend(fontsize=8); ax[0].set_title(f"integrated spectrum, {fam} family (best members)", fontsize=11)
    save(fig, f"spectra_{fam}")

    # ---------------------------------------------------------------- channel maps
    ch = [LINE[int(i)] for i in np.linspace(1, len(LINE) - 2, 4)]
    rws = [("Truth", None, truth)] + [(SHORT[k] + f"\n{short(r)}", k, model(r)) for k, r in A]
    fig, ax = plt.subplots(len(rws), 4, figsize=(4 * 2.1 + 0.8, 2.1 * len(rws)), gridspec_kw=dict(wspace=0.03, hspace=0.03))
    for jc, c in enumerate(ch):
        tc = conv(truth[c:c + 1], kft)[0].cpu().numpy()[C10]; vm = np.percentile(tc, 99.8)
        for ir, (name, k, x) in enumerate(rws):
            a = ax[ir, jc]
            im = a.imshow(conv(x[c:c + 1], kft)[0].cpu().numpy()[C10], origin="lower", extent=EXT, cmap="inferno",
                          norm=matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=vm), interpolation="nearest")
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
            if ir == 0:
                a.set_title(f"{v[c]:.0f} km/s", fontsize=9.5)
            if jc == 0:
                a.text(0.04, 0.96, name, transform=a.transAxes, color="w", ha="left", va="top", fontsize=7.5, fontweight="bold")
                if ir == 0:
                    scalebar(a)
    fig.suptitle(f"channel maps at the extended clean beam, {fam} family (one colour scale per channel)", fontsize=11, y=0.905)
    save(fig, f"channels_{fam}")

    # ---------------------------------------------------------------- power spectra
    def radial(img):
        p = np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2
        ny, nx = img.shape; yy, xx = np.indices(p.shape)
        kr = np.hypot((yy - ny // 2) / (ny * CELL), (xx - nx // 2) / (nx * CELL))           # cycles / arcsec
        edges = np.geomspace(1 / (ny * CELL), 0.5 / CELL, 40); idx = np.digitize(kr.ravel(), edges)
        return np.sqrt(edges[1:] * edges[:-1]), np.array([p.ravel()[idx == i].mean() if (idx == i).any() else np.nan
                                                          for i in range(1, len(edges))])
    tm0 = truth[LINE].sum(0) * DV
    kk, pt = radial(tm0)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for k, r in A:
        lab, c, ls, mk = STYLE[k]
        xm0 = model(r)[LINE].sum(0) * DV
        ax[0].plot(kk, np.sqrt(radial(xm0)[1] / pt), color=c, ls=ls, label=f"{lab} ({short(r)})")
        ax[1].plot(kk, np.sqrt(radial(xm0 - tm0)[1] / pt), color=c, ls=ls)
    for a in ax:
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("spatial frequency, arcsec$^{-1}$")
        for b, nm in ((BC, "compact beam"), (BE, "extended beam")):
            a.axvline(1 / b[0], color=MUTED, lw=0.9, ls=":"); a.text(1 / b[0], 0.97, nm + " ", rotation=90, fontsize=8,
                                                                   color=MUTED, va="top", ha="right",
                                                                   transform=a.get_xaxis_transform())
    ax[0].axhline(1, color=INK, lw=0.8)
    ax[0].set_ylabel("|model| / |truth|  (amplitude transfer)"); ax[1].set_ylabel("|model − truth| / |truth|")
    ax[0].set_title("raw line moment 0: recovered amplitude per frequency", fontsize=10.5)
    ax[1].set_title("error amplitude per frequency", fontsize=10.5)
    ax[0].legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 0.0))
    save(fig, f"power_{fam}")

# ------------------------------------------------------------------ convergence
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
for k in allk:
    r = best(k)
    if r["thr"] == "pdtau" and r["dict"] == "starlet":
        continue
    js = path(r).replace(".npy", ".json")
    if not os.path.exists(js):
        continue
    h = json.load(open(js)).get("history")
    if not h:                                                     # run_model.py (pdtau) keeps no history
        continue
    lab, c, ls, mk = STYLE[k]
    ax[0].plot([e["iter"] for e in h], [e["rel_change"] for e in h], color=c, ls=ls, label=f"{lab} ({short(r)})")
    ax[1].plot([e["iter"] for e in h], [e["flux"] for e in h], color=c, ls=ls)
for f in sorted(glob.glob(os.path.join(D, "sim20k", "*.json"))):
    h = json.load(open(f))["history"]; nm = os.path.basename(f)[:-5]
    k = next((kk for kk in STYLE if nm.startswith(f"{kk[0]}_{kk[1]}_")), None)
    if k:
        ax[0].plot([e["iter"] for e in h], [e["rel_change"] for e in h], color=STYLE[k][1], lw=0.9, alpha=0.6)
        ax[1].plot([e["iter"] for e in h], [e["flux"] for e in h], color=STYLE[k][1], lw=0.9, alpha=0.6,
                   label=f"{nm}, 20,000 iterations")
ax[1].axhline(float(truth.sum()), color=INK, lw=0.9, ls=":", label="truth flux")
ax[0].set_yscale("log"); ax[0].set_ylabel("‖x$_{n+1}$ − x$_n$‖ / ‖x$_{n+1}$‖"); ax[1].set_ylabel("model flux, Jy")
for a in ax:
    a.set_xlabel("iteration")
ax[0].legend(fontsize=7, ncol=1); ax[1].legend(fontsize=7)
fig.suptitle("convergence of the best member of every arm (thin: 20,000-iteration checks)", fontsize=11.5)
save(fig, "convergence")
