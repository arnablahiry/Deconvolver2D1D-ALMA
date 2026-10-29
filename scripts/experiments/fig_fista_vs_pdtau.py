#!/usr/bin/env python
"""
The pipeline's FISTA against its pdtau on the real NGC 3110 compact data (starlet, k = 5, 20,000 iterations, the
models of fista_vs_pdtau.py: data/experiments/24_fista_vs_pdtau/<on|off>/{pdtau,fista_current,fista_exact}.npy,
reweighting on = the production setting), against the joint compact + extended CLEAN over the resolution ladder
(results/NGC3110/ladder_joint/b00..b10), inside each beam's emission mask (the masks of fig_resolution_ladder.py
REF=joint, as in the earlier resolution-ladder figures).

  fista_vs_pdtau_ladder   cube RMSE in the mask per beam, and each solver's difference from pdtau (percentage points),
                          reweighting on and off; with the exact-analysis FISTA (same problem as pdtau) as a reference,
                          and pdtau-syn (pdtau with the dual brought back by the starlet's synthesis W^-1 instead of W^T,
                          data/experiments/23_iht/real/starlet_pdsyn_k*.npy, production reweighting, at its best k in this
                          metric; no known objective)
  fista_vs_pdtau_maps     line moment 0 at the compact lambda/B_max and at the extended clean beam: the joint truth, pdtau,
                          FISTA with W^T, your FISTA (W^-1), pdtau-syn, and each model's error (model - truth); reweighting on
  fista_vs_pdtau_power    azimuthal power spectra of the line moment 0 at the extended clean beam (truth, pdtau, FISTA)
                          and the error amplitude |model - truth| / |truth| per spatial frequency; reweighting on

    fig_fista_vs_pdtau.py  -> figures/experiments/24_fista_vs_pdtau/fista_vs_pdtau_{ladder,maps,power}.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.patches import Ellipse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import beam_ft, conv  # noqa: E402
from common import EXP_DATA, REPO, line_channels  # noqa: E402
from exp23_style import HUE, INK, MUTED, rc  # noqa: E402

D = os.path.join(EXP_DATA, "24_fista_vs_pdtau"); FIG = os.path.join(REPO, "figures", "experiments", "24_fista_vs_pdtau")
os.makedirs(FIG, exist_ok=True)
LINE, DV, CELL, HALF, F4 = line_channels("ngc3110"), 25.0, 0.04, 5.0, np.s_[75:325, 75:325]
rc()
BEAMS, J = [], []
for i in range(11):
    p = os.path.join(REPO, f"results/NGC3110/ladder_joint/b{i:02d}/clean_finished/image.fits")
    h = fits.getheader(p); BEAMS.append((h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"]))
    J.append(np.nan_to_num(fits.getdata(p)).squeeze()[:, 440:840, 440:840][LINE].astype(np.float64))
MZ = np.load(os.path.join(REPO, "results/NGC3110/figures/resolution_ladder_joint/masks.npz"))
MASK = [MZ[f"b{i:02d}"] for i in range(11)]
KFT = [beam_ft((400, 400), b, CELL) for b in BEAMS]
NB, ILB = 11, 5
NAMES = {0: "compact\nclean beam", ILB: "compact\nλ/B$_{max}$", NB - 1: "extended\nclean beam"}
SOL = {"pdtau": ("pdtau (production)", HUE["pipeline"], "-", "o"),
       "fista_current": ("your FISTA (W$^{-1}$ ∘ soft ∘ W)", HUE["network"], "-", "s"),
       "fista_exact": ("FISTA with W$^{T}$ (exact analysis prox, same problem as pdtau)", HUE["starlet"], "--", "D"),
       "pdtau_syn": ("pdtau with W$^{-1}$ instead of W$^{T}$ (pdtau-syn)", HUE["imagenet"], "-", "^")}
M = {(rw, s): np.load(os.path.join(D, rw, f"{s}.npy")).astype(np.float64)[LINE] for rw in ("on", "off") for s in SOL
     if os.path.exists(os.path.join(D, rw, f"{s}.npy"))}
# pdtau-syn: the production reweighting only; its k chosen by the mean masked RMSE over the ladder
_syn = {float(f.split("_k")[1][:-4]): os.path.join(EXP_DATA, "23_iht", "real", f)
        for f in os.listdir(os.path.join(EXP_DATA, "23_iht", "real")) if f.startswith("starlet_pdsyn_k") and f.endswith(".npy")}
def at_beams(x):
    return [conv(x, KFT[i]).cpu().numpy() for i in range(NB)]


def masked(g):
    return [100 * np.linalg.norm((g[i] - J[i]) * MASK[i]) / np.linalg.norm(J[i] * MASK[i]) for i in range(NB)]


native = lambda x: x.reshape(x.shape[0], x.shape[1] // 2, 2, x.shape[2] // 2, 2).sum((2, 4)) if x.shape[-1] == 800 else x
SYN_K = None
if _syn:
    sc = {k: masked(at_beams(native(np.load(p).astype(np.float64))[LINE])) for k, p in _syn.items()}
    SYN_K = min(sc, key=lambda k: np.mean(sc[k]))
    print("pdtau-syn masked mean RMSE per k:", {k: round(float(np.mean(v)), 3) for k, v in sorted(sc.items())}, "-> k =", SYN_K)
    M[("on", "pdtau_syn")] = native(np.load(_syn[SYN_K]).astype(np.float64))[LINE]
    SOL["pdtau_syn"] = (SOL["pdtau_syn"][0] + f", k = {SYN_K:g}",) + SOL["pdtau_syn"][1:]
G = {k: at_beams(x) for k, x in M.items()}                                               # each model at each beam
rmse = {k: [100 * np.linalg.norm((g[i] - J[i]) * MASK[i]) / np.linalg.norm(J[i] * MASK[i]) for i in range(NB)]
        for k, g in G.items()}
for k, v in rmse.items():
    print(k, " ".join(f"{e:5.2f}" for e in v))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=140, bbox_inches="tight"); plt.close(fig)
    print("->", os.path.join(FIG, name + ".png"))


# ------------------------------------------------------------------ ladder
fig, ax = plt.subplots(2, 2, figsize=(16, 9), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1]), constrained_layout=True)
x = np.arange(NB)
for c, rw in enumerate(("on", "off")):
    for s, (lab, col, ls, mk) in SOL.items():
        if (rw, s) not in rmse:
            continue
        ax[0, c].plot(x, rmse[(rw, s)], ls=ls, marker=mk, color=col, mfc="white" if ls != "-" else col, mew=1.5, label=lab)
        if s != "pdtau":
            d = [a - b for a, b in zip(rmse[(rw, s)], rmse[(rw, "pdtau")])]
            ax[1, c].plot(x, d, ls=ls, marker=mk, color=col, mfc="white" if ls != "-" else col, mew=1.5, label=lab)
            for i in (0, ILB, NB - 1):
                ax[1, c].annotate(f"{d[i]:+.2f}", (i, d[i]), xytext=(0, 6 if d[i] >= 0 else -12), textcoords="offset points",
                                  ha="center", fontsize=8, color=col)
    ax[1, c].axhline(0, color=HUE["pipeline"], lw=1.4)
    for a in ax[:, c]:
        a.axvline(ILB, color=MUTED, ls=":", lw=0.9)
    ax[0, c].set_title(f"reweighting {rw}" + (" (production setting)" if rw == "on" else ""), fontsize=11)
    ax[0, c].set_ylabel("cube RMSE vs the joint CLEAN, in its emission mask [%]")
    ax[1, c].set_ylabel("− pdtau's RMSE [percentage points]\n(< 0: better than pdtau)")
    ax[1, c].set_xticks(x)
    ax[1, c].set_xticklabels([f"{b[0]:.2f}″×{b[1]:.2f}″" if abs(b[0] - b[1]) > 1e-3 else f"{b[0]:.2f}″" for b in BEAMS],
                             rotation=30, fontsize=8)
    for i in NAMES:
        ax[1, c].get_xticklabels()[i].set_fontweight("bold")
ax[0, 0].legend(fontsize=8.5)
fig.suptitle("NGC 3110, real data, starlet, k = 5: solvers vs pdtau over the resolution ladder (joint CLEAN truth)",
             fontsize=12)
save(fig, "fista_vs_pdtau_ladder")


# ------------------------------------------------------------------ maps (reweighting on)
def beam(a, b):
    pad = 0.06 * HALF; a.add_patch(Ellipse((HALF - pad - b[0] / 2, -HALF + pad + b[0] / 2), width=b[0], height=b[1],
                                           angle=90 - b[2], facecolor="w", edgecolor="w"))


def scalebar(a):
    xs, ys, cap = -HALF + 0.08 * HALF, HALF - 0.1 * HALF, 0.03 * HALF
    a.plot([xs + 2, xs], [ys, ys], color="w", lw=2.0, solid_capstyle="butt")
    for xe in (xs, xs + 2):
        a.plot([xe, xe], [ys - cap, ys + cap], color="w", lw=1.4, solid_capstyle="butt")
    a.annotate("2″", (xs + 1, ys - cap), xytext=(0, -2.5), textcoords="offset points", color="w", ha="center", va="top",
               fontsize=9, fontweight="bold")


EXT = [HALF, -HALF, -HALF, HALF]
m0 = lambda c: c.sum(0) * DV
NC = 1 + 2 * sum(1 for s_ in ("pdtau", "fista_exact", "fista_current", "pdtau_syn") if ("on", s_) in M)
fig, ax = plt.subplots(2, NC, figsize=(NC * 2.3 + 1, 2 * 2.45 + 0.9), gridspec_kw=dict(wspace=0.04, hspace=0.12))
for r, i in enumerate((ILB, NB - 1)):
    t = m0(J[i])[F4]
    mods = [(nm, s_) for nm, s_ in (("pdtau", "pdtau"), ("FISTA with W$^{T}$", "fista_exact"),
                                    ("your FISTA (W$^{-1}$)", "fista_current"), ("pdtau-syn (W$^{-1}$)", "pdtau_syn"))
            if ("on", s_) in G]
    imgs = {s_: m0(G[("on", s_)][i])[F4] for _, s_ in mods}
    vmax = np.percentile(t, 99.8); lim = np.percentile(np.abs(np.concatenate([(v - t).ravel() for v in imgs.values()])), 99.5)
    panels = [(t, "Joint CLEAN (truth)", "inferno", None)] + [(imgs[s_], nm, "inferno", None) for nm, s_ in mods] + \
             [(imgs[s_] - t, f"{nm}: error {rmse[('on', s_)][i]:.2f}%", "RdBu_r", lim) for nm, s_ in mods]
    for c, (img, lab, cm, lm) in enumerate(panels):
        a = ax[r, c]
        kw = dict(norm=matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=vmax)) if lm is None else dict(vmin=-lm, vmax=lm)
        a.imshow(img, origin="lower", extent=EXT, cmap=cm, interpolation="nearest", **kw)
        if lm is not None:
            a.contour(np.linspace(HALF, -HALF, img.shape[1]), np.linspace(-HALF, HALF, img.shape[0]),
                      MASK[i].any(0)[F4], levels=[0.5], colors=[MUTED], linewidths=0.6)
        a.text(0.04, 0.96, lab, transform=a.transAxes, color="w" if lm is None else INK, ha="left", va="top",
               fontsize=8 if lm is None else 7, fontweight="bold")
        beam(a, BEAMS[i]); a.set_xticks([]); a.set_yticks([])
        if c == 0:
            a.set_ylabel(["compact λ/B$_{max}$ (0.60″)", "extended clean beam"][r], fontsize=10)
            if r == 0:
                scalebar(a)
fig.suptitle("line moment 0, reweighting on (production setting); grey contour: the emission mask of the metric",
             fontsize=11, y=0.98)
save(fig, "fista_vs_pdtau_maps")


# ------------------------------------------------------------------ power spectra at the extended clean beam
def radial(img):
    p = np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2
    yy, xx = np.indices(p.shape); n = img.shape[0]
    kr = np.hypot(yy - n // 2, xx - n // 2) / (n * CELL)
    edges = np.geomspace(1 / (n * CELL), 0.5 / CELL, 36); idx = np.digitize(kr.ravel(), edges)
    return np.sqrt(edges[1:] * edges[:-1]), np.array([p.ravel()[idx == q].mean() if (idx == q).any() else np.nan
                                                      for q in range(1, len(edges))])


tm = m0(J[NB - 1]); kk, pt = radial(tm)
fig, ax = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
ax[0].loglog(kk, pt, color=INK, lw=2.4, label="joint CLEAN (truth)")
for s, (lab, col, ls, mk) in SOL.items():
    if ("on", s) not in G:
        continue
    gm = m0(G[("on", s)][NB - 1])
    ax[0].loglog(kk, radial(gm)[1], color=col, ls=ls, lw=1.6, label=lab)
    ax[1].semilogx(kk, np.sqrt(radial(gm)[1] / pt), color=col, ls=ls, lw=1.6)
    ax[2].loglog(kk, np.sqrt(radial(gm - tm)[1] / pt), color=col, ls=ls, lw=1.6)
ax[1].axhline(1, color=INK, lw=0.8)
for a in ax:
    a.set_xlabel("spatial frequency [arcsec$^{-1}$]")
    for b, nm in ((BEAMS[0], "compact beam"), (BEAMS[ILB], "λ/B$_{max}$"), (BEAMS[-1], "extended beam")):
        a.axvline(1 / b[0], color=MUTED, lw=0.8, ls=":")
        a.text(1 / b[0], 0.97, nm + " ", rotation=90, fontsize=8, color=MUTED, va="top", ha="right",
               transform=a.get_xaxis_transform())
ax[0].set_ylabel("power of the line moment 0"); ax[1].set_ylabel("|model| / |truth|"); ax[2].set_ylabel("|model − truth| / |truth|")
ax[0].set_title("power spectra at the extended clean beam (reweighting on)", fontsize=10.5)
ax[1].set_title("recovered amplitude", fontsize=10.5); ax[2].set_title("error amplitude", fontsize=10.5)
ax[0].legend(fontsize=8)
save(fig, "fista_vs_pdtau_power")
