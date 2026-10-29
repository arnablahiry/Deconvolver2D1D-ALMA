#!/usr/bin/env python
"""
Paper versions (A&A, drawn at final size: 88 mm column / 180 mm page width, fonts 6-9 pt, no titles -- the captions
carry the explanation) of the solver comparison on the real NGC 3110 compact data: the starlet deconvolution with four
solvers, k = 5, the production reweighting, 20,000 iterations, against the joint compact + extended MS-CLEAN
(results/NGC3110/ladder_joint/b00..b10, restored at each beam):

  pdtau, W^T       the production solver: the analysis problem
  FISTA, W^T       FISTA on the same analysis problem (exact prox, inner dual loop with W and W^T)
  pdtau, W^-1      pdtau with the dual brought back by the starlet's synthesis (sum of planes) instead of W^T
  FISTA, W^-1      the pipeline's FISTA, x <- P+(W^-1 soft(W z))

  paper_power_solvers    (180 mm) line moment 0 power spectra in surface brightness, central 16": (a) power, (b) amplitude
                         relative to the joint MS-CLEAN at the same beam, (c) error amplitude against it; the solvers at
                         the extended clean beam (vs b10), the compact dirty (natural weighting, per compact clean beam
                         area) and the compact MS-CLEAN at the compact clean beam (vs b00)
  paper_ladder_solvers   (88 mm) (a) cube RMSE vs the joint MS-CLEAN inside each beam's emission mask (the masks of
                         fig_resolution_ladder.py REF=joint), (b) the difference from pdtau, W^T; the beams drawn below
  paper_maps_solvers     (180 mm) line moment 0 at the extended clean beam: the joint MS-CLEAN and the four solvers (top),
                         each solver's error, model - truth, with the emission mask (bottom)

    fig_paper_solvers.py  -> figures/paper/{paper_power_solvers,paper_ladder_solvers,paper_maps_solvers}.{pdf,png}
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
from exp23_style import HUE, INK, MUTED  # noqa: E402

FIG = os.path.join(REPO, "figures", "paper"); os.makedirs(FIG, exist_ok=True)
MM = 1 / 25.4
plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True, "xtick.major.size": 3, "ytick.major.size": 3,
                     "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
                     "lines.markersize": 3.5, "legend.frameon": True, "legend.framealpha": 0.95, "legend.edgecolor": "0.6",
                     "legend.fancybox": False, "mathtext.fontset": "dejavuserif", "savefig.dpi": 400})
LINE, DV, CELL, HALF, F4 = line_channels("ngc3110"), 25.0, 0.04, 5.0, np.s_[75:325, 75:325]
omega = lambda b: np.pi / (4 * np.log(2)) * b[0] * b[1]
hb = lambda h: (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"])
rd = lambda p: np.nan_to_num(fits.getdata(os.path.join(REPO, p))).squeeze().astype(np.float64)
avg2 = lambda c: c.reshape(c.shape[0], c.shape[1] // 2, 2, c.shape[2] // 2, 2).mean((2, 4))
sum2 = lambda c: c.reshape(c.shape[0], c.shape[1] // 2, 2, c.shape[2] // 2, 2).sum((2, 4))
m0 = lambda c: c.sum(0) * DV                                                    # over the line channels


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig); print("->", os.path.join(FIG, name + ".pdf"))


# ------------------------------------------------------------------ data
BEAMS, J = [], []
for i in range(11):
    p = f"results/NGC3110/ladder_joint/b{i:02d}/clean_finished/image.fits"
    BEAMS.append(hb(fits.getheader(os.path.join(REPO, p)))); J.append(rd(p)[LINE][:, 440:840, 440:840])
NB, ILB = 11, 5
BE, BC, BL = BEAMS[-1], BEAMS[0], BEAMS[ILB]
MZ = np.load(os.path.join(REPO, "results/NGC3110/figures/resolution_ladder_joint/masks.npz"))
MASK = [MZ[f"b{i:02d}"] for i in range(NB)]
KFT = [beam_ft((400, 400), b, CELL) for b in BEAMS]
D24, D23 = os.path.join(EXP_DATA, "24_fista_vs_pdtau", "on"), os.path.join(EXP_DATA, "23_iht", "real")
SOLVERS = [("pdtau, W$^{T}$", os.path.join(D24, "pdtau.npy"), HUE["pipeline"], "-", "o"),
           ("FISTA, W$^{T}$", os.path.join(D24, "fista_exact.npy"), HUE["starlet"], "--", "D"),
           ("pdtau, W$^{-1}$", os.path.join(D23, "starlet_pdsyn_k5.npy"), HUE["imagenet"], "-", "^"),
           ("FISTA, W$^{-1}$", os.path.join(D24, "fista_current.npy"), HUE["network"], "-", "s")]
X = []
for lab, path, col, ls, mk in SOLVERS:
    x = np.load(path).astype(np.float64)
    X.append((sum2(x) if x.shape[-1] == 800 else x)[LINE])
GB = [[conv(x, KFT[i]).cpu().numpy() for i in range(NB)] for x in X]          # each solver at each beam
RMSE = [[100 * np.linalg.norm((g[i] - J[i]) * MASK[i]) / np.linalg.norm(J[i] * MASK[i]) for i in range(NB)] for g in GB]

# ------------------------------------------------------------------ (1) power spectra, 180 mm
truth = m0(J[-1]) / omega(BE)
sols = [m0(g[-1]) / omega(BE) for g in GB]
hC = fits.getheader(os.path.join(REPO, "results/NGC3110/compact/msclean/image.fits"))
clean_c = m0(avg2(rd("results/NGC3110/compact/msclean/image.fits"))[LINE]) / omega(hb(hC))
dirty_c = m0(avg2(rd("data/ngc3110/compact/cube/dirty_cube.fits"))[LINE]) / omega(BC)


def radial(img):
    p_ = np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2
    yy, xx = np.indices(p_.shape); n = img.shape[0]
    kr = np.hypot(yy - n // 2, xx - n // 2) / (n * CELL)
    edges = np.geomspace(1 / (n * CELL), 0.5 / CELL, 40); idx = np.digitize(kr.ravel(), edges)
    return np.sqrt(edges[1:] * edges[:-1]), np.array([p_.ravel()[idx == q].mean() if (idx == q).any() else np.nan
                                                      for q in range(1, len(edges))])


truth_c = m0(J[0]) / omega(BC)                                                # the joint MS-CLEAN at the compact clean beam
kk, pt = radial(truth)
# every curve is compared with the joint MS-CLEAN restored at the same beam: the solvers (convolved to the extended
# clean beam) with b10, the compact MS-CLEAN (restored at the compact clean beam) and the compact dirty with b00
curves = [(lab, img, col, ls, 1.1, truth) for (lab, _, col, ls, _), img in zip(SOLVERS, sols)] + \
         [("compact MS-CLEAN", clean_c, HUE["fire400"], "-.", 1.0, truth_c),
          ("compact dirty", dirty_c, MUTED, ":", 1.2, truth_c)]
fig, ax = plt.subplots(1, 3, figsize=(180 * MM, 58 * MM), gridspec_kw=dict(wspace=0.42))
ax[0].loglog(kk, pt, color=INK, lw=1.8, label=r"joint MS-CLEAN, $\theta_\mathrm{e}$")
ax[0].loglog(kk, radial(truth_c)[1], color=INK, lw=0.9, ls="--", label=r"joint MS-CLEAN, $\theta_\mathrm{c}$")
for lab, img, col, ls, lw, ref in curves:
    pk, pr = radial(img)[1], radial(ref)[1]
    ax[0].loglog(kk, pk, color=col, ls=ls, lw=lw, label=lab)
    ax[1].semilogx(kk, np.sqrt(pk / pr), color=col, ls=ls, lw=lw)
    ax[2].loglog(kk, np.sqrt(radial(img - ref)[1] / pr), color=col, ls=ls, lw=lw)
ax[1].axhline(1, color=INK, lw=0.7)
for a, letter in zip(ax, "abc"):
    a.set_xlabel(r"spatial frequency (arcsec$^{-1}$)")
    for f, nm in ((1 / BC[0], r"$\theta_\mathrm{c}$"), (1 / BL[0], r"$\lambda/B_\mathrm{max}$"), (1 / BE[0], r"$\theta_\mathrm{e}$")):
        a.axvline(f, color=MUTED, lw=0.5, ls=":")
        a.text(f, 1.015, nm, fontsize=6, color=MUTED, ha="center", va="bottom", transform=a.get_xaxis_transform())
    a.text(0.04, 0.96, f"({letter})", transform=a.transAxes, fontsize=7.5, fontweight="bold", va="top")
ax[0].set_ylabel(r"$P(k)$ (Jy$^{2}$ arcsec$^{-4}$ km$^{2}$ s$^{-2}$)")
ax[1].set_ylabel(r"$|X| / |X_\mathrm{joint}|$"); ax[2].set_ylabel(r"$|X - X_\mathrm{joint}| / |X_\mathrm{joint}|$")
ax[1].set_ylim(-0.05, 3.1)
ax[2].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax[2].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[2].set_yticks([0.1, 0.2, 0.5, 1, 2, 3])
ax[1].legend(*ax[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.02, 0.86), fontsize=5.5,
             handlelength=2.2, borderpad=0.3, labelspacing=0.25)               # the empty upper part of (b)
save(fig, "paper_power_solvers")

# ------------------------------------------------------------------ (2) resolution ladder, 88 mm
fig = plt.figure(figsize=(88 * MM, 96 * MM))
a1 = fig.add_axes([0.16, 0.58, 0.82, 0.41]); a2 = fig.add_axes([0.16, 0.25, 0.82, 0.31], sharex=a1)
x = np.arange(NB)
for (lab, _, col, ls, mk), r in zip(SOLVERS, RMSE):
    kw = dict(color=col, ls=ls, marker=mk, mfc="white" if ls != "-" else col, mew=0.8)
    a1.plot(x, r, label=lab, **kw)
    a2.plot(x, [v - p for v, p in zip(r, RMSE[0])], **kw)
a2.axhline(0, color=HUE["pipeline"], lw=0.7)
for a in (a1, a2):
    a.axvline(ILB, color=MUTED, lw=0.5, ls=":")
a1.set_ylabel(r"RMSE (%)"); a2.set_ylabel(r"$\Delta$RMSE (pp)")
a1.tick_params(labelbottom=False)
a2.set_xticks(x)
a2.set_xticklabels([f"{b[0]:.2f}" if abs(b[0] - b[1]) < 1e-3 else f"{b[0]:.2f}×\n{b[1]:.2f}" for b in BEAMS], fontsize=5.5)
a2.set_xlim(-0.6, NB - 0.4)
a1.legend(loc="upper left", fontsize=6)
for a, letter in ((a1, "a"), (a2, "b")):
    a.text(0.97 if letter == "a" else 0.03, 0.06 if letter == "a" else 0.92, f"({letter})", transform=a.transAxes,
           fontsize=7.5, fontweight="bold", ha="right" if letter == "a" else "left", va="bottom" if letter == "a" else "top")
a2.set_xlabel("beam FWHM (arcsec)", labelpad=2)
s = fig.add_axes([0.16, 0.09, 0.82, 0.06]); s.set_xlim(-0.6, NB - 0.4); s.axis("off")
W_in, H_in = 0.82 * 88 * MM, 0.06 * 96 * MM
du = (NB + 0.2) / W_in; s.set_ylim(0, H_in * du)
k_ = 0.55 * (W_in / (NB + 0.2)) / max(b[0] for b in BEAMS) * du               # data units per arcsec
for i, b in enumerate(BEAMS):
    s.add_patch(Ellipse((i, H_in * du * 0.5), width=b[0] * k_, height=b[1] * k_, angle=90 + b[2], facecolor="none",
                        edgecolor=INK if i in (0, ILB, NB - 1) else MUTED, lw=0.7))
save(fig, "paper_ladder_solvers")


# ------------------------------------------------------------------ (3) maps at the extended clean beam, 180 mm
def scalebar(a):
    xs, ys, cap = -HALF + 0.08 * HALF, HALF - 0.1 * HALF, 0.03 * HALF
    a.plot([xs + 2, xs], [ys, ys], color="w", lw=1.2, solid_capstyle="butt")
    for xe in (xs, xs + 2):
        a.plot([xe, xe], [ys - cap, ys + cap], color="w", lw=0.9, solid_capstyle="butt")
    a.annotate("2″", (xs + 1, ys - cap), xytext=(0, -1.5), textcoords="offset points", color="w", ha="center",
               va="top", fontsize=6.5, fontweight="bold")


def beam(a, b, col="w"):
    pad = 0.06 * HALF
    a.add_patch(Ellipse((HALF - pad - b[0] / 2, -HALF + pad + b[0] / 2), width=b[0], height=b[1], angle=90 - b[2],
                        facecolor=col, edgecolor=col))


EXT = [HALF, -HALF, -HALF, HALF]
tm = (m0(J[-1]) / omega(BE))[F4]
ms = [(m0(g[-1]) / omega(BE))[F4] for g in GB]
vmax = np.percentile(tm, 99.8); lim = np.percentile(np.abs(np.concatenate([(m - tm).ravel() for m in ms])), 99.5)
foot = MASK[-1].any(0)[F4]
n = 1 + len(SOLVERS); P = 31 * MM; GAP = 1.2 * MM
fig = plt.figure(figsize=(n * P + (n - 1) * GAP + 12 * MM, 2 * P + GAP + 2 * MM))
FW, FH = fig.get_size_inches()
axes = [[fig.add_axes([(c * (P + GAP)) / FW, (P + GAP if r == 0 else 0) / FH + 1 * MM / FH, P / FW, P / FH])
         for c in range(n)] for r in range(2)]
for c in range(n):
    a = axes[0][c]
    img = tm if c == 0 else ms[c - 1]
    h0 = a.imshow(img, origin="lower", extent=EXT, cmap="inferno", interpolation="nearest",
                  norm=matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=vmax))
    a.text(0.04, 0.96, "Joint\nMS-CLEAN" if c == 0 else SOLVERS[c - 1][0], transform=a.transAxes, color="w", ha="left",
           va="top", fontsize=7, fontweight="bold", linespacing=1.0)
    beam(a, BE)
    if c == 0:
        scalebar(a)
    b = axes[1][c]
    if c == 0:
        b.axis("off")
        b.text(0.5, 0.5, "model − joint\n(same beam)", transform=b.transAxes, ha="center", va="center", fontsize=7)
    else:
        h1 = b.imshow(ms[c - 1] - tm, origin="lower", extent=EXT, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        b.contour(np.linspace(HALF, -HALF, foot.shape[1]), np.linspace(-HALF, HALF, foot.shape[0]), foot, levels=[0.5],
                  colors=["0.35"], linewidths=0.4)
        b.text(0.04, 0.96, f"{RMSE[c - 1][-1]:.1f}%", transform=b.transAxes, color=INK, ha="left", va="top", fontsize=7,
               fontweight="bold")
        beam(b, BE, INK)
    for a_ in (a, b):
        a_.set_xticks([]); a_.set_yticks([])
        for sp in a_.spines.values():
            sp.set_linewidth(0.5)
x0 = (n * (P + GAP)) / FW
c0 = fig.add_axes([x0, (P + GAP) / FH + 1 * MM / FH, 1.6 * MM / FW, P / FH]); cb = fig.colorbar(h0, cax=c0)
cb.set_label(r"Jy arcsec$^{-2}$ km s$^{-1}$", fontsize=6.5); cb.ax.tick_params(labelsize=6)
c1 = fig.add_axes([x0, 1 * MM / FH, 1.6 * MM / FW, P / FH]); cb = fig.colorbar(h1, cax=c1)
cb.set_label(r"Jy arcsec$^{-2}$ km s$^{-1}$", fontsize=6.5); cb.ax.tick_params(labelsize=6)
save(fig, "paper_maps_solvers")
