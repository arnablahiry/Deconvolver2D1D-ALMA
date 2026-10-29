#!/usr/bin/env python
"""
Power spectra of the NGC 3110 line moment 0 (real data, central 16", the 25 km/s line channels), in surface brightness
(Jy/arcsec^2 km/s, each image divided by its own beam area) so that every image is on one scale:

  reference      the joint compact + extended MS-CLEAN, restored at the extended clean beam
                 (results/NGC3110/ladder_joint/b10/clean_finished/image.fits) -- the ground truth
  solvers        the starlet deconvolution of the compact data with four solvers, k = 5, the production reweighting,
                 20,000 iterations (data/experiments/24_fista_vs_pdtau/on/, data/experiments/23_iht/real/), each
                 convolved with the extended clean beam to compare with the truth at its resolution:
                   pdtau with W^T    the production solver (analysis problem)
                   FISTA with W^T    FISTA on the same analysis problem (exact prox by an inner dual loop)
                   pdtau with W^-1   pdtau with the dual brought back by the starlet's synthesis (sum of planes)
                   FISTA with W^-1   the pipeline's FISTA, x <- P+(W^-1 soft(W z))
  compact data   the compact dirty cube (natural weighting; per compact clean beam area) and the compact MS-CLEAN
                 restored at the compact clean beam (results/NGC3110/compact/msclean/image.fits), at their own
                 resolution: what the compact data give without deconvolution to the joint resolution

Panels: the azimuthally averaged power; the amplitude ratio |X| / |truth|; the error amplitude |X - truth| / |truth|.

    fig_power_solvers.py  -> figures/experiments/24_fista_vs_pdtau/power_solvers.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import beam_ft, conv  # noqa: E402
from common import EXP_DATA, REPO, BEAMS, line_channels  # noqa: E402
from exp23_style import HUE, INK, MUTED, rc  # noqa: E402

FIG = os.path.join(REPO, "figures", "experiments", "24_fista_vs_pdtau"); os.makedirs(FIG, exist_ok=True)
LINE, DV, CELL = line_channels("ngc3110"), 25.0, 0.04
rc()
omega = lambda b: np.pi / (4 * np.log(2)) * b[0] * b[1]                      # arcsec^2
hb = lambda h: (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"])
rd = lambda p: np.nan_to_num(fits.getdata(os.path.join(REPO, p))).squeeze().astype(np.float64)
avg2 = lambda c: c.reshape(c.shape[0], c.shape[1] // 2, 2, c.shape[2] // 2, 2).mean((2, 4))   # 0.02" -> 0.04", Jy/beam
sum2 = lambda c: c.reshape(c.shape[0], c.shape[1] // 2, 2, c.shape[2] // 2, 2).sum((2, 4))    # 0.02" -> 0.04", Jy/pixel
m0 = lambda c: c[LINE].sum(0) * DV

# the truth: the joint MS-CLEAN at the extended clean beam, its central 16"
p = "results/NGC3110/ladder_joint/b10/clean_finished/image.fits"
BE = hb(fits.getheader(os.path.join(REPO, p)))
truth = m0(rd(p)[:, 440:840, 440:840]) / omega(BE)
kE = beam_ft((400, 400), BE, CELL)
# the solvers (Jy/pixel at 0.04"), each at the extended clean beam
D24, D23 = os.path.join(EXP_DATA, "24_fista_vs_pdtau", "on"), os.path.join(EXP_DATA, "23_iht", "real")
SOLVERS = [("pdtau with W$^{T}$ (production)", os.path.join(D24, "pdtau.npy"), HUE["pipeline"], "-", 2.0),
           ("FISTA with W$^{T}$", os.path.join(D24, "fista_exact.npy"), HUE["starlet"], "--", 1.8),
           ("pdtau with W$^{-1}$", os.path.join(D23, "starlet_pdsyn_k5.npy"), HUE["imagenet"], "-", 1.6),
           ("FISTA with W$^{-1}$ (your FISTA)", os.path.join(D24, "fista_current.npy"), HUE["network"], "-", 1.6)]
imgs = []
for lab, path, col, ls, lw in SOLVERS:
    x = np.load(path).astype(np.float64)
    x = sum2(x) if x.shape[-1] == 800 else x
    imgs.append((lab, (conv(x[LINE], kE).cpu().numpy().sum(0) * DV) / omega(BE), col, ls, lw))
# the compact data at their own resolution
BC = BEAMS[("ngc3110", "compact")]
hC = fits.getheader(os.path.join(REPO, "results/NGC3110/compact/msclean/image.fits"))
clean_c = m0(avg2(rd("results/NGC3110/compact/msclean/image.fits"))) / omega(hb(hC))            # 16" at 0.02" -> 0.04"
dirty_c = m0(avg2(rd("data/ngc3110/compact/cube/dirty_cube.fits"))) / omega(BC)
REFS = [("compact MS-CLEAN, restored (compact clean beam)", clean_c, HUE["fire400"], "-.", 1.6),
        ("compact dirty (natural)", dirty_c, MUTED, ":", 1.8)]


def radial(img):
    p_ = np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2
    yy, xx = np.indices(p_.shape); n = img.shape[0]
    kr = np.hypot(yy - n // 2, xx - n // 2) / (n * CELL)
    edges = np.geomspace(1 / (n * CELL), 0.5 / CELL, 40); idx = np.digitize(kr.ravel(), edges)
    return np.sqrt(edges[1:] * edges[:-1]), np.array([p_.ravel()[idx == q].mean() if (idx == q).any() else np.nan
                                                      for q in range(1, len(edges))])


kk, pt = radial(truth)
fig, ax = plt.subplots(1, 3, figsize=(19, 5.6), constrained_layout=True)
ax[0].loglog(kk, pt, color=INK, lw=2.8, label="joint MS-CLEAN, restored (extended clean beam): ground truth")
for lab, img, col, ls, lw in imgs + REFS:
    pk = radial(img)[1]
    ax[0].loglog(kk, pk, color=col, ls=ls, lw=lw, label=lab)
    ax[1].semilogx(kk, np.sqrt(pk / pt), color=col, ls=ls, lw=lw)
    ax[2].loglog(kk, np.sqrt(radial(img - truth)[1] / pt), color=col, ls=ls, lw=lw)
ax[1].axhline(1, color=INK, lw=0.9)
LB = 0.6044
for a in ax:
    a.set_xlabel("spatial frequency [arcsec$^{-1}$]")
    for f, nm in ((1 / BC[0], "compact beam"), (1 / LB, "compact λ/B$_{max}$"), (1 / BE[0], "extended beam")):
        a.axvline(f, color=MUTED, lw=0.8, ls=":")
        a.text(f, 0.97, nm + " ", rotation=90, fontsize=8, color=MUTED, va="top", ha="right", transform=a.get_xaxis_transform())
ax[0].set_ylabel("power of the line moment 0 [(Jy arcsec$^{-2}$ km s$^{-1}$)$^2$]")
ax[1].set_ylabel("|X| / |truth|  (recovered amplitude)"); ax[2].set_ylabel("|X − truth| / |truth|  (error amplitude)")
ax[0].set_title("power spectra", fontsize=11); ax[1].set_title("amplitude relative to the joint truth", fontsize=11)
ax[2].set_title("error amplitude relative to the joint truth", fontsize=11)
ax[0].legend(fontsize=8, loc="lower left")
fig.suptitle("NGC 3110, real data: line moment 0 power spectra, central 16″. The four solvers (starlet, k = 5, production "
             "reweighting) at the extended clean beam; the compact dirty and MS-CLEAN at their own resolution", fontsize=11)
fig.savefig(os.path.join(FIG, "power_solvers.png"), dpi=140, bbox_inches="tight")
print("->", os.path.join(FIG, "power_solvers.png"))
