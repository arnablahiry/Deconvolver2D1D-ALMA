#!/usr/bin/env python
"""
NGC 3110: the extended MS-CLEAN continued inside the compact model's per-channel footprint
(scripts/experiments/msclean_compact_mask.py) next to the compact 2D-1D model and the original extended CLEAN, with the
masks as contours. Nothing is masked: the contours show where each CLEAN was allowed to put components.

  cyan      the compact footprint (compact_only_residual.compact_footprint: compact model * extended beam > 1% of the
            channel's peak), = the compact-mask CLEAN's mask in the line channels
  white --  the original extended CLEAN's astrodendro mask (extended/msclean_finished/mask_final.npy)

  compact_mask_mom0     moment 0 (line channels) at the extended clean beam, Jy arcsec^-2 km/s, and the last CLEAN's
                        residual summed over the line channels (Jy/beam km/s, dirty beam); contours: pixels in the
                        mask in >= MIN_CH line channels (the union over all of them covers most of the field)
  compact_mask_chans    eight line channels: one row per model (mJy/beam at the extended clean beam) and the
                        residual row (mJy per dirty beam); contours: that channel's masks

The residual is shown at its native resolution in mJy/beam, not in sigma: in the line channels its rms is 1-4x
(native) and up to 8x (0.6" smoothing) that of the line-free channels (sidelobes and negative bowls of what CLEAN has
not modelled), so a line-free sigma saturates everywhere.

    CLEANS=msclean_finished,msclean_cmask_n3[,msclean_cmask_n2] python scripts/compact_vs_msclean_dendro/fig_compact_mask_clean.py
    -> results/NGC3110/figures/compact_mask_clean/{compact_mask_mom0,compact_mask_chans}.{png,pdf}
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple")); sys.path.insert(0, os.path.join(REPO, "scripts", "experiments"))
from restore import convolve_with_beam  # noqa: E402
from compact_only_residual import compact_footprint  # noqa: E402

R = os.path.join(REPO, "results", "NGC3110")
CLEANS = os.environ.get("CLEANS", "msclean_finished,msclean_cmask_n3").split(",")
NAMES = {"msclean_finished": "Extended, own mask", "msclean_cmask_n3": "Extended, compact mask (3σ)",
         "msclean_cmask_n2": "Extended, compact mask (2σ)"}
OUT = os.path.join(R, "figures", "compact_mask_clean"); os.makedirs(OUT, exist_ok=True)
CYAN, MIN_CH = "#3fd7ff", 3

plt.rcParams.update({"font.family": "serif", "font.size": 13, "xtick.direction": "in", "ytick.direction": "in"})
rd = lambda p: np.nan_to_num(fits.getdata(p).astype(np.float64)).squeeze()
omega = lambda b: np.pi * b[0] * b[1] / (4 * np.log(2))
c = json.load(open(os.path.join(R, "config.json")))
L = json.load(open(os.path.join(R, "line_channels.json")))["line"]
cell, dv, half = c["cell_arcsec"], c["width_kms"], float(c["half_arcsec"])
vel = c["start_kms"] + dv * np.arange(c["nchan"])
h = fits.getheader(os.path.join(R, "extended", CLEANS[0], "image.fits")); B = (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"])

C = compact_footprint(R)
E0 = np.load(os.path.join(R, "extended", "msclean_finished", "mask_final.npy")) > 0
models = {"Compact": np.load(os.path.join(R, "compact", "deconvolved", "model.npy")).astype(np.float64)[L]}
for k in CLEANS:
    models[NAMES.get(k, k)] = rd(os.path.join(R, "extended", k, "model.fits"))[L]       # Jy/pixel, line channels
flux = {n: float(x.sum() * dv) for n, x in models.items()}                               # Jy km/s, whole field
cubes = {n: convolve_with_beam(x, cell, *B) for n, x in models.items()}                  # Jy/beam
res = 1e3 * rd(os.path.join(R, "extended", CLEANS[-1], "residual.fits"))[L]             # mJy per dirty beam
RES_NAME = "Residual" + (", " + NAMES[CLEANS[-1]].split(", ")[1] if ", " in NAMES.get(CLEANS[-1], "") else "")

c0 = res.shape[1] // 2; hp = int(round(half / cell)); sl = np.s_[c0 - hp:c0 + hp, c0 - hp:c0 + hp]
EXT = [half, -half, -half, half]


def decorate(a, label, beam=True, bar=True, fs=13):
    """the house style: bold label top left, beam bottom left, 2" scale bar top right, no ticks"""
    a.set_xticks([]); a.set_yticks([])
    a.text(0.04, 0.96, label, transform=a.transAxes, color="w", ha="left", va="top", fontsize=fs, fontweight="bold")
    if beam:
        pad = 0.06 * half
        a.add_patch(Ellipse((half - pad - B[0] / 2, -half + pad + B[0] / 2), width=B[0], height=B[1], angle=90 - B[2],
                            facecolor="w", edgecolor="w"))
    if bar:
        xs, ys, cap = -half + 0.08 * half, half - 0.1 * half, 0.03 * half
        a.plot([xs + 2, xs], [ys, ys], color="w", lw=2.0, solid_capstyle="butt")
        for xe in (xs, xs + 2):
            a.plot([xe, xe], [ys - cap, ys + cap], color="w", lw=1.4, solid_capstyle="butt")
        a.annotate("2″", (xs + 1, ys - cap), xytext=(0, -2.5), textcoords="offset points", color="w", ha="center",
                   va="top", fontsize=fs - 1, fontweight="bold")


def contours(a, cm, em, lw=1.0, ecol="w"):
    a.contour(cm[sl].astype(float), levels=[0.5], colors=CYAN, linewidths=lw, extent=EXT, origin="lower")
    a.contour(em[sl].astype(float), levels=[0.5], colors=ecol, linewidths=lw, linestyles="--", extent=EXT, origin="lower")


def legend(fig, axes, fs, proj):
    y = max(a.get_position().y1 for a in axes)
    fig.legend(handles=[Line2D([], [], color=CYAN, lw=1.5, label="compact footprint = compact-mask CLEAN's mask" + proj),
                        Line2D([], [], color="k", lw=1.5, ls="--", label="original extended CLEAN mask" + proj)],
               loc="lower center", ncol=2, frameon=False, fontsize=fs, bbox_to_anchor=(0.5, y + 0.005))


# ======================================================================== moment 0
n = len(cubes) + 1; S = 3.6
fig, ax = plt.subplots(1, n, figsize=(S * n + 0.3, S + 1.3), gridspec_kw=dict(wspace=0.04))
m0 = {k: x.sum(0) * dv / omega(B) for k, x in cubes.items()}           # Jy arcsec^-2 km/s
CP, EP = C[L].sum(0) >= MIN_CH, E0[L].sum(0) >= MIN_CH
norm = matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=np.percentile(m0["Compact"][CP], 99.9))
for j, (a, (k, im2)) in enumerate(zip(ax, m0.items())):
    im = a.imshow(im2[sl], origin="lower", extent=EXT, cmap="inferno", norm=norm, interpolation="nearest")
    contours(a, CP, EP); decorate(a, k.replace(", ", ",\n"), bar=j == 0)
    a.text(0.5, -0.03, f"{flux[k]:.0f} Jy km s$^{{-1}}$", transform=a.transAxes, ha="center", va="top", fontsize=12)
cb = fig.colorbar(im, ax=list(ax[:-1]), orientation="horizontal", fraction=0.05, pad=0.1, aspect=60)
cb.set_label(r"Jy arcsec$^{-2}$ km s$^{-1}$ (extended clean beam)")
a = ax[-1]; r0 = res.sum(0) * dv / 1e3                                  # Jy/beam km/s
lim = np.percentile(np.abs(r0[sl]), 99)
im = a.imshow(r0[sl], origin="lower", extent=EXT, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
contours(a, CP, EP, ecol="k"); decorate(a, RES_NAME.replace(", ", ",\n"), beam=False, bar=False)
a.texts[0].set_color("k")
cb = fig.colorbar(im, ax=a, orientation="horizontal", fraction=0.05, pad=0.1); cb.set_label(r"Jy beam$^{-1}$ km s$^{-1}$ (dirty)")
cb.locator = matplotlib.ticker.MaxNLocator(5); cb.update_ticks()
legend(fig, ax, 12, f" (in ≥ {MIN_CH} line channels)")
fig.savefig(os.path.join(OUT, "compact_mask_mom0.png"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "compact_mask_mom0.pdf"), dpi=300, bbox_inches="tight"); plt.close(fig)

# ======================================================================== channel maps
pick = [L[i] for i in np.linspace(1, len(L) - 2, 8).round().astype(int)]
rows = list(cubes) + [RES_NAME]; S = 2.0
fig, ax = plt.subplots(len(rows), len(pick), figsize=(S * len(pick) + 1.2, S * len(rows) + 0.4),
                       gridspec_kw=dict(wspace=0.03, hspace=0.03))
rlim = np.percentile(np.abs(res[[L.index(ch) for ch in pick]][:, sl[0], sl[1]]), 99)
for j, ch in enumerate(pick):
    i = L.index(ch)
    vm = 1e3 * cubes["Compact"][i][C[ch]].max()
    for r, k in enumerate(rows):
        a = ax[r, j]
        if k == RES_NAME:
            im_r = a.imshow(res[i][sl], origin="lower", extent=EXT, cmap="RdBu_r", vmin=-rlim, vmax=rlim, interpolation="nearest")
        else:
            im = a.imshow(1e3 * cubes[k][i][sl], origin="lower", extent=EXT, cmap="inferno",
                          norm=matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=vm), interpolation="nearest")
        contours(a, C[ch], E0[ch], lw=0.7, ecol="k" if k == RES_NAME else "w")
        decorate(a, f"{vel[ch]:.0f} km s$^{{-1}}$" if r == 0 else "", beam=(r == 0 and j == 0), bar=(r == 0 and j == 0), fs=10)
        if j == 0:
            a.set_ylabel(k.replace(", ", ",\n"), fontsize=11)
fig.colorbar(im, ax=ax[:-1, :].ravel().tolist(), fraction=0.015, pad=0.01).set_label(
    "mJy beam$^{-1}$ (range per channel: compact peak)")
fig.colorbar(im_r, ax=ax[-1, :].ravel().tolist(), fraction=0.015, pad=0.01).set_label("mJy beam$^{-1}$ (dirty)")
legend(fig, ax[0], 12, " (this channel)")
fig.savefig(os.path.join(OUT, "compact_mask_chans.png"), dpi=130, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "compact_mask_chans.pdf"), dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved", OUT, {k: round(v, 1) for k, v in flux.items()})
