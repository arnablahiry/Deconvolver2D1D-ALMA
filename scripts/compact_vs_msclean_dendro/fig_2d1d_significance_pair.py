#!/usr/bin/env python
"""
2D-1D significance of the NGC 3110 compact data before and after the deconvolution, as two image grids side by side:
(a) the dirty cube d, (b) the residual d - H x of the pipeline's deconvolved model (results/NGC3110/compact/
deconvolved/model.npy, pdtau, k = 5; saved at 0.02", block-summed back to the 0.04" grid of the deconvolution).
Each grid: n_2D = 1 ... 8 across, n_1D = 1 ... 6 down (simple/gpu_pd.Dict2D1D, J = 7 starlet scales x L = 5 CDF 9/7
levels); each panel |alpha_b| / sigma_b maximised over the sub-band's spectral coefficients, sigma_b = max(sigma_real,
sigma_white) of gpu_pd.noise_lambdas (the deconvolution's thresholds are lambda_b = k sigma_b). One logarithmic
colour scale for both grids; in the panel corner, the peak inside the emission footprint (extended astrodendro mask
over the line channels) -- over the whole field the residual's peaks sit at the edge of the 16" field. Drawn at the A&A text width (523.5 pt).

    [OUT=...] [K=5] [VMAX=2000] python scripts/compact_vs_msclean_dendro/fig_2d1d_significance_pair.py
    -> results/gallery/ngc3110_2d1d_significance_pair.{png,pdf}
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
import gpu_pd as G  # noqa: E402

R = os.path.join(REPO, "results", "NGC3110")
OUT = os.environ.get("OUT", os.path.join(REPO, "results", "gallery", "ngc3110_2d1d_significance_pair.png"))
K, VMAX = float(os.environ.get("K", 5)), float(os.environ.get("VMAX", 2000))
J, L, BIN = 7, 5, 2
plt.rcParams.update({"font.family": "serif", "font.size": 8})

c = json.load(open(os.path.join(R, "config.json")))
line = json.load(open(os.path.join(R, "line_channels.json")))["line"]
rd = lambda p: np.nan_to_num(fits.getdata(p).astype(np.float64)).squeeze()
dirty, psf = G.rebin(rd(os.path.join(R, "compact/cube/dirty_cube.fits")), rd(os.path.join(R, "compact/cube/dirty_beam_cube.fits")), BIN)
nz, ny, nx = dirty.shape
x = np.load(os.path.join(R, "compact/deconvolved/model.npy")).astype(np.float64).reshape(nz, ny, BIN, nx, BIN).sum((2, 4))
resid = (-G.Op(psf, dirty).grad(G.to_t(x))).cpu().numpy()
D = G.Dict2D1D(dirty.shape, J, L, dict_2d="starlet")
_, sr, sw, _ = G.noise_lambdas(D, G.to_t(dirty), [i for i in range(nz) if i not in line], k=1.0)
sigma = np.maximum(sr.cpu().numpy(), sw.cpu().numpy())
M = np.load(os.path.join(R, "extended/msclean_finished/mask_final.npy")).astype(bool)
foot = M[line].any(0).reshape(ny, BIN, nx, BIN).any((1, 3))
bands = dict(G.band_slices(nz, L)); rows = [f"d{lev}" for lev in range(1, L + 1)] + ["a"]

cell, half = c["cell_arcsec"] * BIN, float(c["half_arcsec"])
c0 = ny // 2; hp = int(round(half / cell)); sl = np.s_[c0 - hp:c0 + hp, c0 - hp:c0 + hp]
ext = [half, -half, -half, half]
snr = {k: np.abs(D.W(G.to_t(v)).cpu().numpy()) / sigma for k, v in (("dirty", dirty), ("resid", resid))}

# ======================================================================== layout (inches): A&A text width, 1:1
TW = 523.5307 / 72.27
ML, MR, MT, MB, GAP, BG, CBW, CBP = 0.50, 0.58, 0.50, 0.09, 0.025, 0.16, 0.07, 0.07
S = (TW - ML - MR - BG - 2 * (J + 1 - 1) * GAP) / (2 * (J + 1))
GW = (J + 1) * S + J * GAP
H = len(rows) * S + (len(rows) - 1) * GAP
FH = MT + H + MB
fig = plt.figure(figsize=(TW, FH))
norm = matplotlib.colors.LogNorm(1, VMAX, clip=True)
for g, (key, title) in enumerate((("dirty", "(a) dirty cube"), ("resid", "(b) residual after deconvolution"))):
    xg = ML + g * (GW + BG)
    fig.text((xg + GW / 2) / TW, (FH - 0.04) / FH, title, ha="center", va="top", fontsize=9)
    for r, bk in enumerate(rows):
        seg = bands[bk]
        for j in range(J + 1):
            x0 = xg + j * (S + GAP); y0 = FH - MT - (r + 1) * S - r * GAP
            a = fig.add_axes([x0 / TW, y0 / FH, S / TW, S / FH])
            m = snr[key][j, seg].max(0)
            im = a.imshow(np.maximum(m[sl], 1.0), origin="lower", cmap="inferno", norm=norm, interpolation="nearest",
                          extent=ext)
            pk = m[foot].max()
            a.text(0.06, 0.92, f"{pk:.0f}" if pk >= 10 else f"{pk:.1f}", transform=a.transAxes, ha="left", va="top",
                   fontsize=5.5, color="w", fontweight="bold" if pk >= K else "normal")
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_linewidth(0.4)
            if r == 0:
                a.set_title(rf"$\mathbf{{{j + 1}}}$", fontsize=7, pad=2)
            if j == 0 and g == 0:
                a.text(-0.12, 0.5, rf"$\mathbf{{{r + 1}}}$", transform=a.transAxes, ha="right", va="center", fontsize=7)
    fig.text((xg + GW / 2) / TW, (FH - 0.21) / FH, r"$n_\mathrm{2D}$", ha="center", va="top", fontsize=8)
fig.text((ML - 0.3) / TW, (FH - MT - H / 2) / FH, r"$n_\mathrm{1D}$", rotation=90, ha="center", va="center", fontsize=8)
cax = fig.add_axes([(TW - MR + CBP) / TW, MB / FH, CBW / TW, H / FH])
cb = fig.colorbar(im, cax=cax)
cb.ax.axhline(K, color="#7fdbff", lw=1.0)
cb.set_label(r"$|\alpha_b| / \sigma_b$", fontsize=8, labelpad=2)
cax.tick_params(labelsize=7, length=2)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=400)
fig.savefig(OUT.replace(".png", ".pdf"), dpi=400)
print("saved", OUT)
