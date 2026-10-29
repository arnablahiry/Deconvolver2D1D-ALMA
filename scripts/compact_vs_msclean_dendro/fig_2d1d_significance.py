#!/usr/bin/env python
"""
2D-1D significance of the NGC 3110 compact dirty cube: where each sub-band carries signal above the correlated noise,
i.e. what the deconvolution has to work with. Same grid as fig_2d1d_decomposition.py (0.04" grid, 2 x 2 block mean;
simple/gpu_pd.Dict2D1D, J = 7 starlet scales x L = 5 CDF 9/7 levels; n_2D = 1 ... 8 across, n_1D = 1 ... 6 down),
but each panel is |alpha_b| / sigma_b with sigma_b = max(sigma_real, sigma_white) of sub-band b, the noise of
gpu_pd.noise_lambdas (the thresholds lambda_b = k sigma_b of the deconvolution, Sect. 3.4 of the paper), maximised over
the sub-band's spectral coefficients; one logarithmic colour scale (1 ... VMAX sigma) for all panels, a contour at the pipeline's k = 5, and the
panel's peak significance in its corner. Drawn at the A&A text width (523.5 pt) at its final size.

    [OUT=...] [K=5] [VMAX=2000] python scripts/compact_vs_msclean_dendro/fig_2d1d_significance.py
    -> results/gallery/ngc3110_2d1d_significance.{png,pdf}
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
OUT = os.environ.get("OUT", os.path.join(REPO, "results", "gallery", "ngc3110_2d1d_significance.png"))
K, VMAX = float(os.environ.get("K", 5)), float(os.environ.get("VMAX", 2000))
J, L, BIN = 7, 5, 2
plt.rcParams.update({"font.family": "serif", "font.size": 8})

c = json.load(open(os.path.join(R, "config.json")))
line = json.load(open(os.path.join(R, "line_channels.json")))["line"]
rd = lambda p: np.nan_to_num(fits.getdata(p).astype(np.float64)).squeeze()
dirty, _ = G.rebin(rd(os.path.join(R, "compact/cube/dirty_cube.fits")), rd(os.path.join(R, "compact/cube/dirty_beam_cube.fits")), BIN)
cell, nz = c["cell_arcsec"] * BIN, dirty.shape[0]
noise_ch = [i for i in range(nz) if i not in line]
D = G.Dict2D1D(dirty.shape, J, L, dict_2d="starlet")
d_t = G.to_t(dirty)
_, sig_real, sig_white, _ = G.noise_lambdas(D, d_t, noise_ch, k=1.0)
sigma = np.maximum(sig_real.cpu().numpy(), sig_white.cpu().numpy())  # (J + 1, nz, 1, 1), constant per sub-band
snr = np.abs(D.W(d_t).cpu().numpy()) / sigma
bands = dict(G.band_slices(nz, L))

half = float(c["half_arcsec"])
ny = dirty.shape[1]; c0 = ny // 2; hp = int(round(half / cell)); sl = np.s_[c0 - hp:c0 + hp, c0 - hp:c0 + hp]
rows = [f"d{lev}" for lev in range(1, L + 1)] + ["a"]                 # n_1D = 1 (finest) ... L + 1 (approximation)

# ======================================================================== layout (inches): A&A text width, 1:1
TW = 523.5307 / 72.27
ML, MR, MT, MB, GAP, CBW, CBP = 0.66, 0.50, 0.24, 0.04, 0.05, 0.08, 0.08
S = (TW - ML - MR - (J + 1 - 1) * GAP) / (J + 1)
H = len(rows) * S + (len(rows) - 1) * GAP
FH = MT + H + MB
fig = plt.figure(figsize=(TW, FH))
norm = matplotlib.colors.LogNorm(1, VMAX, clip=True)
for r, bk in enumerate(rows):
    seg = bands[bk]
    for j in range(J + 1):
        x0 = ML + j * (S + GAP); y0 = FH - MT - (r + 1) * S - r * GAP
        a = fig.add_axes([x0 / TW, y0 / FH, S / TW, S / FH])
        m = snr[j, seg].max(0)[sl]                                   # peak over the sub-band's spectral coefficients
        im = a.imshow(np.maximum(m, 1.0), origin="lower", cmap="inferno", norm=norm, interpolation="nearest", extent=[half, -half, -half, half])
        if m.max() >= K:
            a.contour(m, levels=[K], colors="#7fdbff", linewidths=0.5, origin="lower", extent=[half, -half, -half, half])
        a.text(0.05, 0.94, f"{m.max():.0f}", transform=a.transAxes, ha="left", va="top", fontsize=7, color="w",
               fontweight="bold")
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_linewidth(0.5)
        if r == 0:
            a.set_title(rf"$\mathbf{{n_{{2D}} = {j + 1}}}$", fontsize=9, pad=4)
        if j == 0:
            a.text(-0.1, 0.5, rf"$\mathbf{{n_{{1D}} = {r + 1}}}$", transform=a.transAxes, ha="right", va="center", fontsize=9)
cax = fig.add_axes([(TW - MR + CBP) / TW, MB / FH, CBW / TW, H / FH])
cb = fig.colorbar(im, cax=cax)
cb.ax.axhline(K, color="#7fdbff", lw=1.0)
cb.set_label(r"$|\alpha_b| / \sigma_b$", fontsize=9)
cax.tick_params(labelsize=8)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=300)
fig.savefig(OUT.replace(".png", ".pdf"), dpi=300)
peak = {f"n2D={j + 1},n1D={r + 1}": round(float(snr[j, bands[bk]].max()), 1) for r, bk in enumerate(rows) for j in range(J + 1)}
print("saved", OUT, f"(grid {dirty.shape}, {len(noise_ch)} noise channels)")
print("peak |alpha|/sigma per sub-band:", peak)
