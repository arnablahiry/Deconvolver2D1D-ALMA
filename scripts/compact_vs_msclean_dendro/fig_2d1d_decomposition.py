#!/usr/bin/env python
"""
2D-1D decomposition of the NGC 3110 compact dirty cube, as the deconvolution sees it (0.04" grid, 2 x 2 block mean
of the native cube; simple/gpu_pd.Dict2D1D with the pipeline's J = 7 starlet scales x L = 5 CDF 9/7 levels): a grid of
the 48 sub-bands, 2D (starlet) scale along x (w_1 ... w_7, coarse), 1D (spectral) band along y (d_1 ... d_5, a_5).
Each panel is one slice of its sub-band: the spectral coefficient that covers the brightest line channel, on its own
symmetric colour scale. Laid out as the 2D-1D decomposition figure of Lahiry et al. (2025): n_2D = 1 ... J + 1
(J + 1 = coarse) across, n_1D = 1 ... L + 1 (L + 1 = approximation) down. Drawn at the A&A text width (523.5 pt).

    [OUT=...] python scripts/compact_vs_msclean_dendro/fig_2d1d_decomposition.py
    -> results/gallery/ngc3110_2d1d_decomposition.{png,pdf}
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
OUT = os.environ.get("OUT", os.path.join(REPO, "results", "gallery", "ngc3110_2d1d_decomposition.png"))
J, L, BIN = 7, 5, 2
plt.rcParams.update({"font.family": "serif", "font.size": 8})

c = json.load(open(os.path.join(R, "config.json")))
line = json.load(open(os.path.join(R, "line_channels.json")))["line"]
rd = lambda p: np.nan_to_num(fits.getdata(p).astype(np.float64)).squeeze()
dirty, psf = G.rebin(rd(os.path.join(R, "compact/cube/dirty_cube.fits")), rd(os.path.join(R, "compact/cube/dirty_beam_cube.fits")), BIN)
cell, dv, nz = c["cell_arcsec"] * BIN, c["width_kms"], dirty.shape[0]
D = G.Dict2D1D(dirty.shape, J, L, dict_2d="starlet")
alpha = D.W(G.to_t(dirty)).cpu().numpy()                             # (J + 1, nz, ny, nx); spectral axis packed
bands = dict(G.band_slices(nz, L))                                   # "a", "d5", ..., "d1" -> slices of the packed axis
flux = dirty.sum((1, 2))
ch = max(line, key=lambda i: flux[i])                                # brightest line channel
vel = c["start_kms"] + dv * ch

half = float(c["half_arcsec"])
ny = dirty.shape[1]; c0 = ny // 2; hp = int(round(half / cell)); sl = np.s_[c0 - hp:c0 + hp, c0 - hp:c0 + hp]
cols = [(j, rf"$w_{{{j + 1}}}$", f"{cell * 2 ** j:.2g}–{cell * 2 ** (j + 1):.2g}″") for j in range(J)] + \
       [(J, "coarse", f">{cell * 2 ** J:.2g}″")]
rows = [(f"d{lev}", rf"$d_{{{lev}}}$", f"{dv * 2 ** (lev - 1):.0f}–{dv * 2 ** lev:.0f}") for lev in range(1, L + 1)] + \
       [("a", rf"$a_{{{L}}}$", f">{dv * 2 ** L:.0f}")]

# ======================================================================== layout (inches): A&A text width, 1:1
TW = 523.5307 / 72.27
ML, MR, MT, MB, GAP = 0.66, 0.02, 0.24, 0.04, 0.05
S = (TW - ML - MR - (len(cols) - 1) * GAP) / len(cols)
FH = MT + len(rows) * S + (len(rows) - 1) * GAP + MB
fig = plt.figure(figsize=(TW, FH))
for r, (bk, blab, brange) in enumerate(rows):
    seg = bands[bk]; n = seg.stop - seg.start
    k = seg.start + ch * n // nz                                    # the coefficient covering channel ch
    for q, (j, jlab, jrange) in enumerate(cols):
        x0 = ML + q * (S + GAP); y0 = FH - MT - (r + 1) * S - r * GAP
        a = fig.add_axes([x0 / TW, y0 / FH, S / TW, S / FH])
        img = alpha[j, k][sl]
        m = np.percentile(np.abs(img), 99.7) or 1.0
        a.imshow(img, origin="lower", cmap="RdBu_r", vmin=-m, vmax=m, interpolation="nearest",
                 extent=[half, -half, -half, half])
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_linewidth(0.5)
        if r == 0:
            a.set_title(rf"$\mathbf{{n_{{2D}} = {q + 1}}}$", fontsize=9, pad=4)
        if q == 0:
            a.text(-0.1, 0.5, rf"$\mathbf{{n_{{1D}} = {r + 1}}}$", transform=a.transAxes, ha="right", va="center",
                   fontsize=9)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=300)
fig.savefig(OUT.replace(".png", ".pdf"), dpi=300)
print("saved", OUT, f"(channel {ch}, v = {vel:.0f} km/s; grid {dirty.shape}, cell {cell}\")")
