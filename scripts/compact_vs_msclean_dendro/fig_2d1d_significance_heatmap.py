#!/usr/bin/env python
"""
Per-sub-band significance of the NGC 3110 compact data before and after the 2D-1D deconvolution: two 8 x 6 heatmaps
(n_2D = 1 ... 8 across, n_1D = 1 ... 6 down; simple/gpu_pd.Dict2D1D with the pipeline's J = 7 starlet scales x L = 5
CDF 9/7 levels, on the 0.04" grid of the deconvolution) of the peak |alpha_b| / sigma_b inside the emission footprint
(the extended astrodendro mask, extended/msclean_finished/mask_final.npy, over the line channels, binned to 0.04"),
for
  (a) the dirty cube d, and
  (b) the residual d - H x of the pipeline's deconvolved model (results/NGC3110/compact/deconvolved/model.npy, pdtau,
      k = 5, saved at 0.02" and block-summed back to 0.04"),
with sigma_b = max(sigma_real, sigma_white) of gpu_pd.noise_lambdas (the thresholds are lambda_b = k sigma_b). The
values are written in the cells; one logarithmic colour scale, the k = 5 level marked on the colorbar. (Over the
whole field the residual's peaks all sit at the edge of the 16" field, 8-11" from the centre, outside the emission.)
Drawn at the A&A column width (256 pt) at its final size.

    [OUT=...] [K=5] python scripts/compact_vs_msclean_dendro/fig_2d1d_significance_heatmap.py
    -> results/gallery/ngc3110_2d1d_significance_heatmap.{png,pdf}, and .json with the numbers
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
OUT = os.environ.get("OUT", os.path.join(REPO, "results", "gallery", "ngc3110_2d1d_significance_heatmap.png"))
K = float(os.environ.get("K", 5))
J, L, BIN, VMIN, VMAX = 7, 5, 2, 1.0, 2000.0
plt.rcParams.update({"font.family": "serif", "font.size": 8, "xtick.direction": "out", "ytick.direction": "out"})

line = json.load(open(os.path.join(R, "line_channels.json")))["line"]
rd = lambda p: np.nan_to_num(fits.getdata(p).astype(np.float64)).squeeze()
dirty, psf = G.rebin(rd(os.path.join(R, "compact/cube/dirty_cube.fits")), rd(os.path.join(R, "compact/cube/dirty_beam_cube.fits")), BIN)
nz = dirty.shape[0]
x = np.load(os.path.join(R, "compact/deconvolved/model.npy")).astype(np.float64)
x = x.reshape(nz, x.shape[1] // BIN, BIN, x.shape[2] // BIN, BIN).sum((2, 4))     # 0.02" -> 0.04" (block sum)
op = G.Op(psf, dirty)
resid = (-op.grad(G.to_t(x))).cpu().numpy()                                      # d - H x
D = G.Dict2D1D(dirty.shape, J, L, dict_2d="starlet")
_, sr, sw, _ = G.noise_lambdas(D, G.to_t(dirty), [i for i in range(nz) if i not in line], k=1.0)
sigma = np.maximum(sr.cpu().numpy(), sw.cpu().numpy())
M = np.load(os.path.join(R, "extended/msclean_finished/mask_final.npy")).astype(bool)
foot = M[line].any(0)
foot = foot.reshape(foot.shape[0] // BIN, BIN, foot.shape[1] // BIN, BIN).any((1, 3))   # 2D footprint, 0.04" grid
bands = G.band_slices(nz, L)                                                     # [("a", ...), ("d5", ...), ..., ("d1", ...)]
order = [dict(bands)[f"d{lev}"] for lev in range(1, L + 1)] + [dict(bands)["a"]]  # n_1D = 1 (d1) ... L + 1 (a)


def peak_snr(cube):
    s = np.abs(D.W(G.to_t(cube)).cpu().numpy()) / sigma
    return np.array([[s[j, seg][:, foot].max() for j in range(J + 1)] for seg in order])  # (L + 1, J + 1)


S = {"dirty": peak_snr(dirty), "residual": peak_snr(resid)}
json.dump({k: np.round(v, 2).tolist() for k, v in S.items()} | dict(rows="n_1D = 1..6", cols="n_2D = 1..8", k=K,
                                                                     region="emission footprint"),
          open(OUT.replace(".png", ".json"), "w"), indent=1)

# ======================================================================== layout (inches): A&A column width, 1:1
CW = 256.0748 / 72.27
ML, MR, MT, MB, PG, CBW, CBP = 0.42, 0.55, 0.30, 0.36, 0.42, 0.08, 0.08
C = (CW - ML - MR) / (J + 1)                                                     # cell size
H = (L + 1) * C
FH = MT + 2 * H + PG + MB
fig = plt.figure(figsize=(CW, FH))
norm = matplotlib.colors.LogNorm(VMIN, VMAX, clip=True)
cmap = plt.get_cmap("inferno")
for p, (key, title) in enumerate((("dirty", "(a) dirty cube"), ("residual", "(b) residual after deconvolution"))):
    y0 = FH - MT - (p + 1) * H - p * PG
    a = fig.add_axes([ML / CW, y0 / FH, (J + 1) * C / CW, H / FH])
    m = S[key]
    im = a.imshow(np.clip(m, VMIN, VMAX), cmap=cmap, norm=norm, origin="upper", aspect="equal",
                  extent=[0.5, J + 1.5, L + 1.5, 0.5])
    for r in range(L + 1):
        for j in range(J + 1):
            v = m[r, j]
            a.text(j + 1, r + 1, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", va="center", fontsize=5.5,
                   color="k" if norm(max(v, VMIN)) > 0.6 else "w", fontweight="bold" if v >= K else "normal")
    a.set_xticks(range(1, J + 2)); a.set_yticks(range(1, L + 2))
    a.tick_params(labelsize=7, length=2, pad=1.5)
    a.set_ylabel(r"$n_\mathrm{1D}$", fontsize=8, labelpad=2)
    if p == 1:
        a.set_xlabel(r"$n_\mathrm{2D}$", fontsize=8, labelpad=2)
    a.set_title(title, fontsize=8, pad=3)
cax = fig.add_axes([(CW - MR + CBP) / CW, MB / FH, CBW / CW, (2 * H + PG) / FH])
cb = fig.colorbar(im, cax=cax)
cb.ax.axhline(K, color="#7fdbff", lw=1.2)
cb.set_label(r"peak $|\alpha_b| / \sigma_b$ in the emission region", fontsize=8, labelpad=2)
cax.tick_params(labelsize=7, length=2)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=400)
fig.savefig(OUT.replace(".png", ".pdf"))
print("saved", OUT)
for k, v in S.items():
    print(k, "\n", np.array2string(v, precision=1, max_line_width=200, suppress_small=True))
