#!/usr/bin/env python
"""
NGC 3110, forward model at the compact resolution: the compact models seen through the tapered extended dirty beam,
against the tapered extended dirty cube (scripts/experiments/dirty_extended_taper.py). Both sides then carry the
extended array's sampling (its missing short spacings included), so no CLEAN of the extended data is involved and any
difference beyond noise is a disagreement between the compact model and the extended visibilities.

  data   D        the tapered extended dirty cube (Jy per dirty beam)
  model  P = x * psf  for x = the compact 2D-1D model (compact/deconvolved/model.npy) and the compact MS-CLEAN model
                  (data/ngc3110/compact/msclean_complete_v2), Jy/pixel, convolved per channel with the tapered PSF
                  (circular FFT, as tclean's minor cycle; the PSF is the image size)
over the line voxels of the tapered extended run's astrodendro mask (MASK, its mask_final.npy) and of the whole field:
  flux 100 (sum P - sum D) / sum D,  RMSE 100 ||P - D|| / ||D||,  and the RMSE noise alone would give, from the
  line-free channels' rms; plus the scale s minimising ||s P - D||.

Figure 1 x 5, line moment 0 in Jy/dirty beam km/s: D, P(2D-1D), P(MS-CLEAN) on one linear scale, then
100 (P - D) / D for each (+-50%, grey outside the mask's footprint or where D < 5% of its masked peak).

    python scripts/compact_vs_msclean_dendro/fig_forward_taper.py
    -> results/NGC3110/figures/taper_compare/forward_taper.{png,pdf,json}
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse
from numpy.fft import rfft2, irfft2

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(REPO, "results/NGC3110"); D_ = os.path.join(REPO, "data/ngc3110")
DIRTY = os.path.join(D_, "extended/dirty_taper")
MASK = os.environ.get("MASK", os.path.join(D_, "extended/msclean_taper_finished/mask_final.npy"))
MODELS = {"Compact 2D-1D": os.path.join(R, "compact/deconvolved/model.npy"),
          "Compact MS-CLEAN": os.path.join(D_, "compact/msclean_complete_v2/model.fits")}
OUT = os.path.join(R, "figures", "taper_compare"); os.makedirs(OUT, exist_ok=True)
GC = json.load(open(os.path.join(R, "config.json")))
CELL, DV = GC["cell_arcsec"], GC["width_kms"]
L = json.load(open(os.path.join(R, "line_channels.json")))["line"]
F = [c for c in range(GC["nchan"]) if c not in L and 1 < c < GC["nchan"] - 2]
TICK = 13
plt.rcParams.update({"font.family": "serif", "font.size": 13, "xtick.direction": "in", "ytick.direction": "in"})
rd = lambda p: np.nan_to_num(np.load(p) if p.endswith(".npy") else fits.getdata(p)).astype(np.float64).squeeze()
mad = lambda a: 1.4826 * np.median(np.abs(a - np.median(a)))

D = rd(os.path.join(DIRTY, "dirty.fits")); psf = rd(os.path.join(DIRTY, "psf.fits"))
n = D.shape[-1]
pk = np.unravel_index(np.argmax(psf[L[len(L) // 2]]), psf.shape[1:])
fpsf = rfft2(np.roll(psf, (-pk[0], -pk[1]), axis=(1, 2)))                 # PSF peak moved to pixel (0, 0)
M = np.load(MASK).astype(bool)
sig = np.median([mad(D[c]) for c in F])
h = fits.getheader(os.path.join(D_, "extended/msclean_taper_finished/image.fits")); B = (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"])
print(f"tapered dirty: PSF peak at {pk}, line-free noise {sig * 1e3:.2f} mJy/dirty beam; mask {os.path.relpath(MASK, REPO)}")

P, stats = {}, {}
for name, path in MODELS.items():
    x = rd(path)
    P[name] = irfft2(rfft2(x) * fpsf, s=x.shape[1:])                     # x * psf, Jy/dirty beam
    st = {}
    for reg, sel in (("mask", M[L]), ("field", np.ones_like(M[L]))):
        p, d = P[name][L][sel], D[L][sel]
        st[reg] = dict(flux_pct=float(100 * (p.sum() - d.sum()) / d.sum()), rmse_pct=float(100 * np.linalg.norm(p - d) / np.linalg.norm(d)),
                       noise_rmse_pct=float(100 * sig * np.sqrt(d.size) / np.linalg.norm(d)), scale=float((p * d).sum() / (p * p).sum()))
    stats[name] = st
    print(f"{name:17s}: in mask flux {st['mask']['flux_pct']:+.1f}%, RMSE {st['mask']['rmse_pct']:.1f}% (noise alone "
          f"{st['mask']['noise_rmse_pct']:.1f}%), best scale {st['mask']['scale']:.3f} | whole field flux "
          f"{st['field']['flux_pct']:+.1f}%, RMSE {st['field']['rmse_pct']:.1f}% (noise {st['field']['noise_rmse_pct']:.1f}%)")
json.dump(dict(mask=os.path.relpath(MASK, REPO), noise_mJy=sig * 1e3, models={k: os.path.relpath(v, REPO) for k, v in MODELS.items()},
               stats=stats), open(os.path.join(OUT, "forward_taper.json"), "w"), indent=1)

# ======================================================================== figure
m0 = lambda c: c[L].sum(0) * DV                                           # Jy/dirty beam km/s
FOOT = M[L].any(0); C0 = n // 2
HALF = float(GC.get("half_arcsec", n * CELL / 2 - 0.1))
hp = int(round(HALF / CELL)); sl = np.s_[C0 - hp:C0 + hp, C0 - hp:C0 + hp]; EXT = [HALF, -HALF, -HALF, HALF]
d0 = m0(D); dm = m0(D * M)
vmax = np.percentile(d0[sl], 99.9)
CMP = plt.get_cmap("RdBu_r").copy(); CMP.set_bad("0.9")
PANELS = [(d0, "Extended, tapered:\ndirty (data)", "", dict(cmap="inferno", norm=Normalize(-0.25 * vmax, vmax)))]
for name in MODELS:
    PANELS.append((m0(P[name]), f"{name}\n∗ tapered ext. dirty beam", "", dict(cmap="inferno", norm=Normalize(-0.25 * vmax, vmax))))
for name in MODELS:
    pm = m0(P[name] * M)
    pct = np.where(FOOT & (dm > 0.05 * dm.max()), 100 * (pm - dm) / np.where(dm > 0, dm, 1), np.nan)
    s = stats[name]["mask"]
    PANELS.append((pct, f"({name.split()[1]} − data) / data", f"flux {s['flux_pct']:+.1f}%, RMSE {s['rmse_pct']:.1f}%",
                   dict(cmap=CMP, norm=Normalize(-50, 50))))

S, G, ML, MR, MT, MB, CBH, CBG = 3.3, 0.06, 0.1, 0.1, 0.1, 0.95, 0.12, 0.08
FW, FH = ML + 5 * S + 4 * G + MR, MB + S + MT
fig = plt.figure(figsize=(FW, FH))
for j, (im2, lab, sub, kw) in enumerate(PANELS):
    x = ML + j * (S + G)
    a = fig.add_axes([x / FW, MB / FH, S / FW, S / FH])
    im = a.imshow(im2[sl], origin="lower", extent=EXT, interpolation="nearest", **kw)
    a.set_xticks([]); a.set_yticks([])
    a.contour(FOOT[sl].astype(float), levels=[0.5], colors="w" if j < 3 else "k", linewidths=0.5, extent=EXT, origin="lower")
    col = "w" if j < 3 else "k"
    a.text(0.04, 0.96, lab, transform=a.transAxes, color=col, ha="left", va="top", fontsize=TICK - 1, fontweight="bold")
    if sub:
        a.text(0.96, 0.04, sub, transform=a.transAxes, color=col, ha="right", va="bottom", fontsize=TICK - 1)
    pad = 0.06 * HALF
    a.add_patch(Ellipse((HALF - pad - B[0] / 2, -HALF + pad + B[0] / 2), width=B[0], height=B[1], angle=90 - B[2],
                        facecolor="none", edgecolor=col, lw=1.0))
    if j == 0:
        xs, ys, cap = -HALF + 0.08 * HALF, HALF - 0.1 * HALF, 0.03 * HALF
        a.plot([xs + 2, xs], [ys, ys], color="w", lw=2.0, solid_capstyle="butt")
        for xe in (xs, xs + 2):
            a.plot([xe, xe], [ys - cap, ys + cap], color="w", lw=1.4, solid_capstyle="butt")
        a.annotate("2″", (xs + 1, ys - cap), xytext=(0, -2.5), textcoords="offset points", color="w", ha="center",
                   va="top", fontsize=TICK - 1, fontweight="bold")
    if j in (0, 3, 4):
        span = 3 * S + 2 * G if j == 0 else S
        cax = fig.add_axes([(x + 0.1 * S) / FW, (MB - CBG - CBH) / FH, (span - 0.2 * S) / FW, CBH / FH])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.locator = matplotlib.ticker.MaxNLocator(6); cb.update_ticks()
        cax.tick_params(labelsize=TICK - 2)
        cb.set_label("%" if j else r"Jy (dirty beam)$^{-1}$ km s$^{-1}$", fontsize=TICK - 1)
fig.savefig(os.path.join(OUT, "forward_taper.png"), dpi=150)
fig.savefig(os.path.join(OUT, "forward_taper.pdf")); plt.close(fig)
print("saved", os.path.join(OUT, "forward_taper.png"))
