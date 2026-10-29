#!/usr/bin/env python
"""
The two configurations' own multiscale CLEAN models, and the extended one brought to the compact resolution, 1 x 4:
  1  compact MS-CLEAN components (compact/msclean) * the compact clean beam
  2  extended MS-CLEAN components (extended/msclean_finished) * the extended clean beam
  3  the extended restored image at the compact clean beam: the same components * the compact beam + the extended
     residual brought from the extended to the compact beam (beams.to_beam), i.e. all the extended data hold at that
     resolution, including the faint emission CLEAN left in its residual
  4  100 (1 - 3) / 3 inside an astrodendro mask at the compact beam (the convention of fig_resolution_ladder.py,
     relative to the extended CLEAN): both cubes at the compact beam summed over the mask's voxels only, grey outside
     its footprint or where the masked extended map < 5% of its peak; printed: the flux difference and the RMSE
     100 ||1 - 3|| / ||3|| over the mask's voxels. MASK_FROM:
       extended (default)  msclean_complete.dendro_mask of the extended CLEAN image at the compact beam: its
                           components * the compact beam + its residual brought from the extended to the compact beam
                           (beams.to_beam), per-channel sigma from that smoothed residual
       compact             the compact MS-CLEAN's own (compact/msclean/dendro_mask.npy, on its image)
1-3: line moment 0 in Jy arcsec^-2 km/s (moment 0 in Jy/beam km/s over the beam's solid angle), the mask's
footprint as a thin contour; 1 and 2 models only; linear colour scales: 1 and 3 share one, 2 its own,
4 +-100%.

    [GAL_DIR=results/NGC3110] [MASK_FROM=extended|compact] python scripts/compact_vs_msclean_dendro/fig_msclean_compact_extended.py
    -> <GAL_DIR>/figures/msclean_compact_extended/msclean_compact_extended_<MASK_FROM>mask.{png,pdf}
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in ("simple", "scripts", "scripts/compact_vs_msclean_dendro"):
    sys.path.insert(0, os.path.join(REPO, _p))
from restore import convolve_with_beam  # noqa: E402
from msclean_complete import dendro_mask  # noqa: E402
import beams  # noqa: E402

R = os.environ.get("GAL_DIR", os.path.join(REPO, "results/NGC3110"))
MASK_FROM = os.environ.get("MASK_FROM", "extended")
OUT = os.path.join(R, "figures", "msclean_compact_extended"); os.makedirs(OUT, exist_ok=True)
GC = json.load(open(os.path.join(R, "config.json")))
CELL, DV = GC["cell_arcsec"], GC["width_kms"]
L = json.load(open(os.path.join(R, "line_channels.json")))["line"]
TICK = 13
plt.rcParams.update({"font.family": "serif", "font.size": 13, "xtick.direction": "in", "ytick.direction": "in"})
rd = lambda p: np.nan_to_num(fits.getdata(p).astype(np.float64)).squeeze()
beam = lambda p: (lambda h: (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"]))(fits.getheader(p))
omega = beams.omega
m0 = lambda cube, b: cube.sum(0) * DV / omega(b)                           # Jy/beam cube -> Jy arcsec^-2 km/s

DC, DE = os.path.join(R, "compact/msclean"), os.path.join(R, "extended/msclean_finished")
BC, BE = beam(os.path.join(DC, "image.fits")), beam(os.path.join(DE, "image.fits"))
xc, xe = rd(os.path.join(DC, "model.fits"))[L], rd(os.path.join(DE, "model.fits"))[L]     # Jy/pixel
fc, fe = xc.sum() * DV, xe.sum() * DV
Cc = convolve_with_beam(xc, CELL, *BC)                                     # compact CLEAN at its beam
rb = beams.to_beam(rd(os.path.join(DE, "residual.fits"))[L], BE, BC, CELL)    # extended residual at the compact beam
Ek = convolve_with_beam(xe, CELL, *BC)                                     # extended components * compact beam
Ec = Ek + rb                                                               # the extended restored image there
cc, ee, ec = m0(Cc, BC), m0(convolve_with_beam(xe, CELL, *BE), BE), m0(Ec, BC)
if MASK_FROM == "compact":
    M = np.load(os.path.join(DC, "dendro_mask.npy")).astype(bool)[L]
else:                                                                       # the extended CLEAN image at the compact beam
    M, info = dendro_mask(Ec, rb, omega(BC) / CELL ** 2, list(range(len(L))))
    print(f"extended dendro mask at the compact beam: {info['structures']} structures, peak bar {info['peak_bar']}, "
          f"median sigma {np.median(info['sigma_line_mJy']):.2f} mJy/beam")
FOOT = M.any(0)
print(f"{MASK_FROM} mask: {M.mean():.1%} of line voxels, footprint {FOOT.sum() * CELL ** 2:.0f} arcsec2")
cm, em = m0(Cc * M, BC), m0(Ec * M, BC)                                    # summed over the mask's voxels
bpx = omega(BC) / CELL ** 2
fcm, fem = (Cc * M).sum() / bpx * DV, (Ec * M).sum() / bpx * DV
PCT = np.where(FOOT & (em > 0.05 * em.max()), 100 * (cm - em) / np.where(em > 0, em, 1), np.nan)
DF = 100 * (fcm - fem) / fem                                                # flux, % of the extended
RMSE = 100 * np.linalg.norm((Cc - Ec)[M]) / np.linalg.norm(Ec[M])
print(f"compact MS-CLEAN {fc:.1f}, extended MS-CLEAN {fe:.1f} Jy km/s; in the {MASK_FROM} dendro mask (compact beam): "
      f"compact {fcm:.1f}, extended {fem:.1f}, (compact - extended) / extended {DF:+.1f}%, RMSE {RMSE:.1f}%; "
      f"pixel median {np.nanmedian(PCT):+.1f}%")

N = cc.shape[0]; C0 = N // 2
HALF = float(GC.get("half_arcsec", N * CELL / 2 - 0.1))
hp = int(round(HALF / CELL)); sl = np.s_[C0 - hp:C0 + hp, C0 - hp:C0 + hp]; EXT = [HALF, -HALF, -HALF, HALF]
vc = np.percentile(np.maximum(cc, ec)[sl], 99.9)
CMP = plt.get_cmap("RdBu_r").copy(); CMP.set_bad("0.9")
PANELS = [(cc, BC, "Compact MS-CLEAN", f"{fc:.0f} Jy km s$^{{-1}}$", dict(cmap="inferno", norm=Normalize(0, vc))),
          (ee, BE, "Extended MS-CLEAN", f"{fe:.0f} Jy km s$^{{-1}}$",
           dict(cmap="inferno", norm=Normalize(0, np.percentile(ee[sl], 99.9)))),
          (ec, BC, "Extended MS-CLEAN, restored\nat the compact beam", f"{Ec.sum() / bpx * DV:.0f} Jy km s$^{{-1}}$", dict(cmap="inferno", norm=Normalize(0, vc))),
          (PCT, BC, f"(Compact − extended) / extended\nin the {MASK_FROM} mask", f"flux {DF:+.1f}%, RMSE {RMSE:.1f}%",
           dict(cmap=CMP, norm=Normalize(-100, 100)))]

S, G, ML, MR, MT, MB, CBH, CBG = 3.6, 0.06, 0.1, 0.1, 0.1, 0.95, 0.12, 0.08
FW, FH = ML + 4 * S + 3 * G + MR, MB + S + MT
fig = plt.figure(figsize=(FW, FH))
for j, (im2, B, lab, sub, kw) in enumerate(PANELS):
    x = ML + j * (S + G)
    a = fig.add_axes([x / FW, MB / FH, S / FW, S / FH])
    im = a.imshow(im2[sl], origin="lower", extent=EXT, interpolation="nearest", **kw)
    a.set_xticks([]); a.set_yticks([])
    if j != 1 and j != 3:                                                   # the mask's footprint (compact beam)
        a.contour(FOOT[sl].astype(float), levels=[0.5], colors="w", linewidths=0.5, extent=EXT, origin="lower")
    col = "k" if j == 3 else "w"
    a.text(0.04, 0.96, lab, transform=a.transAxes, color=col, ha="left", va="top", fontsize=TICK, fontweight="bold")
    pad = 0.06 * HALF                                                       # beam, bottom left
    a.add_patch(Ellipse((HALF - pad - B[0] / 2, -HALF + pad + B[0] / 2), width=B[0], height=B[1], angle=90 - B[2],
                        facecolor=col if j < 3 else "none", edgecolor=col, lw=1.0))
    a.text(0.96, 0.04, sub, transform=a.transAxes, color=col, ha="right", va="bottom", fontsize=TICK - 1)
    if j == 0:                                                              # scale bar, top right
        xs, ys, cap = -HALF + 0.08 * HALF, HALF - 0.1 * HALF, 0.03 * HALF
        a.plot([xs + 2, xs], [ys, ys], color="w", lw=2.0, solid_capstyle="butt")
        for xe_ in (xs, xs + 2):
            a.plot([xe_, xe_], [ys - cap, ys + cap], color="w", lw=1.4, solid_capstyle="butt")
        a.annotate("2″", (xs + 1, ys - cap), xytext=(0, -2.5), textcoords="offset points", color="w", ha="center",
                   va="top", fontsize=TICK - 1, fontweight="bold")
    cax = fig.add_axes([(x + 0.1 * S) / FW, (MB - CBG - CBH) / FH, 0.8 * S / FW, CBH / FH])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.locator = matplotlib.ticker.MaxNLocator(5); cb.update_ticks()
    cax.tick_params(labelsize=TICK - 2)
    cb.set_label("%" if j == 3 else r"Jy arcsec$^{-2}$ km s$^{-1}$", fontsize=TICK - 1)
name = f"msclean_compact_extended_{MASK_FROM}mask"
fig.savefig(os.path.join(OUT, name + ".png"), dpi=150)
fig.savefig(os.path.join(OUT, name + ".pdf")); plt.close(fig)
print("saved", os.path.join(OUT, name + ".png"))
