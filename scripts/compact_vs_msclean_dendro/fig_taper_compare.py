#!/usr/bin/env python
"""
NGC 3110 at one resolution from both arrays, 1 x 4, line moment 0 of tclean's restored images:
  1  compact MS-CLEAN (COMPACT, default data/ngc3110/compact/msclean_complete_v2), its clean beam
  2  extended MS-CLEAN at full resolution (EXTENDED, default .../extended/msclean_deep), for reference
  3  extended MS-CLEAN with the uv taper matching the compact beam, restored with exactly the compact clean beam
     (TAPERED, default .../extended/msclean_taper_finished; scripts/experiments/msclean_extended_taper.py)
  4  100 (1 - 3) / 3 inside the tapered run's astrodendro mask (its mask_final.npy: the extended data's own mask at
     the compact resolution), grey outside it or where 3 < 5% of its peak; printed: flux difference and RMSE
     100 ||1 - 3|| / ||3|| over the mask's voxels
Panels 1-3 in Jy arcsec^-2 km/s (moment 0 in Jy/beam km/s over the beam's solid angle), linear; 1 and 3 share a
scale. Prints the checks that make the comparison meaningful: the two restoring beams, each run's residual rms in the
line channels against its line-free noise, and image = model * beam + residual.

MODE=uvmatch: the same with both arrays imaged from the 83-341 klambda overlap only (scripts/experiments/
msclean_uvmatch.py): 1 the compact, 3 the extended uv-matched run (both restored with one beam), the mask the extended
uv-matched run's.

    [MODE=taper|uvmatch] [COMPACT=...] [EXTENDED=...] [TAPERED=...] python scripts/compact_vs_msclean_dendro/fig_taper_compare.py
    -> results/NGC3110/figures/taper_compare/<MODE>_compare.{png,pdf,json}
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
for _p in ("simple", "scripts/compact_vs_msclean_dendro"):
    sys.path.insert(0, os.path.join(REPO, _p))
from restore import convolve_with_beam  # noqa: E402
import beams  # noqa: E402

R = os.path.join(REPO, "results/NGC3110")
D = os.path.join(REPO, "data/ngc3110")
MODE = os.environ.get("MODE", "taper")
DEF = dict(taper=("compact/msclean_complete_v2", "extended/msclean_taper_finished"),
           uvmatch=("compact/msclean_uvmatch_finished", "extended/msclean_uvmatch_finished"))[MODE]
RUNS = dict(compact=os.environ.get("COMPACT", os.path.join(D, DEF[0])),
            extended=os.environ.get("EXTENDED", os.path.join(D, "extended/msclean_deep")),
            tapered=os.environ.get("TAPERED", os.path.join(D, DEF[1])))
LAB1, LAB3, LAB4 = dict(taper=("Compact MS-CLEAN", "Extended MS-CLEAN,\ntapered to the compact beam",
                               "(Compact − tapered ext.)\n/ tapered ext."),
                        uvmatch=("Compact MS-CLEAN,\n83–341 kλ only", "Extended MS-CLEAN,\n83–341 kλ only",
                                 "(Compact − extended)\n/ extended, 83–341 kλ"))[MODE]
OUT = os.path.join(R, "figures", "taper_compare"); os.makedirs(OUT, exist_ok=True)
GC = json.load(open(os.path.join(R, "config.json")))
CELL, DV = GC["cell_arcsec"], GC["width_kms"]
L = json.load(open(os.path.join(R, "line_channels.json")))["line"]
F = [c for c in range(GC["nchan"]) if c not in L and 1 < c < GC["nchan"] - 2]
TICK = 13
plt.rcParams.update({"font.family": "serif", "font.size": 13, "xtick.direction": "in", "ytick.direction": "in"})
rd = lambda p: np.nan_to_num(fits.getdata(p).astype(np.float64)).squeeze()
bm = lambda p: (lambda h: (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"]))(fits.getheader(p))
mad = lambda a: 1.4826 * np.median(np.abs(a - np.median(a)))

data, checks = {}, {}
for k, d in RUNS.items():
    B = bm(os.path.join(d, "image.fits"))
    img, mod, res = (rd(os.path.join(d, f"{e}.fits")) for e in ("image", "model", "residual"))
    noise = np.median([mad(res[c]) for c in F])
    restore_err = np.linalg.norm((img - convolve_with_beam(mod, CELL, *B) - res)[L]) / np.linalg.norm(res[L])
    checks[k] = dict(dir=os.path.relpath(d, REPO), beam=B, model_flux_Jy_kms=float(mod[L].sum() * DV),
                     resid_rms_line_over_free=float(np.std(res[L]) / noise),
                     resid_max_line_sigma=float(res[L].max() / noise), resid_min_line_sigma=float(res[L].min() / noise),
                     restore_mismatch=float(restore_err))
    data[k] = (B, img[L])
    print(f"{k:9s} {checks[k]['dir']}: beam {B[0]:.3f}x{B[1]:.3f}\" PA {B[2]:.1f}; components {checks[k]['model_flux_Jy_kms']:.1f} Jy km/s; "
          f"residual rms {checks[k]['resid_rms_line_over_free']:.2f}x noise, extremes {checks[k]['resid_max_line_sigma']:+.0f}/"
          f"{checks[k]['resid_min_line_sigma']:+.0f} sigma; |image - model*beam - residual|/|residual| {restore_err:.3f}")

BC, Ic = data["compact"]; BT, It = data["tapered"]; BE, Ie = data["extended"]
assert np.allclose(BC, BT, rtol=0.01, atol=0.5), (BC, BT)
M = np.load(os.path.join(RUNS["tapered"], "mask_final.npy")).astype(bool)[L]; FOOT = M.any(0)
bpx = beams.omega(BC) / CELL ** 2
fc, ft = Ic[M].sum() / bpx * DV, It[M].sum() / bpx * DV
DF = 100 * (fc - ft) / ft
RMSE = 100 * np.linalg.norm((Ic - It)[M]) / np.linalg.norm(It[M])
m0 = lambda cube, b: cube.sum(0) * DV / beams.omega(b)                        # Jy arcsec^-2 km/s
cc, ee, tt = m0(Ic, BC), m0(Ie, BE), m0(It, BT)
cm, tm = m0(Ic * M, BC), m0(It * M, BT)
PCT = np.where(FOOT & (tm > 0.05 * tm.max()), 100 * (cm - tm) / np.where(tm > 0, tm, 1), np.nan)
print(f"inside the tapered extended mask ({FOOT.sum() * CELL ** 2:.0f} arcsec2): compact {fc:.1f}, tapered extended {ft:.1f} "
      f"Jy km/s -> flux {DF:+.1f}%, RMSE {RMSE:.1f}%, pixel median {np.nanmedian(PCT):+.1f}%")
json.dump(dict(checks=checks, mask_area_arcsec2=float(FOOT.sum() * CELL ** 2), flux_compact=fc, flux_tapered=ft,
               flux_pct=DF, rmse_pct=RMSE), open(os.path.join(OUT, f"{MODE}_compare.json"), "w"), indent=1)

N = cc.shape[0]; C0 = N // 2
HALF = float(GC.get("half_arcsec", N * CELL / 2 - 0.1))
hp = int(round(HALF / CELL)); sl = np.s_[C0 - hp:C0 + hp, C0 - hp:C0 + hp]; EXT = [HALF, -HALF, -HALF, HALF]
vc = np.percentile(np.maximum(cc, tt)[sl], 99.9)
CMP = plt.get_cmap("RdBu_r").copy(); CMP.set_bad("0.9")
PANELS = [(cc, BC, LAB1, dict(cmap="inferno", norm=Normalize(0, vc)), True),
          (ee, BE, "Extended MS-CLEAN", dict(cmap="inferno", norm=Normalize(0, np.percentile(ee[sl], 99.9))), False),
          (tt, BT, LAB3, dict(cmap="inferno", norm=Normalize(0, vc)), True),
          (PCT, BC, LAB4, dict(cmap=CMP, norm=Normalize(-50, 50)), False)]
SUB = [f"{checks['compact']['model_flux_Jy_kms']:.0f} Jy km s$^{{-1}}$", f"{checks['extended']['model_flux_Jy_kms']:.0f} Jy km s$^{{-1}}$",
       f"{checks['tapered']['model_flux_Jy_kms']:.0f} Jy km s$^{{-1}}$", f"flux {DF:+.1f}%, RMSE {RMSE:.1f}%"]

S, G, ML, MR, MT, MB, CBH, CBG = 3.6, 0.06, 0.1, 0.1, 0.1, 0.95, 0.12, 0.08
FW, FH = ML + 4 * S + 3 * G + MR, MB + S + MT
fig = plt.figure(figsize=(FW, FH))
for j, ((im2, B, lab, kw, cont), sub) in enumerate(zip(PANELS, SUB)):
    x = ML + j * (S + G)
    a = fig.add_axes([x / FW, MB / FH, S / FW, S / FH])
    im = a.imshow(im2[sl], origin="lower", extent=EXT, interpolation="nearest", **kw)
    a.set_xticks([]); a.set_yticks([])
    if cont:
        a.contour(FOOT[sl].astype(float), levels=[0.5], colors="w", linewidths=0.5, extent=EXT, origin="lower")
    col = "k" if j == 3 else "w"
    a.text(0.04, 0.96, lab, transform=a.transAxes, color=col, ha="left", va="top", fontsize=TICK, fontweight="bold")
    pad = 0.06 * HALF
    a.add_patch(Ellipse((HALF - pad - B[0] / 2, -HALF + pad + B[0] / 2), width=B[0], height=B[1], angle=90 - B[2],
                        facecolor=col if j < 3 else "none", edgecolor=col, lw=1.0))
    a.text(0.96, 0.04, sub, transform=a.transAxes, color=col, ha="right", va="bottom", fontsize=TICK - 1)
    if j == 0:                                                              # scale bar, top right
        xs, ys, cap = -HALF + 0.08 * HALF, HALF - 0.1 * HALF, 0.03 * HALF
        a.plot([xs + 2, xs], [ys, ys], color="w", lw=2.0, solid_capstyle="butt")
        for xe in (xs, xs + 2):
            a.plot([xe, xe], [ys - cap, ys + cap], color="w", lw=1.4, solid_capstyle="butt")
        a.annotate("2″", (xs + 1, ys - cap), xytext=(0, -2.5), textcoords="offset points", color="w", ha="center",
                   va="top", fontsize=TICK - 1, fontweight="bold")
    cax = fig.add_axes([(x + 0.1 * S) / FW, (MB - CBG - CBH) / FH, 0.8 * S / FW, CBH / FH])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.locator = matplotlib.ticker.MaxNLocator(5); cb.update_ticks()
    cax.tick_params(labelsize=TICK - 2)
    cb.set_label("%" if j == 3 else r"Jy arcsec$^{-2}$ km s$^{-1}$", fontsize=TICK - 1)
fig.savefig(os.path.join(OUT, f"{MODE}_compare.png"), dpi=150)
fig.savefig(os.path.join(OUT, f"{MODE}_compare.pdf")); plt.close(fig)
print("saved", os.path.join(OUT, f"{MODE}_compare.png"))
