#!/usr/bin/env python
"""
40c (NGC 3110): two ways of bringing the compact deconvolution to the extended clean beam, against the extended MS-CLEAN.
Top: the compact deconvolved model (raw, Jy/pixel) * the extended clean beam; the compact deconvolution with a target
resolution of the extended clean beam (gpu_pd.Op target_fwhm_px: sky y = G x, G circular with FWHM = the beam's
geometric mean, 0.195"; shown as is, no further convolution); the extended MS-CLEAN components * the extended clean beam
(one colour scale). Bottom: each compact version - CLEAN (+- the same scale, total error and whole-cube RMSE relative to
CLEAN), and the integrated spectra of all three. Everything x the astrodendro mask (mask_final.npy; MASK= another). Same layout and
style as figure 40.

    [CLEAN=msclean_finished] [MASK=...] [OUT=...png] python scripts/compact_vs_msclean_dendro/fig40c_target_resolution.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.patches import Ellipse
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
from restore import convolve_with_beam  # noqa: E402

R = os.path.join(REPO, "results/NGC3110")
TARGET = os.environ.get("TARGET", os.path.join(R, "compact/deconvolved_target_extbeam/model.npy"))
OUT = os.environ.get("OUT", os.path.join(R, "figures/extended_clean_beam_comparison/40c_target_resolution.png"))
CLEAN = os.environ.get("CLEAN", "msclean_finished")
MASK = os.environ.get("MASK", os.path.join(R, "extended/msclean_finished/mask_final.npy"))
GC = json.load(open(os.path.join(R, "config.json")))
CELL, DV, NCH = GC["cell_arcsec"], GC["width_kms"], GC["nchan"]
vel = GC["vsys_kms"] + DV * (np.arange(NCH) - (NCH - 1) / 2)
L = json.load(open(os.path.join(R, "line_channels.json")))["line"]
HALF = float(GC.get("half_arcsec", 5.5))
BLUE, ORANGE, GREEN, MUTED = "#2a6fdb", "#e07b00", "#1a9e77", "#8a8a8a"
TICK, LAB = 16, 17
SBU = r"Jy arcsec$^{-2}$ km s$^{-1}$"
plt.rcParams.update({"font.family": "serif", "font.size": 12, "xtick.direction": "in", "ytick.direction": "in"})
rd = lambda p: np.nan_to_num(fits.getdata(p).astype(np.float64)).squeeze()
omega = lambda b: np.pi * b[0] * b[1] / (4 * np.log(2))

h = fits.getheader(os.path.join(R, "extended", CLEAN, "image.fits"))
BE = (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"])
TFWHM = float(np.sqrt(BE[0] * BE[1])); BT = (TFWHM, TFWHM, 0.0)
M = np.load(MASK).astype(bool)
FOOT = M[L].any(0); N = FOOT.shape[0]; C0 = N // 2
x = dict(raw=np.load(os.path.join(R, "compact/deconvolved/model.npy")).astype(np.float64) * M,
         tgt=np.load(TARGET).astype(np.float64) * M,
         cln=rd(os.path.join(R, "extended", CLEAN, "model.fits")) * M)
cube = dict(raw=convolve_with_beam(x["raw"][L], CELL, *BE) / omega(BE),     # Jy arcsec^-2 per channel
            tgt=x["tgt"][L] / CELL ** 2,                                   # already at the target resolution
            cln=convolve_with_beam(x["cln"][L], CELL, *BE) / omega(BE))
SB = {k: v.sum(0) * DV for k, v in cube.items()}                           # moment 0, Jy arcsec^-2 km/s
flux = {k: float(v[L].sum() * DV) for k, v in x.items()}
spec = {k: v.sum((1, 2)) for k, v in x.items()}
MV = M[L]
err = {k: dict(total=100 * (flux[k] - flux["cln"]) / flux["cln"],
               rmse=float(100 * np.linalg.norm((cube[k] - cube["cln"])[MV]) / np.linalg.norm(cube["cln"][MV])))
       for k in ("raw", "tgt")}
print({k: {a: round(b, 2) for a, b in v.items()} for k, v in err.items()}, {k: round(v, 1) for k, v in flux.items()})

# ======================================================================== layout (inches), as figure 40
S, GAP, CBH, CBP = 5.2, 0.45, 0.22, 0.16
ML, MR, MB, MT = 0.45, 1.45, 1.35, 1.95
FW, FH = ML + 3 * S + 2 * GAP + MR, MB + 2 * S + GAP + MT
fig = plt.figure(figsize=(FW, FH))
inch = lambda x_, y_, w, h_: fig.add_axes([x_ / FW, y_ / FH, w / FW, h_ / FH])
PX = lambda c: ML + c * (S + GAP)
PY = lambda r: MB + (1 - r) * (S + GAP)


def textbox(a, text, light):
    col = "k" if light else "w"
    head, *rest = text.split("\n")
    lines = [TextArea(head, textprops=dict(color=col, fontsize=17, fontweight="bold"))]
    lines += [TextArea(t, textprops=dict(color=col, fontsize=16)) for t in rest]
    ab = AnchoredOffsetbox(loc="upper left", child=VPacker(children=lines, align="left", pad=0, sep=4),
                           pad=0.45, borderpad=0.6, frameon=True)
    ab.patch.set_boxstyle("round,pad=0.3")
    ab.patch.set(facecolor="white" if light else "black", edgecolor="0.6" if light else "0.35",
                 alpha=0.9 if light else 0.92, linewidth=1.2)
    a.add_artist(ab)


def panel(r, c, img2, beam, text, hi, lo=0.0, cmap="inferno", light=False):
    a = inch(PX(c), PY(r), S, S)
    hp = int(round(HALF / CELL)); sl = np.s_[C0 - hp:C0 + hp, C0 - hp:C0 + hp]; ext = [HALF, -HALF, -HALF, HALF]
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("0.93" if light else "0.12")
    im = a.imshow(np.where(FOOT, img2, np.nan)[sl], origin="lower", extent=ext, cmap=cm, vmin=lo, vmax=hi,
                  interpolation="nearest")
    a.contour(FOOT[sl].astype(float), levels=[0.5], colors="0.55", linewidths=0.6, extent=ext, origin="lower")
    a.set_xticks([]); a.set_yticks([])
    ink = "k" if light else "w"; u = HALF / 5.5
    x0, y0 = -HALF + 0.6 * u, HALF - 0.7 * u
    a.plot([x0 + 1.0, x0], [y0, y0], color=ink, lw=2.5, solid_capstyle="butt")
    a.text(x0 + 0.5, y0 - 0.15 * u, "1″", color=ink, ha="center", va="top", fontsize=17)
    bx, by = HALF - 0.35 * u - beam[0] / 2, -HALF + 0.35 * u + beam[0] / 2
    a.add_patch(Ellipse((bx, by), width=beam[0], height=beam[1], angle=90 - beam[2], facecolor="0.15" if light else "w",
                        edgecolor=ink, lw=1.2))
    btxt = f"{beam[0]:.3f}″ (circular)" if beam[0] == beam[1] else f"{beam[0]:.3f}″ × {beam[1]:.3f}″ @ {beam[2]:.0f}°"
    a.text(bx - beam[0] / 2 - 0.2 * u, by, btxt, color=ink, ha="left", va="center", fontsize=16)
    textbox(a, text, light)
    cax = inch(PX(c), PY(r) + S + CBP if r == 0 else PY(r) - CBP - CBH, S, CBH)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    if r == 0:
        cax.xaxis.set_ticks_position("top"); cax.xaxis.set_label_position("top")
    cax.tick_params(labelsize=TICK, pad=6, direction="out", length=6)
    cb.set_label(SBU, fontsize=LAB, labelpad=8)


ve = np.percentile(SB["cln"][FOOT], 99.9)
panel(0, 0, SB["raw"], BE, "Compact deconvolved\n∗ extended clean beam", ve)
panel(0, 1, SB["tgt"], BT, "Compact deconvolved\ntarget resolution = ext. clean beam", ve)
panel(0, 2, SB["cln"], BE, "Extended MS-CLEAN\n∗ extended clean beam", ve)
for c, (k, t) in enumerate((("raw", "Difference: raw ∗ beam − CLEAN"), ("tgt", "Difference: target − CLEAN"))):
    panel(1, c, SB[k] - SB["cln"], BE if k == "raw" else BT,
          f"{t}\ntotal error {err[k]['total']:+.1f}%, RMSE (cube) {err[k]['rmse']:.1f}%", ve, lo=-ve, cmap="RdBu_r",
          light=True)
a = inch(PX(2), PY(1), S, S)
a.axvspan(vel[L[0]] - DV / 2, vel[L[-1]] + DV / 2, color=MUTED, alpha=0.1, lw=0)
for k, col, lab in (("raw", BLUE, "compact deconvolved"), ("tgt", ORANGE, "compact, target resolution"),
                    ("cln", GREEN, "extended MS-CLEAN")):
    a.step(vel, spec[k], where="mid", color=col, lw=2, label=f"{lab}: {flux[k]:.0f}")
a.axhline(0, color=MUTED, lw=0.8)
a.set_xlabel(r"velocity [km s$^{-1}$]", fontsize=LAB, labelpad=8)
a.set_ylabel("flux density in the mask [Jy]", fontsize=LAB, labelpad=10)
a.yaxis.tick_right(); a.yaxis.set_label_position("right")
a.tick_params(which="both", direction="in", top=True, bottom=True, left=True, right=True, labelleft=False, labelright=True)
a.tick_params(which="major", labelsize=TICK, pad=6, length=7); a.tick_params(which="minor", length=4)
a.minorticks_on()
a.set_ylim(top=1.8 * max(v.max() for v in spec.values()))
textbox(a, "Integrated spectra", light=True)
a.legend(frameon=False, fontsize=13, loc="upper right", bbox_to_anchor=(0.995, 0.86), title="Jy km/s", title_fontsize=13)
fig.text(0.5, 1 - 0.32 / FH, f"{GC['label']}: convolving to vs deconvolving to the extended clean beam", ha="center",
         va="top", fontsize=26, fontweight="bold")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=65)
json.dump(dict(target_fwhm_arcsec=TFWHM, flux_Jy_kms=flux, err_pct=err),
          open(OUT.replace(".png", ".json"), "w"), indent=1)
print("saved", OUT)
