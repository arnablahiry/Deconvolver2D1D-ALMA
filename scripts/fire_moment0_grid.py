#!/usr/bin/env python
"""
Grid of the preprocessed FIRE [CII] cubes (scripts/fire_preprocess.py -> data/fire_cii/): line moment 0, one row per
galaxy, columns grouped by redshift, three inclinations per group. Every panel is the same 2.56" field on the same
0.02" grid (so galaxies shrink with z), its own colour scale with a power stretch, its beam drawn in the corner.

    [STAGE=conv|sky|native] [D=data dir] [SUFFIX=_1x] [GAMMA=0.5] python scripts/fire_moment0_grid.py
    STAGE=native: the learnlet training set (scripts/fire_learnlet_dataset.py), whole box, physical scale
    STAGE=conv (default): the beam-convolved cubes, as surface brightness (Jy km/s arcsec^-2, comparable across beams)
    STAGE=sky: the unconvolved sky model at 0.02" (Jy km/s per pixel / pixel area)
    -> results/gallery/fire_cii_moment0_<stage><suffix>.{png,pdf}
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Ellipse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE = os.environ.get("STAGE", "conv")
D = os.environ.get("D", os.path.join(REPO, "data/fire_cii_learnlet" if STAGE == "native" else "data/fire_cii"))
GAMMA = float(os.environ.get("GAMMA", 0.5))
OUT = os.path.join(REPO, "results/gallery", f"fire_cii_moment0_{STAGE}{os.environ.get('SUFFIX', '')}.png")
idx = json.load(open(os.path.join(D, "index.json")))
cubes = {(c["gal"], c["snap"], c["los"]): c for c in idx["cubes"]}
GALS = sorted({k[0] for k in cubes}); SNAPS = sorted({k[1] for k in cubes}); LOS = sorted({k[2] for k in cubes})
if STAGE == "native":                                     # the training set's preview: moment 0 at 4 x 19.5 pc, whole box
    PREV = np.load(os.path.join(D, "m0_preview.npz"))
    for c in cubes.values():
        c["flux_total_jykms"] = c["flux_jykms"]
    k0 = next(iter(cubes))
    half = PREV[f"{k0[0]}_s{k0[1]}_los{k0[2]}"].shape[0] * 4 * cubes[k0]["pc_per_px_f1"] / 2e3        # kpc
else:
    cell, npix = idx["settings"]["cell"], idx["settings"]["npix"]
    half = npix * cell / 2
plt.rcParams.update({"font.family": "serif"})

S, G_IN, G_Z, ROWG = 1.15, 0.04, 0.22, 0.10                 # panel, gap within / between redshift groups, row gap (inch)
ML, MR, MT, MB = 0.75, 0.15, 1.35, 0.35
ncol = len(SNAPS) * len(LOS)
FW = ML + ncol * S + len(SNAPS) * (len(LOS) - 1) * G_IN + (len(SNAPS) - 1) * G_Z + MR
FH = MT + len(GALS) * S + (len(GALS) - 1) * ROWG + MB
fig = plt.figure(figsize=(FW, FH))
X = lambda si, li: ML + si * (len(LOS) * S + (len(LOS) - 1) * G_IN + G_Z) + li * (S + G_IN)
Y = lambda r: FH - MT - S - r * (S + ROWG)
for r, g in enumerate(GALS):
    fig.text((ML - 0.12) / FW, (Y(r) + S / 2) / FH, g, ha="right", va="center", fontsize=13, fontweight="bold")
    for si, sn in enumerate(SNAPS):
        for li, los in enumerate(LOS):
            c = cubes.get((g, sn, los))
            a = fig.add_axes([X(si, li) / FW, Y(r) / FH, S / FW, S / FH]); a.set_xticks([]); a.set_yticks([])
            if c is None:
                a.set_facecolor("0.85"); continue
            if STAGE == "native":
                m0 = PREV[f"{g}_s{sn}_los{los}"]
            else:
                z = np.load(os.path.join(D, f"{g}_s{sn}_los{los}.npz"))
                bmaj, bmin, bpa = c["beam"]
            if STAGE == "native":
                pass
            elif STAGE == "conv":
                m0 = z["conv"].sum(0) * c["dv_kms"] / (np.pi * bmaj * bmin / (4 * np.log(2)))
            else:
                m0 = z["sky"].sum(0) * c["dv_kms"] / cell ** 2
            a.imshow(m0, origin="lower", extent=[half, -half, -half, half], cmap="inferno",
                     norm=PowerNorm(GAMMA, vmin=0, vmax=max(float(m0.max() if STAGE == "conv" else np.percentile(m0, 99.9)), 1e-30)), interpolation="nearest")
            if STAGE == "conv":
                pad = 0.07 * half
                a.add_patch(Ellipse((half - pad - bmaj / 2, -half + pad + bmaj / 2), bmaj, bmin, angle=90 - bpa,
                                    facecolor="none", edgecolor="w", lw=0.8))
            a.text(0.04, 0.96, f"{c['flux_total_jykms'] * 1e3:.0f} mJy km/s" if c["flux_total_jykms"] >= 0.01
                   else f"{c['flux_total_jykms'] * 1e3:.1f} mJy km/s", transform=a.transAxes, color="w", fontsize=6.5, va="top")
            if r == 0:
                a.set_title(f"i = {c['inclination_deg']:.0f}°", fontsize=9, pad=3)
            if r == 0 and si == 0 and li == 0:                    # scale bar, once
                L, lab = (5.0, "5 kpc") if STAGE == "native" else (1.0, '1"')
                x1 = half - 0.06 * 2 * half
                a.plot([x1, x1 - L], [-half + 0.2 * half, -half + 0.2 * half], color="w", lw=1.5)
                a.text(x1 - L / 2, -half + 0.22 * half, lab, color="w", ha="center", va="bottom", fontsize=7)
    for si, sn in enumerate(SNAPS):
        c = next(cubes[k] for k in sorted(cubes) if k[1] == sn)
        x0 = X(si, 0) + (len(LOS) * S + (len(LOS) - 1) * G_IN) / 2
        fig.text(x0 / FW, (FH - MT + 0.42) / FH, f"z = {c['z']:.2f}" + ("" if STAGE == "native" else f"  ({c['kpc_per_arcsec']:.1f} kpc/″)"), ha="center",
                 va="bottom", fontsize=11, fontweight="bold")
title = {"conv": "FIRE [CII] 158 µm, line moment 0 after preprocessing: observed flux, 0.02″ pixels, 2.56″ field, "
                 "convolved with a beam drawn per cube (0.1–0.8″, ellipse)",
         "sky": "FIRE [CII] 158 µm, line moment 0 after preprocessing: observed flux, 0.02″ pixels, 2.56″ field, "
                "unconvolved sky model",
         "native": "FIRE [CII] 158 µm learnlet training set (preview): line moment 0 of the kept channel maps, whole "
                   "20 kpc box, physical pixels (shown at 78 pc; stored at 19.5, 39, 78 pc)"}[STAGE]
if STAGE != "native":                                     # which simulation cubes were resampled (4x: 78 pc, 1x: 19.5 pc)
    tag = idx["settings"].get("tag", "4x")
    title += f"; from the {tag} cubes ({ {'1x': 19.5, '4x': 78, '16x': 312}.get(tag, '?') } pc pixels)"
fig.text(0.5, 1 - 0.12 / FH, title, ha="center", va="top", fontsize=13, fontweight="bold")
fig.text(0.5, 1 - 0.38 / FH, f"each panel its own scale{'' if STAGE == 'conv' else ' (top 0.1% of pixels saturated)'}, power stretch γ = {GAMMA:g}; label = total [CII] flux in the "
         "cube; C1 omitted (its IFU cubes duplicate C2's)", ha="center", va="top", fontsize=10)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=110); fig.savefig(OUT.replace(".png", ".pdf"))
print("saved", OUT)
