#!/usr/bin/env python
"""
Paper figure: example FIRE [CII] 158 um channel maps from the preprocessed cubes (scripts/fire_preprocess.py), one
column per galaxy (C1 omitted: its IFU cubes duplicate C2's), NROWS rows. In every column the rows are distinct
redshifts in a random order (no galaxy repeats a redshift), each with a random line of sight and a random channel
among those carrying >= CHAN_FRAC of the cube's brightest channel's flux (so no panel is empty). Every panel is the
same field on the same grid (angular, or physical with fire_preprocess.py --cell-pc: then a 1 kpc scale bar), its own colour scale (0 to the VMAX_PCT percentile) with a power stretch; no
labels inside the panels (galaxy names above the columns, one 1" scale bar).

    [STAGE=sky|conv] [D=data/fire_cii_1x] [SUFFIX=] [NROWS=5] [SEED=1] [GAMMA=0.3] [VMAX_PCT=99.5] [CHAN_FRAC=0.25] python scripts/fire_channel_grid.py
    STAGE=sky (default): the unconvolved sky model (Jy/pixel); STAGE=conv: beam-convolved (Jy/beam, beam drawn)
    -> results/gallery/fire_cii_channels_<stage>_<tag><suffix>.{png,pdf}, drawn at A&A full page width (17 cm), final size;
       the draws are printed and saved next to it as .json
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
STAGE = os.environ.get("STAGE", "sky")
D = os.environ.get("D", os.path.join(REPO, "data/fire_cii_1x"))
SEED, GAMMA = int(os.environ.get("SEED", 1)), float(os.environ.get("GAMMA", 0.3))
NROWS, VMAX_PCT = int(os.environ.get("NROWS", 5)), float(os.environ.get("VMAX_PCT", 99.5))
CHAN_FRAC = float(os.environ.get("CHAN_FRAC", 0.25))
idx = json.load(open(os.path.join(D, "index.json")))
TAG = idx["settings"].get("tag", "4x")
OUT = os.path.join(REPO, "results/gallery", f"fire_cii_channels_{STAGE}_{TAG}{os.environ.get('SUFFIX', '')}.png")
cubes = {(c["gal"], c["snap"], c["los"]): c for c in idx["cubes"]}
GALS = sorted({k[0] for k in cubes}); SNAPS = sorted({k[1] for k in cubes}); LOS = sorted({k[2] for k in cubes})
cell, npix = idx["settings"]["cell"], idx["settings"]["npix"]
CELL_PC = idx["settings"].get("cell_pc")                      # physical pixels (fire_preprocess.py --cell-pc): kpc axes
half = npix * CELL_PC / 2e3 if CELL_PC else npix * cell / 2

# ======================================================================== draws
rng = np.random.default_rng(SEED)
draws = []                                                    # draws[r][g]
order = {g: rng.permutation(SNAPS) for g in GALS}
for r in range(NROWS):
    row = []
    for g in GALS:
        sn, los = int(order[g][r]), int(rng.choice(LOS))
        z = np.load(os.path.join(D, f"{g}_s{sn}_los{los}.npz"))
        x = z[STAGE]; f = z["sky"].sum((1, 2))                # channel fluxes from the sky model
        ok = np.flatnonzero(f >= CHAN_FRAC * f.max())
        ch = int(rng.choice(ok))
        row.append(dict(gal=g, snap=sn, los=los, chan=ch, v_kms=float(z["v"][ch]), img=x[ch].astype(np.float64)))
    draws.append(row)

# ======================================================================== layout (inches; A&A full width 17 cm)
FS = 7                                                        # text, pt
FW = 17 / 2.54
ML, MR, MT, MB, G = 0.02, 0.02, 0.17, 0.02, 0.03
S = (FW - ML - MR - (len(GALS) - 1) * G) / len(GALS)
FH = MT + NROWS * S + (NROWS - 1) * G + MB
plt.rcParams.update({"font.family": "serif", "font.size": FS})
fig = plt.figure(figsize=(FW, FH))
for r, row in enumerate(draws):
    for gi, d in enumerate(row):
        c = cubes[(d["gal"], d["snap"], d["los"])]
        a = fig.add_axes([(ML + gi * (S + G)) / FW, (FH - MT - S - r * (S + G)) / FH, S / FW, S / FH])
        a.set_xticks([]); a.set_yticks([])
        m = d["img"]
        a.imshow(m, origin="lower", extent=[half, -half, -half, half], cmap="inferno", interpolation="nearest",
                 norm=PowerNorm(GAMMA, vmin=0, vmax=max(float(np.percentile(m, VMAX_PCT)), 1e-30)))
        if STAGE == "conv":                                   # beam, bottom left, no text
            bmaj, bmin, bpa = c["beam"]; pad = 0.07 * half
            a.add_patch(Ellipse((half - pad - bmaj / 2, -half + pad + bmaj / 2), bmaj, bmin, angle=90 - bpa,
                                facecolor="w", edgecolor="w"))
        if r == 0 and gi == 0:                                # scale bar, top right, once
            L = 1.0; xs, ys, cap = -half + (0.22 if CELL_PC else 0.08) * half, half - 0.12 * half, 0.04 * half   # room for "1 kpc"
            a.plot([xs + L, xs], [ys, ys], color="w", lw=1.2, solid_capstyle="butt")
            for xe in (xs, xs + L):
                a.plot([xe, xe], [ys - cap, ys + cap], color="w", lw=0.9, solid_capstyle="butt")
            a.annotate("1 kpc" if CELL_PC else '1″', (xs + L / 2, ys - cap), xytext=(0, -1.5), textcoords="offset points", color="w",
                       ha="center", va="top", fontsize=FS - 1, fontweight="bold")
        if r == 0:
            a.set_title(d["gal"], fontsize=FS + 1, fontweight="bold", pad=2.5)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=300); fig.savefig(OUT.replace(".png", ".pdf"))
log = [[{k: v for k, v in d.items() if k != "img"} for d in row] for row in draws]
json.dump(dict(seed=SEED, stage=STAGE, source=D, chan_frac=CHAN_FRAC, gamma=GAMMA, vmax_pct=VMAX_PCT, draws=log),
          open(OUT.replace(".png", ".json"), "w"), indent=1)
for row in log:
    print("  ".join(f"{d['gal']} s{d['snap']} los{d['los']} ch{d['chan']:2d}" for d in row))
print("saved", OUT)
