#!/usr/bin/env python
"""
What the 2D filters of the 2D-1D dictionary see: for a few FIRE [CII] cubes (fire_preprocess.py, 32 x 25 km/s, physical
pixels), a PATCH x PATCH crop (the learnlet training patch) -- its line moment 0 on top, and below it one 2D plane per
CDF 9/7 band along velocity (gpu_pd.cdf_fwd, L levels): the spectral approximation a_L (2^L channels per plane,
positive) and the details d_L ... d_1 (signed: where the emission changes with velocity). A band with several planes
shows its most energetic one (its velocity range in the corner). Every panel its own scale: moment 0 and a_L a power
stretch, the details sign(x) |x|^GAMMA on a symmetric diverging scale.

    [D=data/fire_cii_1x_n400_19.5pc] [PATCH=256] [L=5] [GAMMA=0.4] [CUBES=A1_s80_los0,...] python scripts/fire_cdf_bands_figure.py
    -> results/gallery/fire_cii_cdf_bands_<patch>px.{png,pdf}
"""
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "simple"))
from gpu_pd import cdf_fwd, band_slices  # noqa: E402

D = os.environ.get("D", os.path.join(REPO, "data/fire_cii_1x_n400_19.5pc"))
PATCH, L, GAMMA = int(os.environ.get("PATCH", 256)), int(os.environ.get("L", 5)), float(os.environ.get("GAMMA", 0.4))
CUBES = os.environ.get("CUBES", "A1_s80_los0,B2_s85_los2,A4_s60_los1,A8_s75_los1,C2_s70_los0").split(",")
OUT = os.path.join(REPO, "results/gallery", f"fire_cii_cdf_bands_{PATCH}px.png")
idx = {f"{c['gal']}_s{c['snap']}_los{c['los']}": c for c in json.load(open(os.path.join(D, "index.json")))["cubes"]}

cols = []
for name in CUBES:
    z = np.load(os.path.join(D, name + ".npz")); c = idx[name]
    sky, v = z["sky"].astype(np.float64), z["v"]
    n0 = (sky.shape[1] - PATCH) // 2; sky = sky[:, n0:n0 + PATCH, n0:n0 + PATCH]       # central patch (on the peak)
    nz = sky.shape[0]
    b = cdf_fwd(torch.from_numpy(sky)[None], L)[0].numpy()                                # (nz, ny, nx), packed bands
    planes = []
    for lab, sl in band_slices(nz, L):
        k = int(np.argmax((b[sl] ** 2).sum((1, 2))))                                     # most energetic plane
        lev = L if lab == "a" else int(lab[1:]); w = 2 ** lev                           # channels per coefficient
        planes.append((lab, b[sl][k], v[min(k * w, nz - 1)] - c["dv_kms"] / 2, v[min((k + 1) * w, nz) - 1] + c["dv_kms"] / 2,
                       sl.stop - sl.start))
    cols.append((name, c, sky.sum(0) * c["dv_kms"], planes))

dv = cols[0][1]["dv_kms"]
ROWS = ["moment 0"] + [("a" if p[0] == "a" else "d") + f"$_{{{L if p[0] == 'a' else p[0][1:]}}}$" for p in cols[0][3]]
ROWDESC = ["Jy km/s / px"] + [f"approx., {2 ** L * dv:.0f} km/s" if p[0] == "a" else
                              f"detail, {2 ** int(p[0][1:]) * dv:.0f} km/s" for p in cols[0][3]]
S, G, ML, MT, MB, MR = 1.55, 0.04, 1.05, 0.30, 0.05, 0.05
FW, FH = ML + len(cols) * S + (len(cols) - 1) * G + MR, MT + len(ROWS) * S + (len(ROWS) - 1) * G + 0.12 + MB
plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig = plt.figure(figsize=(FW, FH))
pc = cols[0][1]["cell_arcsec"] * cols[0][1]["kpc_per_arcsec"] * 1e3; half = PATCH * pc / 2e3      # kpc
for ci, (name, c, m0, planes) in enumerate(cols):
    for r in range(len(ROWS)):
        y0 = FH - MT - S - r * (S + G) - (0.12 if r else 0)                              # a gap below the moment 0
        a = fig.add_axes([(ML + ci * (S + G)) / FW, y0 / FH, S / FW, S / FH]); a.set_xticks([]); a.set_yticks([])
        if r == 0:
            a.imshow(m0, origin="lower", extent=[half, -half, -half, half], cmap="inferno", interpolation="nearest",
                     norm=PowerNorm(0.3, vmin=0, vmax=np.percentile(m0, 99.9)))
            a.set_title(f"{c['gal']}  z = {c['z']:.2f}  i = {c['inclination_deg']:.0f}°", fontsize=9, fontweight="bold", pad=3)
            if ci == 0:                                                                  # scale bar, top right
                xs, ys, cap = -half + 0.22 * half, half - 0.12 * half, 0.04 * half
                a.plot([xs + 1, xs], [ys, ys], color="w", lw=1.2, solid_capstyle="butt")
                for xe in (xs, xs + 1):
                    a.plot([xe, xe], [ys - cap, ys + cap], color="w", lw=0.9, solid_capstyle="butt")
                a.annotate("1 kpc", (xs + 0.5, ys - cap), xytext=(0, -1.5), textcoords="offset points", color="w",
                           ha="center", va="top", fontsize=8, fontweight="bold")
            continue
        lab, p, v0, v1, n = planes[r - 1]
        if lab == "a":
            a.imshow(p, origin="lower", extent=[half, -half, -half, half], cmap="inferno", interpolation="nearest",
                     norm=PowerNorm(0.3, vmin=0, vmax=max(np.percentile(p, 99.9), 1e-30)))
        else:
            q = np.sign(p) * np.abs(p) ** GAMMA; m = max(np.percentile(np.abs(q), 99.8), 1e-30)
            a.imshow(q, origin="lower", extent=[half, -half, -half, half], cmap="RdBu_r", vmin=-m, vmax=m,
                     interpolation="nearest")
            a.set_facecolor("w")
        if n > 1:
            a.text(0.03, 0.03, f"{v0:+.0f}…{v1:+.0f} km/s".replace("-", "−"), transform=a.transAxes, fontsize=6.5,
                   color="w" if lab == "a" else "k", va="bottom")
for r, (lab, desc) in enumerate(zip(ROWS, ROWDESC)):
    y = FH - MT - S / 2 - r * (S + G) - (0.12 if r else 0)
    fig.text((ML - 0.08) / FW, y / FH + 0.012, lab, ha="right", va="center", fontsize=11, fontweight="bold")
    fig.text((ML - 0.08) / FW, y / FH - 0.008, desc + ("" if r == 0 else f"\n{cols[0][3][r - 1][4]} plane"
                                                         + ("s" if cols[0][3][r - 1][4] > 1 else "")), ha="right",
             va="center", fontsize=7)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200); fig.savefig(OUT.replace(".png", ".pdf"))
print("saved", OUT)
