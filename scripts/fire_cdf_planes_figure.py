#!/usr/bin/env python
"""
Every plane of the CDF 9/7 decomposition along velocity (gpu_pd.cdf_fwd, L levels) of one FIRE [CII] cube
(fire_preprocess.py, 32 x 25 km/s, physical pixels), PATCH x PATCH central crop. Top: the input channel maps. Below, one row per band, finest first: d_1 (nz/2 planes), d_2 (nz/4), ..., d_L, a_L -- the planes the 2D
dictionary (and the learnlet) acts on, together exactly as many as the channels. Each plane sits under the channels
it nominally covers (plane k of level l: channels k 2^l ... (k+1) 2^l - 1), so the rows form a pyramid; the channels
are one row of half-width panels, two per column.
One colour scale per row (comparable within a band): channels and a_L a power stretch, details sign(x) |x|^GAMMA on a
symmetric diverging scale.

    [D=data/fire_cii_1x_n400_19.5pc] [CUBE=B2_s85_los0] [PATCH=256] [L=5] [GAMMA=0.4] python scripts/fire_cdf_planes_figure.py
    -> results/gallery/fire_cii_cdf_planes_<cube>_L<L>_<patch>px.{png,pdf}
    PAPER=1: the paper version, A&A full page width (17 cm) at final size, Times (TeX Gyre Termes), A&A units, no title (channels stacked in pairs above
             the d_1 plane that compares them) -> ..._<patch>px_paper.{png,pdf}
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
CUBE = os.environ.get("CUBE", "B2_s85_los0")
PATCH, L, GAMMA = int(os.environ.get("PATCH", 256)), int(os.environ.get("L", 5)), float(os.environ.get("GAMMA", 0.4))
OUT = os.path.join(REPO, "results/gallery", f"fire_cii_cdf_planes_{CUBE}_L{L}_{PATCH}px.png")
c = {f"{x['gal']}_s{x['snap']}_los{x['los']}": x for x in json.load(open(os.path.join(D, "index.json")))["cubes"]}[CUBE]
z = np.load(os.path.join(D, CUBE + ".npz"))
sky, v = z["sky"].astype(np.float64), z["v"]
n0 = (sky.shape[1] - PATCH) // 2; sky = sky[:, n0:n0 + PATCH, n0:n0 + PATCH]
nz, dv = sky.shape[0], c["dv_kms"]
b = cdf_fwd(torch.from_numpy(sky)[None], L)[0].numpy()
pc = c["cell_arcsec"] * c["kpc_per_arcsec"] * 1e3; half = PATCH * pc / 2e3                # kpc
vr = lambda c0, c1: f"{v[c0] - dv / 2:+.0f}…{v[c1] + dv / 2:+.0f}".replace("-", "−")

# rows: (label, description, [(image, column centre in channel-pair units, velocity text)], kind)
NC = nz // 2                                                  # columns: one per channel pair
rows = [("channels", f"{dv:.0f} km/s\n{nz} maps", [(sky[k], k, vr(k, k)) for k in range(nz)], "chan")]
for lab, sl in band_slices(nz, L)[::-1]:                      # finest detail first, approximation last
    lev = L if lab == "a" else int(lab[1:]); w = 2 ** lev
    planes = [(b[sl][k], (k + 0.5) * w / 2 - 0.5, vr(k * w, min((k + 1) * w, nz) - 1)) for k in range(sl.stop - sl.start)]
    name = f"a$_{{{L}}}$" if lab == "a" else f"d$_{{{lev}}}$"
    desc = (f"approx.\n{w * dv:.0f} km/s" if lab == "a" else f"detail\n{w * dv:.0f} km/s") + f"\n{len(planes)} plane" + ("s" if len(planes) > 1 else "")
    rows.append((name, desc, planes, "pos" if lab == "a" else "signed"))


def paper():
    """A&A full page width (17 cm) at final size, no title (caption): a velocity axis on top, the channels, then one row
    per detail band with each plane as wide as the channels it covers (capped at CAP), d_L and a_L side by side last"""
    FS = 7.5
    KMS = r"km s$^{-1}$"                                      # A&A unit style
    plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "STIXGeneral"],
                         "mathtext.fontset": "stix", "font.size": FS, "axes.linewidth": 0.5,     # A&A: Times
                         "pdf.fonttype": 42,                                                 # embedded TrueType
                         "xtick.major.size": 2.5, "xtick.major.width": 0.5, "xtick.direction": "in"})
    ML, MR, MT, MB, G, RG, CAP = 0.50, 0.02, 0.36, 0.02, 0.02, 0.07, 1.30
    FW = 17 / 2.54
    u = (FW - ML - MR) / NC                                   # inches per channel pair
    det = rows[1:-2]                                          # d_1 ... d_{L-1}; d_L and a_L share the last row
    size = lambda planes: min(u * nz / 2 / len(planes) - G, CAP)
    s = u - G                                                 # channels: each pair stacked above its d_1 plane
    hts = [2 * s + G] + [size(p) for _, _, p, _ in det] + [CAP]
    FH = MT + sum(hts) + RG * (len(hts) - 1) + MB
    fig = plt.figure(figsize=(FW, FH))
    add = lambda x, y, w, h: fig.add_axes([x / FW, y / FH, w / FW, h / FH], xticks=[], yticks=[])
    kw = dict(origin="lower", extent=[half, -half, -half, half], interpolation="nearest")
    tops = FH - MT - np.cumsum(hts) - RG * np.arange(len(hts))                       # bottom edge of each row
    label = lambda y, h, name, sub: (fig.text((ML - 0.05) / FW, (y + h / 2) / FH, name, ha="right", va="bottom",
                                              fontsize=FS + 1.5, fontweight="bold"),
                                     fig.text((ML - 0.05) / FW, (y + h / 2) / FH, sub, ha="right", va="top", fontsize=FS - 1.5))
    # channels: channel 2k above 2k+1, in column k (over d_1 plane k, which compares them); one scale
    vmax = np.percentile(sky, 99.9)
    for k in range(nz):
        a = add(ML + (k // 2) * u, tops[0] + (1 - k % 2) * (s + G), s, s)
        a.imshow(sky[k], cmap="inferno", norm=PowerNorm(0.3, vmin=0, vmax=vmax), **kw)
        for sp in a.spines.values():
            sp.set_visible(False)
    label(tops[0], hts[0], "ch.", f"{dv:.0f} {KMS}")
    ax = fig.add_axes([ML / FW, (tops[0] + hts[0] + 0.03) / FH, (nz // 2 * u - G) / FW, 1e-4])   # velocity axis
    ax.set_xlim(v[0] - dv / 2, v[-1] + dv / 2); ax.set_yticks([]); ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    for sp in ("left", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel(f"$v$ ({KMS})", labelpad=2); ax.tick_params(labelsize=FS - 0.5, pad=1.5)
    # detail rows: each plane centred on its channels
    for r, (name, desc, planes, kind) in enumerate(det, start=1):
        sz = hts[r]; qs = [np.sign(p) * np.abs(p) ** GAMMA for p, _, _ in planes]
        m = max(np.percentile(np.abs(np.stack(qs)), 99.8), 1e-30)
        span = u * nz / 2 / len(planes)
        for i, q in enumerate(qs):
            a = add(ML + i * span + (span - G - sz) / 2, tops[r], sz, sz)
            a.imshow(q, cmap="RdBu_r", vmin=-m, vmax=m, **kw)
        label(tops[r], sz, name, f"{span / u * 2 * dv:.0f} {KMS}")
    # last row: d_L and a_L, both covering every channel, side by side in the middle
    (nd, _, (pd,), _), (na, _, (pa,), _) = rows[-2], rows[-1]
    xc = ML + nz / 2 * u / 2 - G / 2; y = tops[-1]
    qd = np.sign(pd[0]) * np.abs(pd[0]) ** GAMMA; m = max(np.percentile(np.abs(qd), 99.8), 1e-30)
    a = add(xc - CAP - 0.04, y, CAP, CAP); a.imshow(qd, cmap="RdBu_r", vmin=-m, vmax=m, **kw)
    a.text(0.04, 0.96, nd, transform=a.transAxes, ha="left", va="top", fontsize=FS + 1.5, fontweight="bold")
    a = add(xc + 0.04, y, CAP, CAP)
    a.imshow(pa[0], cmap="inferno", norm=PowerNorm(0.3, vmin=0, vmax=np.percentile(pa[0], 99.9)), **kw)
    a.text(0.04, 0.96, na, transform=a.transAxes, ha="left", va="top", fontsize=FS + 1.5, fontweight="bold", color="w")
    xs, ys, cap = -half + 0.6, half - 0.12 * half, 0.03 * half                       # scale bar, top right, on a_L
                                                                                       # (0.6 kpc in: room for "1 kpc")
    a.plot([xs + 1, xs], [ys, ys], color="w", lw=1.0, solid_capstyle="butt")
    for xe in (xs, xs + 1):
        a.plot([xe, xe], [ys - cap, ys + cap], color="w", lw=0.8, solid_capstyle="butt")
    a.annotate("1 kpc", (xs + 0.5, ys - cap), xytext=(0, -1.5), textcoords="offset points", color="w", ha="center",
               va="top", fontsize=FS - 0.5, fontweight="bold")
    label(y, CAP, "", f"{nz * dv:.0f} {KMS}")
    out = OUT.replace(".png", "_paper.png")
    fig.savefig(out, dpi=400); fig.savefig(out.replace(".png", ".pdf"), dpi=400)
    print("saved", out)


if os.environ.get("PAPER") == "1":
    paper()
    sys.exit()

S, G, RG, ML, MT, MB, MR = 0.82, 0.03, 0.10, 0.95, 0.55, 0.05, 0.05
FW, FH = ML + NC * S + (NC - 1) * G + MR, MT + S / 2 + (len(rows) - 1) * (S + RG) + 0.12 + MB
plt.rcParams.update({"font.family": "serif", "font.size": 8})
fig = plt.figure(figsize=(FW, FH))
chan_vmax = np.percentile(sky, 99.9)                          # all channels share one scale
for r, (name, desc, planes, kind) in enumerate(rows):
    y0 = FH - MT - S / 2 if r == 0 else FH - MT - S / 2 - RG - 0.12 - r * S - (r - 1) * RG   # gap below the channels
    s = S / 2 - G / 2 if kind == "chan" else S                   # channels: two half-width panels per column
    if kind == "signed":
        qs = [np.sign(p) * np.abs(p) ** GAMMA for p, _, _ in planes]
        m = max(np.percentile(np.abs(np.stack(qs)), 99.8), 1e-30)
    for i, (p, col, vt) in enumerate(planes):
        x0 = ML + (col // 2) * (S + G) + (col % 2) * (s + G) if kind == "chan" else ML + col * (S + G)   # chan: col = channel
        a = fig.add_axes([x0 / FW, y0 / FH, s / FW, s / FH]); a.set_xticks([]); a.set_yticks([])
        kw = dict(origin="lower", extent=[half, -half, -half, half], interpolation="nearest")
        if kind in ("pos", "chan"):
            a.imshow(p, cmap="inferno", norm=PowerNorm(0.3, vmin=0, vmax=chan_vmax if kind == "chan" else max(np.percentile(p, 99.9), 1e-30)), **kw)
        else:
            a.imshow(qs[i], cmap="RdBu_r", vmin=-m, vmax=m, **kw)
        if kind != "chan" or i % 4 == 0:                      # channels: every 4th labelled (no room)
            a.text(0.5, 0.02, vt, transform=a.transAxes, ha="center", va="bottom", fontsize=4.8 if kind != "chan" else 3.6,
                   color="k" if kind == "signed" else "w")
        if r == 1 and i == 0:                                 # scale bar, top right of the first d_1 plane
            xs, ys, cap = -half + 0.12 * half, half - 0.14 * half, 0.05 * half
            a.plot([xs + 1, xs], [ys, ys], color="k", lw=1.0, solid_capstyle="butt")
            for xe in (xs, xs + 1):
                a.plot([xe, xe], [ys - cap, ys + cap], color="k", lw=0.8, solid_capstyle="butt")
            a.annotate("1 kpc", (xs + 0.5, ys - cap), xytext=(0, -1), textcoords="offset points", color="k",
                       ha="center", va="top", fontsize=5.5, fontweight="bold")
    yc = (y0 + s / 2) / FH
    fig.text((ML - 0.08) / FW, yc + 0.006, name, ha="right", va="center", fontsize=10 if kind != "chan" else 8.5,
             fontweight="bold")
    if desc:
        fig.text((ML - 0.08) / FW, yc - 0.008, desc, ha="right", va="top", fontsize=6)
fig.text(0.5, 1 - 0.12 / FH, f"{c['gal']}  z = {c['z']:.2f}  i = {c['inclination_deg']:.0f}°: the {nz} channel maps "
         f"({dv:.0f} km/s) and every plane of their CDF 9/7 decomposition along velocity (L = {L}), {PATCH} px × "
         f"{pc:.1f} pc = {PATCH * pc / 1e3:.1f} kpc", ha="center", va="top", fontsize=10, fontweight="bold")
fig.text(0.5, 1 - 0.33 / FH, "each plane under the channels it nominally covers (km/s); one colour scale per row; details: "
         f"sign(x)|x|$^{{{GAMMA:g}}}$, red > 0 > blue", ha="center", va="top", fontsize=8)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=170); fig.savefig(OUT.replace(".png", ".pdf"))
print("saved", OUT)
