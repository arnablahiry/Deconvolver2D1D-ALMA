#!/usr/bin/env python
"""
Paper figure (A&A full page width, final size): how the CDF 9/7 transform along velocity (gpu_pd.cdf_fwd) splits a
noisy cube, level by level, on the NCH central channels (NCH = 32: the whole cube, the full L = 5 decomposition of
the solver) of one FIRE [CII] cube (fire_preprocess.py, physical pixels) plus white Gaussian noise (sigma = the
cube's 99.9th percentile / SNR, the same in every pixel and channel).
Top: the channels. Each level below splits the previous approximation a_{l-1} (the channels for l = 1) into the
details d_l (centre) and the approximation a_l (a bit farther out, joined by a faint dashed line, alternating sides);
a darker dotted arrow leads from a_l to the next level's details. Only the details and the last approximation are
kept by the transform; a_1 ... a_{L-1} are the intermediate steps. A group of planes is a grid, in velocity order
left to right, top to bottom, shaped to make its panels as large as possible.
On the free side of every level, the spectrum of an AP x AP pixel box (circled on the maps), chosen where the line
profile is clear: RMIN-3.3 kpc from the bright centre, its noise-free line >= 90% inside the analysed channels,
peaking away from their edges at S/N between 12 and 30, then the most concentrated (one line, no wings) and widest
(channels above half maximum, up to 6) with S/N nearest SPOT_SNR over the whole cube: the noisy spectrum (grey; the noise-free one light grey), the analysed channels shaded
(NCH < 32), and inside them the split: its input (a_{l-1}; dark grey), a_l (steps, each over the channels it
nominally covers) and d_l (stems); one y scale for all.
Channels and approximations share one colour scale (power stretch; a_l are weighted averages of the channels, same
units); each detail row its own symmetric scale, sign(x) |x|^GAMMA.

    [D=data/fire_cii_1x_n400_19.5pc] [CUBE=B2_s85_los0] [NCH=8] [PATCH=400] [SNR=10] [SEED=0] [AP=5] [RMIN=1.2]
        [SPOT_SNR=20] [GAMMA=0.4] [HMAX=...] [SMAX=...] python scripts/fire_cdf_chain_figure.py
    -> results/gallery/fire_cii_cdf_chain_<cube>_<nch>ch_<patch>px_snr<snr>.{png,pdf}
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
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from scipy.ndimage import uniform_filter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "simple"))
from gpu_pd import cdf_fwd  # noqa: E402

D = os.environ.get("D", os.path.join(REPO, "data/fire_cii_1x_n400_19.5pc"))
CUBE = os.environ.get("CUBE", "B2_s85_los0")
NCH, PATCH, GAMMA = int(os.environ.get("NCH", 8)), int(os.environ.get("PATCH", 400)), float(os.environ.get("GAMMA", 0.4))
SNR, SEED, AP = float(os.environ.get("SNR", 10)), int(os.environ.get("SEED", 0)), int(os.environ.get("AP", 5))
RMIN = float(os.environ.get("RMIN", 1.2))                      # kpc: the spectrum's box avoids the bright centre
SPOT_SNR = float(os.environ.get("SPOT_SNR", 20))              # ... and a clearly visible line profile
HMAX = float(os.environ.get("HMAX", 1.6 if NCH <= 8 else 1.05))   # inches: tallest group of planes
SMAX = float(os.environ.get("SMAX", 1.25 if NCH <= 8 else 0.95))  # inches: largest panel
L = int(np.log2(NCH))
OUT = os.path.join(REPO, "results/gallery", f"fire_cii_cdf_chain_{CUBE}_{NCH}ch_{PATCH}px_snr{SNR:g}.png")
c = {f"{x['gal']}_s{x['snap']}_los{x['los']}": x for x in json.load(open(os.path.join(D, "index.json")))["cubes"]}[CUBE]
z = np.load(os.path.join(D, CUBE + ".npz"))
full, vfull = z["sky"].astype(np.float64), z["v"]
n0 = (full.shape[1] - PATCH) // 2; full = full[:, n0:n0 + PATCH, n0:n0 + PATCH]
c0 = (full.shape[0] - NCH) // 2
clean, v = full[c0:c0 + NCH], vfull[c0:c0 + NCH]                                          # the analysed channels
sigma = np.percentile(clean, 99.9) / SNR
fulln = full + sigma * np.random.default_rng(SEED).standard_normal(full.shape)             # the noisy cube
sky = fulln[c0:c0 + NCH]
dv = c["dv_kms"]; half = PATCH * c["cell_arcsec"] * c["kpc_per_arcsec"] / 2               # kpc


def split(cube):
    """{l: a_l} (a_0 = the cube) and {l: d_l}: the first split levels of cdf_fwd"""
    A, Dl = {0: cube}, {}
    for lev in range(1, L + 1):
        b = cdf_fwd(torch.from_numpy(cube)[None], lev)[0].numpy(); n = NCH >> lev
        A[lev], Dl[lev] = b[:n], b[n:2 * n]
    return A, Dl


A, Dl = split(sky)
yy, xx = np.mgrid[:PATCH, :PATCH] - (PATCH - 1) / 2
rr = np.hypot(yy, xx) * 2 * half / PATCH                                                 # kpc
bf = uniform_filter(full, (1, AP, AP)) if AP > 1 else full                              # box-mean spectra, all channels
bx_ = bf[c0:c0 + NCH]; pos = np.clip(bx_, 0, None)
pk, ipk = bx_.max(0), bx_.argmax(0); snr_ = pk / (sigma / AP)
inwin = pos.sum(0) / np.maximum(np.clip(bf, 0, None).sum(0), 1e-30)                     # line flux inside the window
w = max(1, NCH // 10); k = np.arange(NCH)[:, None, None]
conc = (pos * (np.abs(k - ipk[None]) <= w)).sum(0) / np.maximum(pos.sum(0), 1e-30)      # one line, no wings
nhm = (bx_ >= pk[None] / 2).sum(0)                                                       # channels above half maximum
lo, hi = (2, NCH - 3) if NCH <= 8 else (NCH // 4, 3 * NCH // 4)
ok = (rr > RMIN) & (rr < 3.3) & (ipk >= lo) & (ipk <= hi) & (snr_ > 12) & (snr_ < 30) & (inwin > 0.9)
score = conc + 0.05 * np.minimum(nhm, 6) - 0.005 * np.abs(snr_ - SPOT_SNR)
iy, ix = np.unravel_index(np.argmax(np.where(ok, score, -np.inf)), pk.shape)
box = np.s_[:, iy - AP // 2:iy + AP // 2 + 1, ix - AP // 2:ix + AP // 2 + 1]
spec = lambda cube: cube[box].mean((1, 2)) * 1e6                                         # muJy / pixel
px = 2 * half / PATCH; bx, by = half - (ix + 0.5) * px, -half + (iy + 0.5) * px          # box centre, kpc

# ======================================================================== layout (inches)
FS = 7.5
KMS = r"km s$^{-1}$"
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "STIXGeneral"],
                     "mathtext.fontset": "stix", "font.size": FS, "axes.linewidth": 0.5, "pdf.fonttype": 42,
                     "xtick.major.size": 2.5, "xtick.major.width": 0.5, "xtick.direction": "in",
                     "ytick.major.size": 2.5, "ytick.major.width": 0.5, "ytick.direction": "in"})
FW = 17 / 2.54
MT, MB, G, GS, RG, LAB = 0.42, 0.10, 0.03, 0.30, 0.24, 0.20   # top/bottom, panel gap, side gap, row gap, label height
SW, SH, SG = 1.45, 0.85, 0.28                                 # spectrum panel width, height, gap to the details
HW = FW / 2 - 0.03                                            # usable half width
GW = (HW - GS) / 1.5                                          # widest group: half the details + gap + approximation


def grid(n, wmax, hmax, smax):
    """(columns, rows, panel size) for n panels within wmax x hmax, the largest panels (columns a power of 2)"""
    best = None
    for k in [2 ** i for i in range(int(np.log2(n)) + 1)]:
        r = n // k
        s = min((wmax - (k - 1) * G) / k, (hmax - (r - 1) * G) / r, smax)
        if best is None or s > best[2] + 1e-9:
            best = (k, r, s)
    return best


GR = {lev: grid(NCH >> lev, GW, HMAX, SMAX) for lev in range(1, L + 1)}
gsize = lambda g: (g[0] * g[2] + (g[0] - 1) * G, g[1] * g[2] + (g[1] - 1) * G)        # group width, height
H = {lev: max(gsize(GR[lev])[1], SH) for lev in range(1, L + 1)}                        # row height
H[L] = max(H[L], SH + 0.22)                                                               # room for the last x label
CH = grid(NCH, FW - 0.1, 0.85, 0.80)                                                      # the channels: <= 2 rows
FH = MT + gsize(CH)[1] + sum(LAB + RG + H[lev] for lev in range(1, L + 1)) + MB
fig = plt.figure(figsize=(FW, FH))
F = lambda xi, yi: (xi / FW, yi / FH)                         # inches -> figure fraction
cx = FW / 2
vmax = np.percentile(sky, 99.9)                               # channels and approximations: one scale
kw = dict(origin="lower", extent=[half, -half, -half, half], interpolation="nearest")
CA, CD, CI = "#d95f02", "#08306b", "0.35"                     # spectra: approximation, details, the split's input
sall = [spec(fulln), spec(full)] + [spec(A[l]) for l in A] + [spec(Dl[l]) for l in Dl]
lo, hi = min(min(x.min() for x in sall), 0), max(x.max() for x in sall)
YL = (lo - 0.08 * (hi - lo), hi + 0.12 * (hi - lo))
edges = lambda lev: v[0] - dv / 2 + dv * 2 ** lev * np.arange((NCH >> lev) + 1)             # nominal spans
vlab = lambda t: "0" if round(t) == 0 else f"{t:+.0f}".replace("-", "−")


def group(imgs, x0, ytop, g, kind):
    """a grid of panels (g = columns, rows, size) with its top-left at (x0, ytop); returns (axes, box in inches)"""
    k, r, s = g
    if kind == "signed":
        qs = [np.sign(p) * np.abs(p) ** GAMMA for p in imgs]; m = max(np.percentile(np.abs(np.stack(qs)), 99.8), 1e-30)
    axs = []
    for i, p in enumerate(imgs):
        a = fig.add_axes([*F(x0 + (i % k) * (s + G), ytop - (i // k + 1) * s - (i // k) * G), s / FW, s / FH],
                         xticks=[], yticks=[])
        if kind == "signed":
            a.imshow(qs[i], cmap="RdBu_r", vmin=-m, vmax=m, **kw)
        else:
            a.imshow(p, cmap="inferno", norm=PowerNorm(0.3, vmin=0, vmax=vmax), **kw)
            a.add_patch(Circle((bx, by), 0.35, fill=False, ec="w", lw=0.6))                 # the spectrum's box
        axs.append(a)
    w, h = gsize(g)
    return axs, (x0, x0 + w, ytop - h, ytop)


def tag(box, name, sub):
    """bold band name + its velocity scale, above a group"""
    xm = (box[0] + box[1]) / 2
    fig.text(*F(xm, box[3] + 0.03), name + r"$\,$", ha="right", va="bottom", fontsize=FS + 1.5, fontweight="bold")
    fig.text(*F(xm, box[3] + 0.045), " " + sub, ha="left", va="bottom", fontsize=FS - 1, color="0.3")


def arrow(p0, p1):
    """darker dotted arrow a_l -> the next split: a dotted shaft and a solid, filled head"""
    p0, p1 = np.array(p0), np.array(p1); u = (p1 - p0) / np.hypot(*(p1 - p0)); hl = 0.07   # head length, inches
    fig.add_artist(Line2D([p0[0] / FW, (p1[0] - hl * u[0]) / FW], [p0[1] / FH, (p1[1] - hl * u[1]) / FH],
                          transform=fig.transFigure, color="0.25", lw=0.9, ls=(0, (1, 1.6)), dash_capstyle="round"))
    fig.add_artist(FancyArrowPatch(F(*(p1 - hl * u)), F(*p1), transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=7, color="0.25", lw=0.6, shrinkA=0, shrinkB=0, joinstyle="miter"))


def join(b0, b1):                                             # faint dashed line between the two halves of a split
    y = (b0[2] + b0[3]) / 2
    xa, xb = (b0[1], b1[0]) if b1[0] > b0[1] else (b0[0], b1[1])
    fig.add_artist(Line2D([xa / FW, xb / FW], [y / FH, y / FH], transform=fig.transFigure, color="0.6", lw=0.7,
                          ls=(0, (4, 3))))


def spectrum(lev, x0, yc, right):
    """the box spectrum over the whole cube, and split l inside the analysed channels; y axis on the outer side"""
    a = fig.add_axes([*F(x0, yc - SH / 2), SW / FW, SH / FH])
    a.axhline(0, color="0.75", lw=0.5)
    ef = vfull[0] - dv / 2 + dv * np.arange(len(vfull) + 1)
    if NCH < len(vfull):
        a.axvspan(edges(0)[0], edges(0)[-1], color="0.93", lw=0, zorder=0)
    a.stairs(spec(full), ef, color="0.82", lw=0.7)
    a.stairs(spec(fulln), ef, color="0.6", lw=0.7)
    if lev > 1:
        a.stairs(spec(A[lev - 1]), edges(lev - 1), color=CI, lw=0.9)
    a.stairs(spec(A[lev]), edges(lev), color=CA, lw=1.3)
    e = edges(lev); xm = (e[:-1] + e[1:]) / 2
    ml, sl, bl = a.stem(xm, spec(Dl[lev]), linefmt="-", markerfmt="o", basefmt=" ")
    plt.setp(sl, color=CD, lw=0.8); plt.setp(ml, color=CD, ms=2.0 if NCH >> lev > 8 else 2.5)
    a.set_xlim(ef[0], ef[-1]); a.set_ylim(*YL)
    tk = [t for t in (-400, -200, 0, 200, 400) if ef[0] <= t <= ef[-1]]
    a.set_xticks(tk); a.set_xticklabels([vlab(t) for t in tk]); a.tick_params(labelsize=FS - 1.5, pad=1.5)
    if right:
        a.yaxis.tick_right(); a.yaxis.set_label_position("right")
    a.set_ylabel(r"$\mu$Jy px$^{-1}$", fontsize=FS - 1, labelpad=1.5)
    a.set_xlabel(f"$v$ ({KMS})", fontsize=FS - 1, labelpad=1)
    inp = "channels" if lev == 1 else f"a$_{{{lev - 1}}}$"
    a.set_title(f"{inp} → a$_{{{lev}}}$ + d$_{{{lev}}}$" + ("  (1 pixel)" if AP == 1 else f"  ({AP}×{AP} px box)"),
                fontsize=FS - 1, pad=2)
    if lev == 1:                                              # one legend, for every spectrum panel
        hs = [Line2D([], [], color="0.82", lw=0.7), Line2D([], [], color="0.6", lw=0.7),
              Line2D([], [], color=CI, lw=0.9), Line2D([], [], color=CA, lw=1.3),
              Line2D([], [], color=CD, lw=0.8, marker="o", ms=2.5)]
        a.legend(hs, ["noise-free", "noisy", r"input a$_{\ell-1}$", r"a$_\ell$", r"d$_\ell$"], fontsize=FS - 2.5,
                 frameon=False, loc="upper right", handlelength=1.2, borderaxespad=0.2, labelspacing=0.12,
                 ncol=2 if NCH > 8 else 1, columnspacing=0.8)


# channels: one row with a velocity axis, or a grid with the velocity in each panel
w0, h0 = gsize(CH); ytop = FH - MT
axs, prev = group(list(sky), cx - w0 / 2, ytop, CH, "pos")
if CH[1] == 1:
    ax = fig.add_axes([*F(prev[0], prev[3] + 0.03), w0 / FW, 1e-4])
    ax.set_xlim(v[0] - dv / 2 - G / CH[2] * dv / 2, v[-1] + dv / 2 + G / CH[2] * dv / 2); ax.set_yticks([])
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    for sp in ("left", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.set_xticks(v); ax.set_xticklabels([vlab(t) for t in v]); ax.tick_params(labelsize=FS - 0.5, pad=1.5)
    ax.set_xlabel(f"$v$ ({KMS}), {NCH} channels of {dv:.0f} {KMS}, with noise (peak S/N = {SNR:g})", labelpad=2)
else:
    for a, t in zip(axs, v):
        a.text(0.04, 0.03, vlab(t), transform=a.transAxes, color="w", fontsize=FS - 2.5, ha="left", va="bottom")
    fig.text(*F(cx, prev[3] + 0.05), f"{NCH} channels of {dv:.0f} {KMS} ({KMS} in each panel), with noise "
             f"(peak S/N = {SNR:g})", ha="center", va="bottom", fontsize=FS)
src = ((prev[0] + prev[1]) / 2, prev[2])                      # where the next arrow starts
ytop = prev[2]

for lev in range(1, L + 1):
    g = GR[lev]; wd, hd = gsize(g)
    ytop -= RG + LAB; yc = ytop - H[lev] / 2                  # row centre
    _, bd = group(list(Dl[lev]), cx - wd / 2, yc + hd / 2, g, "signed")
    side = 1 if lev % 2 else -1                               # approximation: right, left, right, ...
    aa, ba = group(list(A[lev]), bd[1] + GS if side > 0 else bd[0] - GS - wd, yc + hd / 2, g, "pos")
    join(bd, ba)
    spectrum(lev, bd[0] - SG - SW if side > 0 else bd[1] + SG, yc, right=side < 0)
    arrow(src, ((bd[0] + bd[1]) / 2, bd[3] + LAB + 0.02))                      # ends above the d_l label
    tag(bd, f"d$_{{{lev}}}$", f"{2 ** lev * dv:.0f} {KMS}")
    tag(ba, f"a$_{{{lev}}}$", ("kept" if lev == L else "intermediate") + f", {2 ** lev * dv:.0f} {KMS}")
    src = ((ba[0] + ba[1]) / 2, ba[2])
    ytop -= H[lev]
    if lev == L:                                              # scale bar, top right, on the last approximation
        a = aa[0]; xs, ys, cap = -half + 0.6 * half / 3.9, half - 0.12 * half, 0.03 * half
        a.plot([xs + 1, xs], [ys, ys], color="w", lw=1.0, solid_capstyle="butt")
        for xe in (xs, xs + 1):
            a.plot([xe, xe], [ys - cap, ys + cap], color="w", lw=0.8, solid_capstyle="butt")
        a.annotate("1 kpc", (xs + 0.5, ys - cap), xytext=(0, -1.5), textcoords="offset points", color="w",
                   ha="center", va="top", fontsize=FS - 0.5, fontweight="bold")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=400); fig.savefig(OUT.replace(".png", ".pdf"), dpi=400)
print("saved", OUT, f"({FW * 2.54:.1f} x {FH * 2.54:.1f} cm); sigma = {sigma:.3g} Jy/px; box at px ({iy}, {ix});",
      "grids", {l: GR[l][:2] for l in GR})
