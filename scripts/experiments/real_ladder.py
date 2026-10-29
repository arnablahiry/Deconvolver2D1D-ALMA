#!/usr/bin/env python
"""
Real NGC 3110: every pipeline variant (dictionary x method) against the joint compact + extended CLEAN over the whole
resolution ladder (scripts/experiments/ladder_joint.py: the joint visibilities imaged and restored at each of 11 beams,
results/NGC3110/ladder_joint/b00..b10, from the compact clean beam through the compact lambda/B_max (b05) to the
extended clean beam (b10); 32 x 25 km/s, 0.04"; here the solver's central 16").

At each beam b, over the line channels and the 16" field:
  cube RMSE     100 ||(G_b x - J_b) M_b|| / ||J_b M_b||  (%), G_b the peak-1 Gaussian of b (x in Jy/pixel -> Jy/beam, as
                J_b), M_b the EMISSION MASK of fig_resolution_ladder.py REF=joint at that beam (surface brightness
                >= 0.044 Jy/arcsec^2 and >= 5 sigma, regions >= 2 beams, inside the compact-beam mask; its masks.npz),
                i.e. the metric of the earlier resolution-ladder figures; the whole-field (unmasked) RMSE is kept too
                (real_ladder.json "rmse_full"), which also counts the noise-dominated voxels outside the emission
  noise floor   100 sigma_b sqrt(N_M) / ||J_b M_b||: the truth's own noise (its line-free channels), which no model can beat
  difference    100 (RMSE_arm - RMSE_production) / RMSE_production, against the production pipeline (pdtau, starlet,
                k = 5, results/NGC3110/compact/deconvolved/model.npy)
Each arm is shown at one k (or s): the one with the lowest mean RMSE over the ladder.

Power spectra, at the extended clean beam (b10): azimuthally averaged |FT|^2 of the line moment 0 of G_b10 x and of the
truth; their amplitude ratio; the error amplitude |FT(G x - J)| / |FT J|.

Families (exp23_style colours: hue = where the filters come from):
  pdtau      the pipeline as is, the 2D dictionary swapped              files <dict>_pdtau_k<k>.npy
  pdsyn      pdtau with the dual brought back by W^-1 instead of W^T     <dict>_pdsyn_k<k>.npy
  fista      the pipeline's FISTA (image rebuilt through W^-1)           <dict>_fista_k<k>.npy
  learnt     pdtau with the learnt per-scale thresholds x s             <dict>_pdtaul_s<s>.npy

    real_ladder.py  -> figures/experiments/23_iht/report/real_ladder_<family>.png, real_ladder_best.png,
                       real_power.png; data/experiments/23_iht/real_ladder.json
"""
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.patches import Ellipse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import beam_ft, conv  # noqa: E402
from common import EXP_DATA, REPO, line_channels, noise_channels  # noqa: E402
from exp23_style import HUE, INK, MUTED, rc  # noqa: E402

D = os.path.join(EXP_DATA, "23_iht", "real")
FIG = os.path.join(REPO, "figures/experiments/23_iht/report"); os.makedirs(FIG, exist_ok=True)
LAD = os.path.join(REPO, "results/NGC3110/ladder_joint")
DV, CELL = 25.0, 0.04
LINE, NOISE = line_channels("ngc3110"), noise_channels("ngc3110", "compact")
rc()

# ------------------------------------------------------------------ the ladder's truth cubes
BEAMS, J = [], []
for i in range(11):
    p = os.path.join(LAD, f"b{i:02d}", "clean_finished", "image.fits")
    h = fits.getheader(p)
    BEAMS.append((h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"]))
    J.append(np.nan_to_num(fits.getdata(p)).squeeze()[:, 440:840, 440:840].astype(np.float64))
NB = len(BEAMS); ILB = 5
KFT = [beam_ft((400, 400), b, CELL) for b in BEAMS]
MZ = np.load(os.path.join(REPO, "results/NGC3110/figures/resolution_ladder_joint/masks.npz"))
assert list(MZ["line"]) == list(LINE) and float(MZ["cell"]) == CELL
MASK = [MZ[f"b{i:02d}"] for i in range(11)]                                 # (line channels, 400, 400), the 16" field
JN = [np.linalg.norm(j[LINE] * m) for j, m in zip(J, MASK)]
JNF = [np.linalg.norm(j[LINE]) for j in J]
FLOOR = [100 * (j[NOISE] - j[NOISE].mean(0)).std() * np.sqrt(m.sum()) / n for j, m, n in zip(J, MASK, JN)]
NAMES = {0: "compact\nclean beam", ILB: "compact\nλ/B$_{max}$", NB - 1: "extended\nclean beam"}

# ------------------------------------------------------------------ the arms
DLAB = {"starlet": ("starlet", HUE["starlet"]), "learnlet": ("ImageNet learnlet (64)", HUE["imagenet"]),
        "learnlet_ext32": ("ImageNet, extrapolated (top 32)", HUE["imagenet"]),
        "learnlet_fire_n600": ("FIRE 600 px, per scale", HUE["fire600"]),
        "learnlet_fire_n600_shared": ("FIRE 600 px, shared", HUE["fire600"]),
        "learnlet_fire_shared4": ("FIRE 256 px, shared", HUE["fire400"]),
        "learnlet_fire_aug": ("FIRE scale-augmented, per scale", HUE["fireaug"]),
        "learnlet_fire_aug_shared": ("FIRE scale-augmented, shared", HUE["fireaug"]),
        "learnlet_deconv": ("trained for the deconvolution, per scale", "#6250d6"),
        "learnlet_deconv_shared": ("trained for the deconvolution, shared", "#6250d6"),
        "learnlet_deconv2": ("trained through unrolled FISTA (v2)", "#6250d6"),
        "learnlet_deconv_pd": ("trained through unrolled pdtau", "#6250d6")}
LS = {"learnlet_ext32": "--", "learnlet_fire_n600_shared": "--", "learnlet_fire_shared4": "--",
      "learnlet_fire_aug_shared": "--", "learnlet_deconv_shared": "--", "learnlet_deconv2": "-."}
FAM = {"pdtau": "the pipeline (pdtau), dictionary swapped", "pdsyn": "pdtau with the synthesis W$^{-1}$ instead of W$^{T}$",
       "fista": "the pipeline's FISTA (image rebuilt through W$^{-1}$)", "pdtaul": "pdtau with the learnt per-scale thresholds × s"}
MK = {"pdtau": "o", "pdsyn": "*", "fista": "X", "pdtaul": "P"}
runs = {}                                                           # (dict, family) -> {k: path}
for f in glob.glob(os.path.join(D, "*_*_[ks]*.npy")):
    m = re.match(r"^(?P<d>[a-z0-9_]+?)_(?P<r>pdtau|pdtaul|pdsyn|fista)_[ks](?P<k>[0-9.]+)\.npy$", os.path.basename(f))
    if m and m["d"] in DLAB:
        runs.setdefault((m["d"], m["r"]), {})[float(m["k"])] = f
runs.setdefault(("starlet", "pdtau"), {})[5.0] = os.path.join(REPO, "results/NGC3110/compact/deconvolved/model.npy")
native = lambda x: x.reshape(x.shape[0], 400, 2, 400, 2).sum((2, 4)) if x.shape[-1] == 800 else x


RFULL = {}


def ladder(x, key=None):
    """masked cube RMSE per beam (the main metric); the unmasked one is stored in RFULL[key]"""
    xl = x[LINE]; out, full = [], []
    for i in range(NB):
        e = conv(xl, KFT[i]).cpu().numpy() - J[i][LINE]
        out.append(float(100 * np.linalg.norm(e * MASK[i]) / JN[i])); full.append(float(100 * np.linalg.norm(e) / JNF[i]))
    if key is not None:
        RFULL[key] = full
    return out


R = {}                                                              # (dict, family, k) -> RMSE per beam
for (d, fam), ks in sorted(runs.items()):
    for k, p in sorted(ks.items()):
        R[(d, fam, k)] = ladder(native(np.load(p).astype(np.float64)), (d, fam, k))
        print(f"{d:26s} {fam:6s} {k:<5g} " + " ".join(f"{v:5.1f}" for v in R[(d, fam, k)]), flush=True)
PROD = R[("starlet", "pdtau", 5.0)]
best = {(d, fam): min(ks, key=lambda k: np.mean(R[(d, fam, k)])) for (d, fam), ks in runs.items()}
json.dump(dict(beams=BEAMS, floor=FLOOR, metric="masked (fig_resolution_ladder REF=joint masks)",
               rmse={f"{d}|{fam}|{k:g}": v for (d, fam, k), v in R.items()},
               rmse_full={f"{d}|{fam}|{k:g}": v for (d, fam, k), v in RFULL.items()},
               best={f"{d}|{fam}": k for (d, fam), k in best.items()}),
          open(os.path.join(EXP_DATA, "23_iht", "real_ladder.json"), "w"), indent=1)


def beam_strip(a, H, HS, W):
    """the ladder's beams to scale below the x axis (one arcsec scale for all; RA increases left)"""
    s = a.inset_axes([0, -HS / H - 0.04, 1, HS / H]); s.set_xlim(-0.6, NB - 0.4)
    ang = lambda b: np.radians(90 + b[2])
    ext = lambda b, c, sn: np.hypot(b[0] / 2 * c, b[1] / 2 * sn)
    wide = max(2 * ext(b, np.cos(ang(b)), np.sin(ang(b))) for b in BEAMS)
    tall = max(2 * ext(b, np.sin(ang(b)), np.cos(ang(b))) for b in BEAMS)
    XR = NB + 0.2; ipc = W / XR; BH = 0.62 * HS
    k = min(0.8 * ipc / wide, BH / tall) / ipc
    du = XR / W
    s.set_ylim(0, HS * du)
    for i, b in enumerate(BEAMS):
        s.add_patch(Ellipse((i, (HS - 0.04 - BH / 2) * du), width=b[0] * k, height=b[1] * k, angle=90 + b[2],
                            facecolor="none", edgecolor=INK if i in NAMES else MUTED, lw=1.5))
        if i in NAMES:
            s.text(i, 0.0, NAMES[i], ha="center", va="bottom", fontsize=8, color=MUTED)
    s.axis("off")


def ladder_figure(arms, title, name):
    """arms: list of (label, colour, linestyle, marker, rmse list). Three panels sharing the beam axis: the absolute cube
    RMSE (small, context); the relative difference from the production pipeline (the differences are < 1% of the RMSE,
    invisible on the absolute axis); the same after removing the truth's own noise floor, sqrt(RMSE^2 - floor^2), which
    at the finest beams is up to 60% of the RMSE and compresses every difference there."""
    own = lambda r: [np.sqrt(max(v ** 2 - f ** 2, 0.0)) for v, f in zip(r, FLOOR)]
    po = own(PROD)
    W, H1, H2, HS, G = 11.5, 2.6, 3.0, 1.35, 0.35
    FH = H1 + 2 * H2 + HS + 2 * G + 1.5
    ax_ = lambda y, h: fig.add_axes([1.0 / (W + 1.6), y / FH, W / (W + 1.6), h / FH])
    fig = plt.figure(figsize=(W + 1.6, FH))
    y3 = HS + 0.75; y2 = y3 + H2 + G; y1 = y2 + H2 + G
    a1 = ax_(y1, H1); a2 = ax_(y2, H2); a3 = ax_(y3, H2)
    a2.sharex(a1); a3.sharex(a1)
    x = np.arange(NB)
    a1.fill_between(x, 0, FLOOR, color="0.85", lw=0, label="the truth's own noise floor")
    for lab, c, ls, mk, r in arms:
        kw = dict(ls=ls, marker=mk, color=c, mfc="white" if ls != "-" else c, mew=1.5)
        a1.plot(x, r, ms=5, label=lab, **kw)
        a2.plot(x, [100 * (v - p) / p for v, p in zip(r, PROD)], ms=7, **kw)
        a3.plot(x, [100 * (v - p) / p for v, p in zip(own(r), po)], ms=7, **kw)
    a1.plot(x, PROD, ls="--", color=HUE["pipeline"], lw=2.0, marker="x", ms=6, label="production pipeline (pdtau, starlet, k = 5)")
    for a in (a2, a3):
        a.axhline(0, color=HUE["pipeline"], ls="--", lw=1.6)
        lim = max(1.0, *(abs(v) for line in a.get_lines() for v in line.get_ydata()))
        a.set_ylim(-1.08 * lim, 1.08 * lim)
    for a in (a1, a2, a3):
        a.axvline(ILB, color=MUTED, lw=0.9, ls=":")
    a1.set_ylabel("cube RMSE vs the joint\nCLEAN, in its emission\nmask [%]"); a1.set_ylim(0, None)
    a2.set_ylabel("RMSE vs production [%]\n(< 0: better)")
    a3.set_ylabel("same, the truth's noise\nfloor removed [%]")
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5 + 0.4 / (W + 1.6), (y1 + H1 + 0.95) / FH), ncol=3, fontsize=8)
    for a in (a1, a2):
        a.tick_params(labelbottom=False)
    a3.set_xticks(x)
    a3.set_xticklabels([f"{b[0]:.2f}″×{b[1]:.2f}″" if abs(b[0] - b[1]) > 1e-3 else f"{b[0]:.2f}″" for b in BEAMS], fontsize=8)
    for i in NAMES:
        a3.get_xticklabels()[i].set_fontweight("bold")
    a1.set_xlim(-0.6, NB - 0.4)
    beam_strip(a3, H2, HS, W)
    fig.suptitle(title, fontsize=11, y=(y1 + H1 + 1.35) / FH)
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=140, bbox_inches="tight"); plt.close(fig)
    print("->", os.path.join(FIG, name + ".png"))


for fam in FAM:
    arms = [(f"{DLAB[d][0]} ({'s' if fam == 'pdtaul' else 'k'} = {best[(d, f)]:g})", DLAB[d][1], LS.get(d, "-"), MK[fam],
             R[(d, f, best[(d, f)])]) for (d, f) in sorted(best) if f == fam and not (d == "starlet" and fam == "pdtau"
                                                                                       and best[(d, f)] == 5.0)]
    if arms:
        ladder_figure(arms, f"NGC 3110, real data: {FAM[fam]} — each arm at its best k over the ladder", f"real_ladder_{fam}")
# the best arm of each family, and each family's starlet
pick = []
for fam in FAM:
    cand = [(d, f) for (d, f) in best if f == fam and d != "starlet"]
    if cand:
        d, f = min(cand, key=lambda df: np.mean(R[(*df, best[df])]))
        pick.append((f"{DLAB[d][0]}, {fam} ({'s' if fam == 'pdtaul' else 'k'} = {best[(d, f)]:g})", DLAB[d][1],
                     LS.get(d, "-"), MK[fam], R[(d, f, best[(d, f)])]))
    if ("starlet", fam) in best and fam != "pdtau":
        pick.append((f"starlet, {fam} (k = {best[('starlet', fam)]:g})", HUE["starlet"], ":", MK[fam],
                     R[("starlet", fam, best[("starlet", fam)])]))
ladder_figure(pick, "NGC 3110, real data: the best learnlet arm of each method family", "real_ladder_best")

# ------------------------------------------------------------------ power spectra at the extended clean beam
Jb = J[NB - 1]; tm0 = Jb[LINE].sum(0) * DV


def radial(img):
    p = np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2
    yy, xx = np.indices(p.shape); n = img.shape[0]
    kr = np.hypot(yy - n // 2, xx - n // 2) / (n * CELL)
    edges = np.geomspace(1 / (n * CELL), 0.5 / CELL, 36); idx = np.digitize(kr.ravel(), edges)
    return np.sqrt(edges[1:] * edges[:-1]), np.array([p.ravel()[idx == i].mean() if (idx == i).any() else np.nan
                                                      for i in range(1, len(edges))])


kk, pt = radial(tm0)
show = [("production pipeline (pdtau, starlet, k = 5)", HUE["pipeline"], "--", ("starlet", "pdtau", 5.0))]
for lab, c, ls, mk, r in pick:
    key = next(kk_ for kk_, v in R.items() if v is r)
    show.append((lab, c, ls, key))
fig, ax = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
ax[0].loglog(kk, pt, color=INK, lw=2.4, label="joint CLEAN (truth), extended beam")
for lab, c, ls, key in show:
    d, fam, k = key
    p = runs[(d, fam)][k]
    g = conv(native(np.load(p).astype(np.float64))[LINE], KFT[NB - 1]).cpu().numpy()
    gm0 = g.sum(0) * DV
    ax[0].loglog(kk, radial(gm0)[1], color=c, ls=ls, lw=1.5, label=lab)
    ax[1].semilogx(kk, np.sqrt(radial(gm0)[1] / pt), color=c, ls=ls, lw=1.5)
    ax[2].loglog(kk, np.sqrt(radial(gm0 - tm0)[1] / pt), color=c, ls=ls, lw=1.5)
ax[1].axhline(1, color=INK, lw=0.8)
for a in ax:
    a.set_xlabel("spatial frequency [arcsec$^{-1}$]")
    for b, nm in ((BEAMS[0], "compact beam"), (BEAMS[ILB], "λ/B$_{max}$"), (BEAMS[-1], "extended beam")):
        a.axvline(1 / b[0], color=MUTED, lw=0.8, ls=":")
        a.text(1 / b[0], 0.97, nm + " ", rotation=90, fontsize=8, color=MUTED, va="top", ha="right",
               transform=a.get_xaxis_transform())
ax[0].set_ylabel("power of the line moment 0"); ax[1].set_ylabel("|model| / |truth|"); ax[2].set_ylabel("|model − truth| / |truth|")
ax[0].set_title("power spectra at the extended clean beam", fontsize=10.5)
ax[1].set_title("recovered amplitude", fontsize=10.5); ax[2].set_title("error amplitude", fontsize=10.5)
ax[0].legend(fontsize=7.5)
fig.savefig(os.path.join(FIG, "real_power.png"), dpi=140, bbox_inches="tight"); plt.close(fig)
print("->", os.path.join(FIG, "real_power.png"))
