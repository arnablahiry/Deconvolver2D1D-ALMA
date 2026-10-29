#!/usr/bin/env python
"""
Is the comparison with compact MS-CLEAN fair? NGC 3110, the power spectrum of Fig. 5 (fig_power_joint.py: joint
compact + extended MS-CLEAN at the extended clean beam as the reference, the same window, line channels, 0.08" grid)
for compact MS-CLEAN treated like the 2D-1D model, and with other weightings:

  restored   MS-CLEAN image restored with its own clean beam (components * clean beam + residual), as in Fig. 5
  model      MS-CLEAN components alone (Jy/pixel) convolved with the extended clean beam, exactly as the 2D-1D model
  weighting  natural (consistency_20kms_0.08/clean_compact_finished) and Briggs robust 0.5, 0, -1, -2
             (compact_weighting_20kms/r<R>/clean_finished, compact_weighting_clean.py), whichever exist

    python scripts/compact_vs_msclean_dendro/fig_power_msclean_weighting.py
    -> results/NGC3110/figures/resolution_ladder_joint20/power_msclean_weighting.{png,pdf,json}
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import binary_dilation, convolve1d, gaussian_filter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
from restore import convolve_with_beam  # noqa: E402

R = os.path.join(REPO, "results", "NGC3110"); OUT = os.path.join(R, "figures", "resolution_ladder_joint20")
GC = json.load(open(os.path.join(R, "config.json"))); G = json.load(open(os.path.join(R, "consistency_20kms_0.08/grid.json")))
L, F, CELL, n = G["line"], G["line_free"], 0.08, 640
LB = float(GC["lbmax_compact_arcsec"])
rd = lambda p: np.nan_to_num(fits.getdata(p)).astype(np.float64).squeeze()
bm = lambda p: (lambda h: (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"]))(fits.getheader(p))
om = lambda b: np.pi * b[0] * b[1] / (4 * np.log(2)); px = lambda b: om(b) / CELL ** 2


def crop(a):                                       # the joint cubes: 0.04" (1280 px) -> 0.08", pixel centres aligned
    k = np.array([0.5, 1.0, 0.5])
    return convolve1d(convolve1d(a, k, axis=-1, mode="constant"), k, axis=-2, mode="constant")[..., ::2, ::2] / 4.0


# ======================================================================== the reference and the window, as in Fig. 5
pj = os.path.join(R, "ladder_joint_20kms/b10/clean_finished"); BE = bm(os.path.join(pj, "image.fits"))
J, N = crop(rd(os.path.join(pj, "image.fits"))) / px(BE), crop(rd(os.path.join(pj, "residual.fits"))) / px(BE)
mk = np.load(os.path.join(OUT, "masks.npz"))["b00"].any(0)
W = gaussian_filter(binary_dilation(mk, iterations=int(round(1.0 / CELL))).astype(float), 0.5 / CELL / 2.355)
f = np.fft.fftfreq(n, d=CELL); kk = np.hypot(*np.meshgrid(f, f))
edges = np.arange(0, 1 / (2 * CELL), 2 / (n * CELL)); kc = 0.5 * (edges[1:] + edges[:-1])
idx = np.digitize(kk.ravel(), edges) - 1; ok = (idx >= 0) & (idx < len(kc)); cnt = np.bincount(idx[ok], None, len(kc))
ring = lambda P: np.bincount(idx[ok], P.ravel()[ok], len(kc)) / np.maximum(cnt, 1)
power = lambda cube, ch: ring(np.mean([np.abs(np.fft.fft2(cube[c] * W)) ** 2 for c in ch], 0))
PS = np.clip(power(J, L) - power(N, F), 1e-40, None)
snr1 = 1 / float(kc[np.nonzero((PS / power(N, F) < 1) & (kc > 0.05))[0][0]])

# ======================================================================== the cubes
P, BEAMS = {}, {}
x2d = np.load(os.path.join(R, "pixel_scale/model_0.08_support.npy")).astype(np.float64)
P["2D-1D"] = power(convolve_with_beam(x2d, CELL, *BE) / px(BE), L)
P["dirty"] = power(rd(os.path.join(R, "deconv_inputs/compact_0.08/dirty_cube.fits")) /
                   px(bm(os.path.join(R, "consistency_20kms_0.08/clean_compact_finished/image.fits"))), L)
RUNS = [("natural", os.path.join(R, "consistency_20kms_0.08/clean_compact_finished"))] + \
       [(f"robust {r:g}", os.path.join(R, f"compact_weighting_20kms/r{r:g}/clean_finished")) for r in (0.5, 0, -1, -2)]
for name, d in RUNS:
    if not os.path.exists(os.path.join(d, "image.fits")):
        continue
    b = bm(os.path.join(d, "image.fits")); BEAMS[name] = b
    P[f"{name}, restored"] = power(rd(os.path.join(d, "image.fits")) / px(b), L)
    P[f"{name}, model"] = power(convolve_with_beam(rd(os.path.join(d, "model.fits")), CELL, *BE) / px(BE), L)
    print(f"{name}: clean beam {b[0]:.2f} x {b[1]:.2f}", flush=True)

QB = 2 * np.log(2) / np.pi
def half(r):                                       # the finest equivalent beam reached before the ratio drops below 1/2
    i = np.nonzero((r < 0.5) & (kc > 0.05) & (kc < 1 / snr1))[0]
    return float(QB / kc[i[0]]) if i.size else float("nan")
res = dict(k=kc.tolist(), P_J_signal=PS.tolist(), joint_snr1_arcsec=snr1, beam_extended=BE, clean_beams=BEAMS,
           power={k: v.tolist() for k, v in P.items()},
           half_power_equiv_beam_arcsec={k: half(v / PS) for k, v in P.items()})
json.dump(res, open(os.path.join(OUT, "power_msclean_weighting.json"), "w"), indent=1)
for k, v in res["half_power_equiv_beam_arcsec"].items():
    print(f"  {k:28s} half of the joint power down to an equivalent beam of {v:.2f}\"")

# ======================================================================== figure (A&A column, 88 mm): as Fig. 5
FS, FT_, FL = 8, 7, 6
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": FS, "axes.labelsize": FS,
                     "xtick.labelsize": FT_, "ytick.labelsize": FT_, "xtick.direction": "in", "ytick.direction": "in",
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.fontsize": FL,
                     "lines.linewidth": 1.1})
FW, ML, MR, MB, MT, H1, H2, GAP = 88 / 25.4, 0.55, 0.08, 1.0, 0.4, 1.75, 1.35, 0.1
FH = MB + H2 + GAP + H1 + MT
fig = plt.figure(figsize=(FW, FH)); inch = lambda x, y, w, h: fig.add_axes([x / FW, y / FH, w / FW, h / FH])
W_ = FW - ML - MR; XL = (max(kc[1] * 0.9, 0.08), 1.15 / snr1); KT = [0.2, 0.3, 0.5, 1, 2, 3]; ST = [3, 2, 1, 0.6, 0.4, 0.2]
NF = matplotlib.ticker.NullFormatter()
LEG = dict(frameon=True, framealpha=0.95, edgecolor="0.6", fancybox=False, borderpad=0.4, labelspacing=0.3, handlelength=2.0)
CW = dict(zip(["natural", "robust 0.5", "robust 0", "robust -1", "robust -2"], plt.get_cmap("Oranges")(np.linspace(0.95, 0.4, 5))))


def deco(a, labels):
    a.set_xscale("log"); a.set_xlim(*XL); a.tick_params(which="both", top=False, right=True)
    a.set_xticks(KT); a.set_xticklabels([f"{t:g}" for t in KT] if labels else []); a.xaxis.set_minor_formatter(NF)
    a.axvline(QB / LB, color="0.35", lw=0.7, ls="--")
    sec = a.secondary_xaxis("top", functions=(lambda x: QB / np.maximum(x, 1e-9), lambda x: QB / np.maximum(x, 1e-9)))
    sec.set_xticks(ST); sec.set_xticklabels([f"{t:g}″" for t in ST] if not labels else []); sec.xaxis.set_minor_formatter(NF)
    sec.tick_params(which="both", direction="in", labelsize=FT_, pad=3)
    if not labels:
        sec.set_xlabel("equivalent beam FWHM $0.44/k$ (″)", labelpad=5)


def curves(a, ratio):
    y = (lambda v: v / PS) if ratio else (lambda v: v); s = kc < XL[1]
    if not ratio:
        a.plot(kc[s], PS[s], color="k", lw=1.3, label="joint, signal")
    a.plot(kc[s], y(P["2D-1D"])[s], color="#2a6fdb", lw=1.3, label="2D–1D model ⊗ ext. beam")
    for name in [q for q in BEAMS if q != "robust -2"]:                # robust -2: the same beam and curves as -1
        a.plot(kc[s], y(P[f"{name}, model"])[s], color=CW[name], lw=1.1)
        a.plot(kc[s], y(P[f"{name}, restored"])[s], color=CW[name], lw=0.9, ls="-.")
    a.plot(kc[s], y(P["dirty"])[s], color="0.4", ls=":", lw=1.1, label="compact dirty (natural)")


a = inch(ML, MB + H2 + GAP, W_, H1); deco(a, False); curves(a, False); a.set_yscale("log")
a.set_ylim(PS[kc < XL[1]].min() * 0.3, PS.max() * 30); a.set_ylabel(r"power (Jy$^2$)", labelpad=5)
a.text(0.03, 0.06, "(a) NGC 3110", transform=a.transAxes, ha="left", va="bottom", fontweight="bold")
a.text(QB / LB * 1.06, 0.965, r"$\lambda/B_\mathrm{max}$ beam", transform=a.get_xaxis_transform(), ha="left", va="top",
       fontsize=FL, color="0.3")
a = inch(ML, MB, W_, H2); deco(a, True); curves(a, True)
a.axhline(1, color="0.6", lw=0.6); a.axhline(0.5, color="0.75", lw=0.5, ls="--")
a.set_ylim(0, 2); a.set_ylabel("power / joint signal", labelpad=5); a.set_xlabel(r"spatial frequency $k$ (arcsec$^{-1}$)", labelpad=4)
hs = [(plt.Line2D([], [], color="#2a6fdb", lw=1.3), "2D–1D model ⊗ ext. beam")] + \
     [(plt.Line2D([], [], color=CW[q], lw=1.6), f"{q.replace('-1', '−1, −2')} ({BEAMS[q][0]:.2f}″×{BEAMS[q][1]:.2f}″)")
      for q in BEAMS if q != "robust -2"] + \
     [(plt.Line2D([], [], color="0.3", lw=1.1), "MS-CLEAN: model ⊗ ext. beam"),
      (plt.Line2D([], [], color="0.3", lw=0.9, ls="-."), "MS-CLEAN: restored"),
      (plt.Line2D([], [], color="0.4", lw=1.1, ls=":"), "compact dirty (natural)")]
fig.legend(*zip(*hs), loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, fontsize=FL - 0.5, columnspacing=0.6,
           title="MS-CLEAN weighting (clean beam)", title_fontsize=FL - 0.5, **dict(LEG, handlelength=1.6))
a.text(0.03, 0.93, "(b)", transform=a.transAxes, ha="left", va="top", fontweight="bold")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"power_msclean_weighting.{ext}"), dpi=300 if ext == "png" else None)
print("saved", os.path.join(OUT, "power_msclean_weighting.png"))
