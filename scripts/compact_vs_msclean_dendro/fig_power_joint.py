#!/usr/bin/env python
"""
Spatial-frequency fidelity of the compact 2D-1D model of NGC 3110 against an observed reference: the joint compact +
extended MS-CLEAN image at the extended clean beam (scripts/experiments/ladder_joint.py, b10), both in Jy/pixel on its
0.04" grid (central 16"), line channels, windowed by the compact-beam comparison mask widened by 1" and apodised with a
0.5" Gaussian (the same window for every cube).

  P(k)      azimuthally averaged power, mean over the line channels: 2D-1D model * extended beam (C), the joint image (J),
            its noise (the joint residual in the line-free channels, N; the joint signal power is P_J - P_N), and the
            compact MS-CLEAN restored image (K, its own clean beam: what the compact data give without deconvolution),
            and the compact dirty cube the 2D-1D model is deconvolved from (D, Jy per dirty beam converted with the
            compact clean beam's area, as K)
  coherence gamma(k) = Re<F_X F_J*> / sqrt(P_X (P_J - P_N)): the phase agreement with the joint signal (1: identical
            structure at that scale, up to amplitude; 0: unrelated)
  transfer  T(k) = Re<F_X F_J*> / (P_J - P_N): the fraction of the joint signal's Fourier amplitude recovered in phase

    python scripts/compact_vs_msclean_dendro/fig_power_joint.py
    -> results/NGC3110/figures/resolution_ladder_joint/power_joint.{png,pdf,json}   (A&A column width, 88 mm)
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import binary_dilation, gaussian_filter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
from restore import convolve_with_beam  # noqa: E402

G20 = os.environ.get("GRID") == "20kms"
TGT = os.environ.get("T", "NGC3110")            # other targets: the 20 km/s gallery products (results/<T>/g20), G20 only
G20 = G20 or TGT != "NGC3110"
R = os.path.join(REPO, "results", TGT); OUT = os.path.join(R, "figures", "resolution_ladder_joint" + ("20" if G20 else ""))
os.makedirs(OUT, exist_ok=True)
GC = json.load(open(os.path.join(R, "config.json")))
L = json.load(open(os.path.join(R, "line_channels.json")))["line"]; F = [c for c in range(GC["nchan"]) if c not in L and 1 < c < 30]
if G20:
    _g = json.load(open(os.path.join(R, "consistency_20kms_0.08/grid.json" if TGT == "NGC3110" else "g20/grid.json")))
    L, F = _g["line"], _g["line_free"]
LB = float(GC["lbmax_compact_arcsec"])
rd = lambda p: np.nan_to_num(fits.getdata(p)).astype(np.float64).squeeze()
def bm(p):                                      # the image's beam; per-channel beams: their median
    with fits.open(p) as h:
        if "BMAJ" in h[0].header:
            x = h[0].header; return (x["BMAJ"] * 3600, x["BMIN"] * 3600, x["BPA"])
        b = h["BEAMS"].data; return (float(np.median(b["BMAJ"])), float(np.median(b["BMIN"])), float(np.median(b["BPA"])))
om = lambda b: np.pi * b[0] * b[1] / (4 * np.log(2))
if G20:
    # the 64 x 20 km/s, 0.08" ladder: 2D-1D model of the pixel-scale run, joint cubes (0.04") averaged onto its grid
    # (aligned: 0.08" pixel j = 0.04" pixels 2j-1 (half), 2j, 2j+1 (half)), the compact MS-CLEAN on the same grid
    from scipy.ndimage import convolve1d as _c1
    CELL, n = 0.08, 640
    def crop(a):
        k = np.array([0.5, 1.0, 0.5]); return (_c1(_c1(a, k, axis=-1, mode="constant"), k, axis=-2, mode="constant")[..., ::2, ::2] / 4.0
                                               if a.shape[-1] == 2 * n else a)
    if TGT == "NGC3110":
        pj = os.path.join(R, "ladder_joint_20kms/b10/clean_finished"); pk = os.path.join(R, "consistency_20kms_0.08/clean_compact_finished")
        xc = np.load(os.path.join(R, "pixel_scale/model_0.08_support.npy")).astype(np.float64)
    else:
        pj = os.path.join(R, "ladder_joint_20kms/b10/clean_finished")
        if not os.path.exists(os.path.join(pj, "image.fits")):
            pj = os.path.join(R, "g20/joint_ext")
        pk = os.path.join(R, "g20/clean_compact_finished")
        xc = np.load(os.path.join(R, "g20/model_0.08_support.npy")).astype(np.float64)
else:
    CELL, n = 0.04, 400
    crop = lambda a: a[..., (a.shape[-1] - n) // 2:(a.shape[-1] + n) // 2, (a.shape[-1] - n) // 2:(a.shape[-1] + n) // 2]
    pj = os.path.join(R, "ladder_joint/b10/clean_finished"); pk = os.path.join(REPO, "data/ngc3110/compact/big_finished")
    xc = np.load(os.path.join(R, "compact/deconvolved/model.npy")).astype(np.float64)
    xc = xc.reshape(xc.shape[0], n, 2, n, 2).sum((2, 4))                # 0.02" -> 0.04" (Jy/pixel, block sum)
BE, BK = bm(os.path.join(pj, "image.fits")), bm(os.path.join(pk, "image.fits"))
px = lambda b: om(b) / CELL ** 2
CUBES = dict(C=convolve_with_beam(xc, CELL, *BE) / px(BE), J=crop(rd(os.path.join(pj, "image.fits"))) / px(BE),
             N=crop(rd(os.path.join(pj, "residual.fits"))) / px(BE), K=crop(rd(os.path.join(pk, "image.fits"))) / px(BK))
PD = (os.path.join(R, "deconv_inputs/compact_0.08/dirty_cube.fits") if TGT == "NGC3110" else
      os.path.join(R, "g20/deconv_inputs/dirty_cube.fits")) if G20 else None
if PD and os.path.exists(PD):                                            # the compact dirty cube
    CUBES["D"] = crop(rd(PD)) / px(BK)
if os.path.exists(os.path.join(OUT, "masks.npz")):                       # the ladder's compact-beam mask footprint
    mk = np.load(os.path.join(OUT, "masks.npz"))["b00"].any(0)
else:                                                                    # else: the compact MS-CLEAN's final mask
    mk = np.load(os.path.join(pk, "mask_final.npy")).astype(bool)[L].any(0)
W = gaussian_filter(binary_dilation(mk, iterations=int(round(1.0 / CELL))).astype(float), 0.5 / CELL / 2.355)
f = np.fft.fftfreq(n, d=CELL); kk = np.hypot(*np.meshgrid(f, f))
edges = np.arange(0, 1 / (2 * CELL), 2 / (n * CELL)); kc = 0.5 * (edges[1:] + edges[:-1])
idx = np.digitize(kk.ravel(), edges) - 1; ok = (idx >= 0) & (idx < len(kc)); cnt = np.bincount(idx[ok], None, len(kc))
ring = lambda P: np.bincount(idx[ok], P.ravel()[ok], len(kc)) / np.maximum(cnt, 1)
FT = {k: np.array([np.fft.fft2(CUBES[k][c] * W) for c in (F if k == "N" else L)]) for k in CUBES}
P = {k: ring(np.mean(np.abs(v) ** 2, 0)) for k, v in FT.items()}
X = lambda a: ring(np.mean((FT[a] * np.conj(FT["J"])).real, 0))
PS = np.clip(P["J"] - P["N"], 1e-40, None)                              # the joint signal power
coh = {a: X(a) / np.sqrt(P[a] * PS) for a in ("C", "K")}
T = {a: X(a) / PS for a in ("C", "K")}
first = lambda a, th: float(kc[np.nonzero((a < th) & (kc > 0.05))[0][0]])
# the independent check: the extended configuration alone (no visibility in common with the deconvolution), restored
# at its own beam; the 2D-1D model convolved with that beam
PE = os.path.join(R, "extended_natural_20kms" if G20 else "extended/msclean_finished")
EXT = None
if G20 and os.path.exists(os.path.join(PE, "image.fits")):
    BX = bm(os.path.join(PE, "image.fits")); pxx = om(BX) / CELL ** 2
    _E = crop(rd(os.path.join(PE, "image.fits"))) / pxx; _Er = crop(rd(os.path.join(PE, "residual.fits"))) / pxx
    _C = convolve_with_beam(xc, CELL, *BX) / pxx
    FE = np.array([np.fft.fft2(_E[c] * W) for c in L]); FEr = np.array([np.fft.fft2(_Er[c] * W) for c in F])
    FCx = np.array([np.fft.fft2(_C[c] * W) for c in L])
    PEs = np.clip(ring(np.mean(np.abs(FE) ** 2, 0)) - ring(np.mean(np.abs(FEr) ** 2, 0)), 1e-40, None)
    EXT = ring(np.mean((FCx * np.conj(FE)).real, 0)) / np.sqrt(ring(np.mean(np.abs(FCx) ** 2, 0)) * PEs)
    print(f"vs extended only ({BX[0]:.3f} x {BX[1]:.3f}): coherence < 0.5 at {1 / first(EXT, 0.5):.3f}\"")
res = dict(beam_extended=BE, beam_compact_msclean=BK, lbmax_compact_arcsec=LB, k=kc.tolist(),
           **{f"P_{k}": v.tolist() for k, v in P.items()}, P_J_signal=PS.tolist(),
           **{f"coh_{a}": v.tolist() for a, v in coh.items()}, **{f"T_{a}": v.tolist() for a, v in T.items()},
           thresholds={f"{q}_{a}_{th}": 1 / first(d[a], th) for q, d in (("coh", coh), ("T", T)) for a in ("C", "K")
                       for th in (0.9, 0.5)},
           joint_snr1_arcsec=1 / float(kc[np.nonzero((PS / P["N"] < 1) & (kc > 0.05))[0][0]]),
           joint_signal_power_below_bmax=float((PS * cnt)[kc < 1 / LB].sum() / (PS * cnt).sum()))
if EXT is not None:
    res.update(coh_C_extended_only=EXT.tolist(), coh_ext_05_arcsec=1 / first(EXT, 0.5), coh_ext_09_arcsec=1 / first(EXT, 0.9))
json.dump(res, open(os.path.join(OUT, "power_joint.json"), "w"), indent=1)
print({k: round(v, 3) for k, v in res["thresholds"].items()}, "joint S/N = 1 at", round(res["joint_snr1_arcsec"], 3),
      "power below B/lam", round(res["joint_signal_power_below_bmax"], 4))

# ======================================================================== figure (A&A column, 88 mm)
FS, FT_, FL = 8, 7, 6.5
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": FS, "axes.labelsize": FS,
                     "xtick.labelsize": FT_, "ytick.labelsize": FT_, "xtick.direction": "in", "ytick.direction": "in",
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.5,
                     "ytick.minor.width": 0.5, "xtick.major.pad": 3, "ytick.major.pad": 3, "legend.fontsize": FL,
                     "lines.linewidth": 1.1})
FW, ML, MR, MB, MT, H1, H2, GAP = 88 / 25.4, 0.55, 0.08, 0.42, 0.4, 1.75, 1.15, 0.1
FH = MB + H2 + GAP + H1 + MT
fig = plt.figure(figsize=(FW, FH))
inch = lambda x, y, w, h: fig.add_axes([x / FW, y / FH, w / FW, h / FH])
W_ = FW - ML - MR; XL = (max(kc[1] * 0.9, 0.08), 1.15 / res["joint_snr1_arcsec"])          # up to where the joint S/N reaches 1
C2D, CJ, CK, CN, CD = "#2a6fdb", "k", "#e07b00", "0.55", "0.4"
LEG = dict(frameon=True, framealpha=0.95, edgecolor="0.6", fancybox=False, borderpad=0.4, labelspacing=0.3, handlelength=2.0)


NF = matplotlib.ticker.NullFormatter()
KT = [0.2, 0.3, 0.5, 1, 2, 3]                                           # k ticks; scale ticks on the top axis
ST = [3, 2, 1, 0.6, 0.4, 0.2]                                        # equivalent beam FWHM ticks (arcsec)
QB = 2 * np.log(2) / np.pi                                               # 0.44: a Gaussian of FWHM QB / k has G(k) = 1/2


def topaxis(a, label):
    sec = a.secondary_xaxis("top", functions=(lambda x: QB / np.maximum(x, 1e-9), lambda x: QB / np.maximum(x, 1e-9)))
    sec.set_xticks(ST); sec.set_xticklabels([f"{t:g}″" for t in ST]); sec.xaxis.set_minor_formatter(NF)
    sec.tick_params(which="both", direction="in", labelsize=FT_, pad=3)
    if label:
        sec.set_xlabel("equivalent beam FWHM $0.44/k$ (″)", labelpad=5)
    else:
        sec.tick_params(labeltop=False)


def deco(a, labels):
    a.set_xscale("log"); a.set_xlim(*XL); a.tick_params(which="both", top=False, right=True)
    a.set_xticks(KT); a.set_xticklabels([f"{t:g}" for t in KT]); a.xaxis.set_minor_formatter(NF)
    a.axvline(QB / LB, color="0.35", lw=0.7, ls="--")                     # the lambda/B_max beam (FWHM 0.60")
    if not labels:
        a.tick_params(which="both", labelbottom=False)


a = inch(ML, MB + H2 + GAP, W_, H1); deco(a, False)
sel = kc < XL[1]
a.loglog(kc[sel], PS[sel], color=CJ, lw=1.3, label="joint (compact + extended), signal")
a.loglog(kc, P["C"], color=C2D, label="compact 2D–1D deconvolved")
a.loglog(kc, P["K"], color=CK, ls="-.", label="compact MS-CLEAN (restored)")
if "D" in P:
    a.loglog(kc, P["D"], color=CD, ls=":", lw=1.2, label="compact dirty")
a.set_ylim(PS[kc < XL[1]].min() * 0.3, PS.max() * 30); a.set_ylabel(r"power (Jy$^2$)", labelpad=5)
a.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), **LEG)
a.text(QB / LB * 1.06, 0.965, r"$\lambda/B_\mathrm{max}$ beam", transform=a.get_xaxis_transform(), ha="left",
       va="top", fontsize=FL, color="0.3")                              # right of the dashed line, above the legend
a.text(0.03, 0.06, "(a)" if TGT == "NGC3110" else f"(a) {GC['label']}", transform=a.transAxes, ha="left", va="bottom",
       fontweight="bold")
topaxis(a, True)
a = inch(ML, MB, W_, H2); deco(a, True); topaxis(a, False); a.set_xlabel(r"spatial frequency $k$ (arcsec$^{-1}$)", labelpad=4)
a.axhline(1, color="0.6", lw=0.6, ls="-"); a.axhline(0.5, color="0.75", lw=0.5, ls="--")
sel = kc < XL[1]
kc_ = kc[sel]
a.semilogx(kc_, (P["C"] / PS)[sel], color=C2D, label="compact 2D–1D deconvolved")   # fraction of the observed power recovered
a.semilogx(kc_, (P["K"] / PS)[sel], color=CK, ls="-.", label="compact MS-CLEAN")
if "D" in P:
    a.semilogx(kc_, (P["D"] / PS)[sel], color=CD, ls=":", lw=1.2, label="compact dirty")
# the compact data's natural-weight uv sampling density (|FT| of the compact PSF, line channels), normalised on 3-6"
a.set_ylim(0, 2); a.set_ylabel("power / joint signal", labelpad=5)
a.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), **dict(LEG, handlelength=1.6))
a.text(0.97, 0.94, "(b)", transform=a.transAxes, ha="right", va="top", fontweight="bold")
for ax_, lab_ in zip(fig.axes[:0] or [x for x in fig.axes if x.get_xscale() == "log" and not hasattr(x, "_functions")],
                    (False, True)):
    ax_.set_xlim(*XL); ax_.set_xticks(KT); ax_.xaxis.set_minor_formatter(NF)  # loglog/semilogx reset the formatters
    ax_.set_xticklabels([f"{t:g}" for t in KT] if lab_ else [])
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"power_joint.{ext}"), dpi=300 if ext == "png" else None)
print("saved", os.path.join(OUT, "power_joint.png"), f"({FW * 25.4:.0f} x {FH * 25.4:.0f} mm)")
