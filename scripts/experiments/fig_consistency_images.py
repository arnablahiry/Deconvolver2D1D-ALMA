#!/usr/bin/env python
"""
Figures 2 and 3 of the compact-vs-extended consistency set (NGC 3110), image plane, line only (every cube minus its
line-free mean, line channels, moment 0 x 25 km/s).

Figure 2, why an image made from the extended data alone falls short at the compact beam:
  (a) azimuthally averaged dirty beams: compact, extended (natural), extended tapered to the compact beam
  (b) their integral within radius r, in units of the compact clean beam's area (a beam that integrates to < 1 loses
      the flux of anything that size)
  (c-e) one sky, the compact MS-CLEAN model (51" field): * the compact clean beam (what a perfect image at that beam
      shows), * the compact dirty beam, * the tapered extended dirty beam (what the extended data contain of it);
      the flux in the central 1.5" and in the whole emission region printed
Figure 3, the same data, different images (compact-beam resolution):
  top     compact MS-CLEAN vs the extended MS-CLEAN with the compact-beam taper (both restored, the tapered one brought
          to the compact run's beam): (compact - extended) / extended
  bottom  the tapered extended dirty cube (the data) vs the compact MS-CLEAN model * the tapered extended dirty beam
          (what the extended array would measure if the compact image were the sky): (prediction - data) / data
  flux and RMSE over the tapered extended run's astrodendro mask; RMSE noise alone would give, from the line-free
  channels

Inputs: data/ngc3110/{compact,extended}/dirty_big (82", 0.04"), extended/dirty_big_taper, compact/big_finished,
extended/big_taper_finished (51", 0.04").

    python scripts/experiments/fig_consistency_images.py
    -> results/NGC3110/figures/consistency/{fig2_beams,fig3_clean_vs_data}.{png,pdf}, images.json
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
from numpy.fft import rfft2, irfft2

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in ("simple", "scripts/compact_vs_msclean_dendro"):
    sys.path.insert(0, os.path.join(REPO, _p))
from restore import convolve_with_beam  # noqa: E402
import beams  # noqa: E402

D_ = os.path.join(REPO, "data/ngc3110"); RES = os.path.join(REPO, "results/NGC3110")
OUT = os.path.join(RES, "figures", "consistency"); os.makedirs(OUT, exist_ok=True)
L = json.load(open(os.path.join(RES, "line_channels.json")))["line"]
F = [c for c in range(32) if c not in L and 1 < c < 30]
DV = 25.0
COL = {"compact": "#0b3c6e", "extended": "#3f8fd8", "tapered": "#e07b00"}
TICK, LAB = 12, 14
plt.rcParams.update({"font.family": "serif", "font.size": 13, "xtick.direction": "in", "ytick.direction": "in"})
rd = lambda p: np.nan_to_num(fits.getdata(p)).astype(np.float64).squeeze()
bm = lambda p: (lambda h: (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"]))(fits.getheader(p))
mad = lambda a: 1.4826 * np.median(np.abs(a - np.median(a)))
line = lambda cube: cube[L] - cube[F].mean(0)                                   # line only, line channels
CELL = 0.04
BC = bm(f"{D_}/compact/big_finished/image.fits")
bpx = beams.omega(BC) / CELL ** 2
N = 2048; n = 1280; o = (N - n) // 2


def embed(x):
    """(..., 1280, 1280) on the 51" grid -> (..., 2048, 2048) on the 82" grid (same centre)"""
    out = np.zeros(x.shape[:-2] + (N, N), dtype=x.dtype); out[..., o:o + n, o:o + n] = x
    return out


def centred(P):
    pk = np.unravel_index(np.argmax(P), P.shape)
    return np.roll(P, (-pk[0], -pk[1]), axis=(0, 1))


def conv_psf(x, P):
    """x (nchan, N, N) * PSF (nchan, N, N), circular, PSF peak at (0, 0)"""
    return np.stack([irfft2(rfft2(a) * rfft2(centred(p)), s=a.shape) for a, p in zip(x, P)])


Pc = rd(f"{D_}/compact/dirty_big/psf.fits")[L]
Pe = rd(f"{D_}/extended/dirty_big/psf.fits")[L]
Pt = rd(f"{D_}/extended/dirty_big_taper/psf.fits")[L]
Dt = line(rd(f"{D_}/extended/dirty_big_taper/dirty.fits"))
Dt_free = rd(f"{D_}/extended/dirty_big_taper/dirty.fits")[F]
Mc = embed(line(rd(f"{D_}/compact/big_finished/model.fits")))                  # Jy/pixel, line only
yy, xx = np.indices((N, N)); RAD = np.hypot(yy - N // 2, xx - N // 2) * CELL
cen = RAD < 1.5
info = {}

# ======================================================================== figure 2
rb = np.arange(0, 12.01, 0.04); rm = 0.5 * (rb[1:] + rb[:-1])
prof, cum = {}, {}
for k, P in (("compact", Pc), ("extended", Pe), ("tapered", Pt)):
    p = P[len(L) // 2]; pk = np.unravel_index(np.argmax(p), p.shape)
    r = np.hypot(yy - pk[0], xx - pk[1]) * CELL
    idx = np.digitize(r.ravel(), rb) - 1; ok = (idx >= 0) & (idx < len(rm))
    prof[k] = np.bincount(idx[ok], p.ravel()[ok], len(rm)) / np.maximum(np.bincount(idx[ok], minlength=len(rm)), 1)
    cum[k] = np.cumsum(np.bincount(idx[ok], p.ravel()[ok], len(rm))) / bpx
info["psf_integral_clean_beam_areas"] = {k: {f"r<{R}": float(np.interp(R, rb[1:], cum[k])) for R in (1, 2, 3, 5, 10)} for k in cum}
truth = convolve_with_beam(Mc, CELL, *BC)                                        # Jy/beam
seen = dict(truth=truth, compact=conv_psf(Mc, Pc), tapered=conv_psf(Mc, Pt))
Mmask = embed(np.load(f"{D_}/extended/big_taper_finished/mask_final.npy").astype(bool)[L])
Rg = (truth > 0.02 * truth.max()) | Mmask                                        # the emission region
fl = lambda X, sel: float((X * sel).sum() / bpx * DV)
info["sky_through_beams"] = {k: dict(central=fl(v, cen[None]), region=fl(v, Rg)) for k, v in seen.items()}
for k, v in info["sky_through_beams"].items():
    print(f"fig2 {k:8s}: central 1.5\" {v['central']:7.1f} Jy km/s ({100 * (v['central'] / info['sky_through_beams']['truth']['central'] - 1):+.0f}%), "
          f"region {v['region']:7.1f} ({100 * (v['region'] / info['sky_through_beams']['truth']['region'] - 1):+.0f}%)", flush=True)

H = 8.0; hp = int(H / CELL); sl = np.s_[N // 2 - hp:N // 2 + hp, N // 2 - hp:N // 2 + hp]; EXT = [H, -H, -H, H]
fig = plt.figure(figsize=(17, 9.6))
gs = fig.add_gridspec(2, 6, hspace=0.32, wspace=0.6)
a = fig.add_subplot(gs[0, 0:3])
for k, lab in (("compact", "compact"), ("extended", "extended (natural)"), ("tapered", "extended, tapered to the compact beam")):
    a.plot(rm, prof[k], color=COL[k], lw=2, label=lab)
a.axhline(0, color="0.5", lw=0.8); a.set_xlim(0, 8); a.set_ylim(-0.2, 1.02)
a.set_xlabel("radius (″)", fontsize=LAB); a.set_ylabel("dirty beam (azimuthal mean)", fontsize=LAB); a.tick_params(labelsize=TICK)
a.legend(fontsize=11, frameon=False); a.set_title("(a) the dirty beams", fontsize=LAB)
a = fig.add_subplot(gs[0, 3:6])
for k in ("compact", "extended", "tapered"):
    a.plot(rb[1:], cum[k], color=COL[k], lw=2)
a.axhline(1, color="0.5", lw=0.8, ls="--"); a.text(7.9, 1.03, "= the compact clean beam's area", ha="right", va="bottom", fontsize=10, color="0.4")
a.set_xlim(0, 8); a.set_ylim(-0.1, 1.3)
a.set_xlabel("radius (″)", fontsize=LAB); a.set_ylabel("beam integral within r (compact clean-beam areas)", fontsize=LAB)
a.tick_params(labelsize=TICK)
ct = info["psf_integral_clean_beam_areas"]
a.set_title(f"(b) integral within 3″: compact {ct['compact']['r<3']:.2f}, tapered extended {ct['tapered']['r<3']:.2f}", fontsize=LAB - 1)
m0 = {k: (v.sum(0) * DV)[sl] for k, v in seen.items()}
vmax = np.percentile(m0["truth"], 99.9)
tr = info["sky_through_beams"]["truth"]
for j, (k, lab) in enumerate((("truth", "sky model ∗ clean beam"),
                               ("compact", "∗ compact dirty beam"), ("tapered", "∗ tapered extended dirty beam"))):
    a = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
    im = a.imshow(m0[k], origin="lower", extent=EXT, cmap="inferno", norm=Normalize(-0.25 * vmax, vmax), interpolation="nearest")
    a.set_xticks([]); a.set_yticks([])
    v = info["sky_through_beams"][k]
    a.text(0.04, 0.96, lab, transform=a.transAxes, color="w", ha="left", va="top", fontsize=11, fontweight="bold")
    a.text(0.96, 0.04, f"centre {v['central']:.0f} ({100 * (v['central'] / tr['central'] - 1):+.0f}%)\nregion {v['region']:.0f} "
                       f"({100 * (v['region'] / tr['region'] - 1):+.0f}%) Jy km s$^{{-1}}$", transform=a.transAxes, color="w",
           ha="right", va="bottom", fontsize=10)
    a.contour(cen[sl].astype(float), levels=[0.5], colors="w", linewidths=0.6, extent=EXT, origin="lower")
    if j == 0:
        a.set_title("(c)", fontsize=LAB, loc="left")
cb = fig.colorbar(im, ax=fig.axes[-3:], orientation="horizontal", fraction=0.05, pad=0.04, aspect=60)
cb.set_label("Jy beam$^{-1}$ km s$^{-1}$ (clean or dirty beam)", fontsize=LAB - 1)
fig.savefig(os.path.join(OUT, "fig2_beams.png"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig2_beams.pdf"), dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved fig2", flush=True)

# ======================================================================== figure 3
Ic = embed(line(rd(f"{D_}/compact/big_finished/image.fits")))
BT = bm(f"{D_}/extended/big_taper_finished/image.fits")
It = line(rd(f"{D_}/extended/big_taper_finished/image.fits"))
It = embed(beams.to_beam(It, BT, BC, CELL) if not np.allclose(BT, BC, rtol=0.005) else It)
pred = conv_psf(Mc, Pt)                                                          # compact model through the extended dirty beam
sig_d = np.median([mad(x[RAD > 30]) for x in Dt_free - Dt_free.mean(0)])
_free = rd(f"{D_}/compact/big_finished/image.fits")[F]
sig_i = np.median([mad(x[(RAD > 18) & (RAD < 24)]) for x in embed(_free - _free.mean(0))])


def stat(a, b, sel, sig):
    x, y = a[sel], b[sel]
    return dict(flux_a=float(x.sum() / bpx * DV), flux_b=float(y.sum() / bpx * DV), flux_pct=float(100 * (x.sum() - y.sum()) / y.sum()),
                rmse_pct=float(100 * np.linalg.norm(x - y) / np.linalg.norm(y)),
                noise_rmse_pct=float(100 * sig * np.sqrt(2 * x.size) / np.linalg.norm(y)))


info["clean_vs_clean"] = stat(Ic, It, Mmask, sig_i)
info["forward_vs_data"] = stat(pred, Dt, Mmask, sig_d)
for k in ("clean_vs_clean", "forward_vs_data"):
    m = info[k]
    print(f"fig3 {k:16s}: {m['flux_a']:.1f} vs {m['flux_b']:.1f} -> flux {m['flux_pct']:+.1f}%, RMSE {m['rmse_pct']:.1f}% "
          f"(noise alone {m['noise_rmse_pct']:.1f}%)", flush=True)
json.dump(info, open(os.path.join(OUT, "images.json"), "w"), indent=1)

foot = Mmask.any(0)[sl]
CMP = plt.get_cmap("RdBu_r").copy(); CMP.set_bad("0.9")


def pct(a, b):
    am, bm_ = (a * Mmask).sum(0)[sl], (b * Mmask).sum(0)[sl]
    return np.where(foot & (bm_ > 0.05 * bm_.max()), 100 * (am - bm_) / np.where(bm_ > 0, bm_, 1), np.nan)


rows = [((Ic.sum(0) * DV)[sl], "Compact MS-CLEAN (restored)", (It.sum(0) * DV)[sl], "Extended MS-CLEAN, tapered\nto the compact beam (restored)",
         pct(Ic, It), "(compact − extended) / extended", info["clean_vs_clean"], "Jy beam$^{-1}$ km s$^{-1}$"),
        ((pred.sum(0) * DV)[sl], "Compact MS-CLEAN model\n∗ tapered extended dirty beam", (Dt.sum(0) * DV)[sl],
         "Extended data, tapered:\ndirty image", pct(pred, Dt), "(prediction − data) / data", info["forward_vs_data"],
         "Jy (dirty beam)$^{-1}$ km s$^{-1}$")]
fig, ax = plt.subplots(2, 3, figsize=(15.5, 10.8), gridspec_kw=dict(wspace=0.05, hspace=0.22))
for r_, (A, la, B, lb, P_, lp, st, unit) in enumerate(rows):
    vmax = np.percentile(np.r_[A.ravel(), B.ravel()], 99.9)
    for j, (img, lab) in enumerate(((A, la), (B, lb))):
        a = ax[r_, j]
        im = a.imshow(img, origin="lower", extent=EXT, cmap="inferno", norm=Normalize(-0.25 * vmax, vmax), interpolation="nearest")
        a.contour(foot.astype(float), levels=[0.5], colors="w", linewidths=0.5, extent=EXT, origin="lower")
        a.set_xticks([]); a.set_yticks([])
        a.text(0.04, 0.96, lab, transform=a.transAxes, color="w", ha="left", va="top", fontsize=12, fontweight="bold")
        pad = 0.06 * H
        a.add_patch(Ellipse((H - pad - BC[0] / 2, -H + pad + BC[0] / 2), width=BC[0], height=BC[1], angle=90 - BC[2],
                            facecolor="none" if r_ else "w", edgecolor="w", lw=1.0))
        a.text(0.96, 0.04, f"{st['flux_a' if j == 0 else 'flux_b']:.0f} Jy km s$^{{-1}}$ in the mask", transform=a.transAxes,
               color="w", ha="right", va="bottom", fontsize=11)
    cb = fig.colorbar(im, ax=list(ax[r_, :2]), orientation="horizontal", fraction=0.05, pad=0.02, aspect=50)
    cb.set_label(unit, fontsize=LAB - 1)
    a = ax[r_, 2]
    im = a.imshow(P_, origin="lower", extent=EXT, cmap=CMP, norm=Normalize(-50, 50), interpolation="nearest")
    a.contour(foot.astype(float), levels=[0.5], colors="k", linewidths=0.5, extent=EXT, origin="lower")
    a.set_xticks([]); a.set_yticks([])
    a.text(0.04, 0.96, lp, transform=a.transAxes, color="k", ha="left", va="top", fontsize=12, fontweight="bold")
    a.text(0.96, 0.04, f"flux {st['flux_pct']:+.1f}%, RMSE {st['rmse_pct']:.1f}%\n(noise alone {st['noise_rmse_pct']:.1f}%)",
           transform=a.transAxes, color="k", ha="right", va="bottom", fontsize=11)
    cb = fig.colorbar(im, ax=a, orientation="horizontal", fraction=0.05, pad=0.02); cb.set_label("%", fontsize=LAB - 1)
ax[0, 0].set_title("images made from each array (CLEAN)", fontsize=LAB, loc="left")
ax[1, 0].set_title("the data themselves, compared through the extended array's dirty beam", fontsize=LAB, loc="left")
fig.savefig(os.path.join(OUT, "fig3_clean_vs_data.png"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig3_clean_vs_data.pdf"), dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved fig3", flush=True)
