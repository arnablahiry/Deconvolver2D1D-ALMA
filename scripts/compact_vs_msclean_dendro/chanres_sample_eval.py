#!/usr/bin/env python
"""
The channel-width test over the sample: every target deconvolved on 25, 20, 15, 10, 5 and 2.6 km/s channels (those its
native channels and line width allow; results/<T>/g<DV>_c0.16) with 2D-1D and with the 2D starlet per channel, against
its joint compact + extended images on the same channels at lambda/B_max (b05) and the next ladder step (b07)
(ladder_joint_<DV>kms; 20 km/s: the main ladder, ladder_joint_20kms). Metrics as chanres_eval.py (region: the 20 km/s
ladder mask footprint, line channels): cube, moment-0 RMSE (%), moment-1 and -2 RMSE (km/s).

  figure (A&A text width, square panels): against the channel width, at lambda/B_max (top) and the next ladder step
  (bottom): the mean over the targets that have that width (error bars: standard error; n per width on the top axis),
  and the per-target ratio per channel / 2D-1D, averaged (third row)

    python scripts/compact_vs_msclean_dendro/chanres_sample_eval.py
    -> results/gallery/chanres_sample_eval.json, results/gallery/why2d1d_channels_sample.{png,pdf}
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import convolve1d

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
from restore import convolve_with_beam  # noqa: E402

RES = os.path.join(REPO, "results"); OUTD = os.path.join(RES, "gallery"); CELL = 0.16
WIDTHS = [25, 20, 15, 10, 5, 2.6]; IB = (5, 7)
ALL = ["NGC3110", "IC5179", "MCG-02-33-098", "IC4518W", "ESO297-G011", "NGC3256", "NGC7130", "IC4518E", "IC4687", "IRASF17138-1017",
       "NGC1614", "NGC2369", "NGC5135"]
rd = lambda p: np.nan_to_num(fits.getdata(p)).astype(np.float64).squeeze()


def to016(x):
    k = np.array([0.5, 1, 1, 1, 0.5]) / 4
    return convolve1d(convolve1d(x, k, axis=-1, mode="constant"), k, axis=-2, mode="constant")[..., ::4, ::4]


def moments(cube, foot, thr, v):
    w = np.where(cube >= thr[:, None, None], cube, 0.0); s = w.sum(0); ok = foot & (s > 0)
    m1 = np.where(ok, (w * v[:, None, None]).sum(0) / np.where(ok, s, 1), np.nan)
    m2 = np.where(ok, np.sqrt(np.clip((w * (v[:, None, None] - np.nan_to_num(m1)) ** 2).sum(0) / np.where(ok, s, 1), 0, None)), np.nan)
    return m1, m2, s


out = {}
for t in ALL:
    fd = os.path.join(RES, t, "figures/resolution_ladder_joint20")
    LAD = json.load(open(os.path.join(fd, "resolution_ladder.json"))); MK = np.load(os.path.join(fd, "masks.npz"))
    for dv in WIDTHS:
        gd = os.path.join(RES, t, f"g{dv:g}_c0.16")
        mods = {n: os.path.join(gd, f"model_0.16_support{s}.npy") for n, s in (("2D-1D", ""), ("2D per channel", "_L0"))}
        if not all(os.path.exists(p) for p in mods.values()):
            continue
        G = json.load(open(os.path.join(gd, "grid.json"))); L, F = G["line"], G["line_free"]
        v = (G["start_kms"] + dv * np.arange(G["nchan"]))[L]; X = {n: np.load(p).astype(np.float64)[L] for n, p in mods.items()}
        for ib in IB:
            jd = os.path.join(RES, t, ("ladder_joint_20kms" if dv == 20 else f"ladder_joint_{dv:g}kms") + f"/b{ib:02d}/clean_finished")
            if not os.path.exists(os.path.join(jd, "image.fits")):
                continue
            b = tuple(LAD["rows"][ib][k] for k in ("bmaj", "bmin", "bpa"))
            I = to016(rd(os.path.join(jd, "image.fits"))[L]); res = to016(rd(os.path.join(jd, "residual.fits")))
            foot = MK[f"b{ib:02d}"].any(0)[::2, ::2]; M = np.broadcast_to(foot, I.shape)
            sig = 1.4826 * np.median(np.abs(res - np.median(res, axis=(1, 2), keepdims=True)), axis=(1, 2)); thr = 3 * sig[L]
            j1, j2, j0 = moments(I, foot, thr, v); row = dict(snr=float(I.max() / np.median(sig[[c for c in F if sig[c] > 0]])))
            for n, x in X.items():
                C = convolve_with_beam(x, CELL, *b); c, i = C[M], I[M]; c1, c2, c0 = moments(C, foot, thr, v)
                ok = np.isfinite(c1) & np.isfinite(j1)
                row[n] = dict(rmse=float(100 * np.linalg.norm(c - i) / np.linalg.norm(i)), mom0=float(100 * np.linalg.norm((c0 - j0)[ok]) / np.linalg.norm(j0[ok])),
                              mom1=float(np.sqrt(np.mean((c1 - j1)[ok] ** 2))), mom2=float(np.sqrt(np.mean((c2 - j2)[ok] ** 2))))
            out.setdefault(t, {}).setdefault(f"{dv:g}", {})[f"b{ib:02d}"] = row
            print(f"{t:16s} {dv:4g} km/s b{ib:02d}: 2D-1D / per channel  RMSE {row['2D-1D']['rmse']:.1f}/{row['2D per channel']['rmse']:.1f}%  "
                  f"mom0 {row['2D-1D']['mom0']:.1f}/{row['2D per channel']['mom0']:.1f}%  mom1 {row['2D-1D']['mom1']:.1f}/{row['2D per channel']['mom1']:.1f}", flush=True)
json.dump(out, open(os.path.join(OUTD, "chanres_sample_eval.json"), "w"), indent=1)

# ======================================================================== figure
FS, FT, FL = 8, 7, 6.5
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": FS, "xtick.labelsize": FT,
                     "ytick.labelsize": FT, "xtick.direction": "in", "ytick.direction": "in", "axes.linewidth": 0.6,
                     "legend.fontsize": FL, "lines.linewidth": 1.2, "lines.markersize": 3.2})
MET = [("rmse", "cube RMSE (%)"), ("mom0", "moment-0 RMSE (%)"), ("mom1", r"moment-1 RMSE (km s$^{-1}$)"), ("mom2", r"moment-2 RMSE (km s$^{-1}$)")]
ROWS = [("b05", r"$\lambda/B_\mathrm{max}$", False), ("b07", "next step", False), ("b05", r"ratio at $\lambda/B_\mathrm{max}$", True)]
W = [w for w in WIDTHS if any(f"{w:g}" in out.get(t, {}) for t in out)]; xs = np.arange(len(W))
fig, AX = plt.subplots(3, 4, figsize=(184 / 25.4, 5.6), gridspec_kw=dict(wspace=0.5, hspace=0.3, left=0.08, right=0.985, bottom=0.07, top=0.92))


def agg(bk, key, n, ratio):
    mu, se, cnt = [], [], []
    for w in W:
        vals = [d[f"{w:g}"][bk][n][key] / (d[f"{w:g}"][bk]["2D-1D"][key] if ratio else 1) for d in out.values()
                if bk in d.get(f"{w:g}", {})]
        vals = np.array(vals); cnt.append(vals.size)
        mu.append(vals.mean() if vals.size else np.nan); se.append(vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else np.nan)
    return np.array(mu), np.array(se), cnt


for r, (bk, rl, ratio) in enumerate(ROWS):
    for c, (key, lab) in enumerate(MET):
        a = AX[r, c]
        for n, col, ls, mk, mf, ll in (("2D-1D", "#2a6fdb", "-", "o", None, "2D–1D"), ("2D per channel", "#e07b00", "--", "s", "w", "2D starlet per channel")):
            if ratio and n == "2D-1D":
                continue
            mu, se, cnt = agg(bk, key, n, ratio)
            a.errorbar(xs, mu, yerr=se, color=col, ls=ls, marker=mk, mfc=mf or col, capsize=2, elinewidth=0.7, label=ll)
        a.set_box_aspect(1); a.set_xticks(xs); a.set_xticklabels([f"{w:g}" for w in W] if r == len(ROWS) - 1 else [])
        a.tick_params(which="both", top=True, right=True, pad=2)
        if ratio:
            a.axhline(1, color="0.5", lw=0.6); a.set_xlabel(r"channel width (km s$^{-1}$)", labelpad=2)
        else:
            a.set_ylim(bottom=0)
        if r == 0:
            a.set_title(lab, fontsize=FS - 0.5, pad=11)
            sec = a.secondary_xaxis("top"); sec.set_xticks(xs); sec.set_xticklabels([f"n={k}" for k in cnt]); sec.tick_params(labelsize=FL - 1, pad=1)
        if c == 0:
            a.set_ylabel(rl if not ratio else "per channel / 2D–1D", labelpad=2)
fig.legend(*AX[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=FL + 0.5)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUTD, f"why2d1d_channels_sample.{ext}"), dpi=300 if ext == "png" else None)
print("saved", os.path.join(OUTD, "why2d1d_channels_sample.png"), "targets:", list(out))
