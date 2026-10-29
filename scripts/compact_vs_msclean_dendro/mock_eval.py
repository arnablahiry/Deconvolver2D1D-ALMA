#!/usr/bin/env python
"""
The injection test (mock_deconv.py, MCELL=0.16 TRUTH_BEAM=0.33): 2D-1D against the 2D starlet per channel, against
the known truth, as a function of the truth's scale (S/N). The truth is the sky at 0.33"; at a comparison beam b both
the truth and the model are convolved with the kernel that brings 0.33" to b (sqrt(b^2 - 0.33^2) per axis), so the
comparison is at resolution b exactly. Region: the 20 km/s ladder mask footprint at b, over the line channels.

  per scale, model and beam (compact clean beam, lambda/B_max, 0.40"): cube RMSE (%); the flux recovered in bins of the
  truth's brightness (fraction of its peak: < 3%, 3-10%, 10-30%, > 30%) and the RMSE there; moment-1 and -2 RMSE (km/s)
  over the pixels where the truth's moment 0 exceeds 3% of its peak, voxels above 3% of the truth's channel peak
  figure: these against the peak S/N of the mock dirty cube, at lambda/B_max

    python scripts/compact_vs_msclean_dendro/mock_eval.py
    -> results/NGC3110/mock/mock_eval.json, results/NGC3110/figures/resolution_ladder_joint20/mock_2d1d.{png,pdf}
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
from restore import convolve_with_beam  # noqa: E402

R = os.path.join(REPO, "results/NGC3110"); MD = os.path.join(R, "mock"); FD = os.path.join(R, "figures/resolution_ladder_joint20")
sys.path.insert(0, os.path.join(REPO, "scripts/compact_vs_msclean_dendro"))
from beams import to_beam  # noqa: E402
# MTAG: c0.16_t0.33 (0.16" pixels, the truth at 0.33") or c0.08 (0.08" pixels, the truth at the extended clean beam)
TAG = os.environ.get("MTAG", "c0.16_t0.33"); SUF = "" if TAG == "c0.16_t0.33" else "_" + TAG
if TAG.startswith("c0.08"):
    CELL, SUB, IBS, FIGB = 0.08, 1, (0, 5, 7, 10), ("b05", "b10")
    _h = fits.getheader(os.path.join(R, "ladder_joint_20kms/b10/clean_finished/image.fits")); TB = (_h["BMAJ"] * 3600, _h["BMIN"] * 3600, _h["BPA"])
else:
    CELL, SUB, IBS, FIGB, TB = 0.16, 2, (0, 5, 7), ("b05",), (0.33, 0.33, 0.0)
LAD = json.load(open(os.path.join(FD, "resolution_ladder.json"))); MK = np.load(os.path.join(FD, "masks.npz"))
G = json.load(open(os.path.join(R, "consistency_20kms_0.08/grid.json"))); L = G["line"]
v = (G["start_kms"] + 20 * np.arange(G["nchan"]))[L]
T1 = np.load(os.path.join(MD, f"truth_{TAG}.npy")).astype(np.float64)[L]
BINS = [(0, 0.03), (0.03, 0.1), (0.1, 0.3), (0.3, 10)]


def mom(c, ok):
    s = c.sum(0); m1 = np.where(ok, (c * v[:, None, None]).sum(0) / np.where(ok, s, 1), np.nan)
    m2 = np.where(ok, np.sqrt(np.clip((c * (v[:, None, None] - np.nan_to_num(m1)) ** 2).sum(0) / np.where(ok, s, 1), 0, None)), np.nan)
    return m1, m2


runs = {}
for p in sorted(glob.glob(os.path.join(MD, f"model_{TAG}_s*_L*.npy"))):
    s_, l_ = os.path.basename(p)[len(f"model_{TAG}_s"):-4].split("_L")
    runs.setdefault(float(s_), {})[int(l_)] = p
out = {}
for sc in sorted(runs):
    if set(runs[sc]) != {0, 5}:
        continue
    rj = json.load(open(os.path.join(MD, f"run_{TAG}_s{sc:g}_L5.json")))
    out[sc] = dict(peak_snr=rj["peak_snr"])
    X = {l: np.load(p).astype(np.float64)[L] for l, p in runs[sc].items()}
    for ib in IBS:
        b = tuple(LAD["rows"][ib][k] for k in ("bmaj", "bmin", "bpa"))              # truth and model: TB -> b, the same kernel
        Tb = to_beam(T1 * sc, TB, b, CELL); foot = MK[f"b{ib:02d}"].any(0)[::SUB, ::SUB]; M = np.broadcast_to(foot, Tb.shape)
        rel = Tb / Tb.max(); ok = foot & (Tb.sum(0) > 0.03 * Tb.sum(0).max())
        w = np.where(rel >= 0.03, Tb, 0); t1, t2 = mom(w, ok)
        row = {}
        for l, x in X.items():
            C = to_beam(x, TB, b, CELL); wc = np.where(rel >= 0.03, np.clip(C, 0, None), 0); c1, c2 = mom(wc, ok)
            vv = np.isfinite(c1) & np.isfinite(t1)
            row[f"L{l}"] = dict(rmse=float(100 * np.linalg.norm((C - Tb)[M]) / np.linalg.norm(Tb[M])),
                                mom1=float(np.sqrt(np.mean((c1 - t1)[vv] ** 2))), mom2=float(np.sqrt(np.mean((c2 - t2)[vv] ** 2))),
                                bins=[dict(lo=lo, hi=hi, recovered=float(C[M & (rel >= lo) & (rel < hi)].sum() / Tb[M & (rel >= lo) & (rel < hi)].sum()),
                                           rmse=float(100 * np.linalg.norm((C - Tb)[M & (rel >= lo) & (rel < hi)]) / np.linalg.norm(Tb[M & (rel >= lo) & (rel < hi)])))
                                      for lo, hi in BINS])
        out[sc][f"b{ib:02d}"] = row
        a_, b_ = row["L5"], row["L0"]
        print(f"scale {sc:5g} (peak S/N {rj['peak_snr']:6.0f}) b{ib:02d}: RMSE {a_['rmse']:.1f}/{b_['rmse']:.1f}%  mom1 {a_['mom1']:.1f}/{b_['mom1']:.1f}  "
              f"mom2 {a_['mom2']:.1f}/{b_['mom2']:.1f}  recovered (<3, 3-10, 10-30, >30%): "
              + " ".join(f"{p['recovered']:.2f}/{q['recovered']:.2f}" for p, q in zip(a_["bins"], b_["bins"])), flush=True)
json.dump({str(k): v_ for k, v_ in out.items()}, open(os.path.join(MD, f"mock_eval{SUF}.json"), "w"), indent=1)

# ======================================================================== figure: against S/N, at lambda/B_max
if len(out) >= 2:
    FS, FT, FL = 8, 7, 6.5
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": FS, "xtick.labelsize": FT,
                         "ytick.labelsize": FT, "xtick.direction": "in", "ytick.direction": "in", "axes.linewidth": 0.6,
                         "legend.fontsize": FL, "lines.linewidth": 1.3, "lines.markersize": 3.5})
    nr = len(FIGB); fig, AX = plt.subplots(nr, 4, figsize=(184 / 25.4, 0.3 + 1.85 * nr), squeeze=False,
                                            gridspec_kw=dict(wspace=0.42, hspace=0.45, left=0.06, right=0.985, bottom=0.2 / nr, top=1 - 0.3 / nr))
    sc = sorted(out); snr = [out[s]["peak_snr"] for s in sc]
    BL = {"b05": r"at $\lambda/B_\mathrm{max}$ (0.60″)", "b10": "at the extended beam"}
    for row, bb in enumerate(FIGB):
        for a, (f, ylab, t) in zip(AX[row], ((lambda r: r["rmse"], "cube RMSE (%)", "(a) cube"),
                                            (None, "flux recovered / truth", "(b) flux"),
                                            (lambda r: r["mom1"], r"moment-1 RMSE (km s$^{-1}$)", "(c) velocity"),
                                            (lambda r: r["mom2"], r"moment-2 RMSE (km s$^{-1}$)", "(d) dispersion"))):
            for l, col, ls in ((5, "#2a6fdb", "-"), (0, "#e07b00", "--")):
                if f is None:
                    for ib_ in (2, 1):
                        a.plot(snr, [out[s][bb][f"L{l}"]["bins"][ib_]["recovered"] for s in sc], ls, marker="o" if ib_ == 2 else "s",
                               color=col, mfc=col if ib_ == 2 else "w")
                    continue
                a.plot(snr, [f(out[s][bb][f"L{l}"]) for s in sc], ls, marker="o", color=col)
            a.set_xscale("log"); a.invert_xaxis(); a.set_xlabel("peak S/N per channel", labelpad=2); a.set_ylabel(ylab, labelpad=2)
            a.tick_params(which="both", top=True, right=True)
            a.text(0.04, 0.95, t if nr == 1 else f"{t}, {BL[bb]}", transform=a.transAxes, ha="left", va="top", fontweight="bold",
                   fontsize=FT if nr == 1 else FT - 1.5)
        AX[row][1].axhline(1, color="0.6", lw=0.6); AX[row][1].set_ylim(0.3, 1.6)
    H = [plt.Line2D([], [], color="#2a6fdb", ls="-", marker="o"), plt.Line2D([], [], color="#e07b00", ls="--", marker="o"),
         plt.Line2D([], [], color="0.35", ls="", marker="o"), plt.Line2D([], [], color="0.35", ls="", marker="s", mfc="w")]
    fig.legend(H, ["2D–1D", "2D per channel", "(b) emission at 10–30% of the peak", "(b) at 3–10% of the peak"], loc="upper center",
               ncol=4, bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=FL, columnspacing=1.4, handlelength=2.2)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FD, f"mock_2d1d{SUF}.{ext}"), dpi=300 if ext == "png" else None)
    print("saved", os.path.join(FD, f"mock_2d1d{SUF}.png"))
