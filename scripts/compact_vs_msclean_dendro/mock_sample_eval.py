#!/usr/bin/env python
"""
The injection test over the sample (mock_sample.py): for every target, S/N, noise seed and setting (2D-1D at k = 5, the
2D starlet per channel at k = 5, 4, 3), the deconvolution against the known truth at the target's compact clean beam,
lambda/B_max and the next ladder step (b00, b05, b07). Truth and model are brought from the truth's beam to the
comparison beam with the same kernel; region: the target's 20 km/s ladder mask footprint (masks.npz) over the line channels.

  per run: cube RMSE (%), moment-1 and -2 RMSE (km/s; pixels where the truth's moment 0 exceeds 3% of its peak, voxels
  above 3% of the truth's peak), the flux recovered at 10-30% and 3-10% of the truth's peak
  per target and S/N: the mean over seeds; "2D per channel, best k": the k with the lowest cube RMSE at lambda/B_max
  figure (A&A text width, square panels): at lambda/B_max, against the peak S/N per channel -- top: the mean over the
  targets (error bars: the standard error); bottom: the ratio per channel / 2D-1D, per target, then its mean (error bars)

    python scripts/compact_vs_msclean_dendro/mock_sample_eval.py
    -> results/gallery/mock_sample_eval.json, results/gallery/why2d1d_injection_sample.{png,pdf}
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts/compact_vs_msclean_dendro")); sys.path.insert(0, os.path.join(REPO, "simple"))
from beams import to_beam  # noqa: E402

RES = os.path.join(REPO, "results"); OUTD = os.path.join(RES, "gallery"); CELL = 0.16
SNRS = [330, 110, 33, 12]; CFG = [("L5_k5", "2D–1D"), ("L0_k5", "2D per channel, k = 5"), ("L0_k4", None), ("L0_k3", None)]
BINS = [(0.1, 0.3), (0.03, 0.1)]


def mom(c, ok, v):
    s = c.sum(0); m1 = np.where(ok, (c * v[:, None, None]).sum(0) / np.where(ok, s, 1), np.nan)
    m2 = np.where(ok, np.sqrt(np.clip((c * (v[:, None, None] - np.nan_to_num(m1)) ** 2).sum(0) / np.where(ok, s, 1), 0, None)), np.nan)
    return m1, m2


def target(t):
    md = os.path.join(RES, t, "mock_sample"); fd = os.path.join(RES, t, "figures/resolution_ladder_joint20")
    if not os.path.exists(os.path.join(md, "truth.npy")):
        return None
    G = json.load(open(os.path.join(RES, t, "g20_c0.16/grid.json"))); L = G["line"]
    v = (G["start_kms"] + G["width_kms"] * np.arange(G["nchan"]))[L]
    tb = json.load(open(os.path.join(md, "truth.json")))["truth_beam"]; TB = (tb, tb, 0.0)
    LAD = json.load(open(os.path.join(fd, "resolution_ladder.json"))); MK = np.load(os.path.join(fd, "masks.npz"))
    T1 = np.load(os.path.join(md, "truth.npy")).astype(np.float64)[L]
    TBX = {ib: to_beam(T1, TB, tuple(LAD["rows"][ib][k] for k in ("bmaj", "bmin", "bpa")), CELL) for ib in (0, 5, 7)}
    res = {}
    for p in sorted(glob.glob(os.path.join(md, "model_snr*_seed*_L*_k*.npy"))):
        tag = os.path.basename(p)[6:-4]; snr, seed, l_, k_ = tag.split("_")
        rj = json.load(open(os.path.join(md, f"run_{tag}.json"))); sc = rj["scale"]
        x = np.load(p).astype(np.float64)[L]; cfg = f"{l_}_{k_}"
        for ib in (0, 5, 7):
            b = tuple(LAD["rows"][ib][k] for k in ("bmaj", "bmin", "bpa"))
            Tb = TBX[ib] * sc; C = to_beam(x, TB, b, CELL)
            foot = MK[f"b{ib:02d}"].any(0)[::2, ::2]; M = np.broadcast_to(foot, Tb.shape); rel = Tb / Tb.max()
            ok = foot & (Tb.sum(0) > 0.03 * Tb.sum(0).max())
            t1, t2 = mom(np.where(rel >= 0.03, Tb, 0), ok, v); c1, c2 = mom(np.where(rel >= 0.03, np.clip(C, 0, None), 0), ok, v)
            vv = np.isfinite(c1) & np.isfinite(t1)
            m = dict(rmse=100 * np.linalg.norm((C - Tb)[M]) / np.linalg.norm(Tb[M]), mom1=float(np.sqrt(np.mean((c1 - t1)[vv] ** 2))),
                     mom2=float(np.sqrt(np.mean((c2 - t2)[vv] ** 2))),
                     **{f"rec_{lo:g}": float(C[M & (rel >= lo) & (rel < hi)].sum() / Tb[M & (rel >= lo) & (rel < hi)].sum()) for lo, hi in BINS})
            res.setdefault(snr[3:], {}).setdefault(f"b{ib:02d}", {}).setdefault(cfg, []).append({k: float(v_) for k, v_ in m.items()})
    for snr in res:                                     # the mean over seeds, and the per-channel run with the best k
        for b in res[snr]:
            d = res[snr][b]
            for cfg in list(d):
                d[cfg] = {k: float(np.mean([r[k] for r in d[cfg]])) for k in d[cfg][0]}
    for snr in res:
        ks = [c for c in ("L0_k5", "L0_k4", "L0_k3") if c in res[snr].get("b05", {})]
        if ks:
            best = min(ks, key=lambda c: res[snr]["b05"][c]["rmse"])
            for b in res[snr]:
                res[snr][b]["L0_best"] = dict(res[snr][b][best], k=best)
    print(f"{t}: {len(res)} S/N levels, best per-channel k at lambda/B_max: " +
          ", ".join(f"S/N {s}: {res[s]['b05']['L0_best']['k']}" for s in sorted(res, key=float, reverse=True) if "L0_best" in res[s].get('b05', {})), flush=True)
    return res


ALL = ["NGC3110", "IC5179", "MCG-02-33-098", "IC4518W", "ESO297-G011", "NGC3256", "NGC7130", "IC4518E", "IC4687", "IRASF17138-1017",
       "NGC1614", "NGC2369", "NGC5135"]
D = {t: r for t in ALL if (r := target(t))}
json.dump(D, open(os.path.join(OUTD, "mock_sample_eval.json"), "w"), indent=1)

# ======================================================================== figure: lambda/B_max, mean over the targets
FS, FT, FL = 8, 7, 6.5
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": FS, "xtick.labelsize": FT,
                     "ytick.labelsize": FT, "xtick.direction": "in", "ytick.direction": "in", "axes.linewidth": 0.6,
                     "legend.fontsize": FL, "lines.linewidth": 1.2, "lines.markersize": 3.2})
MET = [("rmse", "cube RMSE (%)"), ("rec_0.1", "flux recovered, 10–30% of peak"), ("mom1", r"moment-1 RMSE (km s$^{-1}$)"),
       ("mom2", r"moment-2 RMSE (km s$^{-1}$)")]
LINES = [("L5_k5", "#2a6fdb", "-", "o", "w", "2D–1D"), ("L0_k5", "#e07b00", "--", "s", "w", "2D per channel, k = 5"),
         ("L0_best", "#b2182b", ":", "^", "full", "2D per channel, best k")]
fig, AX = plt.subplots(2, 4, figsize=(184 / 25.4, 3.9), gridspec_kw=dict(wspace=0.5, hspace=0.32, left=0.07, right=0.985, bottom=0.11, top=0.9))
x = np.array(SNRS, float)


def agg(key, cfg, ratio=False):
    mu, se = [], []
    for s in SNRS:
        vals = []
        for t, r in D.items():
            d = r.get(str(s), {}).get("b05", {})
            if cfg in d and "L5_k5" in d:
                vals.append(d[cfg][key] / d["L5_k5"][key] if ratio else d[cfg][key])
        vals = np.array(vals); mu.append(vals.mean() if vals.size else np.nan)
        se.append(vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else np.nan)
    return np.array(mu), np.array(se)


for c, (key, lab) in enumerate(MET):
    a, ar = AX[0, c], AX[1, c]
    for cfg, col, ls, mk, mf, ll in LINES:
        mu, se = agg(key, cfg); a.errorbar(x, mu, yerr=se, color=col, ls=ls, marker=mk, mfc=col if mf == "full" else mf,
                                           capsize=2, elinewidth=0.7, label=ll)
        if cfg != "L5_k5":
            mu, se = agg(key, cfg, ratio=True)
            ar.errorbar(x, mu, yerr=se, color=col, ls=ls, marker=mk, mfc=col if mf == "full" else mf, capsize=2, elinewidth=0.7)
    for ax_ in (a, ar):
        ax_.set_xscale("log"); ax_.set_xlim(500, 8); ax_.set_box_aspect(1); ax_.tick_params(which="both", top=True, right=True, pad=2)
        ax_.set_xticks([300, 100, 30, 10]); ax_.set_xticklabels(["300", "100", "30", "10"]); ax_.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_title(lab, fontsize=FS - 0.5, pad=3); a.tick_params(labelbottom=False)
    ar.axhline(1, color="0.5", lw=0.6); ar.set_xlabel("peak S/N per channel", labelpad=2)
    if c == 0:
        a.set_ylabel(f"mean of {len(D)} galaxies", labelpad=2); ar.set_ylabel("per channel / 2D–1D", labelpad=2)
fig.legend(*AX[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=FL + 0.5)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUTD, f"why2d1d_injection_sample.{ext}"), dpi=300 if ext == "png" else None)
print("saved", os.path.join(OUTD, "why2d1d_injection_sample.png"), "targets:", list(D))
