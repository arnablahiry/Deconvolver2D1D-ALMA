#!/usr/bin/env python
"""
Experiment 23 scores (runs: iht_dict.py): error vs truth on the compact ground-truth simulation against spurious flux
on pure noise (null test, seed 7), per arm (dictionary x threshold rule) over k or s. Metric as experiment 22
(learnlet_tradeoff.py): ||G*(x - truth)|| / ||G*truth|| at two resolutions, G = the extended clean beam (err) and a
circular Gaussian of the compact array's lambda/B_max, 0.604" (err_lb, results/NGC3110/config.json
lbmax_compact_arcsec: the finest scale the compact data constrain); flux outside the true emission (extended-beam
G*truth < 1% of its peak); per-starlet-scale errors (extended beam). Reference: the pipeline's starlet pdtau.

    iht_score.py [--dir data/experiments/23_iht]  -> <dir>/scores.{json,md}, figures/experiments/23_iht/tradeoff.png
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import beam_ft, conv, ld  # noqa: E402
from common import G, REPO, EXP_DATA, BEAMS  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=os.path.join(EXP_DATA, "23_iht")); A = ap.parse_args()
FIG = os.path.join(REPO, "figures/experiments/23_iht"); os.makedirs(FIG, exist_ok=True)
PAT = re.compile(r"^(?P<d>[a-z0-9_]+?)_(?P<thr>noise|learnt|pdtau)_(?P<p>[ks])(?P<v>[0-9.]+)$")
NAME = {("starlet", "noise"): "starlet, k σ", ("learnlet", "noise"): "learnlet filters, k σ",
        ("learnlet", "learnt"): "learnlet network (learnt thresholds × s)",
        ("starlet", "learnt"): "starlet at the learnt thresholds × s (network with S = 0)"}
COL = {("starlet", "noise"): "#1baf7a", ("learnlet", "noise"): "#2a78d6", ("learnlet", "learnt"): "#eb6834",
       ("starlet", "learnt"): "#7a7a7a"}
DESC = {"learnlet": "ImageNet learnlet", "learnlet_ext32": "ImageNet learnlet, extrapolated, top 32",
        "learnlet_ext": "ImageNet learnlet, scale-3 filters extrapolated", "learnlet_fire": "FIRE learnlet, per scale",
        "learnlet_fire_shared4": "FIRE learnlet, shared, 4 trained scales",
        "learnlet_fire_shared6": "FIRE learnlet, shared, 6 trained scales"}
EXTRA = ["#b8860b", "#9b3fb5", "#d6336c", "#4a6fe3", "#20a0a0", "#8c564b"]

truth = ld(f"{EXP_DATA}/01_simulations/inputs/compact_sim_truth_0.04arcsec.npy")
kft = beam_ft(truth.shape[1:], BEAMS[("ngc3110", "extended")], 0.04)
tb = conv(truth, kft); supp = tb > 0.01 * tb.max()
tpl = G.starlet(tb, 7)
LB = float(json.load(open(os.path.join(REPO, "results/NGC3110/config.json")))["lbmax_compact_arcsec"])
kft_lb = beam_ft(truth.shape[1:], (LB, LB, 0.0), 0.04); tb_lb = conv(truth, kft_lb)


def score(x):
    x = G.to_t(x); e = conv(x, kft) - tb; ep = G.starlet(e, 7)
    return dict(err=float(e.norm() / tb.norm()), err_lb=float((conv(x, kft_lb) - tb_lb).norm() / tb_lb.norm()),
                flux_in=float(x[supp].sum()), flux_out=float(x[~supp].sum()),
                per_scale=[float(ep[j].norm() / tpl[j].norm()) for j in range(8)])


rows = []
for f in sorted(glob.glob(os.path.join(A.dir, "sim", "*.npy"))):
    m = PAT.match(os.path.basename(f)[:-4])
    if not m:
        continue
    key = (m["d"], m["thr"])
    if key not in NAME:                                       # another dictionary: named and coloured on the fly
        NAME[key] = ("pipeline (pdtau) with " if m["thr"] == "pdtau" else "") + DESC.get(m["d"], m["d"]) + \
            {"noise": ", k σ", "learnt": ", learnt thresholds × s", "pdtau": ""}[m["thr"]]
        COL[key] = EXTRA[(len(NAME) - 5) % len(EXTRA)]
    nf = os.path.join(A.dir, "null", os.path.basename(f).replace(".npy", ".json"))
    rows.append(dict(dict=m["d"], thr=m["thr"], par=m["p"], val=float(m["v"]), **score(ld(f)),
                     null_flux=json.load(open(nf))["flux"] if os.path.exists(nf) else None))
for f in sorted(glob.glob(os.path.join(EXP_DATA, "01_simulations", "compact", "pdtau_k*.npy"))):
    k = float(re.search(r"pdtau_k([0-9.]+)\.npy$", f).group(1))
    nf = os.path.join(EXP_DATA, "22_learnlet", "null", f"starlet_k{k:g}.json")
    rows.append(dict(dict="starlet", thr="pdtau", par="k", val=k, **score(ld(f)),
                     null_flux=json.load(open(nf))["flux"] if os.path.exists(nf) else None))
NAME[("starlet", "pdtau")] = "pipeline: starlet, soft + reweighting (pdtau)"; COL[("starlet", "pdtau")] = "#0b3d2a"
ORDER = list(NAME)

md = [f"error = ||G*(x - truth)|| / ||G*truth||, G = extended clean beam ({BEAMS[('ngc3110', 'extended')][0]:.2f}\" x "
      f"{BEAMS[('ngc3110', 'extended')][1]:.2f}\") or the compact lambda/B_max ({LB:.3f}\" circular); per-scale errors at "
      "the extended beam", "",
      "| arm | k or s | error (extended beam) | error (compact λ/B_max) | flux in support | flux outside | null flux (Jy) | " +
      " | ".join(f"scale {j}" for j in range(7)) + " | coarse |", "|---" * 15 + "|"]
for r in sorted(rows, key=lambda r: (ORDER.index((r["dict"], r["thr"])), r["val"])):
    nul = "–" if r["null_flux"] is None else f"{r['null_flux']:.3f}"
    md.append(f"| {NAME[(r['dict'], r['thr'])]} | {r['par']} = {r['val']:g} | {r['err']:.4f} | {r['err_lb']:.4f} | {r['flux_in']:.2f} | "
              f"{r['flux_out']:.2f} | {nul} | "
              + " | ".join(f"{v:.3f}" for v in r["per_scale"]) + " |")
open(os.path.join(A.dir, "scores.md"), "w").write("\n".join(md) + "\n")
json.dump(rows, open(os.path.join(A.dir, "scores.json"), "w"), indent=1)
print("\n".join(md))

plt.rcParams.update({"font.family": "serif", "font.size": 11, "xtick.direction": "in", "ytick.direction": "in"})
fig, ax = plt.subplots(1, 3, figsize=(19, 5.2), constrained_layout=True)
for key in NAME:
    rr = sorted([r for r in rows if (r["dict"], r["thr"]) == key], key=lambda r: r["val"])
    if not rr:
        continue
    pts = [r for r in rr if r["null_flux"] is not None]
    if pts:
        ax[0].plot([r["null_flux"] for r in pts], [r["err"] for r in pts], "o-", color=COL[key], label=NAME[key])
        for r in pts:
            ax[0].annotate(f"{r['par']}={r['val']:g}", (r["null_flux"], r["err"]), xytext=(3, 3),
                           textcoords="offset points", fontsize=8, color=COL[key])
    ax[1].plot([r["flux_out"] for r in rr], [r["err"] for r in rr], "o-", color=COL[key], label=NAME[key])
# per-scale error of each arm's best member
SC = ["0.04–0.08″", "0.08–0.16″", "0.16–0.32″", "0.32–0.64″", "0.64–1.28″", "1.28–2.56″", "2.56–5.12″", "coarse"]
for key in NAME:
    rr = [r for r in rows if (r["dict"], r["thr"]) == key]
    if rr:
        b = min(rr, key=lambda r: r["err"])
        ax[2].plot(range(8), b["per_scale"], "o-", color=COL[key], label=f"{NAME[key]}, {b['par']} = {b['val']:g}")
ax[2].set_xticks(range(8)); ax[2].set_xticklabels(SC, rotation=35, fontsize=9)
ax[2].set_ylabel("error per starlet scale"); ax[2].set_title("best member of each arm", fontsize=11)
ax[2].axvspan(-0.3, 3.3, color="0.92", zorder=0); ax[2].text(1.5, 0.02, "learnlet's trained scales", ha="center",
                                                             transform=ax[2].get_xaxis_transform(), fontsize=9)
ax[2].legend(fontsize=7)
ax[0].set_xscale("log"); ax[0].set_xlabel("spurious flux on pure noise (null test), Jy")
ax[1].set_xlabel("flux outside the true emission, Jy")
for x_ in ax[:2]:
    x_.set_ylabel("error vs truth (extended beam)")
ax[0].legend(fontsize=8)
fig.suptitle("NGC 3110 compact ground-truth simulation: starlet vs ImageNet learnlet, iterative hard thresholding", fontsize=12)
fig.savefig(os.path.join(FIG, "tradeoff.png"), dpi=130)
print("->", os.path.join(FIG, "tradeoff.png"))
