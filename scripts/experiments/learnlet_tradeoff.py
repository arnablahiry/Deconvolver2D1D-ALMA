#!/usr/bin/env python
"""
Experiment 22, the fair comparison of 2D dictionaries: error against the truth on the compact ground-truth simulation
vs spurious flux on pure noise (null test, seed 7), each over k (and, for the FIRE learnlet, the weight w of its
learnt filter planes' lambda, run_model.py --learnt-weight; w = 0 is the starlet prior). A dictionary is better if
its curve lies below the starlet's: less error at the same false-detection level. Also the convergence checks of
the FIRE learnlet at k = 5 (60,000 iterations; warm start from the starlet solution).

  sims  data/experiments/22_learnlet/sim/<tag>[_w<w>]_k<k>[_it60k|_warm].npy, and the starlet's
        data/experiments/01_simulations/compact/pdtau_k<k>.npy (same recipe)
  nulls data/experiments/22_learnlet/null/<tag>[_w<w>]_k<k>.json (total model flux)
  score as learnlet_analyze.py: ||G*(x - truth)|| / ||G*truth|| at the extended clean beam, flux outside the true
        emission (G*truth < 1% of its peak)

    CUDA_VISIBLE_DEVICES=3 python scripts/experiments/learnlet_tradeoff.py
    -> figures/experiments/22_learnlet/tradeoff.png, data/experiments/22_learnlet/tradeoff.{json,md}
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import beam_ft, conv, ld  # noqa: E402
from common import G, REPO, EXP_DATA, BEAMS  # noqa: E402

D = os.path.join(EXP_DATA, "22_learnlet")
FIG = os.path.join(REPO, "figures/experiments/22_learnlet"); os.makedirs(FIG, exist_ok=True)
PAT = re.compile(r"^(?P<tag>[a-z0-9]+?)(?:_w(?P<w>[0-9.]+))?_k(?P<k>[0-9.]+)(?P<var>_it60k|_warm)?$")
NAME = {"starlet": "starlet", "learnlet16": "learnlet:16 (ImageNet)", "learnletfire": "learnlet (FIRE)"}
COL = {("starlet", 1.0): "#1baf7a", ("learnlet16", 1.0): "#2a78d6", ("learnletfire", 1.0): "#9b3fb5",
       ("learnletfire", 0.25): "#d17be0", ("learnletfire", 0.1): "#e8b3ef"}


def key(path):
    m = PAT.match(os.path.basename(path)[:-len(".npy" if path.endswith(".npy") else ".json")])
    return None if m is None else (m["tag"], float(m["w"]) if m["w"] else 1.0, float(m["k"]), (m["var"] or "")[1:])


null = {}
for f in glob.glob(os.path.join(D, "null", "*.json")):
    kk = key(f)
    if kk and not kk[3]:
        null[kk[:3]] = json.load(open(f))["flux"]
sims = {}
for f in glob.glob(os.path.join(D, "sim", "*.npy")):
    kk = key(f)
    if kk:
        sims[kk] = f
for f in glob.glob(os.path.join(EXP_DATA, "01_simulations", "compact", "pdtau_k*.npy")):
    k = float(re.search(r"pdtau_k([0-9.]+)\.npy$", f).group(1))
    sims.setdefault(("starlet", 1.0, k, ""), f)

J, cell = 7, 0.04
truth = ld(f"{EXP_DATA}/01_simulations/inputs/compact_sim_truth_0.04arcsec.npy")
kft = beam_ft(truth.shape[1:], BEAMS[("ngc3110", "extended")], cell)
tb = conv(truth, kft); supp = tb > 0.01 * tb.max()
rows = []
for kk, f in sorted(sims.items()):
    x = G.to_t(ld(f)); e = conv(x, kft) - tb
    rows.append(dict(tag=kk[0], w=kk[1], k=kk[2], var=kk[3], err=float(e.norm() / tb.norm()),
                     flux_out=float(x[~supp].sum()), flux_in=float(x[supp].sum()), null_flux=null.get(kk[:3]), path=f))
    print(f"{kk}: err {rows[-1]['err']:.4f}  flux out {rows[-1]['flux_out']:.2f}  null {rows[-1]['null_flux']}", flush=True)

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.family": "serif", "font.size": 11, "xtick.direction": "in", "ytick.direction": "in"})
fig, ax = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)
lab = lambda t, w: NAME[t] + ("" if w == 1.0 else f", filter planes λ × {w:g}")
for (t, w) in sorted({(r["tag"], r["w"]) for r in rows}):
    rr = sorted([r for r in rows if r["tag"] == t and r["w"] == w and not r["var"]], key=lambda r: r["k"])
    c = COL.get((t, w), "0.4")
    pts = [r for r in rr if r["null_flux"] is not None]
    if pts:
        ax[0].plot([r["null_flux"] for r in pts], [r["err"] for r in pts], "o-", color=c, label=lab(t, w))
        for r in pts:
            ax[0].annotate(f"{r['k']:g}", (r["null_flux"], r["err"]), xytext=(3, 3), textcoords="offset points", fontsize=8, color=c)
    ax[1].plot([r["k"] for r in rr], [r["err"] for r in rr], "o-", color=c, label=lab(t, w))
    ax[2].plot([r["flux_out"] for r in rr], [r["err"] for r in rr], "o-", color=c, label=lab(t, w))
for r in rows:
    if r["var"]:
        m = {"it60k": "D", "warm": "^"}[r["var"]]
        lb = f"{NAME[r['tag']]}, k = {r['k']:g}, " + {"it60k": "60,000 iterations", "warm": "warm start (starlet)"}[r["var"]]
        ax[1].plot(r["k"], r["err"], m, ms=9, mfc="none", color=COL.get((r["tag"], r["w"]), "0.4"), label=lb)
        ax[2].plot(r["flux_out"], r["err"], m, ms=9, mfc="none", color=COL.get((r["tag"], r["w"]), "0.4"), label=lb)
        if null.get((r["tag"], r["w"], r["k"])) is not None:
            ax[0].plot(null[(r["tag"], r["w"], r["k"])], r["err"], m, ms=9, mfc="none", color=COL.get((r["tag"], r["w"]), "0.4"), label=lb)
ax[0].set_xscale("log"); ax[0].set_xlabel("spurious flux on pure noise (null test), Jy"); ax[0].set_ylabel("error vs truth (extended beam)")
ax[0].set_title("the trade-off: lower and to the left is better (numbers: k)", fontsize=11)
ax[1].set_xlabel("k"); ax[1].set_ylabel("error vs truth"); ax[1].set_title("error vs k", fontsize=11)
ax[2].set_xlabel("flux outside the true emission, Jy"); ax[2].set_ylabel("error vs truth")
ax[2].set_title("error vs flux outside the true emission", fontsize=11)
for a in ax:
    a.legend(fontsize=8, frameon=False)
fig.suptitle("NGC 3110 compact ground-truth simulation, pdtau, 20,000 iterations: 2D dictionaries x CDF 9/7", fontsize=13)
fig.savefig(os.path.join(FIG, "tradeoff.png"), dpi=100)
json.dump(rows, open(os.path.join(D, "tradeoff.json"), "w"), indent=1)
md = ["| dictionary | w | k | variant | error | flux outside (Jy) | null flux (Jy) |", "|---|---|---|---|---|---|---|"]
md += [f"| {NAME[r['tag']]} | {r['w']:g} | {r['k']:g} | {r['var'] or '20k'} | {r['err']:.4f} | {r['flux_out']:.2f} | "
       + (f"{r['null_flux']:.3f}" if r["null_flux"] is not None else "–") + " |"
       for r in sorted(rows, key=lambda r: (r["tag"], -r["w"], r["k"], r["var"]))]
open(os.path.join(D, "tradeoff.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md)); print("figure:", os.path.join(FIG, "tradeoff.png"))
