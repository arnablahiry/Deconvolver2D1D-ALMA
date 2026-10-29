#!/usr/bin/env python
"""
Experiment 22 analysis (runs: run_learnlet_queue.py): starlet vs learnlet 2D dictionaries x CDF 9/7, NGC 3110 compact,
pdtau, 20,000 iterations, at (a) k = 5, (b) the calibrated k (same null-test spurious flux as the starlet at k = 5),
(c) the learnlet's own threshold profile on its trained scales.

  null  spurious flux of a pure-noise solve vs k per dictionary; the calibration  -> null.png
  real  line moment 0 of the raw models (power stretch, one scale), residual moment 0, table  -> real.png
  sim   error vs truth as in analyze.py exp01: ||G*(x - truth)|| / ||G*truth||, G = extended clean beam, total and per
        starlet scale; flux inside / outside the true emission (G*truth > 1% of peak); against the starlet's k = 2..5
        curve (data/experiments/01_simulations)  -> sim.png
Everything -> data/experiments/22_learnlet/results.json and summary.md; figures in figures/experiments/22_learnlet/.

    CUDA_VISIBLE_DEVICES=1 python scripts/experiments/learnlet_analyze.py
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
from matplotlib.colors import PowerNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import beam_ft, conv, ld, scale_labels  # noqa: E402
from common import G, REPO, EXP_DATA, BEAMS, problem, to_native  # noqa: E402

D = os.environ.get("EXP22_DIR", os.path.join(EXP_DATA, "22_learnlet"))      # override: testing
FIG = os.path.join(REPO, "figures/experiments/22_learnlet"); os.makedirs(FIG, exist_ok=True)
CAL = json.load(open(os.path.join(D, "calibration.json"))) if os.path.exists(os.path.join(D, "calibration.json")) else {}
TAGS = {"starlet": "starlet", "learnlet16": "learnlet:16", "learnlet": "learnlet (64)", "learnletfire": "learnlet (FIRE)"}
COL = {"starlet": "#1baf7a", "learnlet16": "#2a78d6", "learnlet": "#eb6834", "learnletfire": "#9b3fb5"}
DV, C10, EXT = 25.0, np.s_[75:325, 75:325], [5, -5, -5, 5]
plt.rcParams.update({"font.family": "serif", "font.size": 12, "xtick.direction": "in", "ytick.direction": "in"})
R, md = {}, []
js = lambda p: json.load(open(p.replace(".npy", ".json")))


def variants(kind):
    """(key, label, path) of every model of one kind that exists"""
    top = {"starlet": "ngc3110_compact_pdtau_k5_starlet", "learnlet16": "ngc3110_compact_pdtau_k5_learnlet16",
           "learnlet": "ngc3110_compact_pdtau_k5_learnlet", "learnletfire": "ngc3110_compact_pdtau_k5_learnletfire"}
    out = []
    for t in TAGS:
        k5 = os.path.join(D, top[t] + ".npy") if kind == "real" else os.path.join(D, kind, f"{t}_k5.npy")
        cands = [(f"{t}_k5", f"{TAGS[t]}, k = 5", k5)]
        if t != "starlet":
            cands += [(f"{t}_kcal", f"{TAGS[t]}, k = {CAL.get(t, '?')} (calibrated)", os.path.join(D, kind, f"{t}_kcal.npy")),
                      (f"{t}_profile", f"{TAGS[t]}, learnt thresholds", os.path.join(D, kind, f"{t}_profile.npy"))]
        out += [c for c in cands if os.path.exists(c[2])]
    return out


# ------------------------------------------------------------------ null
pts = {}
for t in TAGS:
    for f in glob.glob(os.path.join(D, "null", f"{t}_k*.json")):
        pts.setdefault(t, {})[float(re.search(r"_k([0-9.]+)\.json$", f).group(1))] = json.load(open(f))["flux"]
prof = {t: js(os.path.join(D, "null", f"{t}_profile.npy"))["flux"] for t in TAGS if t != "starlet"
        if os.path.exists(os.path.join(D, "null", f"{t}_profile.json"))}
R["null"] = dict(flux_vs_k={t: dict(sorted(v.items())) for t, v in pts.items()}, profile=prof, calibration=CAL)
if pts:
    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    for t, v in pts.items():
        k = sorted(v)
        ax.plot(k, [max(v[x], 1e-4) for x in k], "o-", color=COL[t], label=TAGS[t])
        if t in prof:
            ax.axhline(max(prof[t], 1e-4), color=COL[t], ls=":", lw=1.2, label=f"{TAGS[t]}, learnt thresholds")
        if t in CAL and t != "starlet":
            ax.axvline(CAL[t], color=COL[t], ls="--", lw=0.8)
    if "target_flux" in CAL:
        ax.axhline(CAL["target_flux"], color="0.4", lw=0.8)
        ax.text(ax.get_xlim()[0] + 0.05, CAL["target_flux"] * 1.08, "starlet, k = 5", color="0.3", fontsize=10)
    ax.set_yscale("log"); ax.set_xlabel("k (threshold in noise σ per sub-band)"); ax.set_ylabel("spurious flux in a pure-noise cube (Jy)")
    ax.set_title("Null test: model flux when the data are pure noise\n(dashed: calibrated k = same spurious flux as starlet at k = 5)", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    fig.savefig(os.path.join(FIG, "null.png"), dpi=110); plt.close(fig)
    md += ["## Null test (pure noise, measured power spectrum): spurious model flux (Jy)\n",
           "| 2D dictionary | " + " | ".join(f"k = {k:g}" for k in sorted({k for v in pts.values() for k in v})) + " | learnt thresholds | calibrated k |",
           "|---" * (len({k for v in pts.values() for k in v}) + 3) + "|"]
    ks_all = sorted({k for v in pts.values() for k in v})
    for t, v in pts.items():
        cells = [f"{v[k]:.3f}" if k in v else "–" for k in ks_all]
        pr = f"{prof[t]:.3f}" if t in prof else "–"
        kc = (f"{CAL[t]}" + (" (extrapolated)" if CAL.get("extrapolated", {}).get(t) else "")) if t in CAL else "–"
        md.append(f"| {TAGS[t]} | " + " | ".join(cells) + f" | {pr} | {kc} |")
    md.append("")

# ------------------------------------------------------------------ real
real = variants("real")
if real:
    P = problem("ngc3110", "compact", k=5.0)
    line = P.line; lf = [c for c in range(32) if c not in line]
    rows, maps = [], {}
    for key, lab, path in real:
        x = to_native(P, np.load(path)); r = P.op.residual_sky(G.to_t(x)).cpu().numpy()
        rows.append(dict(key=key, label=lab, k=js(path)["k"], flux=float(x.sum()), flux_line=float(x[line].sum()),
                         flux_line_free=float(x[lf].sum()), resid_rms_line=float(r[line].std()),
                         resid_rms_line_free=float(r[lf].std())))
        maps[key] = (x[line].sum(0)[C10] * DV / 0.04 ** 2, r[line].sum(0)[C10] * DV)
    R["real"] = rows
    md += ["## NGC 3110 compact (real data)\n", "| model | flux (Jy) | line | line-free channels | resid rms line | resid rms line-free |",
           "|---|---|---|---|---|---|"]
    md += [f"| {r['label']} | {r['flux']:.2f} | {r['flux_line']:.2f} | {r['flux_line_free']:.3f} | {r['resid_rms_line']:.2e} | "
           f"{r['resid_rms_line_free']:.2e} |" for r in rows] + [""]
    n = len(rows); nc = min(n, 4); nr = -(-n // nc)
    vm = max(np.percentile(m[0], 99.9) for m in maps.values()); rv = max(np.percentile(np.abs(m[1]), 99.5) for m in maps.values())
    fig, axs = plt.subplots(2 * nr, nc, figsize=(4.2 * nc, 8.6 * nr), squeeze=False, constrained_layout=True)
    for i, row in enumerate(rows):
        a, b = axs[2 * (i // nc), i % nc], axs[2 * (i // nc) + 1, i % nc]
        im = a.imshow(maps[row["key"]][0], origin="lower", extent=EXT, cmap="inferno", norm=PowerNorm(0.5, 0, vm))
        a.set_title(row["label"], fontsize=11); a.text(0.97, 0.03, f"line {row['flux_line']:.1f} Jy", transform=a.transAxes, color="w", ha="right", fontsize=10)
        im2 = b.imshow(maps[row["key"]][1], origin="lower", extent=EXT, cmap="RdBu_r", vmin=-rv, vmax=rv)
        b.text(0.97, 0.03, f"resid rms {row['resid_rms_line']:.2e}", transform=b.transAxes, ha="right", fontsize=9)
    for j in range(n, nr * nc):
        axs[2 * (j // nc), j % nc].axis("off"); axs[2 * (j // nc) + 1, j % nc].axis("off")
    fig.colorbar(im, ax=axs[0::2].ravel().tolist(), shrink=0.6, label=r"model moment 0, Jy km s$^{-1}$ arcsec$^{-2}$ (√ stretch)")
    fig.colorbar(im2, ax=axs[1::2].ravel().tolist(), shrink=0.6, label=r"residual moment 0, Jy beam$^{-1}$ km s$^{-1}$")
    fig.suptitle("NGC 3110 compact, pdtau 20k: raw model (rows 1, 3) and residual d − Nx (rows 2, 4), line channels", fontsize=13)
    fig.savefig(os.path.join(FIG, "real.png"), dpi=90); plt.close(fig)

# ------------------------------------------------------------------ sim
sim = variants("sim")
if sim:
    J, cell = 7, 0.04
    truth = ld(f"{EXP_DATA}/01_simulations/inputs/compact_sim_truth_0.04arcsec.npy")
    kft = beam_ft(truth.shape[1:], BEAMS[("ngc3110", "extended")], cell)
    tb = conv(truth, kft); supp = tb > 0.01 * tb.max(); tp = G.starlet(tb, J)

    def score(path):
        x = G.to_t(ld(path)); e = conv(x, kft) - tb; ep = G.starlet(e, J)
        return dict(err=float(e.norm() / tb.norm()), per_scale=[float(ep[j].norm() / tp[j].norm()) for j in range(J + 1)],
                    flux=float(x.sum()), flux_in=float(x[supp].sum()), flux_out=float(x[~supp].sum())), x

    ref = {k: score(f"{EXP_DATA}/01_simulations/compact/pdtau_k{k}.npy")[0] for k in (2, 3, 4, 5)
           if os.path.exists(f"{EXP_DATA}/01_simulations/compact/pdtau_k{k}.npy")}
    rows, m0 = [], {"truth": (tb.cpu().numpy()[P.line if real else slice(3, 24)].sum(0) * DV)}
    for key, lab, path in sim:
        s, x = score(path); s.update(key=key, label=lab, k=js(path)["k"]); rows.append(s)
        m0[key] = conv(x, kft).cpu().numpy()[P.line if real else slice(3, 24)].sum(0) * DV
    R["sim"] = dict(truth_flux=float(truth.sum()), truth_flux_in=float(G.to_t(truth)[supp].sum()),
                    starlet_01_curve=ref, runs=rows)
    md += [f"## Compact simulation vs truth (truth flux {truth.sum():.2f} Jy, {float(G.to_t(truth)[supp].sum()):.2f} inside its support)\n",
           "error = ||G*(x − truth)|| / ||G*truth||, G = extended clean beam\n",
           "| model | error | flux in support | flux outside | " + " | ".join(scale_labels(cell, J)) + " |",
           "|---" * (4 + J + 1) + "|"]
    md += [f"| starlet, k = {k} (01_simulations) | {s['err']:.3f} | {s['flux_in']:.2f} | {s['flux_out']:.2f} | "
           + " | ".join(f"{v:.2f}" for v in s["per_scale"]) + " |" for k, s in ref.items()]
    md += [f"| {r['label']} | {r['err']:.3f} | {r['flux_in']:.2f} | {r['flux_out']:.2f} | "
           + " | ".join(f"{v:.2f}" for v in r["per_scale"]) + " |" for r in rows] + [""]
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.15])
    top = gs[0].subgridspec(1, 2, width_ratios=[1, 1.6])
    a = fig.add_subplot(top[0])
    if ref:
        a.plot(list(ref), [s["err"] for s in ref.values()], "o-", color=COL["starlet"], label="starlet (01_simulations)")
    mk = {"k5": "o", "kcal": "s", "profile": "*"}
    for r in rows:
        t, v = r["key"].rsplit("_", 1)
        a.plot(r["k"] if v != "profile" else 5.0, r["err"], mk[v], ms=11 if v == "profile" else 8, color=COL[t], mfc="none" if v == "kcal" else COL[t],
               ls="none", label=r["label"])
    a.set_xlabel("k"); a.set_ylabel("error vs truth at the extended beam"); a.legend(fontsize=8, frameon=False)
    a.set_title("compact simulation: total error (star: learnt thresholds, plotted at k = 5)", fontsize=11)
    a2 = fig.add_subplot(top[1])
    xs = np.arange(J + 1); w = 0.8 / max(len(rows), 1)
    for i, r in enumerate(rows):
        a2.bar(xs + (i - len(rows) / 2 + 0.5) * w, r["per_scale"], w, color=COL[r["key"].rsplit("_", 1)[0]],
               alpha={"k5": 1.0, "kcal": 0.6, "profile": 0.35}[r["key"].rsplit("_", 1)[1]], label=r["label"])
    a2.set_xticks(xs); a2.set_xticklabels(scale_labels(cell, J), fontsize=9); a2.set_ylabel("error / truth, per starlet scale")
    a2.legend(fontsize=8, frameon=False, ncol=2)
    keys = ["truth"] + [r["key"] for r in rows]
    bot = gs[1].subgridspec(1, len(keys))
    vm = np.percentile(m0["truth"][C10], 99.9)
    for i, k in enumerate(keys):
        a = fig.add_subplot(bot[i]); a.imshow(m0[k][C10], origin="lower", extent=EXT, cmap="inferno", norm=PowerNorm(0.5, 0, vm))
        a.set_xticks([]); a.set_yticks([])
        a.set_title("truth" if k == "truth" else next(r["label"] for r in rows if r["key"] == k).replace(", ", "\n"), fontsize=9)
    fig.suptitle("Compact ground-truth simulation, pdtau 20k: error vs truth, and line moment 0 at the extended clean beam (√ stretch)", fontsize=13)
    fig.savefig(os.path.join(FIG, "sim.png"), dpi=90); plt.close(fig)

json.dump(R, open(os.path.join(D, "results.json"), "w"), indent=1)
open(os.path.join(D, "summary.md"), "w").write("# Experiment 22: starlet vs learnlet 2D dictionaries\n\n" + "\n".join(md))
print("\n".join(md))
print("figures in", FIG)
