#!/usr/bin/env python
"""
Figures of the learnt 2D dictionaries themselves (dictionaries/<name>.npz) and of their training / denoising, for every
model of scripts/experiments/exp23_style.MODELS that exists:

  atoms/<name>        what each coefficient detects and what it puts back, per scale (fig_learnlet_atoms.py)
  thresholds          the learnt thresholds per scale in units of each starlet plane's noise (filled: trained scales,
                      open: extended to untrained scales)
  similarity          per-scale models: how alike the filter sets of two scales are (median |cos| of the best
                      one-to-one matching, sign-free; random 5x5 filters score ~0.45)
  frequency           mean radial frequency response of the effective analysis filters psi_j (*) A_jk, per scale
  share               share of the reconstruction ||S_jk (*) A_jk|| per filter, per scale: how much the learnt
                      correction can move a starlet coefficient
  training            train / validation loss vs epoch (FIRE models)
  denoising           denoising gain on one common test set (eval_learnlet_fire.py --tag n600: test galaxy B1, 600 px
                      data, 256 px patches): per CDF band and per split, plus whole-cube rms; with the ImageNet learnlet
                      and starlet hard thresholding (from the 7-scale evaluation)

    python scripts/fig_learnlet_models.py [--no-atoms]  -> figures/learnlet_models/*.png
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "simple")); sys.path.insert(0, os.path.join(REPO, "scripts", "experiments"))
import gpu_pd as G  # noqa: E402
from exp23_style import MODELS, HUE, INK, MUTED, rc  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--no-atoms", action="store_true"); A = ap.parse_args()
FIG = os.path.join(REPO, "figures", "learnlet_models"); os.makedirs(FIG, exist_ok=True)
rc()
names = [n for n in MODELS if os.path.exists(os.path.join(G.DICT_DIR, n + ".npz"))]
Z = {n: np.load(os.path.join(G.DICT_DIR, n + ".npz")) for n in names}
META = {n: json.loads(str(z["meta"])) for n, z in Z.items()}
ntrained = lambda n: META[n].get("trained_detail_scales", len(Z[n]["analysis"]))
shared = lambda n: bool(META[n].get("shared"))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=140, bbox_inches="tight"); plt.close(fig)
    print("->", os.path.join(FIG, name + ".png"))


if not A.no_atoms:
    for n in names:
        subprocess.run([sys.executable, os.path.join(REPO, "scripts", "fig_learnlet_atoms.py"), "--name", n,
                        "--scales", str(min(len(Z[n]["analysis"]), 7))], check=True)

# ------------------------------------------------------------------ thresholds per scale (plane-noise units)
w = torch.randn((1, 2048, 2048), device=G.DEV, dtype=G.DT, generator=torch.Generator(G.DEV).manual_seed(5))
p = G.starlet(w, 9)[:, 0].flatten(1)
SJ = ((p - p.median(1, keepdim=True).values).abs().median(1).values / 0.6745).cpu().numpy()   # starlet plane j, unit white
fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
for n in names:
    t = Z[n]["threshold_multipliers"]; lab, c, ls = MODELS[n]
    j = np.arange(min(len(t), 7)); f = t[j] / SJ[j]; nt = ntrained(n)
    ax.plot(j, f, color=c, ls=ls, lw=1.8, label=lab)
    ax.plot(j[:nt], f[:nt], "o", color=c, ms=7); ax.plot(j[nt:], f[nt:], "o", color=c, mfc="white", mew=1.6, ms=7)
ax.set_xlabel("starlet scale j  (2$^j$ px)"); ax.set_ylabel("learnt threshold / noise of plane w$_j$  (σ)")
ax.set_title("learnt hard thresholds per scale (filled: trained, open: extended to untrained scales)", fontsize=10.5)
ax.legend(fontsize=8); ax.set_yscale("log")
save(fig, "thresholds")

# ------------------------------------------------------------------ cross-scale similarity (per-scale models)
def unit(F):
    F = F.reshape(len(F), -1); return F / np.linalg.norm(F, axis=1, keepdims=True)


def match(U, V):
    C = np.abs(U @ V.T); r, c = linear_sum_assignment(-C); return float(np.median(C[r, c]))


per = [n for n in names if not shared(n) and ntrained(n) >= 2]
if per:
    rng = np.random.default_rng(0)
    rnd = np.median([match(unit(rng.standard_normal((16, 5, 5))), unit(rng.standard_normal((16, 5, 5)))) for _ in range(20)])
    fig, ax = plt.subplots(1, len(per), figsize=(3.6 * len(per) + 1, 3.6), constrained_layout=True, squeeze=False)
    for a, n in zip(ax[0], per):
        nt = ntrained(n); An = Z[n]["analysis"][:nt]
        M = np.array([[match(unit(An[i]), unit(An[j])) for j in range(nt)] for i in range(nt)])
        im = a.imshow(M, cmap="Blues", vmin=0.3, vmax=1); a.grid(False)
        for i in range(nt):
            for j in range(nt):
                a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if M[i, j] > 0.75 else INK)
        a.set_xticks(range(nt)); a.set_yticks(range(nt)); a.set_xlabel("scale"); a.set_ylabel("scale")
        a.set_title(MODELS[n][0], fontsize=9.5)
    cb = fig.colorbar(im, ax=ax[0, :], fraction=0.02); cb.set_label("median |cos|, best one-to-one match")
    fig.suptitle(f"how alike the analysis filter sets of two scales are (shared models: 1 by construction; "
                 f"random 16-filter sets: {rnd:.2f})", fontsize=10.5)
    save(fig, "similarity")

# ------------------------------------------------------------------ frequency response of psi_j (*) A_jk
fig, ax = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.6), sharey=True, constrained_layout=True, squeeze=False)
N = 1024
for a, n in zip(ax[0], names):
    nsc = min(len(Z[n]["analysis"]), 7)
    fb = G.FilterBank2D(nsc, os.path.join(G.DICT_DIR, n + ".npz"))
    d = torch.zeros((1, N, N), device=G.DEV, dtype=G.DT); d[0, N // 2, N // 2] = 1
    pl = fb.forward(d)[:, 0].cpu().numpy()
    fr = np.hypot(*np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N), indexing="ij"))
    edges = np.geomspace(1 / N, 0.71, 60); idx = np.digitize(fr.ravel(), edges); kc = np.sqrt(edges[1:] * edges[:-1])
    cmap = plt.get_cmap("Blues" if n == "learnlet" else {"#e34948": "Reds", "#008300": "Greens", "#eda100": "YlOrBr"}[MODELS[n][1]])
    i = 0
    for j in range(nsc):
        P = (np.abs(np.fft.fft2(pl[i + 1:i + 1 + fb.A[j].shape[0]])) ** 2).mean(0) if j < len(fb.A) else None
        if P is not None:
            prof = np.array([P.ravel()[idx == m].mean() if (idx == m).any() else np.nan for m in range(1, len(edges))])
            a.plot(kc, prof / np.nanmax(prof), color=cmap(0.35 + 0.6 * j / max(nsc - 1, 1)),
                   ls="-" if j < ntrained(n) else "--", lw=1.6, label=f"j = {j}" + ("" if j < ntrained(n) else " (ext.)"))
            i += 1 + fb.A[j].shape[0]
        else:
            i += 1
    a.set_xscale("log"); a.set_xlabel("frequency, cycles / px"); a.set_title(MODELS[n][0], fontsize=8.5)
    a.legend(fontsize=6.5, ncol=2, loc="upper left")
ax[0, 0].set_ylabel("mean power of ψ$_j$ ∗ A$_{jk}$ (peak 1)")
fig.suptitle("frequency response of the effective analysis filters per scale (each octave band of the starlet, "
             "sharpened by the learnt filters)", fontsize=10.5)
save(fig, "frequency")

# ------------------------------------------------------------------ share of the reconstruction per filter
fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
off = np.linspace(-0.3, 0.3, len(names))
for o, n in zip(off, names):
    z = Z[n]; lab, c, ls = MODELS[n]; nt = min(len(z["analysis"]), 7)
    for j in range(nt):
        a_, s_ = torch.tensor(z["analysis"][j]), torch.tensor(z["synthesis"][j])
        sh = torch.nn.functional.conv2d(a_[None], s_[:, None].flip(-1, -2), padding=a_.shape[-1] - 1,
                                        groups=len(a_))[0].flatten(1).norm(dim=1).numpy()
        ax.scatter(np.full(len(sh), j + o), sh, s=9, color=c, alpha=0.55, marker="o" if j < ntrained(n) else "x",
                   label=lab if j == 0 else None, linewidths=0.8)
        ax.plot([j + o - 0.04, j + o + 0.04], [np.median(sh)] * 2, color=INK, lw=1.4)
ax.set_yscale("log"); ax.set_xticks(range(7)); ax.set_xlabel("starlet scale j"); ax.set_ylabel("‖S$_{jk}$ ∗ A$_{jk}$‖ per filter")
ax.set_title("how much each learnt filter pair contributes to the reconstruction (bar: median; × untrained scales)", fontsize=10.5)
ax.legend(fontsize=8, markerscale=1.8)
save(fig, "share")

# ------------------------------------------------------------------ training curves
fire = [n for n in names if n != "learnlet" and os.path.exists(os.path.join(REPO, "results/learnlet_fire", n, "losses.txt"))]
if fire:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    for n in fire:
        L = np.loadtxt(os.path.join(REPO, "results/learnlet_fire", n, "losses.txt")); lab, c, ls = MODELS[n]
        ax[0].plot(L[:, 0], L[:, 1], color=c, ls=ls, label=lab); ax[1].plot(L[:, 0], L[:, 2], color=c, ls=ls)
        b = int(np.argmin(L[:, 2])); ax[1].plot(L[b, 0], L[b, 2], "o", color=c, ms=7)
    for a, t in zip(ax, ("training loss (MSE / noise², random draws)", "validation loss (A8, fixed draws; dot = best)")):
        a.set_yscale("log"); a.set_xlabel("epoch"); a.set_title(t, fontsize=10.5)
    ax[0].legend(fontsize=8)
    fig.suptitle("FIRE learnlet training (the 256 px and 600 px runs have different epoch sizes: 16,384 vs 4,096 draws)",
                 fontsize=10.5)
    save(fig, "training")

# ------------------------------------------------------------------ denoising on the common test set
ev = {n: json.load(open(p)) for n in fire for p in [os.path.join(REPO, "results/learnlet_fire", n, "eval_n600.json")]
      if os.path.exists(p)}
if ev:
    refn = next((n for n in ("learnlet_fire_n600", "learnlet_fire_n600_shared") if n in ev), next(iter(ev)))
    BANDS = ["d1", "d2", "d3", "d4", "d5", "a"]
    entries = [(MODELS[n][0], MODELS[n][1], MODELS[n][2], ev[n], "fire") for n in ev] + \
              [("ImageNet (64 filters, 4 scales)", HUE["imagenet"], "-", ev[refn], "imagenet"),
               ("starlet, hard 3σ", HUE["starlet"], "-", ev[refn], "starlet3"),
               ("starlet, hard 5σ", HUE["starlet"], "--", ev[refn], "starlet5")]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for lab, c, ls, e, m in entries:
        g = [10 * np.log10(1 / e["bands"]["test"][m]["mse_" + b]) for b in BANDS]
        ax[0].plot(range(6), g, color=c, ls=ls, marker="o", mfc="white" if ls != "-" else c, label=lab)
    ax[0].set_xticks(range(6)); ax[0].set_xticklabels([f"d$_{b[1]}$" if b != "a" else "a$_5$" for b in BANDS])
    ax[0].set_ylabel("denoising gain, dB (test galaxy B1)"); ax[0].set_title("per CDF 9/7 velocity band", fontsize=10.5)
    xs = np.arange(3); wdt = 0.8 / len(entries)
    for i, (lab, c, ls, e, m) in enumerate(entries):
        ax[1].bar(xs + (i - (len(entries) - 1) / 2) * wdt, [e["bands"][s][m]["gain_db"] for s in ("train", "valid", "test")],
                  wdt * 0.92, color=c, hatch="//" if ls != "-" else None, edgecolor="white", linewidth=0.5)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(["train (A1 A2 A4 B2 C2)", "valid (A8)", "test (B1)"])
    ax[1].set_ylabel("denoising gain, dB"); ax[1].set_title("all bands, by split (hatched: shared / 5σ)", fontsize=10.5)
    lo = min(e["bands"][s][m]["gain_db"] for _, _, _, e, m in entries for s in ("train", "valid", "test"))
    ax[1].set_ylim(lo - 1.5, None)
    vals = [(lab, c, ls, e["cubes"]["mean"][m], e["cubes"]["mean"]["m0_" + m]) for lab, c, ls, e, m in entries]
    for i, (lab, c, ls, ch, m0) in enumerate(vals):
        ax[2].barh(i, ch, color=c, hatch="//" if ls != "-" else None, edgecolor="white", height=0.38)
        ax[2].barh(i + 0.4, m0, color=c, alpha=0.45, hatch="//" if ls != "-" else None, edgecolor="white", height=0.38)
        ax[2].text(max(ch, m0) + 0.005, i + 0.2, f"{ch:.3f} / {m0:.3f}", va="center", fontsize=8, color=MUTED)
    ax[2].set_yticks(np.arange(len(vals)) + 0.2); ax[2].set_yticklabels([v[0] for v in vals], fontsize=8.5)
    ax[2].invert_yaxis(); ax[2].set_xlabel("rms error / noise  (solid: channels, light: moment 0)")
    ax[2].set_title("whole test cubes, S/N 30, back in channel space", fontsize=10.5)
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8.5)
    fig.suptitle("denoising FIRE [CII] band planes, one common test set (600 px data, 256 px patches)", fontsize=11.5)
    save(fig, "denoising")
