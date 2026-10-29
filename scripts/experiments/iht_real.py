#!/usr/bin/env python
"""
Starlet vs learnlet dictionaries in the PIPELINE (pdtau: primal-dual, soft thresholding + reweighting, 20,000 iterations)
on the REAL NGC 3110 compact data, scored against the joint compact + extended CLEAN cubes as ground truth (no
simulations). The truth at each scale is the joint visibilities imaged and restored at exactly that beam
(scripts/experiments/ladder_joint.py, 32 x 25 km/s, 0.04", 51.2"; here its central 16", the solver's field):

  extended clean beam (0.221" x 0.172")   results/NGC3110/ladder_joint/b10 (robust -0.5 + taper)
  compact lambda/B_max (0.604")           results/NGC3110/ladder_joint/b05 (robust 0.5 + taper)

  error_b = ||G_b x - J_b|| / ||J_b|| over the line channels, G_b the peak-1 Gaussian of beam b (x in Jy/pixel ->
  Jy/beam, as J_b); also per starlet scale of the extended-beam error, and the compact data residual d - N x.

Models: data/experiments/23_iht/real/<dict>_pdtau_k<k>.npy (SET=pipe_heavy / pipe_light run_iht_queue.sh), the
production starlet model (results/NGC3110/compact/deconvolved/model.npy, k = 5) and experiment 22's two real
learnlet runs (data/experiments/22_learnlet/real/*_kcal.npy, k = 1.5).

  real_k          error vs k per dictionary, at both scales
  real_scales     per-scale error at the extended beam relative to the best starlet
  real_maps       best k of each dictionary: moment 0 (extended beam) | error at the extended beam | error at the
                  compact lambda/B_max | compact data residual; first column the joint truth
  real_spectra    integrated spectra vs the joint truth
  -> data/experiments/23_iht/real_scores.{json,md}

    iht_real.py  -> figures/experiments/23_iht/report/real_*.png
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
from astropy.io import fits
from matplotlib.patches import Ellipse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import beam_ft, conv  # noqa: E402
from common import G, EXP_DATA, REPO, BEAMS, line_channels, noise_channels, read  # noqa: E402
from exp23_style import HUE, INK, MUTED, rc  # noqa: E402

D = os.path.join(EXP_DATA, "23_iht", "real")
FIG = os.path.join(REPO, "figures/experiments/23_iht/report"); os.makedirs(FIG, exist_ok=True)
DV, HALF, F4 = 25.0, 5.0, np.s_[75:325, 75:325]                            # the central 10" at 0.04"
LINE, NOISE = line_channels("ngc3110"), noise_channels("ngc3110", "compact")
BC = BEAMS[("ngc3110", "compact")]
rc()
rd = lambda p: np.nan_to_num(fits.getdata(os.path.join(REPO, p))).squeeze().astype(np.float64)
hb = lambda p: (lambda h: (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"]))(fits.getheader(os.path.join(REPO, p)))
TRUTH = {"ext": "results/NGC3110/ladder_joint/b10/clean_finished/image.fits",
         "lb": "results/NGC3110/ladder_joint/b05/clean_finished/image.fits"}
J = {s: rd(p)[:, 440:840, 440:840] for s, p in TRUTH.items()}             # Jy/beam, 0.04", the solver's 16"
BEAM = {s: hb(p) for s, p in TRUTH.items()}
KFT = {s: beam_ft((400, 400), BEAM[s], 0.04) for s in TRUTH}
SNAME = {"ext": f"extended clean beam ({BEAM['ext'][0]:.2f}″ × {BEAM['ext'][1]:.2f}″)",
         "lb": f"compact λ/B$_{{max}}$ ({BEAM['lb'][0]:.2f}″)"}
dc, pc = G.rebin(read("ngc3110", "compact", "dirty_cube.fits"), read("ngc3110", "compact", "dirty_beam_cube.fits"), 2)
op = G.Op(pc, dc); sig_c = float(G.noise_planes(op.d, NOISE).std())
native = lambda x: x.reshape(x.shape[0], 400, 2, 400, 2).sum((2, 4)) if x.shape[-1] == 800 else x
Jpl = G.starlet(G.to_t(J["ext"][LINE]), 7)
# the truth's own noise (its line-free channels, mean removed) sets an error floor no model can beat:
# floor_b = sigma_b sqrt(N voxels) / ||J_b||; the model's own error is ~ sqrt(error^2 - floor^2)
FLOOR = {s: float((J[s][NOISE] - J[s][NOISE].mean(0)).std() * np.sqrt(J[s][LINE].size) / np.linalg.norm(J[s][LINE]))
         for s in TRUTH}
own = lambda e, s: float(np.sqrt(max(e ** 2 - FLOOR[s] ** 2, 0.0)))

# ------------------------------------------------------------------ models
DICTS = {"starlet": ("starlet (pipeline as is)", HUE["starlet"], "-", "o"),
         "learnlet": ("ImageNet learnlet (64)", HUE["imagenet"], "-", "o"),
         "learnlet_ext32": ("ImageNet learnlet, extrapolated (top 32)", HUE["imagenet"], "--", "D"),
         "learnlet16": ("ImageNet learnlet (top 16), exp 22", HUE["imagenet"], ":", "v"),
         "learnlet_fire_n600": ("FIRE 600 px, per scale", HUE["fire600"], "-", "o"),
         "learnlet_fire_shared4": ("FIRE 256 px, shared", HUE["fire400"], "--", "s"),
         "learnletfire": ("FIRE 256 px, per scale (old thresholds), exp 22", HUE["fire400"], ":", "v"),
         "learnlet_deconv": ("trained for the deconvolution, per scale", "#6250d6", "-", "o"),
         "learnlet_deconv_shared": ("trained for the deconvolution, shared", "#6250d6", "--", "s"),
         "learnlet_deconv2": ("trained through unrolled FISTA (v2)", "#6250d6", "-.", "D"),
         "learnlet_deconv_pd": ("trained through unrolled pdtau", "#6250d6", "-", "P"),
         "learnlet_fire_n600_shared": ("FIRE 600 px, shared", HUE["fire600"], "--", "s"),
         "learnlet_fire_n600@l": ("FIRE 600 px, per scale, learnt thresholds", HUE["fire600"], "-", "P"),
         "learnlet_fire_n600_shared@l": ("FIRE 600 px, shared, learnt thresholds", HUE["fire600"], "--", "P")}
for _d in ("starlet", "learnlet", "learnlet_ext32", "learnlet_fire_n600", "learnlet_deconv", "learnlet_deconv_shared",
           "learnlet_deconv2", "learnlet_deconv_pd"):       # "<dict>@f": the pipeline's FISTA
    _l, _c, _ls, _m = DICTS[_d]                                                     # "<dict>@s": pdtau with W^-1
    DICTS[_d + "@f"] = (_l.replace(" (pipeline as is)", "") + ", FISTA", _c, _ls, "X")
    DICTS[_d + "@s"] = (_l.replace(" (pipeline as is)", "") + ", pdtau with synthesis W$^{-1}$", _c, _ls, "*")
# "<dict>@l": the pipeline with lambda = the dictionary's learnt per-scale thresholds x s (run_model --lam-profile
# learnlet --lam-scale s; files <dict>_pdtaul_s<s>.npy); its parameter is s, not k
runs = [("starlet", 5.0, os.path.join(REPO, "results/NGC3110/compact/deconvolved/model.npy"))]
for f in glob.glob(os.path.join(D, "*_[pf][dis]*_[ks]*.npy")):
    m = re.match(r"^(?P<d>[a-z0-9_]+?)_(?P<r>pdtau|pdtaul|pdsyn|fista)_[ks](?P<k>[0-9.]+)\.npy$", os.path.basename(f))
    d = m and (m["d"] + {"pdtau": "", "pdtaul": "@l", "pdsyn": "@s", "fista": "@f"}[m["r"]])
    if m and d in DICTS:
        runs.append((d, float(m["k"]), f))
for tag, d in (("learnlet16", "learnlet16"), ("learnletfire", "learnletfire")):
    p = os.path.join(EXP_DATA, "22_learnlet", "real", f"{tag}_kcal.npy")
    if os.path.exists(p):
        runs.append((d, 1.5, p))


def evaluate(x):
    xn = native(x.astype(np.float64))
    out = dict(flux_line=float(xn[LINE].sum() * DV))
    for s in TRUTH:
        g = conv(xn[LINE], KFT[s]).cpu().numpy()
        out[f"err_{s}"] = float(np.linalg.norm(g - J[s][LINE]) / np.linalg.norm(J[s][LINE]))
        out[f"own_{s}"] = own(out[f"err_{s}"], s)
        out[f"m0_{s}"] = g.sum(0) * DV
        if s == "ext":
            ep = G.starlet(G.to_t(g - J[s][LINE]), 7)
            out["per_scale"] = [float(ep[j].norm() / Jpl[j].norm()) for j in range(8)]
    r = (-op.grad(G.to_t(xn))).cpu().numpy()
    out["resid"] = float(r[LINE].std() / sig_c); out["rC"] = r[LINE].sum(0) * DV
    return out


E = {}
for d, k, p in sorted(runs):
    E[(d, k)] = evaluate(np.load(p)); E[(d, k)]["path"] = p
    print(f"{d:24s} k {k:<4g} err_ext {E[(d, k)]['err_ext']:.4f}  err_lb {E[(d, k)]['err_lb']:.4f}  "
          f"resid {E[(d, k)]['resid']:.3f} sigma  flux {E[(d, k)]['flux_line']:.0f}", flush=True)
jflux = {s: float(J[s][LINE].sum() / (np.pi / (4 * np.log(2)) * BEAM[s][0] * BEAM[s][1] / 0.04 ** 2) * DV) for s in TRUTH}
dicts = [d for d in DICTS if any(dd == d for dd, _ in E)]
best = {d: min([k for dd, k in E if dd == d], key=lambda k: E[(d, k)]["err_ext"]) for d in dicts}

# ------------------------------------------------------------------ scores table
keep = ("err_ext", "own_ext", "err_lb", "own_lb", "resid", "flux_line")
json.dump({f"{d}_{'s' if d.endswith('@l') else 'k'}{k:g}": dict(dictionary=DICTS[d][0], k=k, **{q: E[(d, k)][q] for q in keep + ("per_scale",)})
           for d, k in E}, open(os.path.join(EXP_DATA, "23_iht", "real_scores.json"), "w"), indent=1)
md = ["Real NGC 3110 compact data, the pipeline (pdtau, 20,000 iterations) with each 2D dictionary. Truth: the joint "
      "compact + extended CLEAN at each beam (results/NGC3110/ladder_joint/b10, b05). error = ||G x - J|| / ||J|| over "
      f"the line channels in the 16\" field. Joint line flux in the field: {jflux['ext']:.0f} Jy km/s (b10), "
      f"{jflux['lb']:.0f} (b05). The truth's own noise floors the error at {FLOOR['ext']:.3f} (extended beam) and "
      f"{FLOOR['lb']:.3f} (lambda/B_max); 'own' = sqrt(error^2 - floor^2), the part due to the model.", "",
      "| dictionary | k or s | error, extended beam | own (ext) | error, compact λ/B_max | own (λ/B_max) | compact residual (σ) | line flux (Jy km/s) |",
      "|---|---|---|---|---|---|---|---|"]
for (d, k) in sorted(E, key=lambda dk: E[dk]["err_ext"]):
    e = E[(d, k)]
    md.append(f"| {DICTS[d][0]} | {'s' if d.endswith('@l') else 'k'} = {k:g} | {e['err_ext']:.4f} | {e['own_ext']:.4f} | {e['err_lb']:.4f} | {e['own_lb']:.4f} | "
              f"{e['resid']:.3f} | {e['flux_line']:.0f} |")
open(os.path.join(EXP_DATA, "23_iht", "real_scores.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=140, bbox_inches="tight"); plt.close(fig)
    print("->", os.path.join(FIG, name + ".png"))


par = lambda d: "s" if d.endswith("@l") else "k"
# ------------------------------------------------------------------ error vs k (and vs s for the learnt thresholds)
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
for d in [d for d in dicts if par(d) == "k" and not d.endswith(("@f", "@s"))]:
    lab, c, ls, mk = DICTS[d]
    ks = sorted(k for dd, k in E if dd == d)
    for a, s in zip(ax, ("ext", "lb")):
        a.plot(ks, [E[(d, k)][f"err_{s}"] for k in ks], color=c, ls=ls, marker=mk, mfc="white" if ls != "-" else c,
               mew=1.6, label=lab)
for a, s in zip(ax, ("ext", "lb")):
    a.axhline(FLOOR[s], color=MUTED, lw=0.9, ls=":")
    a.text(0.99, FLOOR[s], "the truth's noise floor ", transform=a.get_yaxis_transform(), ha="right", va="bottom",
           fontsize=8, color=MUTED)
    a.set_xlabel("k (threshold in units of the noise)"); a.set_ylabel("error vs the joint CLEAN")
    a.set_title(SNAME[s], fontsize=10.5)
ax[0].legend(fontsize=8)
fig.suptitle("NGC 3110, real compact data, the pipeline with each 2D dictionary (lower = better)", fontsize=11.5)
save(fig, "real_k")
for suffix, fname, title in (("@f", "real_fista", "the pipeline's FISTA (synthesis through W$^{-1}$: a learnlet's synthesis filters are used)"),
                             ("@s", "real_syn", "pdtau with the dual brought back by the synthesis W$^{-1}$ instead of W$^{T}$")):
    fd = [d for d in dicts if d.endswith(suffix)]
    if not fd:
        continue
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for d in fd:
        lab, c, ls, mk = DICTS[d]
        ks = sorted(k for dd, k in E if dd == d)
        for a, s in zip(ax, ("ext", "lb")):
            a.plot(ks, [E[(d, k)][f"err_{s}"] for k in ks], color=c, ls=ls, marker=mk, mew=1.6, label=lab)
    for a, s in zip(ax, ("ext", "lb")):
        bs = E[("starlet", 5.0)][f"err_{s}"]
        a.axhline(bs, color=HUE["starlet"], lw=1.2, ls="--")
        a.text(0.01, bs, " production pipeline (pdtau, starlet, k = 5)", transform=a.get_yaxis_transform(), fontsize=8,
               color=MUTED, va="bottom")
        a.set_xlabel("k"); a.set_ylabel("error vs the joint CLEAN"); a.set_title(SNAME[s], fontsize=10.5)
    ax[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=11.5)
    save(fig, fname)
ld = [d for d in dicts if par(d) == "s"]
if ld:
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for d in ld:
        lab, c, ls, mk = DICTS[d]
        ks = sorted(k for dd, k in E if dd == d)
        for a, s in zip(ax, ("ext", "lb")):
            a.plot(ks, [E[(d, k)][f"err_{s}"] for k in ks], color=c, ls=ls, marker=mk, mfc="white" if ls != "-" else c,
                   mew=1.6, label=lab)
    for a, s in zip(ax, ("ext", "lb")):
        bs = E[("starlet", best["starlet"])][f"err_{s}"]
        a.axhline(bs, color=HUE["starlet"], lw=1.2, ls="--"); a.text(0.01, bs, f" best starlet (k = {best['starlet']:g})",
                                                                       transform=a.get_yaxis_transform(), fontsize=8,
                                                                       color=MUTED, va="bottom")
        a.axhline(FLOOR[s], color=MUTED, lw=0.9, ls=":")
        a.set_xlabel("s (the learnt per-scale thresholds x s)"); a.set_ylabel("error vs the joint CLEAN")
        a.set_title(SNAME[s], fontsize=10.5)
    ax[0].legend(fontsize=8)
    fig.suptitle("the pipeline with the FIRE learnlets' own learnt per-scale thresholds", fontsize=11.5)
    save(fig, "real_s")

# ------------------------------------------------------------------ per-scale error relative to the best starlet
SC = ["0.04–0.08″", "0.08–0.16″", "0.16–0.32″", "0.32–0.64″", "0.64–1.28″", "1.28–2.56″", "2.56–5.12″", "coarse"]
ref = E[("starlet", best["starlet"])]["per_scale"]
fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
for d in dicts:
    lab, c, ls, mk = DICTS[d]
    ps = E[(d, best[d])]["per_scale"]
    ax.plot(range(8), [p / r for p, r in zip(ps, ref)], color=c, ls=ls, marker=mk, mfc="white" if ls != "-" else c,
            label=f"{lab}, {par(d)} = {best[d]:g}")
ax.axhline(1, color=MUTED, lw=0.8)
ax.set_xticks(range(8)); ax.set_xticklabels(SC, rotation=30); ax.set_ylabel("error per scale / best starlet's")
ax.set_title("where the error goes, at the extended clean beam (best k of each dictionary; < 1 = better)", fontsize=10.5)
ax.legend(fontsize=8)
save(fig, "real_scales")


# ------------------------------------------------------------------ maps
def beam(a, b):
    pad = 0.06 * HALF; bx, by = HALF - pad - b[0] / 2, -HALF + pad + b[0] / 2
    a.add_patch(Ellipse((bx, by), width=b[0], height=b[1], angle=90 - b[2], facecolor="w", edgecolor="w"))


def scalebar(a):
    xs, ys, cap = -HALF + 0.08 * HALF, HALF - 0.1 * HALF, 0.03 * HALF
    a.plot([xs + 2, xs], [ys, ys], color="w", lw=2.0, solid_capstyle="butt")
    for xe in (xs, xs + 2):
        a.plot([xe, xe], [ys - cap, ys + cap], color="w", lw=1.4, solid_capstyle="butt")
    a.annotate("2″", (xs + 1, ys - cap), xytext=(0, -2.5), textcoords="offset points", color="w", ha="center",
               va="top", fontsize=9, fontweight="bold")


EXT = [HALF, -HALF, -HALF, HALF]
cols = [(d, best[d]) for d in dicts]
Jm = {s: J[s][LINE].sum(0) * DV for s in TRUTH}
vmax = np.percentile(Jm["ext"][F4], 99.9)
lim = {s: max(np.percentile(np.abs(E[c][f"m0_{s}"][F4] - Jm[s][F4]), 99.5) for c in cols) for s in TRUTH}
rlim = max(np.percentile(np.abs(E[c]["rC"][F4]), 99.5) for c in cols)
n = len(cols) + 1; S = 2.2
fig, ax = plt.subplots(4, n, figsize=(S * n + 0.9, 4 * S + 0.6), gridspec_kw=dict(wspace=0.03, hspace=0.05))
h0 = ax[0, 0].imshow(Jm["ext"][F4], origin="lower", extent=EXT, cmap="inferno", interpolation="nearest",
                     norm=matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=vmax))
ax[0, 0].text(0.04, 0.96, "Joint CLEAN (truth)", transform=ax[0, 0].transAxes, color="w", ha="left", va="top",
              fontsize=8, fontweight="bold")
ax[0, 0].text(0.04, 0.84, f"{jflux['ext']:.0f} Jy km/s in 16″", transform=ax[0, 0].transAxes, color="w", ha="left",
              va="top", fontsize=7.5)
beam(ax[0, 0], BEAM["ext"]); scalebar(ax[0, 0])
for row, txt in ((1, "model − truth\nat the extended\nclean beam"), (2, "model − truth\nat the compact\nλ/B$_{max}$"),
                 (3, "compact data:\nd − N x\n(line moment 0,\ncompact\ndirty beam)")):
    ax[row, 0].axis("off")
    ax[row, 0].text(0.5, 0.5, txt, transform=ax[row, 0].transAxes, ha="center", va="center", fontsize=9, color=INK)
for i, (d, k) in enumerate(cols, start=1):
    e = E[(d, k)]
    a = ax[0, i]
    a.imshow(e["m0_ext"][F4], origin="lower", extent=EXT, cmap="inferno", interpolation="nearest",
             norm=matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=vmax))
    a.text(0.04, 0.96, DICTS[d][0].split(",")[0].split(" (")[0], transform=a.transAxes, color="w", ha="left", va="top",
           fontsize=8, fontweight="bold")
    a.text(0.04, 0.84, f"{par(d)} = {k:g}  ·  {e['flux_line']:.0f} Jy km/s", transform=a.transAxes, color="w", ha="left",
           va="top", fontsize=7.5)
    beam(a, BEAM["ext"])
    for row, data, lm, bm, cm, txt in ((1, e["m0_ext"][F4] - Jm["ext"][F4], lim["ext"], BEAM["ext"], "RdBu_r", f"error {e['err_ext']:.3f}"),
                                       (2, e["m0_lb"][F4] - Jm["lb"][F4], lim["lb"], BEAM["lb"], "RdBu_r", f"error {e['err_lb']:.3f}"),
                                       (3, e["rC"][F4], rlim, BC, "PuOr_r", f"rms {e['resid']:.2f} σ")):
        b = ax[row, i]
        hh = b.imshow(data, origin="lower", extent=EXT, cmap=cm, vmin=-lm, vmax=lm, interpolation="nearest")
        b.text(0.04, 0.96, txt, transform=b.transAxes, color=INK, ha="left", va="top", fontsize=8, fontweight="bold")
        beam(b, bm)
        if i == n - 1:
            fig.colorbar(hh, ax=ax[row, :], fraction=0.012, pad=0.01, label="Jy beam$^{-1}$ km s$^{-1}$")
for a in ax.flat:
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
fig.colorbar(h0, ax=ax[0, :], fraction=0.012, pad=0.01, label="moment 0, Jy beam$^{-1}$ km s$^{-1}$")
fig.suptitle("NGC 3110, real compact data, the pipeline with each 2D dictionary at its best k; truth: joint compact + "
             "extended CLEAN at each beam", fontsize=11, y=0.915)
save(fig, "real_maps")

# ------------------------------------------------------------------ spectra
v = 5073.9 + 25.0 * (np.arange(32) - 15.5)
fig, ax = plt.subplots(figsize=(9.5, 5), constrained_layout=True)
om = np.pi / (4 * np.log(2)) * BEAM["ext"][0] * BEAM["ext"][1] / 0.04 ** 2
ax.step(v, J["ext"].sum((1, 2)) / om, where="mid", color=INK, lw=2.4, label="joint CLEAN (truth), 16″ field")
for d, k in cols:
    lab, c, ls, mk = DICTS[d]
    ax.step(v, native(np.load(E[(d, k)]["path"])).sum((1, 2)), where="mid", color=c, ls=ls, lw=1.4, label=f"{lab}, {par(d)} = {k:g}")
ax.set_ylabel("flux density, Jy"); ax.set_xlabel("velocity (radio), km s$^{-1}$")
ax.legend(fontsize=8); ax.set_title("NGC 3110 integrated spectrum, the pipeline with each dictionary", fontsize=11)
save(fig, "real_spectra")
