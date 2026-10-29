#!/usr/bin/env python
"""
Sample-averaged versions of the NGC 3110 resolution ladder (fig_ladder_metrics_column.py) and power-spectrum ratio
(fig_power_joint.py), for the gallery's galaxies: the per-galaxy results (results/<T>/figures/resolution_ladder_joint20/
{resolution_ladder,resolution_ladder_moments,power_joint}.json) on a common axis, the beam (or the equivalent beam
0.44/k) in units of each galaxy's compact lambda/B_max. Thin lines: the galaxies; thick: the median; band: the range.

  ladder  (a) cube RMSE, total flux, moment-0 RMSE (%); (b) moment-1 and moment-2 RMSE (km/s); against the ladder step,
          as in fig_ladder_metrics_column.py (the steps are the same for every galaxy: compact clean beam, 4 circular
          beams, lambda/B_max, 4, extended beam), labelled with the median of theta / (lambda/B_max) at each step
  power   (a) the joint MS-CLEAN signal (at the extended beam, noise subtracted), 2D-1D and compact MS-CLEAN power
          spectra, each galaxy's normalised by its joint signal on the largest scales shown (the top factor 1.5); (b) the 2D-1D
          and compact MS-CLEAN power / joint signal; against 0.44 / (k lambda/B_max), interpolated onto one grid
          centred (in log) on 1 and ending at the finest scale that every galaxy covers (joint signal-to-noise 1)

    python scripts/compact_vs_msclean_dendro/fig_sample_ladder_power.py
    -> results/gallery/sample_ladder_column.{png,pdf}, sample_power_column.{png,pdf}, sample_ladder_power.json
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TARGETS = os.environ.get("TARGETS", "IC5179,MCG-02-33-098,NGC3110,IC4518W,ESO297-G011,NGC3256,NGC7130").split(",")
OUT = os.path.join(REPO, "results", "gallery")
QB = 2 * np.log(2) / np.pi
D = lambda t: os.path.join(REPO, "results", t, "figures", "resolution_ladder_joint20")
have = [t for t in TARGETS if all(os.path.exists(os.path.join(D(t), f)) for f in
                                  ("resolution_ladder.json", "resolution_ladder_moments.json", "power_joint.json"))]
print("galaxies:", have, "missing:", [t for t in TARGETS if t not in have])

# ======================================================================== ladder: per step, every galaxy
LAD = {}
for t in have:
    a = json.load(open(os.path.join(D(t), "resolution_ladder.json"))); m = json.load(open(os.path.join(D(t), "resolution_ladder_moments.json")))
    lb = a["lbmax_compact"]
    LAD[t] = dict(x=np.array([np.sqrt(r["bmaj"] * r["bmin"]) / lb for r in a["rows"]]),
                  rmse=np.array([r["rmse"] for r in a["rows"]]), flux=np.array([r["flux"] for r in a["rows"]]),
                  m0=np.array([r["mom0_rmse_pct"] for r in m["rows"]]), m1=np.array([r["mom1_rmse_kms"] for r in m["rows"]]),
                  m2=np.array([r["mom2_rmse_kms"] for r in m["rows"]]))
stack = lambda k: np.array([LAD[t][k] for t in have])
X = np.median(stack("x"), 0)
# ======================================================================== power: onto one grid of 0.44 / (k lambda/B_max)
POW, XR = {}, []
for t in have:                                                           # the range every galaxy covers
    p = json.load(open(os.path.join(D(t), "power_joint.json"))); k = np.array(p["k"])
    ok = (k > 0) & (k < 1 / p["joint_snr1_arcsec"]); xe = QB / (k[ok] * p["lbmax_compact_arcsec"])
    XR.append((xe.min(), xe.max()))
GLO = max(r[0] for r in XR); GHI = 1 / GLO                              # common to all galaxies, centred on 1
G = np.geomspace(GLO, GHI, 120)                                          # equivalent beam / (lambda/B_max)
for t in have:
    p = json.load(open(os.path.join(D(t), "power_joint.json"))); lb = p["lbmax_compact_arcsec"]
    k = np.array(p["k"]); PS = np.array(p["P_J_signal"]); ok = (k > 0) & (k < 1 / p["joint_snr1_arcsec"])
    xe = QB / (k[ok] * lb)
    on = lambda y: np.exp(np.interp(np.log(G), np.log(xe[::-1]), np.log(np.clip(y[ok], 1e-30, None))[::-1],
                                    left=np.nan, right=np.nan))
    for key in [q for q in ("C", "K", "D") if f"P_{q}" in p]:
        r = (np.array(p[f"P_{key}"]) / PS)[ok]
        POW.setdefault(key, {})[t] = np.interp(np.log(G), np.log(xe[::-1]), r[::-1], left=np.nan, right=np.nan)
    ref = np.nanmean(on(PS)[G >= GHI / 1.5])                             # the joint signal on the largest scales
    for key, y in [("J", PS)] + [(q, np.array(p[f"P_{q}"])) for q in ("C", "K", "D") if f"P_{q}" in p]:
        POW.setdefault("n" + key, {})[t] = on(y) / ref
json.dump(dict(galaxies=have, ladder_x=X.tolist(), **{f"ladder_{k}_median": np.median(stack(k), 0).tolist()
                                                     for k in ("rmse", "flux", "m0", "m1", "m2")},
               power_x=G.tolist(), **{f"power_{k}_median": np.nanmedian(np.array(list(POW[k].values())), 0).tolist()
                                      for k in ("C", "K", "D") if len(POW.get(k, {})) == len(have)}),
          open(os.path.join(OUT, "sample_ladder_power.json"), "w"), indent=1)

# ======================================================================== figures (A&A column, 88 mm)
FS, FT, FL = 8, 7, 6
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": FS, "axes.labelsize": FS,
                     "xtick.labelsize": FT, "ytick.labelsize": FT, "xtick.direction": "in", "ytick.direction": "in",
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.fontsize": FL,
                     "lines.linewidth": 1.1})
LEG = dict(frameon=True, framealpha=0.95, edgecolor="0.6", fancybox=False, borderpad=0.35, labelspacing=0.25,
           handlelength=2.0, columnspacing=0.8, handletextpad=0.4)
FW = 88 / 25.4; ML, MR, MB, MT, H, GP = 0.5, 0.08, 0.42, 0.08, 1.5, 0.12
FH = MB + 2 * H + GP + MT
fig = plt.figure(figsize=(FW, FH)); W = FW - ML - MR
inch = lambda x, y, w, h: fig.add_axes([x / FW, y / FH, w / FW, h / FH])


NS = len(X); S = np.arange(NS); ILB = int(np.argmin(abs(X - 1)))        # ladder steps; lambda/B_max in the middle


def deco(a, last):
    a.set_xlim(-0.5, NS - 0.5); a.set_xticks(S); a.tick_params(which="both", top=True, right=True)
    a.tick_params(axis="x", which="minor", bottom=False, top=False); a.axvline(ILB, color="0.45", lw=0.6, ls=":")
    if last:
        a.set_xticklabels(["1" if i == ILB else f"{v:.2f}" for i, v in enumerate(X)], fontsize=FL)
        for i_ in (0, ILB, NS - 1):
            a.get_xticklabels()[i_].set_fontweight("bold")
        a.set_xlabel(r"comparison beam FWHM / ($\lambda/B_\mathrm{max}$ beam), median", labelpad=3)
    else:
        a.tick_params(labelbottom=False)


def band(a, key, col, lab, ls="-", mk="o"):
    s = stack(key)
    for i in range(len(s)):
        a.plot(S, s[i], color=col, lw=0.4, alpha=0.35)
    a.fill_between(S, s.min(0), s.max(0), color=col, alpha=0.1, lw=0)
    a.plot(S, np.median(s, 0), ls, marker=mk, color=col, lw=1.4, ms=2.6, label=lab)


a = inch(ML, MB + H + GP, W, H); deco(a, False)
a.axhline(10, color="0.45", lw=0.6, ls="--"); a.axhline(5, color="0.75", lw=0.5, ls="--"); a.axhline(0, color="0.5", lw=0.5)
band(a, "rmse", "k", "cube: RMSE", mk="D"); band(a, "m0", "#7a3e9d", "mom. 0: RMSE"); band(a, "flux", "0.4", "cube: total flux", ls="--", mk="s")
a.set_ylabel("compact − joint (%)", labelpad=4); a.set_ylim(top=max(40, a.get_ylim()[1]))
a.text(0.03, 0.95, "(a) cube and moment 0" + (f" ({len(have)} galaxies)" if len(have) > 1 else ""), transform=a.transAxes, ha="left", va="top", fontweight="bold", fontsize=FT)
a.legend(loc="upper left", bbox_to_anchor=(0.012, 0.86), **LEG)
a = inch(ML, MB, W, H); deco(a, True)
a.axhline(5, color="0.45", lw=0.6, ls="--"); a.axhline(2.5, color="0.75", lw=0.5, ls="--"); a.axhline(0, color="0.5", lw=0.5)
band(a, "m1", "#2a6fdb", "mom. 1: RMSE"); band(a, "m2", "#e07b00", "mom. 2: RMSE")
a.set_ylabel(r"compact − joint (km s$^{-1}$)", labelpad=4); a.set_ylim(bottom=-0.5, top=max(14, a.get_ylim()[1]))
a.text(0.03, 0.95, "(b) moments 1 and 2", transform=a.transAxes, ha="left", va="top", fontweight="bold", fontsize=FT)
a.legend(loc="upper left", bbox_to_anchor=(0.012, 0.86), **LEG)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"sample_ladder_column.{ext}"), dpi=300 if ext == "png" else None)
plt.close(fig)

HA, HB = 1.75, 1.05
FH = MB + HB + GP + HA + MT
fig = plt.figure(figsize=(FW, FH))
CJ, C2D, CK, CD = "k", "#2a6fdb", "#e07b00", "0.4"


def pdeco(a, last):
    a.set_xscale("log"); a.set_xlim(GHI, GLO); a.tick_params(which="both", top=True, right=True)
    a.axvline(1, color="0.45", lw=0.6, ls=":")
    a.set_xticks([3, 2, 1, 0.5, 0.3]); a.set_xticklabels(["3", "2", "1", "0.5", "0.3"] if last else [])
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def pband(a, key, col, lab, ls, lw=1.5):
    r = np.array(list(POW[key].values()))
    for row in r:
        a.plot(G, row, color=col, lw=0.4, alpha=0.35, ls=ls)
    n = np.isfinite(r).sum(0); m = np.where(n == len(r), np.nanmedian(r, 0), np.nan)
    a.fill_between(G, np.nanmin(r, 0), np.nanmax(r, 0), where=n >= 2, color=col, alpha=0.1, lw=0)
    a.plot(G, m, color=col, lw=lw, ls=ls, label=lab)


a = inch(ML, MB + HB + GP, W, HA); pdeco(a, False); a.set_yscale("log")
pband(a, "nJ", CJ, "joint MS-CLEAN (compact + extended), signal", "-", 1.3)
pband(a, "nC", C2D, "compact 2D–1D deconvolved", "-")
pband(a, "nK", CK, "compact MS-CLEAN (restored)", "-.")
if len(POW.get("nD", {})) == len(have):
    pband(a, "nD", CD, "compact dirty", ":", 1.3)
nJ = np.array(list(POW["nJ"].values()))
a.set_ylim(np.nanmin(nJ) * 0.3, 300); a.set_ylabel("normalised power", labelpad=4)
a.text(0.94, 0.965, r"$\lambda/B_\mathrm{max}$ beam", transform=a.get_xaxis_transform(), ha="left", va="top",
       fontsize=FL, color="0.35")
a.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9), **LEG)
a.text(0.03, 0.05, "(a)" + (f" {len(have)} galaxies" if len(have) > 1 else ""), transform=a.transAxes, ha="left",
       va="bottom", fontweight="bold", fontsize=FT)
a = inch(ML, MB, W, HB); pdeco(a, True)
a.axhline(1, color="0.6", lw=0.6); a.axhline(0.5, color="0.75", lw=0.5, ls="--")
pband(a, "C", C2D, "compact 2D–1D deconvolved", "-"); pband(a, "K", CK, "compact MS-CLEAN", "-.")
if len(POW.get("D", {})) == len(have):
    pband(a, "D", CD, "compact dirty", ":", 1.3)
a.set_ylim(0, 2); a.set_ylabel("power / joint signal", labelpad=4)
a.set_xlabel(r"equivalent beam FWHM [($0.44/k$) / ($\lambda/B_\mathrm{max}$ beam)]", labelpad=3)
a.legend(loc="upper left", bbox_to_anchor=(0.012, 0.965), **LEG)
a.text(0.97, 0.93, "(b)", transform=a.transAxes, ha="right", va="top", fontweight="bold", fontsize=FT)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"sample_power_column.{ext}"), dpi=300 if ext == "png" else None)
print("saved", os.path.join(OUT, "sample_ladder_column.png"), os.path.join(OUT, "sample_power_column.png"))
