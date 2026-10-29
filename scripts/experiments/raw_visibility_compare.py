#!/usr/bin/env python
"""
Figure 1 of the compact-vs-extended consistency set (NGC 3110): do the two arrays measure the same line visibilities
where both measure? Raw visibilities (raw_visibilities.py: line-integrated, continuum-subtracted, per baseline and
integration; null = a line-free window of the same width) compared through one common model.

The two arrays' tracks almost never cross on a fine uv grid, and a single line-integrated visibility has S/N ~ 1, so
visibilities are not compared one to one. Instead both are referred to the same model, the compact MS-CLEAN on the
51" field (data/ngc3110/compact/big_finished: line channels minus the line-free mean, x 25 km/s), predicted at every
visibility (FFT of the model, zero-padded x2, bilinear interpolation). Per baseline bin and array:
  r(q)  = sum w V_data conj(V_model) / sum w |V_model|^2      (complex; |r| the data/model scale, arg r a phase offset)
  chi2  = sum w |V_data - r V_model|^2 / (2 N - 2), with w the visibility weights put on an absolute scale by the null
          visibilities (sigma^2 per component = k / w, k = mean(Re(null)^2 w)), scaled to the line window's noise:
          its continuum comes from 138 channels, the null's from 74, so k_line = k_null (1 + 207/138) / (1 + 207/74)
The model is only a common reference (an interpolator of the sky between the two samplings): the test is whether
r_extended(q) = r_compact(q) where both have data, i.e. their ratio, which does not depend on the model being right.
The sign/axis convention of the prediction is the one that best matches the compact data (checked on the compact data
only, so it cannot bias the comparison).

  (a) uv coverage   (b) where each array has its weight, with the compact clean beam's transfer function
  (c) data rotated by the model phase, Re(V e^{-i arg V_model}), binned, both arrays, with the model and noise floors
  (d) |r| per bin for each array, and their ratio extended/compact
  (e) chi2/dof per bin for each array (data vs the scaled model), the null visibilities for reference

    python scripts/experiments/raw_visibility_compare.py
    -> results/NGC3110/figures/consistency/fig1_visibilities.{png,pdf,json}
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from numpy.fft import fft2, ifftshift
from scipy.ndimage import map_coordinates

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(REPO, "results/NGC3110")
OUT = os.path.join(RES, "figures", "consistency"); os.makedirs(OUT, exist_ok=True)
GC = json.load(open(os.path.join(RES, "config.json")))
L = json.load(open(os.path.join(RES, "line_channels.json")))["line"]
F = [c for c in range(32) if c not in L and 1 < c < 30]
Z = np.load(os.path.join(REPO, "data/ngc3110/raw_visibilities.npz"))
MODEL = os.path.join(REPO, "data/ngc3110/compact/big_finished/model.fits")
COL = {"compact": "#0b3c6e", "extended": "#3f8fd8"}
RED, PUR = "#c0392b", "#7a4fbf"
TICK, LAB = 12, 14
plt.rcParams.update({"font.family": "serif", "font.size": 13, "xtick.direction": "in", "ytick.direction": "in"})
ARCSEC = np.pi / 180 / 3600

bc = GC["beams_natural"]["compact"]
theta = np.sqrt(bc[0] * bc[1]) * ARCSEC
T = lambda qq: np.exp(-(np.pi * theta * qq * 1e3) ** 2 / (4 * np.log(2)))
QB = 2 * np.sqrt(np.log(2) * np.log(10)) / (np.pi * theta) / 1e3                # T = 10%

# ---------------------------------------------------------------- the model's line visibilities on a fine uv grid
h = fits.getheader(MODEL); cell = abs(h["CDELT1"]) * 3600 * ARCSEC
M = np.nan_to_num(fits.getdata(MODEL)).squeeze().astype(np.float64)
Ml = (M[L] - M[F].mean(0)).sum(0) * 25.0                                        # Jy km/s per pixel, line only
n0 = Ml.shape[0]; N = 2 * n0
pad = np.zeros((N, N)); pad[(N - n0) // 2:(N - n0) // 2 + n0, (N - n0) // 2:(N - n0) // 2 + n0] = Ml
FM = fft2(ifftshift(pad))                                                        # sum I exp(-2pi i k (x - centre) / N)
print(f"model {os.path.relpath(MODEL, REPO)}: line flux {Ml.sum():.1f} Jy km/s; grid {N} px, uv cell "
      f"{1 / (N * cell) / 1e3:.2f} klambda", flush=True)


def predict(u, v, su, sv):
    """model visibility at (u, v) klambda; su, sv = +-1 choose the axis convention"""
    kx = (su * u * 1e3 * N * cell) % N; ky = (sv * v * 1e3 * N * cell) % N
    re = map_coordinates(FM.real, [ky, kx], order=1, mode="grid-wrap")
    im = map_coordinates(FM.imag, [ky, kx], order=1, mode="grid-wrap")
    return re + 1j * im


KLINE = (1 + 207 / 138) / (1 + 207 / 74)


def load(cfg):
    u, v, lr, li, nr, ni, cr, ci, w = Z[cfg].astype(np.float64)
    # the null window's continuum comes from 74 channels, the line's from 138 (raw_visibilities.py): per visibility
    # var = n sigma^2 (1 + n / n_cont) for an n-channel sum, so the line noise is the null's x KLINE
    k_null = float(np.mean(nr ** 2 * w))
    return dict(u=u, v=v, q=np.hypot(u, v), line=lr + 1j * li, null=nr + 1j * ni, w=w, k_null=k_null, k=k_null * KLINE)


D = {c: load(c) for c in ("compact", "extended")}
# the axis convention: the one whose prediction best matches the compact data (compact only)
best = None
for su in (1, -1):
    for sv in (1, -1):
        d = D["compact"]; s = d["q"] < 60
        vm = predict(d["u"][s], d["v"][s], su, sv)
        r = (d["w"][s] * d["line"][s] * np.conj(vm)).sum() / (d["w"][s] * np.abs(vm) ** 2).sum()
        res = (d["w"][s] * np.abs(d["line"][s] - r * vm) ** 2).sum()
        print(f"convention u x {su:+d}, v x {sv:+d}: compact (< 60 klambda) |r| {np.abs(r):.3f}, phase {np.degrees(np.angle(r)):+.1f} deg, "
              f"weighted residual {res:.4g}", flush=True)
        if best is None or res < best[0]:
            best = (res, su, sv)
_, SU, SV = best
for c, d in D.items():
    d["model"] = predict(d["u"], d["v"], SU, SV)
print(f"using u x {SU:+d}, v x {SV:+d}", flush=True)

lo, hi = float(np.percentile(D["extended"]["q"], 5)), float(D["compact"]["q"].max())
bins = np.geomspace(8, 2000, 34); mid = np.sqrt(bins[1:] * bins[:-1])


def per_bin(d, a, b, kind="line"):
    s = (d["q"] >= a) & (d["q"] < b)
    if s.sum() < 30:
        return None
    w = d["w"][s]; x = d[kind][s]; vm = d["model"][s]
    W = w / d["k" if kind == "line" else "k_null"]                             # 1 / sigma^2 per component
    r = (W * x * np.conj(vm)).sum() / (W * np.abs(vm) ** 2).sum()
    rerr = 1 / np.sqrt((W * np.abs(vm) ** 2).sum())                            # sigma of r per component
    rot = (x * np.exp(-1j * np.angle(vm))).real
    return dict(n=int(s.sum()), r=complex(r), ratio=float(np.abs(r)), ratio_err=float(rerr), phase_deg=float(np.degrees(np.angle(r))),
                chi2=float((W * np.abs(x - r * vm) ** 2).sum() / (2 * s.sum() - 2)),
                rot=float((W * rot).sum() / W.sum()), rot_err=float(1 / np.sqrt(W.sum())),
                model_rot=float((W * np.abs(vm)).sum() / W.sum()))


P = {c: [per_bin(D[c], a, b) for a, b in zip(bins[:-1], bins[1:])] for c in D}
PN = {c: [per_bin(D[c], a, b, "null") for a, b in zip(bins[:-1], bins[1:])] for c in D}


def summary(c, qa, qb, kind="line"):
    return per_bin(D[c], qa, qb, kind)


stats = dict(model=os.path.relpath(MODEL, REPO), convention=[SU, SV], overlap_klambda=[lo, hi], compact_beam_klambda=QB)
for name, (qa, qb) in (("compact_beam_band_overlap", (lo, QB)), ("overlap", (lo, hi)), ("compact_beam_band", (8, QB))):
    stats[name] = {}
    for c in D:
        m = summary(c, qa, qb); mn = summary(c, qa, qb, "null")
        stats[name][c] = dict(ratio=m["ratio"], ratio_err=m["ratio_err"], phase_deg=m["phase_deg"], chi2=m["chi2"], n=m["n"],
                              null_chi2=mn["chi2"])
    e, cpt = stats[name]["extended"], stats[name]["compact"]
    stats[name]["extended_over_compact"] = e["ratio"] / cpt["ratio"]
    stats[name]["extended_over_compact_err"] = e["ratio"] / cpt["ratio"] * np.hypot(e["ratio_err"] / e["ratio"], cpt["ratio_err"] / cpt["ratio"])
    print(f"{name:26s} ({qa:.0f}-{qb:.0f} klambda): data/model compact {cpt['ratio']:.3f} (chi2 {cpt['chi2']:.2f}, null {cpt['null_chi2']:.2f}), "
          f"extended {e['ratio']:.3f} (chi2 {e['chi2']:.2f}, null {e['null_chi2']:.2f}) -> extended/compact "
          f"{stats[name]['extended_over_compact']:.3f} +- {stats[name]['extended_over_compact_err']:.3f}; phases "
          f"{cpt['phase_deg']:+.1f} / {e['phase_deg']:+.1f} deg", flush=True)
for c in D:
    stats[f"weight_in_band_{c}"] = float(D[c]["w"][D[c]["q"] < QB].sum() / D[c]["w"].sum())
json.dump(dict(stats=stats, bins_klambda=mid.tolist(),
               per_bin={c: [None if p is None else {k: (v if not isinstance(v, complex) else [v.real, v.imag]) for k, v in p.items()}
                            for p in P[c]] for c in P}), open(os.path.join(OUT, "fig1_visibilities.json"), "w"), indent=1)

# ======================================================================== figure 2 x 3
fig = plt.figure(figsize=(18, 10.5))
gs = fig.add_gridspec(2, 3, wspace=0.42, hspace=0.32)
a = fig.add_subplot(gs[0, 0])                                                    # (a) uv coverage
rng = np.random.default_rng(1)
for c in ("extended", "compact"):
    u_, v_ = D[c]["u"], D[c]["v"]; i = rng.choice(u_.size, min(u_.size, 60000), replace=False)
    a.scatter(np.r_[u_[i], -u_[i]], np.r_[v_[i], -v_[i]], s=0.3, color=COL[c], alpha=0.6, rasterized=True, linewidths=0)
    a.plot([], [], "s", color=COL[c], label=c)
for r_, ls, c_ in ((lo, "--", "0.2"), (hi, "--", "0.2"), (QB, ":", RED)):
    a.add_patch(plt.Circle((0, 0), r_, fill=False, ls=ls, color=c_, lw=1.3))
a.text(0, -QB * 1.05, "compact-beam scales", color=RED, ha="center", va="top", fontsize=10)
Lm = 420; a.set_xlim(Lm, -Lm); a.set_ylim(-Lm, Lm); a.set_aspect("equal"); a.tick_params(labelsize=TICK)
a.set_xlabel("u (kλ)", fontsize=LAB); a.set_ylabel("v (kλ)", fontsize=LAB)
a.legend(loc="upper left", fontsize=11, frameon=True, framealpha=0.9, edgecolor="none", markerscale=1.5)
a.set_title(f"(a) uv coverage; dashed: {lo:.0f} and {hi:.0f} kλ", fontsize=LAB - 1)

a = fig.add_subplot(gs[0, 1])                                                    # (b) weight per baseline length
hb = np.geomspace(8, 2000, 50)
for c in D:
    hh, _ = np.histogram(D[c]["q"], hb, weights=D[c]["w"])
    a.step(np.sqrt(hb[1:] * hb[:-1]), hh / hh.sum(), where="mid", color=COL[c], lw=2,
           label=f"{c}: {stats[f'weight_in_band_{c}']:.0%} of its weight < {QB:.0f} kλ")
a2 = a.twinx(); qq = np.geomspace(8, 2000, 300)
a2.plot(qq, T(qq), color=RED, lw=1.5, ls=":"); a2.set_ylim(0, 1.05)
a2.set_ylabel("compact clean beam: transfer", color=RED, fontsize=LAB - 1); a2.tick_params(colors=RED, labelsize=TICK)
a.axvspan(lo, hi, color="0.93", zorder=0)
a.set_xscale("log"); a.set_xlabel("baseline length (kλ)", fontsize=LAB); a.set_ylabel("fraction of weight", fontsize=LAB)
a.tick_params(labelsize=TICK); a.legend(loc="upper left", fontsize=10, frameon=False)
a.set_title("(b) where each array has its data", fontsize=LAB)

a = fig.add_subplot(gs[0, 2])                                                    # (c) model-phase-rotated amplitude
for c in D:
    ok = [p is not None for p in P[c]]
    qm = mid[ok]; y = np.array([p["rot"] for p in P[c] if p]); e = np.array([p["rot_err"] for p in P[c] if p])
    good = y > 3 * e
    a.errorbar(qm[good], y[good], yerr=e[good], fmt="o", ms=4, color=COL[c], capsize=0, label=f"{c} data")
    a.plot(qm[~good], np.abs(y[~good]), "o", ms=4, mfc="none", color=COL[c])
    a.plot(qm, [p["model_rot"] for p in P[c] if p], "-", color=COL[c], lw=1.2, alpha=0.7)
a.plot([], [], "k-", lw=1.2, alpha=0.7, label="model at each array's baselines"); a.plot([], [], "ko", mfc="none", ms=4, label="|value| < 3σ")
a.axvspan(lo, hi, color="0.93", zorder=0); a.axvline(QB, color=RED, ls=":", lw=1.2)
a.set_xscale("log"); a.set_yscale("log"); a.tick_params(labelsize=TICK)
a.set_xlabel("baseline length (kλ)", fontsize=LAB); a.set_ylabel("Re(V e$^{-i\\,\\arg V_{\\rm model}}$) (Jy km s$^{-1}$)", fontsize=LAB)
a.set_ylim(0.5, 600); a.set_xlim(8, 2000)
a.legend(fontsize=9.5, frameon=False, loc="upper right"); a.set_title("(c) line visibilities, model-phase rotated", fontsize=LAB)

a = fig.add_subplot(gs[1, 0:2])                                                  # (d) data/model per bin
rat = {}
for c in D:
    ok = np.array([p is not None and p["ratio"] > 3 * p["ratio_err"] for p in P[c]])
    rat[c] = (ok, np.array([p["ratio"] if p else np.nan for p in P[c]]), np.array([p["ratio_err"] if p else np.nan for p in P[c]]))
    a.errorbar(mid[ok], rat[c][1][ok], yerr=rat[c][2][ok], fmt="o-", ms=5, color=COL[c], capsize=0, lw=1.2, label=f"{c} data / model")
both = rat["compact"][0] & rat["extended"][0]
rr = rat["extended"][1] / rat["compact"][1]
re_ = rr * np.hypot(rat["extended"][2] / rat["extended"][1], rat["compact"][2] / rat["compact"][1])
a.errorbar(mid[both], rr[both], yerr=re_[both], fmt="D", ms=7, color="k", capsize=0, label="extended / compact (same bins)")
m = stats["compact_beam_band"]
a.axvspan(8, QB, color="#fbe9e7", zorder=0)
a.axhline(1, color="0.6", lw=1); a.axhline(m["extended_over_compact"], color="k", ls="--", lw=1.2,
                                           label=f"extended / compact, all baselines < {QB:.0f} kλ: {m['extended_over_compact']:.3f} ± {m['extended_over_compact_err']:.3f}")
a.axvline(QB, color=RED, ls=":", lw=1.2)
a.text(9, 0.45, "compact-beam scales", color=RED, fontsize=11, va="bottom")
a.set_xscale("log"); a.set_xlim(8, 400); a.set_ylim(0.4, 1.6); a.tick_params(labelsize=TICK)
a.set_xlabel("baseline length (kλ)", fontsize=LAB); a.set_ylabel("|data / model|", fontsize=LAB)
a.legend(fontsize=10, frameon=False, loc="upper right", ncol=2)
a.set_title("(d) both arrays against one common model: extended / compact (black) is the test", fontsize=LAB)

a = fig.add_subplot(gs[1, 2])                                                    # (e) chi2 per bin
for c in D:
    ok = [p is not None for p in P[c]]
    a.plot(mid[ok], [p["chi2"] for p in P[c] if p], "o-", color=COL[c], ms=4, lw=1.2, label=f"{c}: line")
    okn = [p is not None for p in PN[c]]
    a.plot(mid[okn], [p["chi2"] for p in PN[c] if p], "x", color=COL[c], ms=5, label=f"{c}: null (no line)")
a.axhline(1, color="0.5", lw=1); a.axvspan(lo, hi, color="0.93", zorder=0); a.axvline(QB, color=RED, ls=":", lw=1.2)
a.set_xscale("log"); a.set_ylim(0, 3); a.tick_params(labelsize=TICK)
a.set_xlabel("baseline length (kλ)", fontsize=LAB); a.set_ylabel("χ²/dof vs the scaled model (clipped at 3)", fontsize=LAB)
a.legend(fontsize=9.5, frameon=False); a.set_title("(e) fit quality per bin", fontsize=LAB)
fig.savefig(os.path.join(OUT, "fig1_visibilities.png"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig1_visibilities.pdf"), dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved", os.path.join(OUT, "fig1_visibilities.png"))
