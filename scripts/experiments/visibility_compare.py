#!/usr/bin/env python
"""
Figure 1 of the compact-vs-extended consistency set (NGC 3110): the two arrays' line visibilities against baseline
length.

Gridded visibilities from the 82" dirty cubes (dirty_bigfield.py: natural weighting, same channels, phase centre at
the image centre): per channel F[dirty] = S V and F[PSF] = S (S the weighted sampling, on a 1/82" = 2.5 klambda
grid). Line only: FD = sum over the line channels of (F[dirty] - its line-free mean) x 25 km/s, so |V| = |FD| / S
in Jy km/s. Per baseline bin and array, the amplitude is the S-weighted mean of |V| over the cells that array really
samples (S > 5% of its 99th percentile), its noise from the line-free channels. Cells are not compared one to one:
the gridding kernel spreads each array's sparse sampling over neighbouring cells, on the same ~10 klambda scale on
which the visibilities of this ~20" source change, so cell values of two different samplings are not comparable.
A cell counts as sampled where F[PSF] exceeds 10x its artefact floor (the grid correction amplifies the image edges,
giving F[PSF] a low floor everywhere; measured beyond the array's longest baseline) and lies within that baseline.

  (a) uv coverage, with the overlap (the extended array's 5th percentile to the compact array's longest baseline) and
      the compact-beam scales (where the compact clean beam's transfer function exceeds 10%)
  (b) fraction of each array's visibilities per baseline length, with the compact beam's transfer function
  (c) line visibility amplitude vs baseline length, each array's noise floor (the mean |noise| its line-free channels
      give, dashed) and, open, bins below 5x that floor; error bars the cell-to-cell spread / sqrt(N_eff) (the arrays
      sample different azimuths of an asymmetric source)
  (d) amplitude ratio compact / extended per bin where both are >= 5x their floors, and its weighted mean over the
      compact-beam scales and over the whole overlap

    python scripts/experiments/visibility_compare.py
    -> results/NGC3110/figures/consistency/fig1_visibilities.{png,pdf,json}
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from numpy.fft import rfft2, fftfreq, rfftfreq, ifftshift

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D_ = os.path.join(REPO, "data/ngc3110"); RES = os.path.join(REPO, "results/NGC3110")
OUT = os.path.join(RES, "figures", "consistency"); os.makedirs(OUT, exist_ok=True)
GC = json.load(open(os.path.join(RES, "config.json")))
L = json.load(open(os.path.join(RES, "line_channels.json")))["line"]
F = [c for c in range(32) if c not in L and 1 < c < 30]
DV = 25.0
COL = {"compact": "#0b3c6e", "extended": "#3f8fd8"}
RED = "#c0392b"
TICK, LAB = 12, 14
plt.rcParams.update({"font.family": "serif", "font.size": 13, "xtick.direction": "in", "ytick.direction": "in"})
ARCSEC = np.pi / 180 / 3600
SNR_MIN = 5.0


def grid_vis(cfg):
    """FD (line, Jy km/s x sampling), S (sampling), sigma (per-cell noise of FD), q (klambda) on the rfft grid"""
    hd = fits.getheader(f"{D_}/{cfg}/dirty_big/dirty.fits"); cell = abs(hd["CDELT1"]) * 3600
    D = np.nan_to_num(fits.getdata(f"{D_}/{cfg}/dirty_big/dirty.fits")).astype(np.float32).squeeze()
    P = np.nan_to_num(fits.getdata(f"{D_}/{cfg}/dirty_big/psf.fits")).astype(np.float32).squeeze()
    fd = {c: rfft2(ifftshift(D[c])) for c in L + F}
    S = np.mean([rfft2(ifftshift(P[c])).real for c in L + F], axis=0)
    cont = np.mean([fd[c] for c in F], axis=0)
    FD = np.sum([fd[c] - cont for c in L], axis=0) * DV
    sig = np.sqrt(np.mean([np.abs(fd[c] - cont) ** 2 for c in F], axis=0) * len(F) / (len(F) - 1)) * DV * np.sqrt(len(L))
    n = D.shape[-1]
    u = fftfreq(n, d=cell * ARCSEC)[:, None] / 1e3; v = rfftfreq(n, d=cell * ARCSEC)[None, :] / 1e3
    return FD, S, sig, np.hypot(u, v) * np.ones_like(S)


uvz = np.load(os.path.join(RES, "uv_coverage.npz"))
rc, re = np.hypot(uvz["compact_u"], uvz["compact_v"]), np.hypot(uvz["extended_u"], uvz["extended_v"])
lo, hi = float(np.percentile(re, 5)), float(rc.max())
bc = GC["beams_natural"]["compact"]
theta = np.sqrt(bc[0] * bc[1]) * ARCSEC
T = lambda qq: np.exp(-(np.pi * theta * qq * 1e3) ** 2 / (4 * np.log(2)))          # compact clean beam transfer
qbeam = 2 * np.sqrt(np.log(2) * np.log(10)) / (np.pi * theta) / 1e3                # T = 10%
print(f"overlap {lo:.0f}-{hi:.0f} klambda; compact beam {np.sqrt(bc[0] * bc[1]):.2f}\": T > 10% for q < {qbeam:.0f} klambda")

bins = np.geomspace(8, 2000, 36); mid = np.sqrt(bins[1:] * bins[:-1])
prof = {}
QMAX = dict(compact=hi, extended=float(re.max()))
for k in ("compact", "extended"):
    FD, S, sig, q = grid_vis(k)
    # F[PSF] has a low artefact floor everywhere (the grid correction amplifies the image edges): cells count as sampled
    # only well above that floor, measured beyond the array's longest baseline, and only within that baseline
    art = np.percentile(np.abs(S[q > 1.1 * QMAX[k]]), 99)
    good = (S > 10 * art) & (q <= QMAX[k])
    print(f"{k}: sampling artefact floor {art / S.max():.1e} of the peak; {good.sum()} sampled cells", flush=True)
    amp, err, snr, floor = [], [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        s_ = good & (q >= a) & (q < b)
        if s_.sum() < 3:
            amp.append(np.nan); err.append(np.nan); snr.append(np.nan); floor.append(np.nan); continue
        w = S[s_]; x = np.abs(FD[s_]) / w                                         # |V| per cell
        A = float((w * x).sum() / w.sum())                                        # S-weighted mean |V|
        fl = float((np.sqrt(np.pi) / 2 * sig[s_]).sum() / w.sum())               # mean |noise| alone (Rayleigh)
        neff = w.sum() ** 2 / (w ** 2).sum()
        e = float(np.sqrt((w * (x - A) ** 2).sum() / w.sum() / neff))             # cell-to-cell spread (azimuth + noise)
        amp.append(A); err.append(e); snr.append(A / fl); floor.append(fl)
    prof[k] = dict(amp=np.array(amp), err=np.array(err), snr=np.array(snr), floor=np.array(floor),
                   weight_below_overlap=float(S[q < lo].sum() / S.sum()), weight_in_beam_band=float(S[q < qbeam].sum() / S.sum()))
    print(f"{k}: weight below the overlap {prof[k]['weight_below_overlap']:.1%}, in the compact-beam band "
          f"{prof[k]['weight_in_beam_band']:.1%}", flush=True)

pc, pe = prof["compact"], prof["extended"]
ok = (pc["snr"] >= SNR_MIN) & (pe["snr"] >= SNR_MIN)
ratio = pc["amp"] / pe["amp"]
rerr = ratio * np.hypot(pc["err"] / pc["amp"], pe["err"] / pe["amp"])


def mean_ratio(sel):
    w = 1 / rerr[sel] ** 2
    return dict(bins=int(sel.sum()), ratio=float((w * ratio[sel]).sum() / w.sum()), err=float(1 / np.sqrt(w.sum())),
                scatter=float(np.std(ratio[sel])), range_klambda=[float(mid[sel].min()), float(mid[sel].max())])


stats = dict(overlap_klambda=[lo, hi], compact_beam_klambda=qbeam,
             compact_beam_scales=mean_ratio(ok & (mid < qbeam)), overlap=mean_ratio(ok & (mid >= lo) & (mid <= hi)),
             all_shared=mean_ratio(ok),
             weight_below_overlap={k: v["weight_below_overlap"] for k, v in prof.items()},
             weight_in_beam_band={k: v["weight_in_beam_band"] for k, v in prof.items()})
for k in ("compact_beam_scales", "overlap", "all_shared"):
    m = stats[k]
    print(f"{k:20s}: {m['bins']} bins {m['range_klambda'][0]:.0f}-{m['range_klambda'][1]:.0f} klambda, |V| ratio compact/extended "
          f"{m['ratio']:.3f} +- {m['err']:.3f} (bin scatter {m['scatter']:.3f})", flush=True)
json.dump(dict(stats=stats, bins_klambda=mid.tolist(), **{f"amp_{k}": v["amp"].tolist() for k, v in prof.items()},
               **{f"err_{k}": v["err"].tolist() for k, v in prof.items()}, ratio=ratio.tolist(), ratio_err=rerr.tolist()),
          open(os.path.join(OUT, "fig1_visibilities.json"), "w"), indent=1)

# ======================================================================== figure 2 x 2
fig, ax = plt.subplots(2, 2, figsize=(12.5, 10.5), gridspec_kw=dict(wspace=0.32, hspace=0.3))
a = ax[0, 0]                                                                     # (a) uv coverage
rng = np.random.default_rng(1)
for k in ("extended", "compact"):
    u_, v_ = uvz[f"{k}_u"], uvz[f"{k}_v"]; i = rng.choice(u_.size, min(u_.size, 60000), replace=False)
    a.scatter(np.r_[u_[i], -u_[i]], np.r_[v_[i], -v_[i]], s=0.3, color=COL[k], alpha=0.6, rasterized=True, linewidths=0)
    a.plot([], [], "s", color=COL[k], label=k)
for r_, ls, c_ in ((lo, "--", "0.2"), (hi, "--", "0.2"), (qbeam, ":", RED)):
    a.add_patch(plt.Circle((0, 0), r_, fill=False, ls=ls, color=c_, lw=1.3))
a.text(0, -qbeam * 1.04, "compact-beam scales", color=RED, ha="center", va="top", fontsize=10)
a.text(0, hi * 1.03, f"{hi:.0f} kλ", color="0.2", ha="center", va="bottom", fontsize=10)
a.text(0, lo * 1.05, f"{lo:.0f} kλ", color="0.2", ha="center", va="bottom", fontsize=10)
Lm = 420; a.set_xlim(Lm, -Lm); a.set_ylim(-Lm, Lm); a.set_aspect("equal")
a.set_xlabel("u (kλ)", fontsize=LAB); a.set_ylabel("v (kλ)", fontsize=LAB); a.tick_params(labelsize=TICK)
a.legend(loc="upper left", fontsize=11, frameon=True, framealpha=0.9, edgecolor="none", markerscale=1.5)
a.set_title("(a) uv coverage (inner 420 kλ)", fontsize=LAB)

a = ax[0, 1]                                                                     # (b) weight per baseline length
hb = np.geomspace(8, 2000, 50)
for k, r_ in (("compact", rc), ("extended", re)):
    hh, _ = np.histogram(r_, hb)
    a.step(np.sqrt(hb[1:] * hb[:-1]), hh / hh.sum(), where="mid", color=COL[k], lw=2,
           label=f"{k}: {stats['weight_in_beam_band'][k]:.0%} of its data < {qbeam:.0f} kλ")
a2 = a.twinx(); qq = np.geomspace(8, 2000, 300)
a2.plot(qq, T(qq), color=RED, lw=1.5, ls=":"); a2.set_ylim(0, 1.05)
a2.set_ylabel("compact clean beam: transfer", color=RED, fontsize=LAB - 1); a2.tick_params(colors=RED, labelsize=TICK)
a.axvspan(lo, hi, color="0.93", zorder=0)
a.set_xscale("log"); a.set_xlabel("baseline length (kλ)", fontsize=LAB); a.set_ylabel("fraction of visibilities", fontsize=LAB)
a.tick_params(labelsize=TICK); a.legend(loc="upper left", fontsize=10, frameon=False)
a.set_title("(b) where each array has its data", fontsize=LAB)

a = ax[1, 0]                                                                     # (c) amplitude profile
for k in ("compact", "extended"):
    p = prof[k]; hiS = p["snr"] >= SNR_MIN
    a.errorbar(mid[hiS], p["amp"][hiS], yerr=p["err"][hiS], fmt="o-", ms=5, color=COL[k], lw=1.5, capsize=0, label=k)
    a.plot(mid[~hiS], p["amp"][~hiS], "o", ms=5, mfc="none", color=COL[k])
    a.plot(mid, p["floor"], "--", color=COL[k], lw=1, alpha=0.8)
a.plot([], [], "k--", lw=1, label="noise floor (line-free channels)"); a.plot([], [], "ko", mfc="none", label=f"< {SNR_MIN:.0f}x the floor")
a.axvspan(lo, hi, color="0.93", zorder=0); a.axvline(qbeam, color=RED, ls=":", lw=1.2)
a.set_xscale("log"); a.set_yscale("log"); a.tick_params(labelsize=TICK)
a.set_xlabel("baseline length (kλ)", fontsize=LAB); a.set_ylabel("line visibility amplitude (Jy km s$^{-1}$)", fontsize=LAB)
a.legend(fontsize=11, frameon=False); a.set_title("(c) the same visibilities where both measure", fontsize=LAB)

a = ax[1, 1]                                                                     # (d) ratio
a.errorbar(mid[ok], ratio[ok], yerr=rerr[ok], fmt="o", color="k", ms=5, capsize=0)
a.axhline(1, color="0.6", lw=1)
m = stats["overlap"]
a.axhspan(m["ratio"] - m["err"], m["ratio"] + m["err"], color="0.85", zorder=0)
a.axhline(m["ratio"], color="k", ls="--", lw=1.2, label=f"overlap mean {m['ratio']:.3f} ± {m['err']:.3f}")
mb = stats["compact_beam_scales"]
a.axvspan(mb["range_klambda"][0] / 1.1, qbeam, color="#f6dcd8", zorder=0, label=f"compact-beam scales: {mb['ratio']:.3f} ± {mb['err']:.3f}")
a.axvline(qbeam, color=RED, ls=":", lw=1.2)
a.set_xscale("log"); a.set_ylim(0.6, 1.4); a.tick_params(labelsize=TICK)
a.set_xlabel("baseline length (kλ)", fontsize=LAB); a.set_ylabel("amplitude ratio compact / extended", fontsize=LAB)
a.legend(fontsize=10, frameon=False, loc="upper left"); a.set_title(f"(d) ratio where both are ≥ {SNR_MIN:.0f}x their noise floor", fontsize=LAB)
fig.savefig(os.path.join(OUT, "fig1_visibilities.png"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig1_visibilities.pdf"), dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved", os.path.join(OUT, "fig1_visibilities.png"))
