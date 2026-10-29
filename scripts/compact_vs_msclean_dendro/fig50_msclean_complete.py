#!/usr/bin/env python
"""
50: the extended MS-CLEAN before (data/ngc3110/extended/ms_clean) and after it was finished inside the growing
dendrogram mask (scripts/msclean_complete.py -> msclean_complete): moment 0 of the emission only (image - residual =
components * clean beam, Jy/beam), of the residual, and the residual rms per channel against the line-free noise.

    python scripts/compact_vs_msclean_dendro/fig50_msclean_complete.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GAL_DIR = os.environ.get("GAL_DIR")                  # results/<galaxy>/ of scripts/pipeline/run_target.py
if GAL_DIR:
    GC = json.load(open(os.path.join(GAL_DIR, "config.json")))
    LABEL, EXT = GC["label"], os.path.join(GAL_DIR, "extended")
    BEFORE, AFTER = "msclean", "msclean_finished"
    OUT = os.path.join(GAL_DIR, "figures", "50_msclean_finished.png")
    CELL, DV, NCH = GC["cell_arcsec"], GC["width_kms"], GC["nchan"]
    vel = GC["vsys_kms"] + DV * (np.arange(NCH) - (NCH - 1) / 2)
    L = json.load(open(os.path.join(GAL_DIR, "line_channels.json")))["line"]
else:
    LABEL, EXT = "NGC 3110", os.path.join(REPO, "data/ngc3110/extended")
    BEFORE, AFTER = "ms_clean", "msclean_complete"
    OUT = os.path.join(REPO, "figures/compact_vs_msclean_dendro/50_msclean_complete.png")
    CELL, DV, NCH = 0.02, 25.0, 32
    vel = 5073.9 + 25.0 * (np.arange(32) - 15.5)
    L = [i for i, v in enumerate(vel) if 4750 <= v <= 5280]
FREE = [i for i in range(NCH) if i not in L]
rd = lambda p: np.nan_to_num(fits.getdata(p).astype(np.float64)).squeeze()
h = fits.getheader(os.path.join(EXT, AFTER, "image.fits"))
N = int(h["NAXIS1"]); C0 = N // 2
BEAM_PX = np.pi * h["BMAJ"] * h["BMIN"] * 3600 ** 2 / (4 * np.log(2)) / CELL ** 2
UNIT = r"Jy beam$^{-1}$ km s$^{-1}$"

runs = {}
for name, lab in ((BEFORE, "original MS-CLEAN (nsigma 3, auto-multithresh)"),
                  (AFTER, "finished in the dendrogram mask")):
    d = os.path.join(EXT, name)
    img, res = rd(os.path.join(d, "image.fits")), rd(os.path.join(d, "residual.fits"))
    em = img - res
    runs[name] = dict(lab=lab, em=em, res=res, m0=em[L].sum(0) * DV, r0=res[L].sum(0) * DV,
                      flux=em[L].sum() / BEAM_PX * DV, rflux=res[L].sum() / BEAM_PX * DV,
                      rms=1e3 * res.std(axis=(1, 2)))
sig = 1e3 * float(runs[BEFORE]["res"][FREE].std())
mf = os.path.join(EXT, AFTER, "mask_final.npy")
mask = np.load(mf if os.path.exists(mf) else os.path.join(EXT, AFTER, "mask_round1.npy")).astype(bool)[L].any(0)
rounds = json.load(open(os.path.join(EXT, AFTER, "rounds.json")))

plt.rcParams.update({"font.family": "serif", "font.size": 12, "xtick.direction": "in", "ytick.direction": "in"})
fig = plt.figure(figsize=(24, 11.5), constrained_layout=True)
gs = fig.add_gridspec(2, 5)
half = min(6.0 * CELL / 0.02 if GAL_DIR else 6.0, N * CELL / 2 - 0.1); hp = int(half / CELL)
sl = np.s_[C0 - hp:C0 + hp, C0 - hp:C0 + hp]; ext = [half, -half, -half, half]
vm = np.percentile(runs[AFTER]["m0"][sl], 99.9)
rl = np.percentile(np.abs(runs[BEFORE]["r0"][sl]), 99.5)
panels = [(runs[BEFORE]["m0"], f"original: emission only\n{runs[BEFORE]['flux']:.0f} Jy km/s", "inferno", 0, vm),
          (runs[AFTER]["m0"], f"finished: emission only\n{runs[AFTER]['flux']:.0f} Jy km/s", "inferno", 0, vm),
          (runs[AFTER]["m0"] - runs[BEFORE]["m0"], "added by finishing", "inferno", 0, 0.5 * vm),
          (runs[BEFORE]["r0"], f"original: residual ({runs[BEFORE]['rflux']:.0f} Jy km/s)", "RdBu_r", -rl, rl),
          (runs[AFTER]["r0"], f"finished: residual ({runs[AFTER]['rflux']:.0f} Jy km/s)", "RdBu_r", -rl, rl)]
for k, (im2, t, cm, lo, hi) in enumerate(panels):
    a = fig.add_subplot(gs[0, k])
    im = a.imshow(im2[sl], origin="lower", extent=ext, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
    a.contour(mask[sl].astype(float), levels=[0.5], colors="cyan" if cm == "RdBu_r" else "0.6", linewidths=0.6,
              extent=ext, origin="lower")
    a.set_title(t, fontsize=12); a.set_xlabel(r'$\Delta$RA ["]'); a.set_ylabel(r'$\Delta$Dec ["]')
    plt.colorbar(im, ax=a, shrink=0.85, label=UNIT)
a = fig.add_subplot(gs[1, :3])
a.axvspan(vel[L[0]] - DV / 2, vel[L[-1]] + DV / 2, color="0.5", alpha=0.1, lw=0)
for name, col in ((BEFORE, "0.45"), (AFTER, "#1a9e77")):
    a.plot(vel, runs[name]["rms"], "-o", color=col, lw=2, ms=6, label=runs[name]["lab"])
a.axhline(sig, color="k", lw=1, ls="--"); a.text(vel[-1], sig, f" line-free noise {sig:.3f}", ha="right", va="bottom")
a.set_xlabel(r"velocity [km s$^{-1}$]"); a.set_ylabel("residual rms per channel [mJy/beam]"); a.grid(alpha=0.25)
a.legend(frameon=False); a.set_title("residual rms per channel, whole 16″ field", fontsize=12)
a = fig.add_subplot(gs[1, 3:])
for name, col in ((BEFORE, "0.45"), (AFTER, "#1a9e77")):
    a.step(vel, runs[name]["em"].sum((1, 2)) / BEAM_PX, where="mid", color=col, lw=2, label=f"{runs[name]['lab']}")
a.set_xlabel(r"velocity [km s$^{-1}$]"); a.set_ylabel("flux density [Jy]"); a.grid(alpha=0.25)
a.set_title("integrated spectrum of the emission (image − residual)", fontsize=12); a.legend(frameon=False, fontsize=10)
steps = "; ".join(f"round {r['round']}: {r['comps_line_Jy_kms']:.0f} Jy km/s, resid rms {r['resid_rms_line_mJy']:.2f}" for r in rounds)
fig.suptitle(f"{LABEL} extended multiscale CLEAN, finished inside the growing astrodendro mask (outline) — emission only = "
             f"components ∗ clean beam ({h['BMAJ'] * 3600:.3f}″×{h['BMIN'] * 3600:.3f}″)\n{steps}", fontsize=13)
fig.savefig(OUT, dpi=65)
print("saved", OUT)
