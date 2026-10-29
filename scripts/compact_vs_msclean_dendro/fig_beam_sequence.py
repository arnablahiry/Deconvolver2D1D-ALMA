#!/usr/bin/env python
"""
How the compact deconvolution's agreement with the extended MS-CLEAN depends on the resolution of the comparison, on
one fixed region: the mask of the finished extended MS-CLEAN (extended/msclean_finished/mask_final.npy).

A sequence of circular beams, log-spaced in FWHM from the compact clean beam (geometric mean) down to the extended clean
beam (geometric mean). At each beam b, C and K are the compact deconvolved model and the extended MS-CLEAN components,
each x the mask, * b. Then
  cube      100 ||C - K|| / ||K|| over every masked line voxel                                  (%)
  moment 0  the same on the moment-0 maps, over the mask's footprint                             (%)
  moment 1  rms of C - K in km/s (intensity-weighted mean velocity, positive voxels), over the footprint pixels where
            both moment-0 maps exceed 5% of CLEAN's peak
  moment 2  the same for the intensity-weighted dispersion                                      (km/s)

Writes <GAL_DIR>/figures/beam_sequence/rmse_vs_beam.png and beam_sequence.json.

    GAL_DIR=results/NGC3110 [NBEAM=12] [CLEAN=msclean_finished] [MASK=...] [MASK_DESC=...] [OUT_DIR=...] \
        python scripts/compact_vs_msclean_dendro/fig_beam_sequence.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
from restore import convolve_with_beam  # noqa: E402

R = os.environ.get("GAL_DIR", os.path.join(REPO, "results/NGC3110"))
NBEAM = int(os.environ.get("NBEAM", 12))
OUT = os.environ.get("OUT_DIR", os.path.join(R, "figures", "beam_sequence")); os.makedirs(OUT, exist_ok=True)
CLEAN = os.environ.get("CLEAN", "msclean_finished")
MASK = os.environ.get("MASK", os.path.join(R, "extended/msclean_finished/mask_final.npy"))
MASK_DESC = os.environ.get("MASK_DESC", "the finished extended MS-CLEAN")
GC = json.load(open(os.path.join(R, "config.json")))
CELL, DV, NCH = GC["cell_arcsec"], GC["width_kms"], GC["nchan"]
vel = GC["vsys_kms"] + DV * (np.arange(NCH) - (NCH - 1) / 2)
L = json.load(open(os.path.join(R, "line_channels.json")))["line"]
BLUE, ORANGE, GREEN, PURPLE, MUTED = "#2a6fdb", "#e07b00", "#1a9e77", "#7a4fbf", "#8a8a8a"
TICK, LAB = 15, 16
plt.rcParams.update({"font.family": "serif", "font.size": 13, "xtick.direction": "in", "ytick.direction": "in"})
rd = lambda p: np.nan_to_num(fits.getdata(p).astype(np.float64)).squeeze()
geo = lambda b: float(np.sqrt(b[0] * b[1]))

hE = fits.getheader(os.path.join(R, "extended", CLEAN, "image.fits"))
BE = (hE["BMAJ"] * 3600, hE["BMIN"] * 3600, hE["BPA"])
hC = fits.getheader(os.path.join(R, "compact/msclean/image.fits"))
BC = (hC["BMAJ"] * 3600, hC["BMIN"] * 3600, hC["BPA"])
FWHM = np.geomspace(geo(BC), geo(BE), NBEAM)                               # compact -> extended
M = np.load(MASK).astype(bool)
MV = M[L]; FOOT = MV.any(0)
xc = np.load(os.path.join(R, "compact/deconvolved/model.npy")).astype(np.float64)[L] * MV
xk = rd(os.path.join(R, "extended", CLEAN, "model.fits"))[L] * MV
print(f"{GC['label']}: {NBEAM} circular beams {FWHM[0]:.3f}\" -> {FWHM[-1]:.3f}\", fixed mask {MV.sum()} voxels "
      f"({FOOT.sum() * CELL ** 2:.1f} arcsec2)", flush=True)


def moments(cube, sel):
    """intensity-weighted mean velocity and dispersion over the line channels (positive voxels), on pixels `sel`"""
    w = np.clip(cube, 0, None); s = np.maximum(w.sum(0), 1e-30)
    v = vel[L][:, None, None]
    m1 = (w * v).sum(0) / s
    m2 = np.sqrt(np.clip((w * (v - m1) ** 2).sum(0) / s, 0, None))
    return m1[sel], m2[sel]


rows = []
for b in FWHM:
    C, K = convolve_with_beam(xc, CELL, b, b, 0.0), convolve_with_beam(xk, CELL, b, b, 0.0)
    cube = 100 * np.linalg.norm((C - K)[MV]) / np.linalg.norm(K[MV])
    C0, K0 = C.sum(0) * DV, K.sum(0) * DV
    mom0 = 100 * np.linalg.norm((C0 - K0)[FOOT]) / np.linalg.norm(K0[FOOT])
    sel = FOOT & (C0 > 0.05 * K0.max()) & (K0 > 0.05 * K0.max())
    (c1, c2), (k1, k2) = moments(C, sel), moments(K, sel)
    rows.append(dict(fwhm=float(b), rmse_cube=float(cube), rmse_mom0=float(mom0),
                     rmse_mom1_kms=float(np.sqrt(np.mean((c1 - k1) ** 2))),
                     rmse_mom2_kms=float(np.sqrt(np.mean((c2 - k2) ** 2))), mom12_pixels=int(sel.sum()),
                     # as percentages: moment 1 relative to CLEAN's velocity field about v_sys (the rotation signal),
                     # moment 2 relative to CLEAN's dispersion
                     rmse_mom1=float(100 * np.linalg.norm(c1 - k1) / np.linalg.norm(k1 - GC["vsys_kms"])),
                     rmse_mom2=float(100 * np.linalg.norm(c2 - k2) / np.linalg.norm(k2))))
    print("  " + ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in rows[-1].items()), flush=True)
json.dump(dict(galaxy=GC["name"], mask=os.path.relpath(MASK, R), clean=CLEAN, compact_beam=BC, extended_beam=BE,
               lbmax_compact=GC.get("lbmax_compact_arcsec"), rows=rows),
          open(os.path.join(OUT, "beam_sequence.json"), "w"), indent=1)

# ======================================================================== figure: two panels sharing the beam axis
f = np.array([r["fwhm"] for r in rows])
MARKS = [(geo(BC), "compact clean beam"), (GC.get("lbmax_compact_arcsec"), "λ/B$_{\\rm max,\\,compact}$"),
         (geo(BE), "extended clean beam")]
fig, a = plt.subplots(figsize=(10, 7), constrained_layout=True)
for key, mk, col, lab in (("rmse_cube", "-o", BLUE, "cube (every voxel)"), ("rmse_mom0", "-s", ORANGE, "moment 0"),
                          ("rmse_mom1", "-^", GREEN, "moment 1 (velocity, relative to v − v$_{\\rm sys}$)"),
                          ("rmse_mom2", "-D", PURPLE, "moment 2 (dispersion)")):
    a.plot(f, [r[key] for r in rows], mk, color=col, lw=2.2, ms=7, label=lab)
a.set_ylabel("relative RMSE [%]", fontsize=LAB, labelpad=8)
a.set_xlabel("comparison beam FWHM [″] (circular)", fontsize=LAB, labelpad=8)
a.set_xscale("log"); a.set_ylim(bottom=0)
a.tick_params(which="both", direction="in", top=True, right=True)
a.tick_params(which="major", labelsize=TICK, pad=6, length=7); a.tick_params(which="minor", length=4)
a.set_xticks([1.0, 0.7, 0.5, 0.3, 0.2])
a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a.invert_xaxis()                                                           # compact (coarse) left -> extended right
for xv, t in MARKS:
    if xv:
        a.axvline(xv, color=MUTED, lw=1, ls=":")
        a.text(xv, 0.03, t + " ", rotation=90, va="bottom", ha="right", fontsize=12, color="0.35",
               transform=a.get_xaxis_transform())
a.legend(frameon=False, fontsize=13, loc="upper center")
a.set_title(f"{GC['label']}: agreement vs comparison resolution\n(fixed mask: " + MASK_DESC + ")",
            fontsize=14, fontweight="bold")
for old in ("rmse_voxel_vs_beam.png", "rmse_moments_vs_beam.png"):
    if os.path.exists(os.path.join(OUT, old)):
        os.remove(os.path.join(OUT, old))
fig.savefig(os.path.join(OUT, "rmse_vs_beam.png"), dpi=80); plt.close(fig)
print("saved", os.path.join(OUT, "rmse_vs_beam.png"))
