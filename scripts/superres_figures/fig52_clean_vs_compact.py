"""
52: extended multiscale CLEAN parameter study. Each CLEAN's components convolved with the extended clean
beam, against the compact deconvolved model convolved with the same beam (the independent reference; it is
used only to score the CLEANs, never to constrain them: every CLEAN here uses the extended data alone).
Moment 0 over the line channels, central 10", Jy km/s arcsec^-2. Top: the maps; bottom: CLEAN - compact.
Prints per CLEAN: component flux, flux left in the residual (approximate: Jy/beam / beam area), relative
rms difference and correlation with the compact reference, and the CLEAN mask's coverage.

    CLEANS=ms_clean,msclean_deep_auto,... python scripts/superres_figures/fig52_clean_vs_compact.py
"""
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

EXT = os.path.join(REPO, "data/ngc3110/extended")
CLEANS = os.environ.get("CLEANS", "ms_clean,msclean_auto_n2,msclean_deep_auto,msclean_nomask_n2").split(",")
LABELS = {"ms_clean": "original: pipeline auto-mask,\nnsigma 3, scales 0-60 px",
          "msclean_auto_n2": "pipeline auto-mask,\nnsigma 2, scales 0-120 px",
          "msclean_deep_auto": "looser auto-mask,\nnsigma 1.5, scales 0-120 px",
          "msclean_nomask_n2": "no mask,\nnsigma 2, scales 0-120 px"}
REF = os.environ.get("REF", os.path.join(REPO, "data/experiments/04_pd_tau_reweighting/compact/pdtau_k5.npy"))
OUT = os.environ.get("OUT", os.path.join(REPO, "figures/experiments/14_clean_parameters/52_clean_vs_compact.png"))
CELL = 0.02
vel = 5073.9 + 25.0 * (np.arange(32) - 15.5)
L = [i for i, v in enumerate(vel) if 4750 <= v <= 5280]
C10 = np.s_[150:650, 150:650]
h = fits.getheader(os.path.join(EXT, "ms_clean/image.fits"))
BE = (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"])
area = np.pi * BE[0] * BE[1] / (4 * np.log(2))                                     # arcsec^2
rd = lambda p: np.nan_to_num(fits.getdata(p).astype(np.float64)).squeeze()
m0 = lambda c: c[L].sum(0) * 25.0


def at_beam(x):
    """Jy/pixel cube -> moment 0 at the extended clean beam, Jy km/s arcsec^-2"""
    return m0(convolve_with_beam(x, CELL, *BE)) / area


ref = at_beam(np.load(REF).astype(np.float64))[C10]
rows = []
for name in CLEANS:
    d = os.path.join(EXT, name)
    if not os.path.exists(os.path.join(d, "model.fits")):
        print("missing", d); continue
    comp = at_beam(rd(os.path.join(d, "model.fits")))[C10]
    resid = m0(rd(os.path.join(d, "residual.fits")))[C10] / area
    mask = rd(os.path.join(d, "mask.fits")) > 0
    diff = comp - ref
    r = dict(flux=comp.sum() * CELL ** 2, resid=resid.sum() * CELL ** 2, rel=np.linalg.norm(diff) / np.linalg.norm(ref),
             corr=np.corrcoef(comp.ravel(), ref.ravel())[0, 1], mask=mask.mean(), mask_line=mask[L].mean())
    rows.append((name, comp, diff, r))
print(f"compact reference (x extended beam): {ref.sum() * CELL ** 2:.0f} Jy km/s in the central 10\"")
for name, _, _, r in rows:
    print(f"{name:20s} components {r['flux']:6.0f} Jy km/s ({r['flux'] / (ref.sum() * CELL ** 2):.0%} of the reference), "
          f"residual {r['resid']:5.0f} | rel. rms difference {r['rel']:.3f}, correlation {r['corr']:.3f} | "
          f"mask {r['mask']:.1%} of voxels ({r['mask_line']:.1%} of line channels)")

plt.rcParams.update({"font.family": "serif", "font.size": 11, "xtick.direction": "in", "ytick.direction": "in"})
n = len(rows) + 1
fig, ax = plt.subplots(2, n, figsize=(4.6 * n + 1, 9.4), constrained_layout=True)
ext = [5, -5, -5, 5]; vm = np.percentile(ref, 99.9); dv = 0.5 * vm
ax[0, 0].imshow(ref, origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=vm)
ax[0, 0].set_title("compact pdtau * extended beam\n(reference)", fontsize=11)
ax[1, 0].axis("off")
for c, (name, comp, diff, r) in enumerate(rows, start=1):
    ax[0, c].imshow(comp, origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=vm)
    ax[0, c].set_title(f"{LABELS.get(name, name)}: components {r['flux']:.0f}, residual {r['resid']:.0f} Jy km/s", fontsize=10)
    im = ax[1, c].imshow(diff, origin="lower", extent=ext, cmap="RdBu_r", vmin=-dv, vmax=dv)
    ax[1, c].set_title(f"CLEAN − compact: rel. rms {r['rel']:.2f}, corr {r['corr']:.2f}", fontsize=10)
for a in ax.ravel():
    a.set_xlabel(r'$\Delta$RA ["]'); a.set_ylabel(r'$\Delta$Dec ["]')
fig.colorbar(im, ax=ax[1, 1:], shrink=0.8, label=r"Jy km s$^{-1}$ arcsec$^{-2}$")
fig.suptitle("NGC 3110 extended multiscale CLEAN (components ∗ extended clean beam) vs the compact deconvolved model, "
             "moment 0, central 10″", fontsize=13)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=60)
print("saved", OUT)
