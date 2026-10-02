"""
46: where the extended CLEAN's diffuse emission went. Top: extended CASA multiscale CLEAN --
clean components * clean beam | residual (cyan: union of the auto-multithresh mask) | restored image
(components + residual). Bottom: extended dirty cube | the compact deconvolved model observed by the
extended array (N_ext x_compact: convolved with the extended dirty beam) | difference.
Moment 0 over the line channels, central 10", all in Jy km/s arcsec^-2 (Jy/beam maps / beam area).

    MODEL=... OUT=... python scripts/superres_figures/fig46_extended_clean_budget.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import fftconvolve

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
from restore import convolve_with_beam  # noqa: E402

MODEL = os.environ.get("MODEL", os.path.join(REPO, "data/experiments/04_pd_tau_reweighting/compact/pdtau_k5.npy"))
OUT = os.environ.get("OUT", os.path.join(REPO, "figures/experiments/04_pd_tau_reweighting/5_sigma/46_extended_clean_budget.png"))
EC = os.path.join(REPO, "data/ngc3110/extended/ms_clean")
CELL = 0.02
vel = 5073.9 + 25.0 * (np.arange(32) - 15.5)
L = [i for i, v in enumerate(vel) if 4750 <= v <= 5280]
FS = 15
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": FS, "axes.labelsize": FS,
                     "xtick.labelsize": FS - 2, "ytick.labelsize": FS - 2, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True, "axes.linewidth": 1.2})
UNIT = r"Jy km s$^{-1}$ arcsec$^{-2}$"
rd = lambda p: np.nan_to_num(fits.getdata(p).astype(np.float64)).squeeze()
mom0 = lambda c: c[L].sum(0) * 25.0

h = fits.getheader(os.path.join(EC, "image.fits"))
BE = (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"])
area = np.pi * BE[0] * BE[1] / (4 * np.log(2))                    # arcsec^2
mod, res, img, msk = (rd(os.path.join(EC, f"{n}.fits")) for n in ("model", "residual", "image", "mask"))
dirty = rd(os.path.join(REPO, "data/ngc3110/extended/cube/dirty_cube.fits"))
psf = rd(os.path.join(REPO, "data/ngc3110/extended/cube/dirty_beam_cube.fits"))
xc = np.load(MODEL).astype(np.float64)
comp = convolve_with_beam(mom0(mod)[None], CELL, *BE)[0] / area
sim = sum(fftconvolve(xc[i], psf[i], mode="same") for i in L) * 25.0 / area
maps = [comp, mom0(res) / area, mom0(img) / area, mom0(dirty) / area, sim, (mom0(dirty) / area) - sim]
h2 = 250; s = np.s_[400 - h2:400 + h2, 400 - h2:400 + h2]; ext = [5, -5, -5, 5]
flux = lambda m: m[s].sum() * CELL ** 2                             # Jy km/s in the central 10"
mask_u = msk[L].max(0)[s]
titles = [("Extended CLEAN", "components $\\ast$ clean beam"), ("Extended CLEAN", "residual (cyan: CLEAN mask)"),
          ("Extended CLEAN", "image = components + residual"), ("Extended dirty", "observed"),
          ("Compact deconvolved", "observed by the extended array"), ("Difference", "dirty $-$ compact observed")]

P, g, CBH, CG = 5.2, 0.35, 0.2, 0.12
YL, XL, CBL, TM = 1.05, 0.72, 0.78, 1.05
W = YL + 3 * P + 2 * g + 0.25
H = TM + CBL + CBH + CG + P + g + P + XL + CG + CBH + CBL + 0.1
fig = plt.figure(figsize=(W, H))
ax_at = lambda x, y, w, hh, **kw: fig.add_axes([x / W, y / H, w / W, hh / H], **kw)
y_bot = 0.1 + CBL + CBH + CG + XL; y_top = y_bot + P + g
vtop = max(maps[0][s].max(), maps[2][s].max())
vdir = max(maps[3][s].max(), maps[4][s].max())
first = None
for k, (m, (t1, t2)) in enumerate(zip(maps, titles)):
    r, c = divmod(k, 3)
    x0, y0 = YL + c * (P + g), (y_top, y_bot)[r]
    a = ax_at(x0, y0, P, P, **({} if first is None else dict(sharex=first, sharey=first))); first = first or a
    if k in (1, 5):
        v = np.abs(m[s]).max() if k == 1 else vdir
        im = a.imshow(m[s], origin="lower", extent=ext, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
    else:
        v = vtop if r == 0 else vdir
        im = a.imshow(m[s], origin="lower", extent=ext, cmap="inferno", vmin=min(0, m[s].min()) if r else 0, vmax=v,
                      interpolation="nearest")
    if k == 1:
        a.contour(mask_u, levels=[0.5], extent=ext, origin="lower", colors="c", linewidths=1.2)
    tc = "k" if k in (1, 5) else "w"
    a.text(0.03, 0.97, t1, transform=a.transAxes, color=tc, fontsize=FS, fontweight="bold", va="top")
    a.text(0.03, 0.905, t2, transform=a.transAxes, color=tc, fontsize=FS - 3, fontweight="bold", va="top")
    a.text(0.03, 0.03, f"flux {flux(m):.0f} Jy km s$^{{-1}}$", transform=a.transAxes, color=tc, fontsize=FS - 3,
           fontweight="bold", va="bottom")
    a.set_xlim(5, -5); a.set_ylim(-5, 5); a.set_xticks([4, 2, 0, -2, -4]); a.set_yticks([-4, -2, 0, 2, 4])
    if r == 0:
        plt.setp(a.get_xticklabels(), visible=False)
    else:
        a.set_xlabel(r"$\Delta$RA [$^{\prime\prime}$]")
    if c > 0:
        plt.setp(a.get_yticklabels(), visible=False)
    else:
        a.set_ylabel(r"$\Delta$Dec [$^{\prime\prime}$]")
    cy = y0 + P + CG if r == 0 else y0 - XL - CG - CBH
    cax = ax_at(x0, cy, P, CBH); cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    pos = "top" if r == 0 else "bottom"
    cax.xaxis.set_ticks_position(pos); cax.xaxis.set_label_position(pos); cb.set_label(UNIT)
    cax.tick_params(which="both", direction="out", top=(r == 0), bottom=(r == 1), labeltop=(r == 0), labelbottom=(r == 1))
fig.text(0.5, 1 - 0.12 / H, "NGC 3110 CO(2-1), moment 0 (4750-5280 km/s), central 10$^{\\prime\\prime}$ -- extended multiscale CLEAN flux budget\n"
         "bottom: Jy/beam maps (dirty-beam weighted) divided by the clean-beam area; fluxes are approximate there",
         ha="center", va="top", fontsize=FS + 1)
fig.savefig(OUT, dpi=60)
print("saved", OUT)
for (t1, t2), m in zip(titles, maps):
    print(f"  {t1}: {t2:40s} {flux(m):7.1f} Jy km/s")
print("  CLEAN mask: fraction of the 10\" box (union over line channels)", round(float(mask_u.mean()), 3))
