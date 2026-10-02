"""
45: the compact deconvolved model restored with a sequence of beams, from the resolution of the
extended array's longest baseline up to the extended clean beam, between the two CASA multiscale CLEAN
results -- all as surface brightness (Jy km/s arcsec^-2), moment 0 over the line channels, central 10".

Panels: compact CLEAN (clean components * compact clean beam, no residual) | compact model, raw |
compact model * {1, 1.25, 1.5} x lambda/B_max beam | compact model * extended clean beam |
extended CLEAN (clean components * extended clean beam, no residual) | extended CLEAN image (with residual).
The lambda/B_max beam keeps the extended clean beam's shape (axis ratio, PA), scaled so its geometric
mean is lambda/B_max of the extended array (0.1134" at 226.885 GHz, B_max = 2403.1 m).

    MODEL=... OUT=... LABEL=... python scripts/superres_figures/fig45_beam_sequence.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
from restore import convolve_with_beam  # noqa: E402

MODEL = os.environ.get("MODEL", os.path.join(REPO, "data/experiments/04_pd_tau_reweighting/compact/pdtau_k5.npy"))
OUT = os.environ.get("OUT", os.path.join(REPO, "figures/experiments/04_pd_tau_reweighting/5_sigma/45_beam_sequence_compact.png"))
LABEL = os.environ.get("LABEL", r"compact model: primal-dual ($\tau\lambda$ reweighting), k = 5, 20,000 iterations")
CLEAN = {c: os.path.join(REPO, f"data/ngc3110/{c}/ms_clean") for c in ("compact", "extended")}
CELL, LBMAX = 0.02, 0.1134
vel = 5073.9 + 25.0 * (np.arange(32) - 15.5)
L = [i for i, v in enumerate(vel) if 4750 <= v <= 5280]
FS = 15
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": FS, "axes.labelsize": FS,
                     "xtick.labelsize": FS - 2, "ytick.labelsize": FS - 2, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True, "axes.linewidth": 1.2})
UNIT = r"Jy km s$^{-1}$ arcsec$^{-2}$"
area = lambda b: np.pi * b[0] * b[1] / (4 * np.log(2))
restore = lambda m0, b: convolve_with_beam(m0, CELL, *b)[0] / area(b)        # Jy/pixel km/s -> Jy km/s arcsec^-2


def clean_components(cfg):
    """CASA clean components (Jy/pixel), moment 0, restored with that run's clean beam -- no residual"""
    h = fits.getheader(os.path.join(CLEAN[cfg], "image.fits"))
    b = (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"])
    mod = np.nan_to_num(fits.getdata(os.path.join(CLEAN[cfg], "model.fits")).astype(np.float64)).squeeze()
    return restore(mod[L].sum(0)[None] * 25.0, b), b


m0 = np.load(MODEL).astype(np.float64)[L].sum(0)[None] * 25.0
cc, bcc = clean_components("compact"); ce, BE = clean_components("extended")     # BE: the extended clean beam, from CASA
s0 = LBMAX / np.sqrt(BE[0] * BE[1]); b1 = (BE[0] * s0, BE[1] * s0, BE[2])
panels = [("Compact: CASA multiscale CLEAN", "clean components $\\ast$ compact clean beam", cc, bcc, False),
          ("Compact (deconvolved)", "raw model, 0.02$^{\\prime\\prime}$ pixels", m0[0] / CELL ** 2, None, True)]
for f in (1.0, 1.25, 1.5):
    b = (b1[0] * f, b1[1] * f, BE[2])
    panels.append(("Compact (deconvolved)", f"$\\ast$ {f:g} $\\times$ $\\lambda/B_{{\\rm max}}$ beam", restore(m0, b), b, False))
panels.append(("Compact (deconvolved)", "$\\ast$ extended clean beam", restore(m0, BE), BE, False))
panels.append(("Extended: CASA multiscale CLEAN", "clean components $\\ast$ extended clean beam", ce, BE, False))
img = np.nan_to_num(fits.getdata(os.path.join(CLEAN["extended"], "image.fits")).astype(np.float64)).squeeze()
panels.append(("Extended: CASA multiscale CLEAN", "CLEAN image (components + residual)", img[L].sum(0) * 25.0 / area(BE), BE, False))

# ---- exact layout (inches): square maps P, one gap g between all maps, shared axes,
#      horizontal colorbars above the top row and below the bottom row, each as wide as its map
P, g, CBH, CG = 5.2, 0.35, 0.2, 0.12            # map side, gap, colorbar height, colorbar-map gap
YL, XL, CBL, TM = 1.05, 0.72, 0.78, 1.05        # y-label width, x-label height, colorbar label height, title
W = YL + 4 * P + 3 * g + 0.25
H = TM + CBL + CBH + CG + P + g + P + XL + CG + CBH + CBL + 0.1
fig = plt.figure(figsize=(W, H))
ax_at = lambda x, y, w, hh, **kw: fig.add_axes([x / W, y / H, w / W, hh / H], **kw)
y_bot = 0.1 + CBL + CBH + CG + XL; y_top = y_bot + P + g
h = 250; s = np.s_[400 - h:400 + h, 400 - h:400 + h]; ext = [5, -5, -5, 5]
first = None
for k, (t1, t2, img, b, cap) in enumerate(panels):
    r, c = divmod(k, 4)
    x0, y0 = YL + c * (P + g), (y_top, y_bot)[r]
    a = ax_at(x0, y0, P, P, **({} if first is None else dict(sharex=first, sharey=first))); first = first or a
    vmax = np.percentile(img[s], 99.9) if cap else img[s].max()            # raw: a few bright pixels would set the scale
    im = a.imshow(img[s], origin="lower", extent=ext, cmap="inferno", vmin=min(0, img[s].min()) if "residual" in t2 else 0, vmax=vmax,
                 interpolation="nearest")                                # noisy CLEAN image: no 0 floor
    a.text(0.03, 0.97, t1, transform=a.transAxes, color="w", fontsize=FS, fontweight="bold", va="top")
    a.text(0.03, 0.905, t2 + ("; scale capped at 99.9%" if cap else ""), transform=a.transAxes, color="w", fontsize=FS - 3,
           fontweight="bold", va="top")
    if b is not None:
        big = b[0] > 1
        bx, by = (3.0, -3.2) if big else (3.3, -3.9)
        a.add_patch(Ellipse((bx, by), width=b[0], height=b[1], angle=90 - b[2], fc="none", ec="w", lw=1.6))
        a.text(bx, by - (0.85 if big else 0.35), f"{b[0]:.3f}$^{{\\prime\\prime}}$ $\\times$ {b[1]:.3f}$^{{\\prime\\prime}}$", color="w",
               ha="center", va="top", fontsize=FS - 4, fontweight="bold")
    for sp in a.spines.values():
        sp.set_visible(True)
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
    for sp in cax.spines.values():
        sp.set_visible(True)
fig.text(0.5, 1 - 0.12 / H, f"NGC 3110 CO(2-1), moment 0 (4750-5280 km/s), central 10$^{{\\prime\\prime}}$ -- {LABEL}\n"
         f"restoring beams from the extended array's longest-baseline resolution ($\\lambda/B_{{\\rm max}}$ = {LBMAX}$^{{\\prime\\prime}}$) "
         f"to the extended clean beam; CLEAN components shown without the residual, last panel: the CLEAN image", ha="center", va="top", fontsize=FS + 1)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=60)
print("saved", OUT, "| compact CLEAN beam", [round(v, 4) for v in bcc], "| extended clean beam", [round(v, 4) for v in BE])
