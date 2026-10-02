"""
48: the SHARPEST resolution of the compact deconvolved model -- per-knot effective resolution theta
(data/experiments/09_knot_resolution/knot_resolution.json, from knot_resolution.py), measured against
the independently observed extended dirty cube.
(a) raw compact model (moment 0, central 10"), knots coloured by theta;
(b) theta against the knot's extended S/N, real data and the simulation (noisy and exact reference);
(c) the corresponding beams in Fourier space: sharpest knot, median knot, the map-average (fig 47),
    the extended and compact clean beams, lambda/B_max.

    python scripts/superres_figures/fig48_knot_resolution.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from astropy.io import fits

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = json.load(open(os.path.join(REPO, "data/experiments/09_knot_resolution/knot_resolution.json")))
OUT = os.environ.get("OUT", os.path.join(REPO, "figures/experiments/04_pd_tau_reweighting/5_sigma/48_knot_resolution_compact.png"))
LABEL = os.environ.get("LABEL", r"compact model: primal-dual ($\tau\lambda$ reweighting), k = 5, 20,000 iterations")
D = os.path.join(REPO, "data/ngc3110")
CELL, N, AVG = 0.02, 800, float(os.environ.get("AVG_FWHM", 0.349))       # map-average beam from fig 47
LBMAX_E = 0.1134
vel = 5073.9 + 25.0 * (np.arange(32) - 15.5)
L = [i for i, v in enumerate(vel) if 4750 <= v <= 5280]
FS = 17
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": FS, "axes.labelsize": FS + 1,
                     "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True, "axes.linewidth": 1.3, "legend.fontsize": FS - 3})
beam = lambda c: (lambda h: (h["BMAJ"] * 3600, h["BMIN"] * 3600))(fits.getheader(os.path.join(D, c, "ms_clean/image.fits")))
BE, BC = beam("extended"), beam("compact")
geo = lambda b: float(np.sqrt(b[0] * b[1]))
good = lambda ks: [k for k in ks if not k["at_edge"]]
real, sim, sim0 = good(RES["real"]["knots"]), good(RES["sim"]["knots"]), RES["sim_exact"]["knots"]
th_r = np.array([k["theta"] for k in real])
best = real[int(np.argmin(th_r))]
med = float(np.median(th_r))
n_edge = len(RES["real"]["knots"]) - len(real)

# ---- layout: three square panels
P, g, YL, XL, TM, CB = 6.6, 1.25, 1.2, 0.95, 1.5, 0.75
W, H = YL + 3 * P + 2 * g + 0.3, XL + P + CB + TM
fig = plt.figure(figsize=(W, H))
ax = [fig.add_axes([(YL + i * (P + g)) / W, XL / H, P / W, P / H]) for i in range(3)]

# (a) raw model + knots
m0 = np.load(RES["model"]).astype(np.float64)[L].sum(0) * 25.0 / CELL ** 2
s = np.s_[150:650, 150:650]
a = ax[0]
a.imshow(m0[s], origin="lower", extent=[5, -5, -5, 5], cmap="gray_r", vmin=0, vmax=np.percentile(m0[s], 99.7), interpolation="nearest")
norm = Normalize(0.15, 0.5)
cmap = plt.get_cmap("plasma_r")
a.scatter([k["x"] for k in real], [k["y"] for k in real], s=170, marker="o", facecolors="none",
          edgecolors=cmap(norm(th_r)), linewidths=2.6)
sc = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
for k in RES["real"]["knots"]:
    if k["at_edge"]:
        a.plot(k["x"], k["y"], "x", color="0.4", ms=9, mew=2)
a.annotate(f"sharpest: {best['theta']:.2f}$^{{\\prime\\prime}}$", (best["x"], best["y"]), xytext=(best["x"] + 1.2, best["y"] + 1.3),
           fontsize=FS - 2, fontweight="bold", arrowprops=dict(arrowstyle="-", lw=1.4))
a.set_xlim(5, -5); a.set_ylim(-5, 5)
a.set_xlabel(r"$\Delta$RA [$^{\prime\prime}$]"); a.set_ylabel(r"$\Delta$Dec [$^{\prime\prime}$]")
a.text(0.03, 0.97, "(a) compact deconvolved (raw)", transform=a.transAxes, va="top", fontsize=FS, fontweight="bold")
a.text(0.03, 0.91, "knots of the extended data; x: fit at the grid edge", transform=a.transAxes, va="top", fontsize=FS - 4,
       fontweight="bold")
pos = a.get_position()
cax = fig.add_axes([pos.x0, pos.y1 + 0.12 / H, pos.width, 0.2 / H])
cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
cax.xaxis.set_ticks_position("top"); cax.xaxis.set_label_position("top")
cb.set_label(r"effective resolution $\theta$ [$^{\prime\prime}$]"); cax.tick_params(direction="out", top=True, bottom=False)

# (b) theta vs S/N
a = ax[1]
err = lambda ks: np.array([[k["theta"] - k["theta_16"] for k in ks], [k["theta_84"] - k["theta"] for k in ks]]).clip(0)
a.errorbar([k["snr"] for k in real], th_r, err(real), fmt="o", color="tab:red", ms=8, capsize=2, label="NGC 3110 (real)")
a.errorbar([k["snr"] for k in sim], [k["theta"] for k in sim], err(sim), fmt="s", color="tab:blue", ms=7, mfc="none", mew=1.6,
           capsize=2, label="simulation")
snr_sim = {(round(k["x"], 2), round(k["y"], 2)): k["snr"] for k in sim}
x0 = [snr_sim.get((round(k["x"], 2), round(k["y"], 2))) for k in sim0]
a.plot([x for x in x0 if x], [k["theta"] for k, x in zip(sim0, x0) if x], "+", color="tab:blue", ms=11, mew=1.8,
       label="simulation, exact (vs truth)")
for v, lab, st in ((geo(BE), "extended clean beam", "--"), (LBMAX_E, r"$\lambda/B_{\rm max}$", "-."), (AVG, "map average (fig. 47)", ":")):
    a.axhline(v, color="k", ls=st, lw=1.5); a.text(108, v + 0.006, lab, ha="right", va="bottom", fontsize=FS - 4)
a.set_xscale("log"); a.set_xlim(10, 110); a.set_ylim(0.0, 0.55)
a.set_xlabel("knot S/N (extended dirty, moment 0)"); a.set_ylabel(r"effective resolution $\theta$ (FWHM) [$^{\prime\prime}$]")
a.text(0.03, 0.97, "(b) per-knot resolution", transform=a.transAxes, va="top", fontsize=FS, fontweight="bold")
a.legend(loc="lower left", frameon=True)

# (c) Fourier space
a = ax[2]
k = np.geomspace(0.05, 25, 500)
G = lambda t: np.exp(-(np.pi * t * k) ** 2 / (4 * np.log(2)))
a.semilogx(k, G(geo(BC)), color="tab:blue", ls="--", lw=1.6, label=f"compact clean beam ({geo(BC):.2f}$^{{\\prime\\prime}}$)")
a.semilogx(k, G(AVG), color="tab:red", ls=":", lw=2.2, label=f"deconvolved, map average ({AVG:.2f}$^{{\\prime\\prime}}$)")
a.semilogx(k, G(med), color="tab:red", ls="--", lw=2.2, label=f"deconvolved, median knot ({med:.2f}$^{{\\prime\\prime}}$)")
a.semilogx(k, G(best["theta"]), color="tab:red", lw=3.0, label=f"deconvolved, sharpest knot ({best['theta']:.2f}$^{{\\prime\\prime}}$)")
a.semilogx(k, G(geo(BE)), color="k", ls="--", lw=1.6, label=f"extended clean beam ({geo(BE):.2f}$^{{\\prime\\prime}}$)")
a.semilogx(k, G(LBMAX_E), color="0.45", ls="-.", lw=1.6, label=r"$\lambda/B_{\rm max}$" + f" ({LBMAX_E:.2f}$^{{\\prime\\prime}}$)")
a.axhline(0.5, color="0.75", lw=1, zorder=0)
a.set_xlim(0.05, 25); a.set_ylim(-0.08, 1.9)
a.set_xlabel(r"spatial frequency $k$ [cycles arcsec$^{-1}$]"); a.set_ylabel(r"transfer function $T(k)$")
a.text(0.03, 0.03, "(c) beams in Fourier space", transform=a.transAxes, va="bottom", fontsize=FS, fontweight="bold")
a.legend(loc="upper right", frameon=True, fontsize=FS - 4)
fig.text(0.5, 1 - 0.12 / H, f"NGC 3110 CO(2-1), moment 0: sharpest resolution of the compact deconvolution, knot by knot -- {LABEL}\n"
         f"per knot: dirty$_{{\\rm ext}}$ $\\ast$ Gauss($\\theta$) = model $\\ast$ PSF$_{{\\rm ext}}$ in a 0.5$^{{\\prime\\prime}}$ window "
         f"(independent extended data, no CLEAN); {len(real)} knots, {n_edge} unconstrained",
         ha="center", va="top", fontsize=FS - 1)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=60)
print("saved", OUT, "| sharpest", best, "| median", round(med, 3), "| 5 sharpest", sorted(th_r)[:5])
