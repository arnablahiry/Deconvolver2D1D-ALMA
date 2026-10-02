"""
49: point-source response of the compact deconvolution (experiment 10). 12 unresolved sources with known
flux were injected into the real compact dirty cube through the compact PSF and deconvolved with the
same recipe; response = model(data + injected) - model(data), moment 0, 0.04" solve grid.
Top: responses for four S/N. Bottom: (a) recovered FWHM against S/N; (b) the point-source transfer
function |FT(response)| / flux against the beams.

    python scripts/superres_figures/fig49_point_injection.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E = os.path.join(REPO, "data/experiments/10_point_injection")
REF = os.path.join(REPO, "data/experiments/04_pd_tau_reweighting/compact/pdtau_k5.npy")
OUT = os.environ.get("OUT", os.path.join(REPO, "figures/experiments/10_point_injection/49_point_injection_compact.png"))
D = os.path.join(REPO, "data/ngc3110")
RES = json.load(open(os.path.join(E, "results.json")))["sources"]
vel = 5073.9 + 25.0 * (np.arange(32) - 15.5)
L = [i for i, v in enumerate(vel) if 4750 <= v <= 5280]
C, H, LBMAX_E = 0.04, 16, 0.1134
SPIKES = (0.05, 0.11)                     # FWHM range of the narrowest peaks in the model (not reproducible)
FS = 17
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": FS, "axes.labelsize": FS + 1,
                     "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True, "axes.linewidth": 1.3, "legend.fontsize": FS - 4})
beam = lambda c: (lambda h: (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"]))(fits.getheader(os.path.join(D, c, "ms_clean/image.fits")))
BE, BC = beam("extended"), beam("compact")
geo = lambda b: float(np.sqrt(b[0] * b[1]))
m0 = lambda p: (np.load(p).astype(np.float64)[L].sum(0) * 25.0).reshape(400, 2, 400, 2).sum((1, 3))
resp = m0(os.path.join(E, "pdtau_k5_injected.npy")) - m0(REF)
cut = lambda s, h=H: resp[s["j"] // 2 - h:s["j"] // 2 + h + 1, s["i"] // 2 - h:s["i"] // 2 + h + 1]
snr = np.array([s["snr_peak_channel"] for s in RES])
fw = np.array([s["fwhm_geo"] for s in RES])
ok = snr >= 8                                                             # S/N 5: not recovered (10% of the flux)

# ---- layout
Ps, gs, Pb, gb, YL, XL, TM, MID = 3.55, 0.3, 7.0, 1.6, 1.25, 0.95, 1.45, 1.05
W = YL + 2 * Pb + gb + 0.35
gs = (2 * Pb + gb - 4 * Ps) / 3
H_ = XL + Pb + MID + Ps + 0.55 + TM
fig = plt.figure(figsize=(W, H_))
yb, yt = XL, XL + Pb + MID
for n, s in enumerate([r for r in RES if r["snr_peak_channel"] in (12, 30, 130, 1000)]):
    a = fig.add_axes([(YL + n * (Ps + gs)) / W, yt / H_, Ps / W, Ps / H_])
    c = cut(s, 12)
    a.imshow(c / C ** 2, origin="lower", extent=[0.5, -0.5, -0.5, 0.5], cmap="inferno", vmin=0, vmax=c.max() / C ** 2,
             interpolation="nearest")
    a.text(0.05, 0.95, f"S/N {s['snr_peak_channel']}", transform=a.transAxes, color="w", va="top", fontsize=FS - 1, fontweight="bold")
    a.text(0.05, 0.05, f"FWHM {s['fwhm_geo']:.2f}$^{{\\prime\\prime}}$\nflux {100 * s['flux_ratio']:.0f}%", transform=a.transAxes,
           color="w", va="bottom", fontsize=FS - 3, fontweight="bold")
    a.set_xticks([0.4, 0, -0.4]); a.set_yticks([-0.4, 0, 0.4])
    a.set_xlabel(r"$\Delta$RA [$^{\prime\prime}$]", fontsize=FS - 2)
    if n == 0:
        a.set_ylabel(r"$\Delta$Dec [$^{\prime\prime}$]", fontsize=FS - 2)
    else:
        plt.setp(a.get_yticklabels(), visible=False)
fig.text((YL + 2 * Pb + gb) / 2 / W + YL / 2 / W, (yt + Ps + 0.12) / H_,
         "injected point sources in the compact deconvolution (response, moment 0, 0.04$^{\\prime\\prime}$ pixels; each its own scale)",
         ha="center", va="bottom", fontsize=FS, fontweight="bold")

# (a) FWHM vs S/N
a = fig.add_axes([YL / W, yb / H_, Pb / W, Pb / H_])
a.axhspan(*SPIKES, color="0.85", zorder=0)
a.text(9, np.mean(SPIKES), "narrowest peaks in the model\n(not reproducible)", va="center", fontsize=FS - 4, color="0.3")
a.semilogx(snr[ok], fw[ok], "o-", color="tab:red", ms=9, lw=2, label="injected point source, recovered")
for v, lab, st in ((geo(BE), "extended clean beam", "--"), (LBMAX_E, r"$\lambda/B_{\rm max}$ (extended)", "-."),
                   (C, "0.04$^{\\prime\\prime}$ solve pixel", ":")):
    a.axhline(v, color="k", ls=st, lw=1.5); a.text(1150, v + 0.005, lab, ha="right", va="bottom", fontsize=FS - 4)
a.set_xlim(6, 1300); a.set_ylim(0, 0.42)
a.set_xlabel("injected peak S/N per channel (compact dirty)"); a.set_ylabel(r"recovered FWHM [$^{\prime\prime}$]")
a.text(0.97, 0.97, f"(a) point-source size\ncompact clean beam: {geo(BC):.2f}$^{{\\prime\\prime}}$", transform=a.transAxes,
       ha="right", va="top", fontsize=FS, fontweight="bold")
a.legend(loc="upper left", bbox_to_anchor=(0.0, 0.8), frameon=True)

# (b) point-source transfer function
a = fig.add_axes([(YL + Pb + gb) / W, yb / H_, Pb / W, Pb / H_])
NP = 256                                                                  # zero-padded FFT (finer k sampling)
kx, ky = np.meshgrid(np.fft.fftfreq(NP, C), np.fft.fftfreq(NP, C))
kr = np.hypot(kx, ky).ravel()
edges = np.geomspace(0.15, 12.6, 22)
kc = np.sqrt(edges[1:] * edges[:-1])
yy, xx = np.indices((2 * H + 1, 2 * H + 1)) - H
ap = np.hypot(xx, yy) * C < 0.6


def tf(s):
    c = np.zeros((NP, NP)); c[NP // 2 - H:NP // 2 + H + 1, NP // 2 - H:NP // 2 + H + 1] = cut(s) * ap
    Fk = np.abs(np.fft.fft2(np.fft.ifftshift(c))).ravel() / c.sum()
    return np.array([Fk[(kr >= lo) & (kr < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])])


k = np.geomspace(0.1, 12.5, 400)
G = lambda t: np.exp(-(np.pi * t * k) ** 2 / (4 * np.log(2)))
hi_t = np.mean([tf(s) for s in RES if s["snr_peak_channel"] >= 130], axis=0)
lo_t = np.mean([tf(s) for s in RES if 12 <= s["snr_peak_channel"] <= 80], axis=0)
fhi = float(np.median(fw[snr >= 130]))
a.semilogx(k, G(geo(BC)), color="tab:blue", ls="--", lw=1.6, label=f"compact clean beam ({geo(BC):.2f}$^{{\\prime\\prime}}$)")
a.semilogx(kc, lo_t, "s-", color="tab:orange", ms=7, lw=2, label="point source, S/N 12-80")
a.semilogx(kc, hi_t, "o-", color="tab:red", ms=8, lw=2.6, label="point source, S/N 130-1000")
a.semilogx(k, G(geo(BE)), color="k", ls="--", lw=1.6, label=f"extended clean beam ({geo(BE):.2f}$^{{\\prime\\prime}}$)")
a.semilogx(k, G(LBMAX_E), color="0.45", ls="-.", lw=1.6, label=r"$\lambda/B_{\rm max}$" + f" ({LBMAX_E:.2f}$^{{\\prime\\prime}}$)")
a.semilogx(k, G(0.05), color="0.6", ls=":", lw=2, label="0.05$^{\\prime\\prime}$ (narrowest model peaks)")
a.axhline(0.5, color="0.75", lw=1, zorder=0)
a.set_xlim(0.1, 12.5); a.set_ylim(-0.05, 1.55)
a.set_xlabel(r"spatial frequency $k$ [cycles arcsec$^{-1}$]"); a.set_ylabel(r"transfer function $T(k)$")
a.text(0.03, 0.03, "(b) point-source beam in Fourier space", transform=a.transAxes, va="bottom", fontsize=FS, fontweight="bold")
a.legend(loc="upper right", frameon=True)
fig.text(0.5, 1 - 0.12 / H_, "NGC 3110 compact: 12 point sources injected into the real compact dirty cube, deconvolved with the same recipe\n"
         "primal-dual ($\\tau\\lambda$ reweighting), k = 5, 20,000 iterations; "
         f"bright sources come back at FWHM {fhi:.2f}$^{{\\prime\\prime}}$",
         ha="center", va="top", fontsize=FS - 1)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=60)
print("saved", OUT, "| median FWHM S/N>=130:", round(fhi, 3), "| T(k) hi:", np.round(hi_t, 2), "k", np.round(kc, 2))
