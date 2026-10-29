"""
50: deconvolving to a target resolution vs convolving afterwards. The compact pdtau model (k = 5), raw; the
same model convolved with the lambda/B_max beam (fig 45, panel 3); the model deconvolved WITH a target
resolution of lambda/B_max (gpu_pd.Op target_fwhm_px: sky = G x, G a circular Gaussian of that FWHM), raw
-- no convolution afterwards; and target minus convolved. Moment 0 over the line channels, Jy km/s arcsec^-2.
Top: central 10"; bottom: the central bar, 3". Prints the size distribution of the peaks of each map.

    MODEL=... TARGET_MODEL=... OUT=... python scripts/superres_figures/fig50_target_resolution.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import maximum_filter
from scipy.optimize import curve_fit

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
from restore import convolve_with_beam  # noqa: E402

MODEL = os.environ.get("MODEL", os.path.join(REPO, "data/experiments/04_pd_tau_reweighting/compact/pdtau_k5.npy"))
TARGET_MODEL = os.environ.get("TARGET_MODEL", os.path.join(REPO, "data/experiments/11_target_resolution/pdtau_k5_t0.1134.npy"))
OUT = os.environ.get("OUT", os.path.join(REPO, "figures/experiments/11_target_resolution/50_target_resolution_pdtau_k5.png"))
LABEL = os.environ.get("LABEL", "pdtau k = 5")
CELL, LBMAX = 0.02, 0.1134
vel = 5073.9 + 25.0 * (np.arange(32) - 15.5)
L = [i for i, v in enumerate(vel) if 4750 <= v <= 5280]
h = fits.getheader(os.path.join(REPO, "data/ngc3110/extended/ms_clean/image.fits"))
BE = (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"])
s0 = LBMAX / np.sqrt(BE[0] * BE[1]); B1 = (BE[0] * s0, BE[1] * s0, BE[2])      # fig 45's lambda/B_max beam
area = lambda b: np.pi * b[0] * b[1] / (4 * np.log(2))                          # arcsec^2
m0 = lambda x: x[L].sum(0, keepdims=True) * 25.0                                # Jy/pixel km/s

x = np.load(MODEL).astype(np.float64)
y = np.load(TARGET_MODEL).astype(np.float64)
raw = m0(x)[0] / CELL ** 2
conv = convolve_with_beam(m0(x), CELL, *B1)[0] / area(B1)
targ = m0(y)[0] / CELL ** 2
diff = targ - conv


def g2(X, a, x0, y0, s1, s2, b):
    return a * np.exp(-0.5 * ((X[0] - x0) ** 2 / s1 ** 2 + (X[1] - y0) ** 2 / s2 ** 2)) + b


def peak_fwhm(m, frac=0.05, hw=8):
    """Gaussian FWHM (arcsec) of each local maximum above frac x the max, central 12"; 0.02" grid"""
    loc = (m == maximum_filter(m, size=7)) & (m > frac * m.max())
    out = []
    for j, i in zip(*np.nonzero(loc)):
        if not (100 <= j < 700 and 100 <= i < 700):
            continue
        cut = m[j - hw:j + hw + 1, i - hw:i + hw + 1]; yy, xx = np.indices(cut.shape) - hw
        try:
            p, _ = curve_fit(g2, (xx.ravel(), yy.ravel()), cut.ravel(), p0=(cut.max(), 0, 0, 2, 2, 0), maxfev=4000)
            fw = 2.3548 * np.sqrt(abs(p[3] * p[4])) * CELL
            if fw < 1.0:
                out.append(fw)
        except RuntimeError:
            pass
    return np.array(out)


bins = [0, 0.11, 0.2, 0.3, 1.0]
for name, m in (("raw model", raw), ("model * lambda/Bmax beam", conv), ("target lambda/Bmax", targ)):
    fw = peak_fwhm(m)
    hist = [int(((fw >= a) & (fw < b)).sum()) for a, b in zip(bins[:-1], bins[1:])]
    print(f"{name:26s} {len(fw):3d} peaks; FWHM <0.11\": {hist[0]}, 0.11-0.2: {hist[1]}, 0.2-0.3: {hist[2]}, >0.3: {hist[3]}; "
          f"narrowest {fw.min():.3f}\", flux {m.sum() * CELL ** 2:.1f} Jy km/s")

plt.rcParams.update({"font.family": "serif", "font.size": 12, "xtick.direction": "in", "ytick.direction": "in"})
fig, ax = plt.subplots(2, 4, figsize=(19, 9.6), constrained_layout=True)
vm = np.percentile(raw, 99.9)
vc = max(np.percentile(conv, 99.95), np.percentile(targ, 99.95))
dv = np.percentile(np.abs(diff), 99.9)
panels = [("raw model (0.02″ pixels)", raw, "inferno", 0, vm),
          (f"model ∗ λ/B$_{{\\rm max}}$ beam ({B1[0]:.3f}″×{B1[1]:.3f}″)", conv, "inferno", 0, vc),
          (f"deconvolved with target λ/B$_{{\\rm max}}$ = {LBMAX}″ (raw)", targ, "inferno", 0, vc),
          ("target − convolved", diff, "RdBu_r", -dv, dv)]
for row, (half, lab) in enumerate(((250, "central 10″"), (75, "bar, 3″"))):
    c = 400; sl = np.s_[c - half:c + half, c - half:c + half]; ext = [half * CELL, -half * CELL, -half * CELL, half * CELL]
    for k, (title, img, cm, lo, hi) in enumerate(panels):
        a = ax[row, k]
        im = a.imshow(img[sl], origin="lower", extent=ext, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
        a.set_title(title if row == 0 else lab, fontsize=12)
        a.set_xlabel(r'$\Delta$RA ["]'); a.set_ylabel(r'$\Delta$Dec ["]')
        fig.colorbar(im, ax=a, shrink=0.85, label=r"Jy km s$^{-1}$ arcsec$^{-2}$")
fig.suptitle(f"NGC 3110 compact, {LABEL}, moment 0: convolving with the λ/B$_{{\\rm max}}$ beam afterwards vs "
             "deconvolving to a λ/B$_{\\rm max}$ target resolution", fontsize=14)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=70)
print("saved", OUT)
