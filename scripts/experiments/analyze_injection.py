"""
Experiment 10: response of the compact deconvolution (pdtau, k = 5) to injected point sources.
response = model(data + N x_inj) - model(data), moment 0 of the line channels, on the 0.04" solve grid.
Per source: recovered flux (r < 0.6"), FWHM of a 2D Gaussian fit, peak offset, fraction of the
flux in the brightest pixel, and the point-source transfer function T(k) = |FT(response)| / flux.

    python scripts/experiments/analyze_injection.py -> data/experiments/10_point_injection/results.json
"""
import json
import os

import numpy as np
from scipy.optimize import curve_fit

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E = os.path.join(REPO, "data/experiments/10_point_injection")
REF = os.path.join(REPO, "data/experiments/04_pd_tau_reweighting/compact/pdtau_k5.npy")
vel = 5073.9 + 25.0 * (np.arange(32) - 15.5)
L = [i for i, v in enumerate(vel) if 4750 <= v <= 5280]
C, H = 0.04, 16                                                     # solve-grid cell, cutout half-size [px]
m0 = lambda p: (np.load(p).astype(np.float64)[L].sum(0) * 25.0).reshape(400, 2, 400, 2).sum((1, 3))
resp = m0(os.path.join(E, "pdtau_k5_injected.npy")) - m0(REF)
src = json.load(open(os.path.join(E, "sources.json")))["sources"]
y, x = np.indices((2 * H + 1, 2 * H + 1)) - H
r = np.hypot(x, y) * C
kx, ky = np.meshgrid(np.fft.fftfreq(2 * H + 1, C), np.fft.fftfreq(2 * H + 1, C))
kr = np.hypot(kx, ky)
kb = np.linspace(0, 12.5, 21)


def g2(X, a, x0, y0, s1, s2, b):
    return a * np.exp(-0.5 * ((X[0] - x0) ** 2 / s1 ** 2 + (X[1] - y0) ** 2 / s2 ** 2)) + b


out = []
for s in src:
    j, i = s["j"] // 2, s["i"] // 2
    cut = resp[j - H:j + H + 1, i - H:i + H + 1]
    f = float(cut[r < 0.6].sum())
    try:
        p, _ = curve_fit(g2, (x.ravel(), y.ravel()), cut.ravel(), p0=[cut.max(), 0, 0, 1, 1, 0],
                         bounds=([0, -5, -5, 0.15, 0.15, -np.inf], [np.inf, 5, 5, 15, 15, np.inf]))
        fw = [2.3548 * p[3] * C, 2.3548 * p[4] * C]; off = float(np.hypot(p[1], p[2]) * C)
    except RuntimeError:
        fw, off = [np.nan, np.nan], np.nan
    Fk = np.abs(np.fft.fft2(np.fft.ifftshift(cut * (r < 0.6))))
    T = [float(Fk[(kr >= a) & (kr < b)].mean() / max(f, 1e-12)) for a, b in zip(kb[:-1], kb[1:])]
    out.append(dict(s, flux_rec=f, flux_ratio=f / s["flux_mom0"], fwhm=fw, fwhm_geo=float(np.sqrt(fw[0] * fw[1])),
                    offset=off, peak_pixel_frac=float(cut.max() / f) if f > 0 else np.nan, T=T))
    o = out[-1]
    print(f"S/N {s['snr_peak_channel']:5d}: flux ratio {o['flux_ratio']:5.2f}  FWHM {o['fwhm_geo']:.3f}\" "
          f"({fw[0]:.3f} x {fw[1]:.3f})  offset {off * 1000:5.1f} mas  peak-pixel fraction {o['peak_pixel_frac']:.2f}  "
          f"T(k=5) {T[8]:.2f}")
json.dump(dict(k_bins=kb.tolist(), cell=C, sources=out), open(os.path.join(E, "results.json"), "w"), indent=1)
