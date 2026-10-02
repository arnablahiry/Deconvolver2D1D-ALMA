"""
Point sources to inject into the real compact dirty cube (experiment 10): 12 unresolved sources
(one 0.02" pixel), Gaussian spectra (FWHM 100 km/s, centred on the systemic velocity), peak
per-channel S/N 5 ... 1000 in the compact dirty cube, at the emptiest places of the reference model
at r = 5.5-7.5" (>= 2.5" apart, > 2x the compact beam). Output: the 0.02" model cube of the
sources (Jy/pixel), which run_model.py --inject puts through the compact PSF and adds to the data.

    python scripts/experiments/make_injection.py
"""
import json
import os

import numpy as np
from astropy.io import fits
from scipy.ndimage import uniform_filter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "data/experiments/10_point_injection")
REF = os.path.join(REPO, "data/experiments/04_pd_tau_reweighting/compact/pdtau_k5.npy")
vel = 5073.9 + 25.0 * (np.arange(32) - 15.5)
L = [i for i, v in enumerate(vel) if 4750 <= v <= 5280]
F = [i for i in range(32) if i not in L]
SNR = [5, 8, 12, 20, 30, 50, 80, 130, 200, 350, 600, 1000]

d = np.nan_to_num(fits.getdata(os.path.join(REPO, "data/ngc3110/compact/cube/dirty_cube.fits")).astype(np.float64)).squeeze()
sigma = float(d.reshape(32, 400, 2, 400, 2).mean((2, 4))[F].std())          # per channel, on the 0.04" solve grid
m = np.load(REF)[L].sum(0)
loc = uniform_filter(m, 61)                                                    # local model flux in 1.2" boxes
yy, xx = np.indices(m.shape)
r = np.hypot(xx - 400, yy - 400) * 0.02
cand = np.argwhere((r > 5.5) & (r < 7.5) & ((xx % 2) == 0) & ((yy % 2) == 0))   # coarse-pixel aligned
cand = cand[np.argsort(loc[cand[:, 0], cand[:, 1]])]
pos = []
for j, i in cand:
    if all(np.hypot(j - a, i - b) * 0.02 >= 2.5 for a, b in pos):
        pos.append((int(j), int(i)))
    if len(pos) == len(SNR):
        break
spec = np.exp(-4 * np.log(2) * (vel - 5073.9) ** 2 / 100.0 ** 2)
cube = np.zeros((32, 800, 800), np.float32)
src = []
for (j, i), s in zip(pos, SNR):
    cube[:, j, i] += (s * sigma * spec).astype(np.float32)
    src.append(dict(j=j, i=i, x=float(-(i - 400) * 0.02), y=float((j - 400) * 0.02), snr_peak_channel=s,
                    peak_jy=s * sigma, flux_mom0=float(s * sigma * spec[L].sum() * 25.0),
                    local_ref_flux=float(loc[j, i] * 61 ** 2)))
os.makedirs(OUT, exist_ok=True)
np.save(os.path.join(OUT, "inject_cube.npy"), cube)
json.dump(dict(sigma_chan=sigma, fwhm_kms=100.0, sources=src), open(os.path.join(OUT, "sources.json"), "w"), indent=1)
for s in src:
    print(s)
