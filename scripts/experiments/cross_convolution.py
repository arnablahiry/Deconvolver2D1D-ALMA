#!/usr/bin/env python
"""
Are NGC 3110's compact and extended data the same at the compact resolution? A model-free test: for one sky S,

    D_c = S * P_c + n_c,   D_e = S * P_e + n_e   ->   X_c = D_c * P_e  and  X_e = D_e * P_c  both equal S * P_c * P_e

(+ noise): the same effective PSF (resolution ~ the compact beam) and the same effective uv weighting, the product of
the two samplings, i.e. only the spatial frequencies both arrays measure. No CLEAN, model, taper or mask enters.
Inputs: the 82" dirty cubes and PSFs of scripts/experiments/dirty_bigfield.py (natural weighting, same channels and
phase centre), convolved circularly (FFT, as tclean) with the PSF peak moved to pixel (0, 0).

Metrics over a region R of line voxels (R: where (X_c + X_e) / 2 > K sigma, sigma from an emission-free annulus):
  flux   100 (sum X_c - sum X_e) / sum X_e        RMSE  100 ||X_c - X_e|| / ||X_e||
  noise  the RMSE noise alone gives: 100 sigma_diff sqrt(N) / ||X_e||, sigma_diff the rms of X_c - X_e in the annulus
  scale  s minimising ||X_c - s X_e||, and the RMSE after it
Experiments (results/NGC3110/figures/cross_convolution/cross_convolution.json, and the figure):
  E1  main, K = 3, 5, 10 sigma          E2  per line channel          E3  both smoothed further (coarser beams)
  E4  spectral shift of +-1 channel     E5  spatial shift (moment-0 cross-correlation)
  E6  the same test on the old 16" field (both cubes and PSFs cropped), to show what the small field did

    python scripts/experiments/cross_convolution.py
"""
import json
import os
import sys

import numpy as np
from astropy.io import fits
from numpy.fft import rfft2, irfft2

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
from restore import convolve_with_beam  # noqa: E402

D_ = os.path.join(REPO, "data/ngc3110"); R = os.path.join(REPO, "results/NGC3110")
OUT = os.path.join(R, "figures", "cross_convolution"); os.makedirs(OUT, exist_ok=True)
L = json.load(open(os.path.join(R, "line_channels.json")))["line"]
DV = 25.0
rd = lambda p: np.nan_to_num(fits.getdata(p)).astype(np.float32).squeeze()
Dc, Pc = rd(f"{D_}/compact/dirty_big/dirty.fits"), rd(f"{D_}/compact/dirty_big/psf.fits")
De, Pe = rd(f"{D_}/extended/dirty_big/dirty.fits"), rd(f"{D_}/extended/dirty_big/psf.fits")
CELL = abs(fits.getheader(f"{D_}/compact/dirty_big/dirty.fits")["CDELT1"]) * 3600
N = Dc.shape[-1]; C0 = N // 2
yy, xx = np.indices((N, N)); RAD = np.hypot(yy - C0, xx - C0) * CELL                # arcsec from the phase centre


def centred(P):
    """PSF cube with its peak moved to pixel (0, 0), for circular convolution"""
    pk = np.unravel_index(np.argmax(P[P.shape[0] // 2]), P.shape[1:])
    return np.roll(P, (-pk[0], -pk[1]), axis=(1, 2))


def xconv(A, P):
    return irfft2(rfft2(A[L]) * rfft2(centred(P)[L]), s=A.shape[1:]).astype(np.float32)


def metrics(Xc, Xe, K=5.0, ann=(30, 38)):
    a = (RAD >= ann[0]) & (RAD < ann[1])
    sig = np.median([1.4826 * np.median(np.abs(x[a] - np.median(x[a]))) for x in 0.5 * (Xc + Xe)])
    sigd = np.median([1.4826 * np.median(np.abs(x[a] - np.median(x[a]))) for x in Xc - Xe])
    Rg = 0.5 * (Xc + Xe) > K * sig
    c, e = Xc[Rg], Xe[Rg]
    s = float((c * e).sum() / (e * e).sum())
    return dict(K=K, sigma=float(sig), sigma_diff=float(sigd), voxels=int(Rg.sum()),
                area_arcsec2=float(Rg.any(0).sum() * CELL ** 2),
                flux_pct=float(100 * (c.sum() - e.sum()) / e.sum()), rmse_pct=float(100 * np.linalg.norm(c - e) / np.linalg.norm(e)),
                noise_rmse_pct=float(100 * sigd * np.sqrt(c.size) / np.linalg.norm(e)), scale=s,
                rmse_scaled_pct=float(100 * np.linalg.norm(c - s * e) / np.linalg.norm(s * e))), Rg


def show(tag, m):
    print(f"{tag:34s} K {m['K']:>4}: area {m['area_arcsec2']:6.0f} arcsec2 | flux {m['flux_pct']:+6.2f}% | RMSE {m['rmse_pct']:5.2f}% "
          f"(noise {m['noise_rmse_pct']:4.2f}%) | scale {m['scale']:.3f} -> RMSE {m['rmse_scaled_pct']:5.2f}%", flush=True)


res = {}
Xc, Xe = xconv(Dc, Pe), xconv(De, Pc)
Peff = irfft2(rfft2(centred(Pc)[L[len(L) // 2]]) * rfft2(centred(Pe)[L[len(L) // 2]]), s=(N, N))
Peff = np.roll(Peff, (C0, C0), axis=(0, 1)); Peff /= Peff.max()
fwhm = 2 * np.sqrt((Peff > 0.5).sum() * CELL ** 2 / np.pi)
res["effective_beam_fwhm_arcsec"] = float(fwhm)
print(f"field {N} px x {CELL}\" = {N * CELL:.0f}\"; effective PSF P_c * P_e: FWHM ~{fwhm:.2f}\" (geometric mean), "
      f"min {Peff.min():+.3f}", flush=True)

# E1 main
res["E1"] = []
for K in (3.0, 5.0, 10.0):
    m, Rg = metrics(Xc, Xe, K); res["E1"].append(m); show("E1 main (82\" field)", m)
    if K == 5.0:
        R5 = Rg
# E2 per channel
res["E2"] = []
for i, c in enumerate(L):
    s = R5[i]
    if s.sum() < 50:
        continue
    a, b = Xc[i][s], Xe[i][s]
    res["E2"].append(dict(channel=c, flux_pct=float(100 * (a.sum() - b.sum()) / b.sum()),
                          rmse_pct=float(100 * np.linalg.norm(a - b) / np.linalg.norm(b))))
print("E2 per channel flux %:", [round(r["flux_pct"], 1) for r in res["E2"]], flush=True)
print("E2 per channel RMSE %:", [round(r["rmse_pct"], 1) for r in res["E2"]], flush=True)
# E3 extra smoothing
res["E3"] = []
for b in (0.5, 1.0, 1.5, 2.5):
    sc = convolve_with_beam(Xc, CELL, b, b, 0.0) if b else Xc
    se = convolve_with_beam(Xe, CELL, b, b, 0.0) if b else Xe
    m, _ = metrics(sc, se, 5.0); m["extra_fwhm"] = b; m["total_fwhm"] = float(np.hypot(fwhm, b)); res["E3"].append(m)
    show(f"E3 + {b}\" (total ~{np.hypot(fwhm, b):.2f}\")", m)
# E4 spectral shift
res["E4"] = {}
for k in (-1, 0, 1):
    a, b = (Xc[max(0, k):len(L) + min(0, k)], Xe[max(0, -k):len(L) + min(0, -k)])
    m, _ = metrics(a, b, 5.0); res["E4"][k] = m; show(f"E4 compact ch = extended ch {k:+d}", m)
# E5 spatial shift
m0c, m0e = Xc.sum(0), Xe.sum(0)
cc = irfft2(rfft2(m0c) * np.conj(rfft2(m0e)), s=(N, N)); iy, ix = np.unravel_index(np.argmax(cc), cc.shape)
sub = lambda v, i: i + 0.5 * (v[(i - 1) % N] - v[(i + 1) % N]) / (v[(i - 1) % N] - 2 * v[i] + v[(i + 1) % N])
dy, dx = (sub(cc[:, ix], iy) + C0) % N - C0, (sub(cc[iy, :], ix) + C0) % N - C0
res["E5_shift_mas"] = [float(dy * CELL * 1e3), float(dx * CELL * 1e3)]
print(f"E5 spatial offset compact vs extended: dy {dy * CELL * 1e3:+.1f} mas, dx {dx * CELL * 1e3:+.1f} mas", flush=True)
# E6 the old 16" field
h = int(round(8 / CELL)); cut = np.s_[:, C0 - h:C0 + h, C0 - h:C0 + h]
RAD_big = RAD; RAD = RAD_big[C0 - h:C0 + h, C0 - h:C0 + h]
Xc16, Xe16 = xconv(Dc[cut], Pe[cut]), xconv(De[cut], Pc[cut])
res["E6"] = []
for K in (5.0,):
    m, _ = metrics(Xc16, Xe16, K, ann=(6.5, 8)); res["E6"].append(m); show("E6 old 16\" field (cropped)", m)
RAD = RAD_big
json.dump(res, open(os.path.join(OUT, "cross_convolution.json"), "w"), indent=1)
np.savez_compressed(os.path.join(OUT, "moment0.npz"), Xc=m0c * DV, Xe=m0e * DV, R5=R5.any(0), cell=CELL)
print("saved", OUT, flush=True)
