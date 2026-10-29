#!/usr/bin/env python
"""
Follow-up checks on cross_convolution.py (same inputs and cross-convolved cubes X_c = D_c * P_e, X_e = D_e * P_c):
  S1  the compact-extended offset measured on the emission only (moment-0 cross-correlation inside the 20" box,
      masked to where the mean moment 0 > 5 sigma), and the cube metrics after shifting X_e by it (Fourier shift)
  S2  moment-0 metrics (signal adds over channels): flux, RMSE, noise RMSE over the pixels where the mean > 5 sigma
  S3  null test: the same cube metrics in the line-free channels (no line: RMSE / flux there show the noise level)

    python scripts/experiments/cross_convolution_checks.py   -> results/NGC3110/figures/cross_convolution/checks.json
"""
import json
import os

import numpy as np
from astropy.io import fits
from numpy.fft import rfft2, irfft2, fftfreq, rfftfreq

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D_ = os.path.join(REPO, "data/ngc3110"); RES = os.path.join(REPO, "results/NGC3110")
OUT = os.path.join(RES, "figures", "cross_convolution")
L = json.load(open(os.path.join(RES, "line_channels.json")))["line"]
F = [c for c in range(32) if c not in L and 1 < c < 30]
rd = lambda p: np.nan_to_num(fits.getdata(p)).astype(np.float32).squeeze()
mad = lambda a: 1.4826 * np.median(np.abs(a - np.median(a)))
Dc, Pc = rd(f"{D_}/compact/dirty_big/dirty.fits"), rd(f"{D_}/compact/dirty_big/psf.fits")
De, Pe = rd(f"{D_}/extended/dirty_big/dirty.fits"), rd(f"{D_}/extended/dirty_big/psf.fits")
CELL = abs(fits.getheader(f"{D_}/compact/dirty_big/dirty.fits")["CDELT1"]) * 3600
N = Dc.shape[-1]; C0 = N // 2
yy, xx = np.indices((N, N)); RAD = np.hypot(yy - C0, xx - C0) * CELL
ANN = (RAD >= 30) & (RAD < 38)


def centred(P):
    pk = np.unravel_index(np.argmax(P[P.shape[0] // 2]), P.shape[1:])
    return np.roll(P, (-pk[0], -pk[1]), axis=(1, 2))


def xconv(A, P, ch):
    return irfft2(rfft2(A[ch]) * rfft2(centred(P)[ch]), s=A.shape[1:])


def metrics(a_cube, b_cube, Rg):
    a, b = a_cube[Rg], b_cube[Rg]
    sigd = np.median([mad(x[ANN]) for x in (a_cube - b_cube).reshape(-1, N, N)])
    s = float((a * b).sum() / (b * b).sum())
    return dict(voxels=int(Rg.sum()), flux_pct=float(100 * (a.sum() - b.sum()) / b.sum()),
                rmse_pct=float(100 * np.linalg.norm(a - b) / np.linalg.norm(b)),
                noise_rmse_pct=float(100 * sigd * np.sqrt(a.size) / np.linalg.norm(b)), scale=s,
                rmse_scaled_pct=float(100 * np.linalg.norm(a - s * b) / np.linalg.norm(s * b)))


def fshift(cube, dy, dx):
    ky = fftfreq(N)[:, None]; kx = rfftfreq(N)[None, :]
    return irfft2(rfft2(cube) * np.exp(-2j * np.pi * (ky * dy + kx * dx)), s=(N, N))


Xc, Xe = xconv(Dc, Pe, L), xconv(De, Pc, L)
sig = np.median([mad(x[ANN]) for x in 0.5 * (Xc + Xe)])
R5 = 0.5 * (Xc + Xe) > 5 * sig
out = {"unshifted": metrics(Xc, Xe, R5)}
# S1 offset on the emission only
m0c, m0e = Xc.sum(0), Xe.sum(0); s0 = mad(0.5 * (m0c + m0e)[ANN])
W = (0.5 * (m0c + m0e) > 5 * s0) & (RAD < 10)
best = (np.inf, 0, 0)
for dy in np.arange(-6, 6.01, 0.25):                                       # pixels (0.04")
    for dx in np.arange(-6, 6.01, 0.25):
        e = fshift(m0e, dy, dx)
        r = np.linalg.norm((m0c - e * (m0c[W] * e[W]).sum() / (e[W] ** 2).sum())[W])
        if r < best[0]:
            best = (r, dy, dx)
_, dy, dx = best
out["offset_mas"] = dict(dy=float(dy * CELL * 1e3), dx=float(dx * CELL * 1e3))
print(f"S1 offset (moment 0, emission only, least squares): dy {dy * CELL * 1e3:+.0f} mas, dx {dx * CELL * 1e3:+.0f} mas", flush=True)
Xe_s = np.stack([fshift(x, dy, dx) for x in Xe])
out["shifted"] = metrics(Xc, Xe_s, R5)
for k in ("unshifted", "shifted"):
    m = out[k]
    print(f"S1 cubes {k:9s}: flux {m['flux_pct']:+.2f}%, RMSE {m['rmse_pct']:.2f}% (noise {m['noise_rmse_pct']:.2f}%), "
          f"scale {m['scale']:.3f} -> {m['rmse_scaled_pct']:.2f}%", flush=True)
# S2 moment 0
for k, E in (("unshifted", Xe), ("shifted", Xe_s)):
    a, b = Xc.sum(0), E.sum(0)
    Rm = 0.5 * (a + b) > 5 * mad(0.5 * (a + b)[ANN])
    m = metrics(a[None], b[None], Rm[None]); m["area_arcsec2"] = float(Rm.sum() * CELL ** 2)
    out[f"moment0_{k}"] = m
    print(f"S2 moment 0 {k:9s} ({m['area_arcsec2']:.0f} arcsec2): flux {m['flux_pct']:+.2f}%, RMSE {m['rmse_pct']:.2f}% "
          f"(noise {m['noise_rmse_pct']:.2f}%), scale {m['scale']:.3f} -> {m['rmse_scaled_pct']:.2f}%", flush=True)
# S3 null test in the line-free channels, over the same spatial footprint
Fc, Fe = xconv(Dc, Pe, F), xconv(De, Pc, F)
foot = R5.any(0)
a, b = Fc[:, foot], Fe[:, foot]
out["null_linefree"] = dict(rms_diff_over_rms_mean=float(np.std(a - b) / np.std(0.5 * (a + b))), mean_c=float(a.mean()), mean_e=float(b.mean()),
                            rms_c=float(a.std()), rms_e=float(b.std()))
print(f"S3 null (line-free channels, same footprint): mean compact {a.mean() * 1e3:.3f}, extended {b.mean() * 1e3:.3f} (x1e-3); "
      f"rms of difference / rms of each {np.std(a - b) / np.std(0.5 * (a + b)):.2f}", flush=True)
json.dump(out, open(os.path.join(OUT, "checks.json"), "w"), indent=1)
