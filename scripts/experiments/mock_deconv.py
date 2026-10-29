#!/usr/bin/env python
"""
Injection test with a known sky, from NGC 3110's own data, to compare the 2D-1D dictionary with the 2D starlet per
channel (L1D = 0) as a function of S/N, without the noise of an observed reference:

  truth   the joint compact + extended MS-CLEAN components at the extended clean beam (ladder_joint_20kms/b10), convolved
          with that beam (0.22" x 0.17"), on the 0.08" grid (Jy/pixel), line channels only, times SCALE
  dirty   each channel of the truth convolved with the compact PSF of that channel (deconv_inputs/compact_0.08), plus
          PSF-filtered noise, independent per channel, at the line-free noise of the real compact dirty cube
  solve   exactly as pixel_scale_deconv.py SUPPORT=1 (J = 6, CDF 9/7 L1D levels, k = 5, 20000 iterations with
          reweighting, positivity, the compact MS-CLEAN support)

    SCALE=0.1 L1D=0 CUDA_VISIBLE_DEVICES=0 python scripts/experiments/mock_deconv.py
    -> results/NGC3110/mock/model_s<SCALE>_L<L1D>.npy (+ truth_s<SCALE>.npy, the noise-free truth), run json
"""
import json
import os
import sys
import time

import numpy as np
from astropy.io import fits
from scipy.ndimage import convolve1d

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
import gpu_pd as G  # noqa: E402
import torch  # noqa: E402
from restore import convolve_with_beam  # noqa: E402

SCALE = float(os.environ.get("SCALE", 1)); L1D = int(os.environ.get("L1D", 5)); K = 5.0; N_ITER = int(os.environ.get("N_ITER", 20000))
CELL = float(os.environ.get("MCELL", 0.08)); J = 6 if CELL == 0.08 else 5          # 0.16": 4x less GPU memory
TB = float(os.environ.get("TRUTH_BEAM", 0)) or None                        # the truth's resolution (circular, "); default: extended beam
TAG = f"c{CELL:g}" + (f"_t{TB:g}" if TB else "")
R = os.path.join(REPO, "results/NGC3110"); OUT = os.path.join(R, "mock"); os.makedirs(OUT, exist_ok=True)
grid = json.load(open(os.path.join(R, "consistency_20kms_0.08/grid.json"))); line, free = grid["line"], grid["line_free"]
rd = lambda p: np.nan_to_num(fits.getdata(p)).astype(np.float64).squeeze()
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] mock s{SCALE:g} L{L1D}: {m}", flush=True)

# ---------------------------------------------------------------- the truth (noise-free, 0.08" grid, Jy/pixel)
TF = os.path.join(OUT, f"truth_{TAG}.npy")
if not os.path.exists(TF):
    jd = os.path.join(R, "ladder_joint_20kms/b10/clean_finished"); h = fits.getheader(os.path.join(jd, "image.fits"))
    BE = (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"])
    m = rd(os.path.join(jd, "model.fits"))                               # Jy/pixel on 0.04" pixels, 1280 px
    f_ = int(round(CELL / 0.04)); k = np.r_[0.5, np.ones(f_ - 1), 0.5]     # summed onto the grid, pixel centres aligned
    m = convolve1d(convolve1d(m, k, axis=-1, mode="constant"), k, axis=-2, mode="constant")[..., ::f_, ::f_]
    if TB:
        BE = (TB, TB, 0.0)
    t = np.zeros_like(m); t[line] = np.clip(m[line], 0, None)
    om = np.pi * BE[0] * BE[1] / (4 * np.log(2))
    t = convolve_with_beam(t, CELL, *BE) * CELL ** 2 / om                 # the sky at the extended beam, Jy/pixel
    np.save(TF, t.astype(np.float32)); log(f"truth: {t[line].sum() * 20:.1f} Jy km/s, beam {BE[0]:.2f} x {BE[1]:.2f}")
truth = np.load(TF).astype(np.float64) * SCALE

# ---------------------------------------------------------------- the mock compact dirty cube
D_IN = os.path.join(R, f"deconv_inputs/compact_{CELL:g}")
psf = rd(os.path.join(D_IN, "dirty_beam_cube.fits")); real = rd(os.path.join(D_IN, "dirty_cube.fits"))
empty = np.nonzero(psf.max((1, 2)) < 0.5)[0]
for c in empty:
    psf[c] = psf[c - 1] if c > 0 else psf[c + 1]
n = truth.shape[-1]; c0 = psf.shape[-1] // 2
sig0 = float(np.median([real[c].std() for c in free if c not in empty]))
rng = np.random.default_rng(int(1000 * SCALE) + 7)
F2 = lambda a: np.fft.rfft2(np.fft.ifftshift(a))
dirty = np.zeros_like(truth)
for c in range(truth.shape[0]):
    P = psf[c][c0 - n // 2:c0 + n // 2, c0 - n // 2:c0 + n // 2]; fp = F2(P)
    if truth[c].any():
        dirty[c] = np.fft.irfft2(np.fft.rfft2(truth[c]) * fp, s=(n, n))
    e = np.fft.irfft2(np.fft.rfft2(rng.standard_normal((n, n))) * fp, s=(n, n))
    dirty[c] += sig0 * e / e.std()
for c in empty:
    dirty[c] = 0.0
log(f"dirty: noise {sig0 * 1e3:.3f} mJy/beam, peak S/N {dirty[line].max() / sig0:.0f}")

# ---------------------------------------------------------------- the deconvolution, as pixel_scale_deconv.py SUPPORT=1
t0 = time.time()
op = G.Op(psf, dirty); D = G.Dict2D1D(op.d.shape, J, L1D)
lam, _, _, _ = G.noise_lambdas(D, op.d, [c for c in free if c not in empty], k=K)
beta = G.field_lipschitz(op); tau = 1.0 / beta
msk = np.load(os.path.join(R, f"consistency_20kms_{CELL:g}/clean_compact_finished/mask_final.npy")).astype(np.float32)
sup = torch.tensor(msk, device=G.DEV).to(G.DT)
x, _, hist, _ = G.solve_pd(op, D, lam, n_iter=N_ITER, beta=beta, reweight=dict(start=6000, every=2000, eps=1.0, ref=tau * lam),
                           print_every=4000, log=log, support=sup)
xm = x.cpu().numpy().astype(np.float32)
np.save(os.path.join(OUT, f"model_{TAG}_s{SCALE:g}_L{L1D}.npy"), xm)
json.dump(dict(scale=SCALE, L1D=L1D, k=K, cell=CELL, truth_beam=TB, noise_mJy=sig0 * 1e3, peak_snr=float(dirty[line].max() / sig0),
               truth_flux=float(truth[line].sum() * 20), model_flux=float(xm[line].sum() * 20), seconds=round(time.time() - t0)),
          open(os.path.join(OUT, f"run_{TAG}_s{SCALE:g}_L{L1D}.json"), "w"), indent=1)
log(f"done: model {xm[line].sum() * 20:.1f} vs truth {truth[line].sum() * 20:.1f} Jy km/s, {time.time() - t0:.0f} s")
