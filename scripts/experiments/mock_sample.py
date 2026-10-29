#!/usr/bin/env python
"""
The injection test of mock_deconv.py for any target T of the sample, on its 20 km/s, 0.16" compact products
(results/<T>/g20_c0.16: gallery20_casa.py CHANKMS=20 GCELL=0.16), to compare the 2D-1D dictionary with the 2D starlet
per channel at a set S/N, over noise realisations and threshold multiples:

  truth   the target's joint compact + extended MS-CLEAN components at its extended clean beam (g20/joint_ext; NGC 3110:
          ladder_joint_20kms/b10), convolved to max(extended beam, 0.33") (sampled by the 0.16" grid), line channels only,
          scaled so that the noise-free mock dirty cube has a peak of SNR times the noise
  dirty   each channel of the truth convolved with the compact PSF of that channel, plus PSF-filtered noise, independent
          per channel (seed SEED), at the line-free noise of the target's real compact dirty cube
  solve   2D-1D (L1D = 5) or the 2D starlet per channel (L1D = 0), J = 5, threshold k = K, 20000 iterations with
          reweighting, positivity, the support of the compact MS-CLEAN (g20_c0.16/clean_compact_finished)

    T=IC5179 SNR=33 SEED=1 L1D=0 K=4 CUDA_VISIBLE_DEVICES=0 python scripts/experiments/mock_sample.py
    -> results/<T>/mock_sample/model_snr<SNR>_seed<SEED>_L<L1D>_k<K>.npy, truth.npy (unscaled), run json
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

T = os.environ["T"]; SNR = float(os.environ["SNR"]); SEED = int(os.environ.get("SEED", 1)); L1D = int(os.environ.get("L1D", 5))
K = float(os.environ.get("K", 5)); N_ITER = int(os.environ.get("N_ITER", 20000)); CELL, J = 0.16, 5
R = os.path.join(REPO, "results", T); GD = os.path.join(R, "g20_c0.16"); OUT = os.path.join(R, "mock_sample"); os.makedirs(OUT, exist_ok=True)
grid = json.load(open(os.path.join(GD, "grid.json"))); line, free = grid["line"], grid["line_free"]
rd = lambda p: np.nan_to_num(fits.getdata(p)).astype(np.float64).squeeze()
TAG = f"snr{SNR:g}_seed{SEED}_L{L1D}_k{K:g}"
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {T} {TAG}: {m}", flush=True)

# ---------------------------------------------------------------- the truth (noise-free, 0.16" grid, Jy/pixel), cached
TF = os.path.join(OUT, "truth.npy")
if not os.path.exists(TF):
    jd = os.path.join(R, "ladder_joint_20kms/b10/clean_finished" if T == "NGC3110" else "g20/joint_ext")
    h = fits.getheader(os.path.join(jd, "image.fits")); BE = (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"])
    m = rd(os.path.join(jd, "model.fits"))                               # Jy/pixel, 0.04" (1280 px)
    k = np.array([0.5, 1, 1, 1, 0.5])                                     # summed onto 0.16", pixel centres aligned
    m = convolve1d(convolve1d(m, k, axis=-1, mode="constant"), k, axis=-2, mode="constant")[..., ::4, ::4]
    t = np.zeros_like(m); t[line] = np.clip(m[line], 0, None)
    tb = max(0.33, float(np.sqrt(BE[0] * BE[1])))
    # the sky convolved with the truth beam, a unit-sum Gaussian (Jy/pixel kept)
    n = t.shape[-1]; yy, xx = np.mgrid[-n // 2:n // 2, -n // 2:n // 2] * CELL
    gk = np.exp(-4 * np.log(2) * (xx ** 2 + yy ** 2) / tb ** 2); gk /= gk.sum()
    fk = np.fft.rfft2(np.fft.ifftshift(gk))
    t = np.stack([np.fft.irfft2(np.fft.rfft2(c) * fk, s=(n, n)) for c in t])
    np.save(TF, t.astype(np.float32)); json.dump(dict(truth_beam=tb, extended_beam=BE), open(os.path.join(OUT, "truth.json"), "w"))
    log(f"truth: {t[line].sum() * 20:.1f} Jy km/s at {tb:.2f}\"")
truth = np.load(TF).astype(np.float64)

# ---------------------------------------------------------------- the mock dirty cube at the requested S/N
psf = rd(os.path.join(GD, "deconv_inputs/dirty_beam_cube.fits")); real = rd(os.path.join(GD, "deconv_inputs/dirty_cube.fits"))
empty = np.nonzero(psf.max((1, 2)) < 0.5)[0]
for c in empty:
    ok = [i for i in range(len(psf)) if i not in empty]; psf[c] = psf[min(ok, key=lambda i: abs(i - c))]
n = truth.shape[-1]; c0 = psf.shape[-1] // 2
sig0 = float(np.median([real[c].std() for c in free if c not in empty]))
FP = [np.fft.rfft2(np.fft.ifftshift(psf[c][c0 - n // 2:c0 + n // 2, c0 - n // 2:c0 + n // 2])) for c in range(len(psf))]
clean = np.zeros_like(truth)
for c in line:
    clean[c] = np.fft.irfft2(np.fft.rfft2(truth[c]) * FP[c], s=(n, n))
scale = SNR * sig0 / clean.max(); rng = np.random.default_rng(1000 * SEED + 17)
dirty = clean * scale
for c in range(len(dirty)):
    e = np.fft.irfft2(np.fft.rfft2(rng.standard_normal((n, n))) * FP[c], s=(n, n)); dirty[c] += sig0 * e / e.std()
for c in empty:
    dirty[c] = 0.0
log(f"scale {scale:.3g}, noise {sig0 * 1e3:.3f} mJy/beam, peak S/N {SNR:g}")

# ---------------------------------------------------------------- the deconvolution
t0 = time.time()
op = G.Op(psf, dirty); D = G.Dict2D1D(op.d.shape, J, L1D)
lam, _, _, _ = G.noise_lambdas(D, op.d, [c for c in free if c not in empty], k=K)
beta = G.field_lipschitz(op); tau = 1.0 / beta
msk = np.load(os.path.join(GD, "clean_compact_finished/mask_final.npy")).astype(np.float32)
sup = torch.tensor(msk, device=G.DEV).to(G.DT)
x, _, hist, _ = G.solve_pd(op, D, lam, n_iter=N_ITER, beta=beta, reweight=dict(start=6000, every=2000, eps=1.0, ref=tau * lam),
                           print_every=5000, log=log, support=sup)
xm = x.cpu().numpy().astype(np.float32)
np.save(os.path.join(OUT, f"model_{TAG}.npy"), xm)
json.dump(dict(T=T, snr=SNR, seed=SEED, L1D=L1D, k=K, scale=scale, noise_mJy=sig0 * 1e3, truth_flux=float(truth[line].sum() * 20 * scale),
               model_flux=float(xm[line].sum() * 20), seconds=round(time.time() - t0)),
          open(os.path.join(OUT, f"run_{TAG}.json"), "w"), indent=1)
log(f"done: model {xm[line].sum() * 20:.1f} vs truth {truth[line].sum() * 20 * scale:.1f} Jy km/s, {time.time() - t0:.0f} s")
