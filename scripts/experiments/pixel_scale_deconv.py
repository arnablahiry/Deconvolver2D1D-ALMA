#!/usr/bin/env python
"""
Pixel-scale experiment: the same 2D-1D deconvolution (pdtau: k = 5, reweighting every 2000 iterations from 6000,
20000 iterations, positivity, no support mask) of the NGC 3110 compact dirty cube on two grids of the same 51" field,
0.16" (320 px) and 0.08" (640 px), 64 x 20 km/s (results/NGC3110/deconv_inputs/compact_<cell>/, PSF at 2x the field).
Starlet scales J chosen so the coarsest is the same on the sky: 2^J px = 5.1" (J = 5 at 0.16", J = 6 at 0.08"); 1D CDF
9/7 with 5 levels along velocity. Lambdas from the line-free channels (gpu_pd.noise_lambdas). The empty last channel
(no visibilities at the band edge) is kept with its neighbour's PSF.

    CELL=0.16 CUDA_VISIBLE_DEVICES=1 python scripts/experiments/pixel_scale_deconv.py
    -> results/NGC3110/pixel_scale/model_<cell>.npy (Jy/pixel, (64, n, n)) and run_<cell>.json
"""
import json
import os
import sys
import time

import numpy as np
from astropy.io import fits

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
import gpu_pd as G  # noqa: E402

CELL = os.environ["CELL"]; N_ITER = int(os.environ.get("N_ITER", 20000)); K = float(os.environ.get("K", 5))
J = {"0.16": 5, "0.08": 6}[CELL]
SUPPORT = os.environ.get("SUPPORT") == "1"    # x = 0 outside the compact MS-CLEAN's final mask (same grid), every iteration
L1D = int(os.environ.get("L1D", 5))           # levels of the 1D CDF 9/7 along velocity; 0: the 2D starlet per channel
DIL = float(os.environ.get("SUPPORT_DILATE", 0))   # grow the support mask by this many arcsec in each channel
NEG = os.environ.get("NEGATE") == "1"               # null test: deconvolve -dirty (positivity: any model flux is false)
DICT = os.environ.get("DICT_2D", "starlet")
NX = float(os.environ.get("NOISE_X", 1))          # lower S/N: add PSF-filtered noise, independent per channel, so that the
                                                   # line-free noise becomes NX times the original
TAG = (("_support" if SUPPORT else "") + ("" if L1D == 5 else f"_L{L1D}") + ("" if K == 5 else f"_k{K:g}") +
       (f"_dil{DIL:g}" if DIL else "") + ("_neg" if NEG else "") + ("" if NX == 1 else f"_noise{NX:g}") + ("" if DICT == "starlet" else "_" + DICT.replace(":", "-")))
R = os.path.join(REPO, "results/NGC3110"); OUT = os.path.join(R, "pixel_scale"); os.makedirs(OUT, exist_ok=True)
D_IN = os.path.join(R, "deconv_inputs", f"compact_{CELL}")
grid = json.load(open(os.path.join(R, "consistency_20kms_0.16/grid.json")))
rd = lambda p: np.nan_to_num(fits.getdata(p)).astype(np.float64).squeeze()
dirty, psf = rd(os.path.join(D_IN, "dirty_cube.fits")), rd(os.path.join(D_IN, "dirty_beam_cube.fits"))
# all 64 channels (the 1D CDF 9/7 needs a multiple of 2^5); an empty channel (no data at the band edge) gets its
# neighbour's PSF, its dirty map is zero
empty = np.nonzero(psf.max((1, 2)) < 0.5)[0]
for c in empty:
    psf[c] = psf[c - 1] if c > 0 else psf[c + 1]; dirty[c] = 0.0
if NEG:
    dirty = -dirty
if NX != 1:
    from scipy.signal import fftconvolve
    rng = np.random.default_rng(12345); n_ = dirty.shape[-1]; c_ = psf.shape[-1] // 2
    sig0 = float(np.median([dirty[c].std() for c in grid["line_free"] if c not in empty]))
    for c in range(dirty.shape[0]):
        if c in empty:
            continue
        e = fftconvolve(rng.standard_normal((n_, n_)), psf[c][c_ - n_ // 2:c_ + n_ // 2, c_ - n_ // 2:c_ + n_ // 2], mode="same")
        dirty[c] += np.sqrt(NX ** 2 - 1) * sig0 * e / e.std()
keep = np.arange(dirty.shape[0])
line = [int(np.nonzero(keep == c)[0][0]) for c in grid["line"] if c in keep]
noise = [int(np.nonzero(keep == c)[0][0]) for c in grid["line_free"] if c in keep]
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {CELL}\": {m}", flush=True)
log(f"dirty {dirty.shape}, PSF {psf.shape}, J = {J}, {len(line)} line / {len(noise)} line-free channels, k = {K}")

t0 = time.time()
op = G.Op(psf, dirty)
D = G.Dict2D1D(op.d.shape, J, L1D)
lam, _, _, _ = G.noise_lambdas(D, op.d, noise, k=K)
beta = G.field_lipschitz(op); tau = 1.0 / beta
rw = dict(start=6000, every=2000, eps=1.0, ref=tau * lam)
sup = None
if SUPPORT:
    import torch
    m = np.load(os.path.join(R, f"consistency_20kms_{CELL}/clean_compact_finished/mask_final.npy")).astype(np.float32)
    if DIL:
        from scipy.ndimage import binary_dilation
        r_ = int(round(DIL / float(CELL))); yy, xx = np.mgrid[-r_:r_ + 1, -r_:r_ + 1]
        m = np.array([binary_dilation(c_ > 0, structure=(xx ** 2 + yy ** 2 <= r_ ** 2)) for c_ in m]).astype(np.float32)
    sup = torch.tensor(m, device=G.DEV).to(G.DT); log(f"support mask: {m.mean():.2%} of voxels")
x, _, hist, _ = G.solve_pd(op, D, lam, n_iter=N_ITER, beta=beta, reweight=rw, print_every=4000, log=log, support=sup)
xm = x.cpu().numpy().astype(np.float32)
np.save(os.path.join(OUT, f"model_{CELL}{TAG}.npy"), xm)
r = (op.d - op.apply(x)).cpu().numpy()
json.dump(dict(cell=float(CELL), J=J, L1D=L1D, k=K, support_dilate=DIL, negate=NEG, dict_2d=DICT, noise_x=NX, n_iter=N_ITER, channels_kept=keep.tolist(), line=line, noise=noise,
               flux_line_Jy_kms=float(xm[line].sum() * grid["width_kms"]), resid_rms_line=float(r[line].std()),
               resid_rms_free=float(r[noise].std()), seconds=round(time.time() - t0), history=hist),
          open(os.path.join(OUT, f"run_{CELL}{TAG}.json"), "w"), indent=1)
log(f"done: line flux {xm[line].sum() * grid['width_kms']:.1f} Jy km/s, {time.time() - t0:.0f} s")
