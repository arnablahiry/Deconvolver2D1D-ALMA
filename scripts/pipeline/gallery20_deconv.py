#!/usr/bin/env python
"""
2D-1D deconvolution of a target's compact cube for the 20 km/s gallery (paper Fig. 5), exactly as the NGC 3110
0.08" run of the pixel-scale test (scripts/experiments/pixel_scale_deconv.py): 64 x 20 km/s, 0.08" pixels, 51" field,
starlet J = 6 x CDF 9/7 (5 levels), k = 5, reweighting every 2000 iterations from 6000, 20000 iterations, positivity,
x = 0 outside the compact MS-CLEAN's final mask (results/<T>/g20/clean_compact_finished/mask_final.npy). Empty channels
(no visibilities, e.g. at the edge of the band) keep the neighbouring PSF and a zero dirty map.

    T=IC5179 CUDA_VISIBLE_DEVICES=2 python scripts/pipeline/gallery20_deconv.py   -> results/<T>/g20/model_0.08_support.npy
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
import torch  # noqa: E402

T = os.environ["T"]; N_ITER = int(os.environ.get("N_ITER", 20000)); K = float(os.environ.get("K", 5))
GCELL = float(os.environ.get("GCELL", 0.08)); JJ = 6 if GCELL == 0.08 else 5      # coarsest starlet scale 5.1" on both grids
OUT = os.path.join(REPO, "results", T, f"g{float(os.environ.get('CHANKMS', 20)):g}" + ("" if GCELL == 0.08 else f"_c{GCELL:g}"))
D = os.path.join(OUT, "deconv_inputs")
grid = json.load(open(os.path.join(OUT, "grid.json")))
rd = lambda p: np.nan_to_num(fits.getdata(p)).astype(np.float64).squeeze()
dirty, psf = rd(os.path.join(D, "dirty_cube.fits")), rd(os.path.join(D, "dirty_beam_cube.fits"))
empty = np.nonzero(psf.max((1, 2)) < 0.5)[0]
for c in empty:                                                          # nearest channel with a PSF
    ok = [i for i in range(len(psf)) if i not in empty]; j = min(ok, key=lambda i: abs(i - c))
    psf[c] = psf[j]; dirty[c] = 0.0
line, noise = grid["line"], [c for c in grid["line_free"] if c not in empty]
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {T}: {m}", flush=True)
log(f"dirty {dirty.shape}, PSF {psf.shape}, empty channels {empty.tolist()}, {len(line)} line / {len(noise)} line-free")
t0 = time.time()
L1D = int(os.environ.get("L1D", 5)); SUF = "" if L1D == 5 else f"_L{L1D}"     # 0: the 2D starlet per channel
op = G.Op(psf, dirty); Dic = G.Dict2D1D(op.d.shape, JJ, L1D)
lam, _, _, _ = G.noise_lambdas(Dic, op.d, noise, k=K)
beta = G.field_lipschitz(op); tau = 1.0 / beta
m = np.load(os.path.join(OUT, "clean_compact_finished/mask_final.npy")).astype(np.float32)
sup = torch.tensor(m, device=G.DEV).to(G.DT); log(f"support {m.mean():.2%} of voxels")
x, _, hist, _ = G.solve_pd(op, Dic, lam, n_iter=N_ITER, beta=beta, reweight=dict(start=6000, every=2000, eps=1.0, ref=tau * lam),
                           print_every=4000, log=log, support=sup)
xm = x.cpu().numpy().astype(np.float32)
np.save(os.path.join(OUT, f"model_{GCELL:g}_support{SUF}.npy"), xm)
json.dump(dict(k=K, n_iter=N_ITER, empty=empty.tolist(), flux_line_Jy_kms=float(xm[line].sum() * grid["width_kms"]),
               seconds=round(time.time() - t0), history=hist), open(os.path.join(OUT, f"deconv_run{SUF}.json"), "w"), indent=1)
log(f"done: line flux {xm[line].sum() * grid['width_kms']:.1f} Jy km/s, {time.time() - t0:.0f} s")
