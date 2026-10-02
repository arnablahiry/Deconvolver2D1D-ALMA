#!/usr/bin/env python
"""
Extended-configuration 2D-1D primal-dual model (simple/gpu_pd.py), with the same
reweighting schedule as the compact recipe:

    full 0.02" grid (800 x 800 x 32), 8 starlet scales x 5 CDF 9/7 levels,
    lambda_b = 3 x max(real, white) noise MAD (noise channels v < 4780 or v > 5180 km/s),
    x = 0 outside the support mask data/ngc3110/extended/pd/support_mask.npy (grown earlier,
    see data/ngc3110/extended/pd/README.txt; frozen here),
    reweighting (eps = 1) every 2000 iterations from 6000, 20,000 iterations from x = 0.

    CUDA_VISIBLE_DEVICES=0 /scratch/alahiry/conda/envs/denoise2d1d/bin/python scripts/make_extended_pd.py

Writes data/ngc3110/extended/pd/model.npy (the previous model, without reweighting, is kept
as model_no_reweight.npy the first time this runs), and the evolution GIFs
figures/pd/extended/pd_extended.gif, pd_extended_model_hires.gif (+ frames/).
"""
import os, sys, time, shutil
import numpy as np
from astropy.io import fits
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "simple"))
import gpu_pd as G
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evolution_gifs import FrameGrabber, render

PD = os.path.join(REPO, "data/ngc3110/extended/pd")
vel = 5073.9 + 25.0 * (np.arange(32) - 15.5)
LINE = [i for i, v in enumerate(vel) if 4750 <= v <= 5280]
NOISE = [i for i, v in enumerate(vel) if v < 4780 or v > 5180]
K, N_ITER = 3.0, 20000
REWEIGHT = dict(start=6000, every=2000, eps=1.0)

read = lambda f: np.nan_to_num(fits.getdata(os.path.join(REPO, "data/ngc3110/extended/cube", f)).astype(np.float64)).squeeze()
op = G.Op(read("dirty_beam_cube.fits"), read("dirty_cube.fits"))
D = G.Dict2D1D(op.d.shape, 8, 5)
lam, _, _, _ = G.noise_lambdas(D, op.d, NOISE, k=K)
support = torch.tensor(np.load(os.path.join(PD, "support_mask.npy")), device=G.DEV).to(G.DT)
print(f"support mask: {float(support.mean()):.1%} of voxels", flush=True)

grab = FrameGrabber(np.unique(np.geomspace(1, N_ITER, 140).astype(int)), LINE, cell=0.02, crop=np.s_[150:650, 150:650])
t0 = time.time()
x, _, hist, _ = G.solve_pd(op, D, lam, n_iter=N_ITER, support=support, reweight=REWEIGHT, print_every=2000,
                           log=lambda m: print(m, flush=True), callback=grab)
print(f"{N_ITER} iterations in {time.time() - t0:.0f} s", flush=True)

old, keep = os.path.join(PD, "model.npy"), os.path.join(PD, "model_no_reweight.npy")
if not os.path.exists(keep):
    shutil.copy(old, keep)
xn = x.cpu().numpy().astype(np.float32)
prev = np.load(keep).astype(np.float64)
np.save(old, xn)
r = -op.grad(x)
print(f"flux: reweighted {xn.sum():.2f} Jy vs no reweighting {prev.sum():.2f} Jy (sum over voxels); "
      f"relative difference {np.linalg.norm(xn - prev) / np.linalg.norm(prev):.3f}; "
      f"residual rms {float(r[LINE].std()):.3e} (line channels)", flush=True)
print("saved", old, flush=True)

np.savez_compressed(os.path.join(PD, "pd_extended_frames.npz"), it=np.array([f[0] for f in grab.frames]),
                    model=np.array([f[1] for f in grab.frames], dtype=np.float32),
                    resid=np.array([f[2] for f in grab.frames], dtype=np.float32),
                    flux=np.array([f[3] for f in grab.frames]), trace=np.array(grab.trace))
render(grab.frames, grab.trace, os.path.join(REPO, "figures/pd/extended"), "pd_extended",
       "NGC 3110 extended, 2D-1D primal-dual deconvolution",
       lambda it: "k = 3, support mask" + (", reweighted" if it > REWEIGHT["start"] else ""), N_ITER,
       reweight_iters=range(REWEIGHT["start"], N_ITER, REWEIGHT["every"]))
