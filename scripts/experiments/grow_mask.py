#!/usr/bin/env python
"""
Grow the extended-configuration support mask from the data, with the recipe that made
data/ngc3110/extended/pd/support_mask.npy: starlet planes 2-5 of the velocity-smoothed
(3-channel) residual, significant where > max(5 sigma of matched noise, 1.5 x PSF sidelobe
ratio x the plane's peak), dilated 10 px and +-1 channel; 200 primal-dual iterations
(k = 3) inside the mask per cycle, until the mask grows by < 0.01% of voxels.

    grow_mask.py --galaxy ngc2369 --out data/ngc2369/extended/pd/support_mask.npy
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import G, problem  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--galaxy", default="ngc2369"); ap.add_argument("--out", required=True)
ap.add_argument("--k", type=float, default=3.0)
a = ap.parse_args()
SC = (2, 3, 4, 5)
# the problem without a support mask (the mask is what is being made)
import common  # noqa: E402
psf, dirty = common.read(a.galaxy, "extended", "dirty_beam_cube.fits"), common.read(a.galaxy, "extended", "dirty_cube.fits")
noise_ch = common.noise_channels(a.galaxy, "extended")
op = G.Op(psf, dirty); D = G.Dict2D1D(op.d.shape, 8, 5)
lam, _, _, _ = G.noise_lambdas(D, op.d, noise_ch, k=a.k)
beta = G.field_lipschitz(op)

nz, ny, nx = op.d.shape
vs = lambda c: F.avg_pool1d(c.permute(1, 2, 0).reshape(-1, 1, nz), 3, 1, 1, count_include_pad=False).reshape(ny, nx, nz).permute(2, 0, 1)
noise = G.noise_with_ps(G.measured_ps(G.noise_planes(op.d, noise_ch)), op.d.shape, seed=11)
pn = G.starlet(vs(noise), max(SC) + 1)
sig_j = {j: float(1.4826 * (pn[j] - pn[j].median()).abs().median()) for j in SC}; del pn, noise
p0 = psf[16]; cy, cx = np.unravel_index(np.argmax(p0), p0.shape)
pp = G.starlet(G.to_t(p0[cy - 400:cy + 400, cx - 400:cx + 400])[None], max(SC) + 1)[:, 0]
yy, xx = np.indices((800, 800)); rr = torch.tensor(np.hypot(yy - 400, xx - 400), device=G.DEV)
side_j = {j: float(pp[j][rr > max(3 * 2 ** j, 15)].max() / pp[j][400, 400]) for j in SC}
print(f"noise channels {noise_ch}; plane sigma {sig_j}; sidelobe ratio {side_j}", flush=True)

mask = torch.zeros_like(op.d, dtype=torch.bool); x = torch.zeros_like(op.d); u = None
for cyc in range(15):
    before = int(mask.sum())
    mask = G.grow_support(mask, -op.grad(x), sig_j, side_j, SC, nsig=5.0, slfac=1.5, grow_px=10)
    added = int(mask.sum()) - before
    print(f"cycle {cyc}: mask {float(mask.float().mean()):.2%} of voxels (+{added})", flush=True)
    if cyc > 0 and added < 1e-4 * mask.numel():
        break
    x, u, _, _ = G.solve_pd(op, D, lam, n_iter=200, beta=beta, x0=x, u0=u, support=mask.to(G.DT), print_every=10 ** 9,
                            log=lambda m: None)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
np.save(a.out, mask.cpu().numpy())
print(f"saved {a.out}: {float(mask.float().mean()):.2%} of voxels  DONE", flush=True)
