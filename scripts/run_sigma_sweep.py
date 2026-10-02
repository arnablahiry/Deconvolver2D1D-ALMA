#!/usr/bin/env python
"""
One run of the k-sigma sweep: method (pd | fista) x configuration (compact | extended) x k.

    CUDA_VISIBLE_DEVICES=0 /scratch/alahiry/conda/envs/denoise2d1d/bin/python scripts/run_sigma_sweep.py pd compact 3
    ... run_sigma_sweep.py pd compact 3 fine      (compact on the native 0.02" grid, 8 starlet scales)

Common to every run: 2D-1D starlet x CDF 9/7 (5 levels), lambda_b = k x max(real, white)
noise MAD per sub-band, step 1/beta, reweighting (eps = 1) every 2000 iterations from
6000, 20,000 iterations from x = 0.
  compact : 0.04" grid (2 x 2 rebinned), 7 starlet scales, noise channels = all line-free
            channels (v < 4750 or v > 5280 km/s); model upsampled back to 0.02"
  extended: 0.02" grid, 8 starlet scales, noise channels v < 4780 or v > 5180 km/s, and
            x = 0 outside data/ngc3110/extended/pd/support_mask.npy
  pd    : Condat-Vu primal-dual (simple/gpu_pd.solve_pd)
  fista : the loop of simple/deconvolve.py: x = max(0, W^-1 soft(W(z - tau grad), tau lambda)),
          FISTA momentum (thresholds reweighted as tau lambda / (|W x| / (tau lambda) + eps))

Writes
  data/ngc3110/<cfg>/<method>/model_k<k>_20k.npy       final model, Jy/pixel, 0.02"
  data/ngc3110/<cfg>/<method>/frames_k<k>_20k.npz      frame cache
  figures/<method>/<k>_sigma/<cfg>/<method>_<cfg>.gif, <method>_<cfg>_model_hires.gif, frames/
With "fine" (compact only): data/ngc3110/compact/<method>/0.02arcsec/... and
  figures/0.02arcsec/<method>/<k>_sigma/compact/...
"""
import os, sys, time
import numpy as np
from astropy.io import fits
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "simple"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_pd as G
from evolution_gifs import FrameGrabber, render

METHOD, CFG, K = sys.argv[1], sys.argv[2], float(sys.argv[3])
FINE = len(sys.argv) > 4 and sys.argv[4] == "fine"            # compact on the native 0.02" grid
assert METHOD in ("pd", "fista") and CFG in ("compact", "extended") and not (FINE and CFG != "compact")
TAG = f"{K:g}"
N_ITER = int(os.environ.get("SWEEP_N_ITER", 20000))     # override only for smoke tests
RW = dict(start=6000, every=2000, eps=1.0)
vel = 5073.9 + 25.0 * (np.arange(32) - 15.5)
LINE = [i for i, v in enumerate(vel) if 4750 <= v <= 5280]
read = lambda f: np.nan_to_num(fits.getdata(os.path.join(REPO, f"data/ngc3110/{CFG}/cube", f)).astype(np.float64)).squeeze()
if CFG == "compact" and FINE:
    dirty, psf = read("dirty_cube.fits"), read("dirty_beam_cube.fits")
    CELL, BIN, J, CROP = 0.02, 1, 8, np.s_[150:650, 150:650]
    NOISE = [i for i in range(32) if i not in LINE]
    support = None
elif CFG == "compact":
    dirty, psf = G.rebin(read("dirty_cube.fits"), read("dirty_beam_cube.fits"), 2)
    CELL, BIN, J, CROP = 0.04, 2, 7, np.s_[75:325, 75:325]
    NOISE = [i for i in range(32) if i not in LINE]
    support = None
else:
    dirty, psf = read("dirty_cube.fits"), read("dirty_beam_cube.fits")
    CELL, BIN, J, CROP = 0.02, 1, 8, np.s_[150:650, 150:650]
    NOISE = [i for i, v in enumerate(vel) if v < 4780 or v > 5180]
    support = torch.tensor(np.load(os.path.join(REPO, "data/ngc3110/extended/pd/support_mask.npy")), device=G.DEV).to(G.DT)

op = G.Op(psf, dirty); D = G.Dict2D1D(op.d.shape, J, 5)
lam, _, _, _ = G.noise_lambdas(D, op.d, NOISE, k=K)
beta = G.field_lipschitz(op)
grab = FrameGrabber(np.unique(np.geomspace(1, N_ITER, 140).astype(int)), LINE, cell=CELL, crop=CROP)
if FINE:
    CFG_LABEL = "compact 0.02arcsec"
print(f"[{METHOD} {CFG}{' fine' if FINE else ''} k={TAG}] start", flush=True)
t0 = time.time()
if METHOD == "pd":
    x, _, _, _ = G.solve_pd(op, D, lam, n_iter=N_ITER, beta=beta, support=support, reweight=RW, print_every=4000,
                            log=lambda m: print(f"[{METHOD} {CFG} k={TAG}] {m}", flush=True), callback=grab)
else:
    tau = 1.0 / beta; T0 = tau * lam; T = T0
    soft = lambda c, t: torch.sign(c) * torch.clamp(c.abs() - t, min=0)
    x = torch.zeros_like(op.d); z = x.clone(); t = 1.0
    for it in range(1, N_ITER + 1):
        if it > RW["start"] and (it - 1 - RW["start"]) % RW["every"] == 0:
            T = T0 / (D.W(x).abs() / T0 + RW["eps"])
        xn = D.Winv(soft(D.W(z - tau * op.grad(z)), T)).clamp_(min=0)
        if support is not None:
            xn.mul_(support)
        tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t)); z = xn + ((t - 1) / tn) * (xn - x); x, t = xn, tn
        grab(it, x, op.grad(x) if it in grab.marks else None)
        if it % 4000 == 0:
            print(f"[{METHOD} {CFG} k={TAG}] iter {it}  flux {float(x.sum()):.3f}  change {grab.trace[-1][1]:.2e}", flush=True)
print(f"[{METHOD} {CFG} k={TAG}] {N_ITER} iterations in {time.time() - t0:.0f} s; flux {float(x.sum()):.2f} Jy", flush=True)

dat = os.path.join(REPO, f"data/ngc3110/{CFG}/{METHOD}" + ("/0.02arcsec" if FINE else "")); os.makedirs(dat, exist_ok=True)
xm = x.cpu().numpy().astype(np.float64)
np.save(os.path.join(dat, f"model_k{TAG}_20k.npy"), (G.upsample(xm, BIN) if BIN > 1 else xm).astype(np.float32))
np.savez_compressed(os.path.join(dat, f"frames_k{TAG}_20k.npz"), it=np.array([f[0] for f in grab.frames]),
                    model=np.array([f[1] for f in grab.frames], dtype=np.float32),
                    resid=np.array([f[2] for f in grab.frames], dtype=np.float32),
                    flux=np.array([f[3] for f in grab.frames]), trace=np.array(grab.trace))
name = {"pd": "primal-dual", "fista": "FISTA"}[METHOD]
render(grab.frames, grab.trace, os.path.join(REPO, ("figures/0.02arcsec/" if FINE else "figures/") + f"{METHOD}/{TAG}_sigma/{CFG}"),
       f"{METHOD}_{CFG}", f"NGC 3110 {CFG}{' (0.02 arcsec grid)' if FINE else ''}, 2D-1D {name} deconvolution",
       lambda it: f"k = {TAG}" + (", support mask" if support is not None else "") + (", reweighted" if it > RW["start"] else ""),
       N_ITER, reweight_iters=range(RW["start"], N_ITER, RW["every"]))
print(f"[{METHOD} {CFG} k={TAG}] DONE in {time.time() - t0:.0f} s", flush=True)
