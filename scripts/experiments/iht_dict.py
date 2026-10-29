#!/usr/bin/env python
"""
Experiment 23: starlet vs the pretrained ImageNet learnlet in ONE deconvolution scheme, iterative hard thresholding
(forward-backward with a hard-thresholding denoiser):

    x <- P+( W^-1 H_theta( W (x - tau (N x - d)) ) )

W the 2D dictionary x CDF 9/7 (5 levels), H_theta hard thresholding, P+ positivity. The noise in the gradient step
z = x - tau (N x - d) is tau x the dirty-cube noise, so plane b's noise is tau sigma_b, sigma_b = max(real, white)
noise MAD of sub-band b (gpu_pd.noise_lambdas with k = 1, as the pipeline's lambda).

  --dict starlet|<name>     W: the starlet, or a learnlet filter bank dictionaries/<name>.npz (learnlet: ImageNet, 64
                            filters, 4 scales; learnlet_ext: the same with its scale-3 filters reused at scales 4-8;
                            learnlet_fire*: trained on FIRE, train_learnlet_fire.py):
                            gpu_pd.FilterBank2D's inverse w_j = T(w_j) + sum_k S_k (T(A_k w_j) - A_k T(w_j)) is exactly
                            the learnlet network's own synthesis, so with the learnt thresholds W^-1 H W IS the
                            pretrained network (checked by --check against vicbonj/learnlet).
  --thr noise               theta_b = k tau sigma_b on every plane (the pipeline's rule; "learnt filters" arm)
  --thr learnt              on the learnlet's 4 trained scales, every plane of scale j at s t_j tau sigma_{w_j} / s_j:
                            t_j the network's learnt thresholds (units of a white pixel noise), s_j the MAD of starlet
                            plane j for unit white noise (so sigma_{w_j} / s_j = the white-equivalent pixel noise at
                            scale j); s = --s. With --dict learnlet this is the network (PnP); with --dict starlet it is
                            the network with S = 0 (starlet hard thresholding at the learnt thresholds), the ablation.
                            Deeper scales and the coarse plane: --k-deep x tau sigma_b.

    iht_dict.py --dict learnlet --thr learnt --s 1 --sim --out data/experiments/23_iht/sim/learnlet_learnt_s1.npy
    iht_dict.py --dict starlet --thr noise --k 4 --null 7 --out data/experiments/23_iht/null/starlet_noise_k4.npy
    iht_dict.py --check
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import G, problem, to_saved  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dict", default="starlet", help="starlet or a dictionaries/<name>.npz (learnlet, learnlet_ext, ...)")
ap.add_argument("--thr", choices=["noise", "learnt"], default="noise")
ap.add_argument("--k", type=float, default=4.0); ap.add_argument("--s", type=float, default=1.0)
ap.add_argument("--k-deep", type=float, default=4.0)
ap.add_argument("--sim", action="store_true"); ap.add_argument("--null", type=int)
ap.add_argument("--galaxy", default="ngc3110"); ap.add_argument("--cfg", default="compact")
ap.add_argument("--n-iter", type=int, default=10000); ap.add_argument("--out")
ap.add_argument("--check", action="store_true", help="W^-1 H W with the learnt thresholds vs the learnlet network")
a = ap.parse_args()
LEARNLET = os.path.expanduser("~/repos/learnlet")
WEIGHTS = os.path.join(LEARNLET, "weights/LEARNLET_FINAL_64_5_sc5_True_hard.pth")
NPZ = np.load(os.path.join(G.DICT_DIR, ("learnlet" if a.dict == "starlet" else a.dict) + ".npz"))
T_J = NPZ["threshold_multipliers"]          # --dict starlet --thr learnt: the ImageNet learnlet's (its S = 0 ablation)
log = lambda m: print(f"[{os.path.basename(a.out or 'check')}] {m}", flush=True)


def white_mad(t2d, n=1024, seed=5):
    """MAD-sigma of every 2D plane of t2d for unit white noise"""
    w = torch.randn((1, n, n), device=G.DEV, dtype=G.DT, generator=torch.Generator(G.DEV).manual_seed(seed))
    p = t2d.forward(w)[:, 0].flatten(1)
    return ((p - p.median(1, keepdim=True).values).abs().median(1).values / 0.6745).cpu().numpy()


def learnt_factor(D, s):
    """per 2D plane: theta / (tau sigma_{w_j}) on the trained scales (s t_j / s_j); NaN elsewhere"""
    sj = white_mad(G.Starlet2D(D.J))
    nt = min(len(T_J), D.J)                                     # never the coarse plane (scale D.J)
    return np.array([s * T_J[j] / sj[j] if j < nt else np.nan for j in D.plane_scale])


if a.check:   # the network vs W^-1 H W, one white-noise image, 4 scales (coarse untouched as in the network)
    sys.path.insert(0, LEARNLET)
    from learnlet.learnlet import Learnlet
    net = Learnlet(pretrained=False).to(G.DEV)
    net.load_state_dict(torch.load(WEIGHTS, map_location=G.DEV, weights_only=True)); net.eval()
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:256, :256]
    clean = np.exp(-((yy - 128) ** 2 + (xx - 100) ** 2) / 300.) + 0.5 * (np.hypot(yy - 60, xx - 180) < 20)
    noisy = G.to_t(clean + 0.1 * rng.standard_normal(clean.shape))[None]
    with torch.no_grad():
        ref = net(noisy[None].float(), 0.1)[0].double()
    D = G.Dict2D1D((1, 256, 256), 4, 0, "learnlet")
    c = D.t2d.forward(noisy)
    fac = learnt_factor(D, 1.0)
    sj = white_mad(G.Starlet2D(4))
    th = torch.tensor([T_J[j] * 0.1 if j < 4 else 0.0 for j in D.plane_scale], device=G.DEV, dtype=G.DT)
    out = D.t2d.inverse(c * (c.abs() > th.view(-1, 1, 1, 1)))
    e = float((out - ref).norm() / ref.norm())
    print(f"||W^-1 H W - learnlet network|| / ||network|| = {e:.2e}; MSE vs clean: network "
          f"{float(((ref - G.to_t(clean)) ** 2).mean()):.3e}, noisy {0.01:.3e}")
    print("learnt factors per scale (theta / plane noise):", [round(float(T_J[j] / sj[j]), 2) for j in range(4)])
    sys.exit()

def noise_lambdas_banded(D, dirty_t, noise_ch, k, seed=1):
    """gpu_pd.noise_lambdas, the same numbers (same noise draws), without ever holding W x whole: the CDF first, then
    the 2D stage band by band (they commute); a learnlet with many planes would not fit otherwise"""
    npl = G.noise_planes(dirty_t, noise_ch)
    s_pix = float(npl.std())
    real = G.noise_with_ps(G.measured_ps(npl), dirty_t.shape, seed=seed)
    white = torch.randn(dirty_t.shape, device=G.DEV, dtype=G.DT, generator=torch.Generator(G.DEV).manual_seed(seed + 100)) * s_pix
    out = []
    for nz in (real, white):
        cz = G.cdf_fwd(nz[None], D.L)[0]
        sig = torch.zeros((len(D.plane_scale), D.nz, 1, 1), device=G.DEV, dtype=G.DT)
        for _, sl in D.bands:
            c = D.t2d.forward(cz[sl])
            for j in range(c.shape[0]):
                b = c[j].flatten()
                if b.numel() > 2 ** 24:
                    b = b[torch.randperm(b.numel(), device=G.DEV)[: 2 ** 24]]
                sig[j, sl] = (b - b.median()).abs().median() / 0.6745
            del c
        out.append(sig)
    return k * torch.maximum(*out), out[0], out[1], s_pix


G.noise_lambdas = noise_lambdas_banded                         # problem() calls it

t0 = time.time()
data = None
if a.null is not None:
    P0 = problem(a.galaxy, a.cfg, k=1.0)
    ps = G.measured_ps(G.noise_planes(P0.op.d, P0.noise))
    data = G.noise_with_ps(ps, tuple(P0.op.d.shape), seed=a.null).cpu().numpy().astype(np.float64)
    del P0, ps; torch.cuda.empty_cache()
G.DICT_2D = a.dict
P = problem(a.galaxy, a.cfg, sim=a.sim, k=1.0, data=data)        # P.lam = sigma_b (k = 1)
D, tau = P.D, P.tau
if a.thr == "noise":
    theta = a.k * tau * P.lam
else:
    fac = learnt_factor(D, a.s)
    wplane = {}                                                  # scale j -> index of its starlet plane w_j
    for i, j in enumerate(D.plane_scale):
        wplane.setdefault(j, i)
    idx = [wplane[j] for j in D.plane_scale]
    sig_w = P.lam[idx]                                           # sigma of the starlet plane of each plane's scale
    f = torch.tensor(np.where(np.isnan(fac), 0.0, fac), device=G.DEV, dtype=G.DT).view(-1, 1, 1, 1)
    trained = torch.tensor(~np.isnan(fac), device=G.DEV).view(-1, 1, 1, 1)
    theta = torch.where(trained, f * tau * sig_w, a.k_deep * tau * P.lam)
    log("learnt theta / (tau sigma_w) per trained scale: " + str([round(float(a.s * T_J[j] / white_mad(G.Starlet2D(D.J))[j]), 2) for j in range(min(len(T_J), D.J))]))
log(f"IHT {a.dict} thr={a.thr} k={a.k:g} s={a.s:g} k_deep={a.k_deep:g} sim={a.sim} null={a.null} n_iter={a.n_iter} "
    f"planes={len(D.plane_scale)} (setup {time.time() - t0:.0f} s)")

# the CDF along velocity and the 2D stage act on different axes and commute: the CDF once on the cube, then the 2D
# stage on every band channel, gives D.W's coefficients (~5x faster with the learnlet's 264 planes); the 2D stage
# then works on CHUNK band channels at a time (identical result, a fraction of the memory)
CHUNK = 8


def denoise(z):
    """W^-1 H_theta W z"""
    cz = G.cdf_fwd(z[None], D.L)[0]
    out = torch.empty_like(cz)
    for c0 in range(0, cz.shape[0], CHUNK):
        sl = slice(c0, c0 + CHUNK)
        c = D.t2d.forward(cz[sl])
        out[sl] = D.t2d.inverse(c.mul_(c.abs() > theta[:, sl]))
        del c
    return G.cdf_inv(out[None], D.L)[0]


nv = 2 * 2 ** (D.J - 1) + 8                                     # the coarsest scale's reflect padding fits
v = torch.randn((D.nz, nv, nv), device=G.DEV, dtype=G.DT)
Ds = G.Dict2D1D(v.shape, D.J, D.L, a.dict)
cw = Ds.t2d.forward(G.cdf_fwd(v[None], D.L)[0])
log(f"commuted W vs D.W: rel. diff {float((cw - Ds.W(v)).norm() / Ds.W(v).norm()):.1e}; "
    f"W^-1 W: {float((G.cdf_inv(Ds.t2d.inverse(cw)[None], D.L)[0] - v).norm() / v.norm()):.1e}")
del v, Ds, cw

x = torch.zeros_like(P.op.d)          # plain forward-backward: FISTA momentum diverges with hard thresholding
hist = []
for it in range(1, a.n_iter + 1):
    xn = denoise(x - tau * P.op.grad(x)).clamp_(min=0)
    if it % 500 == 0 or it == a.n_iter:
        rel = float((xn - x).norm() / max(float(xn.norm()), 1e-30))
        hist.append(dict(iter=it, flux=float(xn.sum()), rel_change=rel))
        log(f"iter {it:6d}  flux {hist[-1]['flux']:8.3f}  change {rel:.2e}  ({time.time() - t0:.0f} s)")
    x = xn

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
np.save(a.out, to_saved(P, x))
r = P.op.residual_sky(x)
diag = dict(dict_2d=a.dict, thr=a.thr, k=a.k, s=a.s, k_deep=a.k_deep, sim=a.sim, null=a.null, n_iter=a.n_iter,
            flux=float(x.sum()), flux_line=float(x[P.line].sum()),
            flux_noise_channels=float(x[[c for c in range(x.shape[0]) if c not in P.line]].sum()),
            resid_rms_line=float(r[P.line].std()), history=hist, seconds=round(time.time() - t0))
json.dump(diag, open(a.out.replace(".npy", ".json"), "w"), indent=1)
log(f"saved {a.out}: flux {diag['flux']:.2f} Jy, {diag['seconds']} s")
