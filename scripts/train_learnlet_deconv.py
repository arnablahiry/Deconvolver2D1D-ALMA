#!/usr/bin/env python
"""
Train the learnlet FOR the deconvolution (experiment 24): its filters (analysis A, synthesis S) and per-scale thresholds
are learnt end to end through an unrolled version of the pipeline's FISTA step with the real NGC 3110 compact dirty
beam, instead of for one-step white-noise denoising (train_learnlet_fire.py). The trained bank is then exported as a
gpu_pd dictionary and used inside the pipeline itself (pdtau / FISTA) like any other.

Training samples (one CDF 9/7 velocity-band plane of a FIRE [CII] cube, as everywhere else):
  sky     a 256 x 256 crop of a band plane of the scale-augmented FIRE sets (5 pixel scales), random flip / rotation,
          scaled so the cube's brightest compact-beam-smoothed channel has S/N log-uniform in [SNR_MIN, SNR_MAX] (the
          real compact cube: ~1300), in units of the plane's noise
  dirty   PSF (*) sky + noise: the real compact dirty beam (rebinned to 0.04", the central channel; it varies by 0.4%
          across the band), linear convolution (512 px FFT grid); noise with the real compact cube's measured power
          spectrum (line-free channels), unit standard deviation (the plane's own noise)
  target  the sky at the two evaluation resolutions, G_ext (*) sky and G_lb (*) sky (unit-sum Gaussians of the extended
          clean beam and the compact lambda/B_max) -- what is scored against the joint CLEAN cubes on the real data

Model (the pipeline's FISTA iteration, unrolled):
  x_0     Wiener deconvolution of the dirty plane, F^-1[conj(OTF) D / (|OTF|^2 + mu)], mu learnt
  x_t+1   c_t L(z_t / c_t),  z_t = x_t - tau (N x_t - d),  tau = 1 / max|OTF|,  c_t = tau softplus(m_t)
          L = the learnlet (train_learnlet_fire.Learnlet2D: starlet split, analysis bank, SOFT thresholds as the
          pipeline's, synthesis with the exact-reconstruction term), initialised from the trained scale-augmented FIRE
          learnlet (learnlet_fire_aug[_shared]); the noise in z_t is tau x the dirty noise, so z_t / c_t has unit noise
          scaled by 1 / softplus(m_t), a learnt per-iteration threshold level
  output  [G_ext (*) x_K, G_lb (*) x_K]; loss MSE against the target (both terms in the plane's noise units)

    CUDA_VISIBLE_DEVICES=0 python scripts/train_learnlet_deconv.py [--shared] [--iters 10] [--epochs 30]
        [--name learnlet_deconv]   -> results/learnlet_deconv/<name>/, dictionaries/<name>.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--shared", action="store_true", help="one filter bank for every scale (from learnlet_fire_aug_shared)")
ap.add_argument("--solver", choices=["fista", "pdtau"], default="fista",
                help="the iteration unrolled: the pipeline's FISTA (W^-1 soft W) or its pdtau (primal-dual, W and W^T)")
ap.add_argument("--iters", type=int, default=10, help="unrolled steps after the Wiener start")
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--samples-per-epoch", type=int, default=2048)
ap.add_argument("--batch-size", type=int, default=8)
ap.add_argument("--target-lr", type=float, default=1e-3,
                help="the EFFECTIVE Adam learning rate (train_common.fit scales its base rate by batch / 64; the first "
                     "run of 2026-10-23 used base 2e-4 -> 2.5e-5 and barely moved the filters)")
ap.add_argument("--patience", type=int, default=8); ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--snr-min", type=float, default=50.0); ap.add_argument("--snr-max", type=float, default=2000.0)
ap.add_argument("--valid-samples", type=int, default=512); ap.add_argument("--test-samples", type=int, default=1024)
ap.add_argument("--data", nargs="+", default=[os.path.join(REPO, "data", d) for d in (
    "fire_cii_1x_n600_19.5pc", "fire_cii_1x_n600_27.621pc", "fire_cii_1x_n600_39.0625pc", "fire_cii_1x_n600_55.243pc",
    "fire_cii_4x_n600_78.1pc")])
ap.add_argument("--init", help="weights_best.pt to start from (default: the scale-augmented FIRE learnlet)")
ap.add_argument("--name", default=None)
E = ap.parse_args()
E.name = E.name or ("learnlet_deconv" + ("_pd" if E.solver == "pdtau" else "") + ("_shared" if E.shared else ""))
PATCH, PAD = 256, 512

# the FIRE learnlet's pieces (data loading, model, export); its own argument parser sees the matching options
argv, sys.argv = sys.argv, [sys.argv[0], "--data", *E.data, "--patch", str(PATCH), "--scales", "8", "--filters", "16",
                            "--thr-units", "plane", "--thresh", "soft", "--beam-px", "29.4", "--name", E.name] + \
    (["--shared", "--export-scales", "9"] if E.shared else [])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_learnlet_fire as T  # noqa: E402
sys.argv = argv
sys.path.insert(0, os.path.join(REPO, "scripts", "experiments"))
from train_common import fit, log, seed_everything, setup_logging  # noqa: E402
from losses import get_loss  # noqa: E402
import gpu_pd as G  # noqa: E402
from common import read, noise_channels, BEAMS  # noqa: E402
from analyze import beam_ft  # noqa: E402

OUT = os.path.join(REPO, "results", "learnlet_deconv", E.name)
T.OUT = OUT


# ======================================================================== the forward model's pieces (numpy, CPU)
def kernels():
    """OTFs on the 512 px grid (rfft2 layout), peak (0, 0): the compact dirty beam, and unit-sum Gaussians of the
    extended clean beam and the compact lambda/B_max; plus a bank of unit-variance noise fields with the compact cube's
    measured power spectrum"""
    d, p = G.rebin(read("ngc3110", "compact", "dirty_cube.fits"), read("ngc3110", "compact", "dirty_beam_cube.fits"), 2)
    psf = p[p.shape[0] // 2] / p[p.shape[0] // 2].max()
    py, px = np.unravel_index(np.argmax(psf), psf.shape)
    h = PAD // 2
    k = np.zeros((PAD, PAD)); k[:] = psf[py - h:py + h, px - h:px + h]
    otf = np.fft.rfft2(np.roll(k, (-h, -h), axis=(0, 1)))
    lb = float(json.load(open(os.path.join(REPO, "results/NGC3110/config.json")))["lbmax_compact_arcsec"])
    g = {}
    for nm, b in (("ext", BEAMS[("ngc3110", "extended")]), ("lb", (lb, lb, 0.0))):
        f = beam_ft((PATCH, PATCH), b, 0.04).cpu().numpy()
        g[nm] = f / f[0, 0].real                                          # unit sum
    npl = G.noise_planes(G.to_t(d), noise_channels("ngc3110", "compact"))
    ps = G.measured_ps(npl)
    bank = torch.stack([G.noise_with_ps(ps, (1,) + tuple(d.shape[1:]), seed=1000 + i)[0] for i in range(256)])
    bank = (bank / bank.std()).cpu().numpy().astype(np.float32)
    return otf.astype(np.complex64), g["ext"].astype(np.complex64), g["lb"].astype(np.complex64), bank


def conv_np(x, otf):
    big = np.zeros((PAD, PAD), np.float32); big[:PATCH, :PATCH] = x
    return np.fft.irfft2(np.fft.rfft2(big) * otf, s=(PAD, PAD))[:PATCH, :PATCH].astype(np.float32)


class DeconvPlanes(Dataset):
    """(dirty, [G_ext sky, G_lb sky]) band-plane pairs in units of the plane's noise; fixed=True: the same draws"""

    def __init__(self, bands, peak, n, nk, K, fixed=False, seed=0):
        self.b, self.peak, self.n, self.nk, self.fixed, self.seed = bands, peak, n, nk, fixed, seed
        self.otf, self.gE, self.gL, self.bank = K
        self.omega = 2 * np.pi * (T.A.beam_px / 2.3548) ** 2                # unit-sum Gaussian peak -> peak-1 (Jy/beam)

    def __len__(self):
        return self.n

    def draw(self, i):
        rng = np.random.default_rng((self.seed, i) if self.fixed else None)
        ci, k = rng.integers(len(self.b)), rng.integers(self.b.shape[1])
        y0, x0 = rng.integers(0, self.b.shape[2] - PATCH + 1, 2)
        p = self.b[ci, k, y0:y0 + PATCH, x0:x0 + PATCH]
        p = np.rot90(p, rng.integers(4)); p = p[:, ::-1] if rng.integers(2) else p
        S = np.exp(rng.uniform(np.log(E.snr_min), np.log(E.snr_max)))
        sky = (np.ascontiguousarray(p) * (S / (self.peak[ci] * self.omega)) / self.nk[k]).astype(np.float32)
        nb = self.bank[rng.integers(len(self.bank))]
        ny, nx = rng.integers(0, nb.shape[0] - PATCH + 1, 2)
        dirty = conv_np(sky, self.otf) + nb[ny:ny + PATCH, nx:nx + PATCH]
        return dirty, np.stack([conv_np(sky, self.gE), conv_np(sky, self.gL)]), k, S

    def __getitem__(self, i):
        d, t, k, S = self.draw(i)
        return torch.from_numpy(d)[None], torch.from_numpy(t), torch.tensor([k, S]), torch.zeros(1)


# ======================================================================== the unrolled deconvolver
class Unrolled(nn.Module):
    def __init__(self, K, iters):
        super().__init__()
        otf, gE, gL, _ = K
        self.register_buffer("otf", torch.from_numpy(otf)); self.register_buffer("gE", torch.from_numpy(gE))
        self.register_buffer("gL", torch.from_numpy(gL))
        self.tau = float(1.0 / np.abs(otf).max())
        self.learnlet = T.Learnlet2D(T.A.scales, T.A.kernel, T.A.filters, T.A.thresh, T.A.shared, T.A.thr_units)
        self.m = nn.Parameter(torch.full((iters,), 0.5413))                # softplus(0.5413) = 1
        self.log_mu = nn.Parameter(torch.tensor(float(np.log(1e-2 * np.abs(otf).max() ** 2))))

    def conv(self, x, otf):
        big = torch.nn.functional.pad(x, (0, PAD - PATCH, 0, PAD - PATCH))
        return torch.fft.irfft2(torch.fft.rfft2(big) * otf, s=(PAD, PAD))[..., :PATCH, :PATCH]

    def forward(self, d):
        D = torch.fft.rfft2(torch.nn.functional.pad(d, (0, PAD - PATCH, 0, PAD - PATCH)))
        x = torch.fft.irfft2(D * self.otf.conj() / (self.otf.abs() ** 2 + self.log_mu.exp()), s=(PAD, PAD))[..., :PATCH, :PATCH]
        for m in self.m:
            z = x - self.tau * (self.conv(x, self.otf) - d)
            c = self.tau * nn.functional.softplus(m)
            x = c * self.learnlet(z / c)
        return torch.cat([self.conv(x, self.gE), self.conv(x, self.gL)], 1)


class UnrolledPD(Unrolled):
    """the pipeline's pdtau unrolled (Condat-Vu, analysis prior, no positivity in the band domain, no reweighting):
    x <- x - tau (N x - d + W'u),  u <- clip(u + sigma W(2 x_new - x), +-lambda),  from the Wiener start and u = 0.
    W = the learnlet's ANALYSIS (starlet planes + filtered copies; its synthesis filters are not used, as in pdtau);
    W' its exact adjoint (autograd). lambda per plane = softplus(w_j) x the plane's noise (W of a unit noise patch with
    the compact noise spectrum), one learnt w_j per scale, initialised to 1.5 (the learnlets' best pdtau k)"""

    def __init__(self, K, iters, noise_patch):
        super().__init__(K, iters)
        self.register_buffer("npatch", torch.from_numpy(noise_patch)[None, None])
        self.w = nn.Parameter(torch.full((T.A.scales,), float(np.log(np.expm1(1.5)))))   # softplus(w) = 1.5
        self.sigma = None                                                   # set_step(), once the filters are loaded

    def set_step(self):
        """sigma = 1 / (2 tau ||W||^2 x 1.1): Condat-Vu's tau (beta/2 + sigma ||W||^2) <= 1 with tau = 1/beta; ||W||^2
        by power iteration on W'W (A is L2-normalised, so it changes little during training)"""
        v = torch.randn(1, 1, PATCH, PATCH, device=self.otf.device); v /= v.norm(); n2 = 0.0
        for _ in range(30):
            g = self.WT(self.W(v).detach(), v).detach()
            n2 = float(g.norm()); v = g / n2
        self.sigma = 1.0 / (self.tau * 2 * n2 * 1.1)
        return n2

    def W(self, x):
        planes, c = [], x
        for b in self.learnlet.blocks:
            s = b._coarsen(c); w = c - s; c = s
            planes += [w, b._analyse(w)]
        return torch.cat(planes + [c], 1)

    def WT(self, u, like):
        with torch.enable_grad():                                           # also inside fit's no-grad validation
            x = torch.zeros_like(like).requires_grad_(True)
            g, = torch.autograd.grad(self.W(x), x, grad_outputs=u, create_graph=torch.is_grad_enabled() and u.requires_grad)
        return g

    def lam(self):
        sig = self.W(self.npatch).std(dim=(0, 2, 3)).detach()                # each plane's noise
        scale = []
        for j, b in enumerate(self.learnlet.blocks):
            scale += [j] * (1 + b.analysis.weight.shape[0])
        scale += [len(self.learnlet.blocks) - 1]                             # the coarse plane: the last scale's
        return (nn.functional.softplus(self.w)[scale] * sig).view(1, -1, 1, 1)

    def forward(self, d):
        D = torch.fft.rfft2(torch.nn.functional.pad(d, (0, PAD - PATCH, 0, PAD - PATCH)))
        x = torch.fft.irfft2(D * self.otf.conj() / (self.otf.abs() ** 2 + self.log_mu.exp()), s=(PAD, PAD))[..., :PATCH, :PATCH]
        lam = self.lam(); u = torch.zeros_like(self.W(x))
        for _ in self.m:
            xn = x - self.tau * (self.conv(x, self.otf) - d + self.WT(u, x))
            u = torch.maximum(torch.minimum(u + self.sigma * self.W(2 * xn - x), lam), -lam)
            x = xn
        return torch.cat([self.conv(x, self.gE), self.conv(x, self.gL)], 1)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    setup_logging(os.path.join(OUT, "logs"), E.name)
    log.info(f"Command: {' '.join(sys.argv)}")
    for k_, v in sorted(vars(E).items()):
        log.info(f"  {k_:20s} {v}")
    seed_everything(0)
    t0 = time.time()
    K = kernels()
    Wm = np.stack([G.cdf_fwd(torch.eye(32, dtype=torch.float64)[c].reshape(1, 32, 1, 1), 5)[0, :, 0, 0].numpy() for c in range(32)], 1)
    nk = np.sqrt((Wm ** 2).sum(1)).astype(np.float32)
    data = {s: T.load(g) for s, g in T.SPLIT.items()}
    log.info(f"data: {', '.join(f'{s} {len(d[2])} cubes' for s, d in data.items())} ({time.time() - t0:.0f} s); "
             f"tau = {1 / np.abs(K[0]).max():.3e}")
    tr = DeconvPlanes(data["train"][0], data["train"][1], E.samples_per_epoch, nk, K)
    va = DeconvPlanes(data["valid"][0], data["valid"][1], E.valid_samples, nk, K, fixed=True, seed=1)
    te = DeconvPlanes(data["test"][0], data["test"][1], E.test_samples, nk, K, fixed=True, seed=2)
    kw = dict(batch_size=E.batch_size, num_workers=E.workers, pin_memory=True, persistent_workers=True)
    trl = DataLoader(tr, shuffle=False, drop_last=True, **kw); val = DataLoader(va, shuffle=False, **kw)
    model = (UnrolledPD(K, E.iters, K[3][0, :PATCH, :PATCH]) if E.solver == "pdtau" else Unrolled(K, E.iters)).cuda()
    init = E.init or os.path.join(REPO, "results", "learnlet_fire",
                                  "learnlet_fire_aug_shared" if E.shared else "learnlet_fire_aug", "weights_best.pt")
    model.learnlet.load_state_dict(torch.load(init, map_location="cuda"))
    if E.solver == "pdtau":
        log.info(f"pdtau unrolled: ||W||^2 = {model.set_step():.3f}, sigma = {model.sigma:.3e}, tau = {model.tau:.3e}")
    log.info(f"Unrolled deconvolver: Wiener start + {E.iters} learnlet {E.solver} steps, learnlet from {init}; "
             f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    fit(E.epochs, model, get_loss("mse"), trl, val, torch.device("cuda"), OUT, scheduler_name="cosine_warmup",
        warmup_epochs=2, min_lr=1e-6, base_lr=E.target_lr * 64 / E.batch_size, patience=E.patience)

    # test (B1): the trained deconvolver vs the same unrolling with the initial (denoising-trained) learnlet
    model.load_state_dict(torch.load(os.path.join(OUT, "weights_best.pt"), map_location="cuda")); model.eval()
    base = (UnrolledPD(K, E.iters, K[3][0, :PATCH, :PATCH]) if E.solver == "pdtau" else Unrolled(K, E.iters)).cuda()
    base.learnlet.load_state_dict(torch.load(init, map_location="cuda")); base.eval()
    if E.solver == "pdtau":
        model.set_step(); base.set_step()
    errs = {"trained": [], "initial": []}
    with torch.set_grad_enabled(E.solver == "pdtau"):                     # pdtau's W' needs autograd
        for i0 in range(0, E.test_samples, 16):
            batch = [te.draw(i) for i in range(i0, min(i0 + 16, E.test_samples))]
            d = torch.from_numpy(np.stack([b[0] for b in batch]))[:, None].cuda()
            t = torch.from_numpy(np.stack([b[1] for b in batch])).cuda()
            for nm, mdl in (("trained", model), ("initial", base)):
                o = mdl(d).detach()
                errs[nm].append(((o - t) ** 2).flatten(2).sum(2) / (t ** 2).flatten(2).sum(2).clamp(min=1e-12))
    res = {nm: {s: float(torch.cat(v)[:, j].sqrt().median()) for j, s in enumerate(("ext", "lb"))} for nm, v in errs.items()}
    json.dump(res, open(os.path.join(OUT, "test.json"), "w"), indent=1)
    log.info(f"test (B1) median relative error, trained vs initial learnlet: {res}")
    if E.solver == "pdtau":                                                 # the learnt per-scale lambda / noise
        kj = nn.functional.softplus(model.w).detach().cpu().numpy().tolist()
        json.dump(dict(k_per_scale=kj), open(os.path.join(OUT, "pdtau_k.json"), "w"), indent=1)
        log.info(f"learnt pdtau lambda / noise per scale: {np.round(kj, 3).tolist()}")
    path = os.path.join(REPO, "dictionaries", f"{E.name}.npz")
    thr, e_inv, e_a = T.export(model.learnlet, path)
    log.info(f"wrote {path}: thresholds {np.round(thr, 4).tolist()}; gpu_pd check |W^-1 W x - x| {e_inv:.1e}, "
             f"|analysis - model| {e_a:.1e}; total {(time.time() - t0) / 60:.1f} min")
