#!/usr/bin/env python
"""
Train the 2D learnlet of the 2D-1D dictionary on what it acts on in the solver: the CDF 9/7 band planes (along
velocity, gpu_pd.cdf_fwd, L levels) of FIRE [CII] cubes (fire_preprocess.py: 400 x 400 px of 19.5 pc, 32 x 25 km/s),
with noise as in the real data. The model, training loop and loss are Denoiser2D1D-improved's (LearnletBlock2D:
fixed B3 starlet split, L2-normalised analysis bank, depthwise synthesis with the exact-reconstruction term, MiniNet
thresholds k_min + softplus; train_common.fit; losses MSE); this script adds the data and the export.

Data (built once in memory; every sample fresh):
  * band planes: cdf_fwd of each clean cube; one sample = one of the nz planes of one cube, a random PATCH x PATCH
    crop, a random flip / 90 degree rotation
  * noise: white Gaussian in the channels (the sky model's noise is roughly white inside the uv coverage; the
    solver measures the real, coloured noise per sub-band for its thresholds) with sigma_c = peak / S, peak = the
    cube's brightest channel smoothed by a beam-like Gaussian (FWHM BEAM_PX px, as the real cubes' ~6.7 px beams)
    and S ~ log-uniform [SNR_MIN, SNR_MAX] (the real cubes' brightest channels: S/N 22-250, measured on the
    compact and extended MS-CLEAN cubes of results/*). A band plane k then has white noise sigma_c * n_k,
    n_k = the norm of its 1D filter, exactly; the sample is divided by it, so the model always sees unit noise
    (the learnlet is scale-equivariant: thresholds ~ sigma) and its thresholds come out in sigma units
  * split by galaxy (C1 excluded: duplicate of C2): train A1 A2 A4 B2 C2, valid A8 (fixed draws), test B1

--shared: ONE analysis / synthesis filter bank for every detail scale (a learnt mother-wavelet family: scale j applies
it with dilation 2**j, as the starlet its B3 spline), so it extends to any number of scales; thresholds stay per scale.
The export then writes the shared bank for --export-scales scales (the untrained ones hold the last threshold).

Output: <out>/weights_best.pt, losses.txt, test.json (MSE per band vs the noisy input and the ImageNet learnlet),
and dictionaries/<name>.npz for gpu_pd (dict_2d="<name>"), checked to reproduce the model's transform.

    CUDA_VISIBLE_DEVICES=1 python scripts/train_learnlet_fire.py [--filters 16] [--scales 5] [--epochs 80]
        [--samples-per-epoch 16384] [--batch-size 32] [--name learnlet_fire] [--shared --export-scales 9]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEN = os.path.expanduser("~/repos/Denoiser2D1D-improved")
for p in (os.path.join(DEN, "src"), os.path.join(DEN, "methods", "learnlets"), os.path.join(DEN, "methods", "shared")):
    sys.path.insert(0, p)
sys.path.insert(0, os.path.join(REPO, "simple"))
import data.compat  # noqa: E402,F401  (train_common's dataset imports need it)
from learnlets import LearnletBlock2D, MiniNet  # noqa: E402
from train_common import fit, log, seed_everything, setup_logging  # noqa: E402
from losses import get_loss  # noqa: E402
from gpu_pd import band_slices, cdf_fwd  # noqa: E402

SPLIT = dict(train=["A1", "A2", "A4", "B2", "C2"], valid=["A8"], test=["B1"])

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", nargs="+", default=[os.path.join(REPO, "data/fire_cii_1x_n400_19.5pc")],
                help="preprocessed set(s); several = scale augmentation (the same cubes at other pixel scales, pooled)")
ap.add_argument("--levels", type=int, default=5, help="CDF levels along velocity (the solver's L)")
ap.add_argument("--patch", type=int, default=256)
ap.add_argument("--snr-min", type=float, default=20.0); ap.add_argument("--snr-max", type=float, default=250.0)
ap.add_argument("--beam-px", type=float, default=6.7)
ap.add_argument("--scales", type=int, default=5, help="starlet scales incl. the coarse one (learnt: scales - 1)")
ap.add_argument("--filters", type=int, default=16); ap.add_argument("--kernel", type=int, default=5)
ap.add_argument("--thresh", default="hard", choices=["hard", "soft", "garrote"])
ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--samples-per-epoch", type=int, default=16384)
ap.add_argument("--valid-samples", type=int, default=2048); ap.add_argument("--test-samples", type=int, default=4096)
ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--patience", type=int, default=15); ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--shared", action="store_true", help="one filter bank shared by every detail scale")
ap.add_argument("--thr-units", choices=["pixel", "plane"], default="pixel",
                help="the MiniNet threshold (k_min + softplus, k_min = 0.5) in units of the input's pixel noise (as "
                     "trained before 2026-10-22), or of each starlet plane's own noise (its floor then 0.5 sigma of "
                     "THAT plane: in pixel units the floor is 0.5 / ||psi_j||, 12 sigma of plane 3, ~100 of plane 6)")
ap.add_argument("--export-scales", type=int, help="--shared: detail scales written to the .npz (default: the trained ones)")
ap.add_argument("--name", default="learnlet_fire", help="dictionaries/<name>.npz, results/learnlet_fire/<name>/")
ap.add_argument("--imagenet", default=os.path.expanduser("~/repos/learnlet"), help="vicbonj/learnlet repo (baseline)")
A = ap.parse_args()
OUT = os.path.join(REPO, "results", "learnlet_fire", A.name)


# ======================================================================== data
def load(gals):
    """clean band planes (n_cubes, nz, ny, nx) float32, the per-cube channel noise scale (peak), names"""
    bands, peak, names = [], [], []
    for d in A.data:                                            # every set: the same cubes at its own pixel scale
        idx = json.load(open(os.path.join(d, "index.json")))
        tag = f"{idx['settings'].get('cell_pc', 0):.1f}pc" if len(A.data) > 1 else ""
        for c in idx["cubes"]:
            if c["gal"] not in gals:
                continue
            sky = np.load(os.path.join(d, f"{c['gal']}_s{c['snap']}_los{c['los']}.npz"))["sky"].astype(np.float32)
            bands.append(cdf_fwd(torch.from_numpy(sky)[None], A.levels)[0].numpy())
            peak.append(max(gaussian_filter(ch, A.beam_px / 2.3548).max() for ch in sky))
            names.append(f"{c['gal']}_s{c['snap']}_los{c['los']}" + (f"_{tag}" if tag else ""))
    return np.stack(bands), np.array(peak, np.float32), names


class Planes(Dataset):
    """noisy / clean band plane pairs in units of the plane's noise; fixed=True: the same draws every epoch"""

    def __init__(self, bands, peak, n, nk, fixed=False, seed=0):
        self.b, self.peak, self.n, self.nk, self.fixed, self.seed = bands, peak, n, nk, fixed, seed

    def __len__(self):
        return self.n

    def draw(self, i):
        rng = np.random.default_rng((self.seed, i) if self.fixed else None)
        ci, k = rng.integers(len(self.b)), rng.integers(self.b.shape[1])
        y0, x0 = rng.integers(0, self.b.shape[2] - A.patch + 1, 2)
        p = self.b[ci, k, y0:y0 + A.patch, x0:x0 + A.patch]
        p = np.rot90(p, rng.integers(4)); p = p[:, ::-1] if rng.integers(2) else p
        S = np.exp(rng.uniform(np.log(A.snr_min), np.log(A.snr_max)))
        sig = self.peak[ci] / S * self.nk[k]                                         # the plane's noise
        clean = (np.ascontiguousarray(p, np.float32) / np.float32(sig)).astype(np.float32)
        noisy = clean + rng.standard_normal(clean.shape).astype(np.float32)
        return noisy, clean, k, S

    def __getitem__(self, i):
        noisy, clean, k, S = self.draw(i)
        return torch.from_numpy(noisy)[None], torch.from_numpy(clean)[None], torch.tensor([k, S]), torch.zeros(1)


# ======================================================================== model
def plane_noise(n):
    """noise std of starlet planes 0..n-1 for unit white noise: ||psi_j||_2 (B3 starlet, gpu_pd.starlet)"""
    import gpu_pd
    N = 8 * 2 ** n + 1
    d = torch.zeros((1, N, N), dtype=torch.float64); d[0, N // 2, N // 2] = 1
    return gpu_pd.starlet(d.to(gpu_pd.DEV, gpu_pd.DT), n)[:n, 0].flatten(1).norm(dim=1).cpu().numpy()


class Learnlet2D(nn.Module):
    """the 2D learnlet (Denoiser2D1D-improved's LearnletBlock2D per detail scale), for unit-noise input: thresholds
    are the MiniNet multipliers themselves (sigma = 1)"""

    def __init__(self, scales, kernel, filters, thresh, shared=False, thr_units="pixel"):
        super().__init__()
        unit = plane_noise(scales - 1) if thr_units == "plane" else np.ones(scales - 1)
        self.register_buffer("unit", torch.tensor(unit, dtype=torch.float32), persistent=False)
        self.blocks = nn.ModuleList([LearnletBlock2D(2 ** j, kernel, filters, exact_rec=True) for j in range(scales - 1)])
        if shared:                       # tie every scale's weights to scale 0's (the dilation stays per block)
            for b in self.blocks[1:]:
                b.analysis.weight, b.synthesis.weight = self.blocks[0].analysis.weight, self.blocks[0].synthesis.weight
        self.mininet = MiniNet(scales - 1)
        self.register_buffer("k_init", torch.ones(1))
        self.mode = thresh

    def thresholds(self):
        """per detail scale, in units of the input's (pixel) noise"""
        return self.mininet(self.k_init) * self.unit

    def forward(self, x):
        t, out = self.thresholds(), 0
        for j, blk in enumerate(self.blocks):
            d, x = blk(x, t[j], self.mode)
            out = out + d
        return out + x


def export(model, path):
    """the gpu_pd FilterBank2D file, checked against the model's own analysis and exact inverse"""
    from torch.nn import functional as F
    import gpu_pd
    m = model.cpu().eval()
    with torch.no_grad():
        An = np.stack([(b.analysis.weight / (torch.linalg.vector_norm(b.analysis.weight, dim=(2, 3), keepdim=True)
                                             + 1e-8))[:, 0].numpy() for b in m.blocks])
        Sy = np.stack([b.synthesis.weight[:, 0].numpy() for b in m.blocks])
        thr = m.thresholds().numpy()
    n_tr = len(thr)
    if A.shared and A.export_scales and A.export_scales > n_tr:          # the shared bank at untrained scales
        An, Sy = np.repeat(An[:1], A.export_scales, 0), np.repeat(Sy[:1], A.export_scales, 0)
        u = plane_noise(A.export_scales)                    # the last threshold held in units of each plane's noise
        thr = np.r_[thr, thr[-1] / u[n_tr - 1] * u[n_tr:]]
    meta = dict(kind="starlet_filterbank", source=f"FIRE [CII] CDF 9/7 band planes ({', '.join(os.path.basename(d.rstrip('/')) for d in A.data)}, L = {A.levels}), "
                f"noise S/N {A.snr_min:g}-{A.snr_max:g}; Denoiser2D1D-improved LearnletBlock2D", filters=A.filters,
                kernel_size=A.kernel, trained_detail_scales=A.scales - 1, thresh=A.thresh, split=SPLIT, shared=A.shared, thr_units=A.thr_units,
                exported_detail_scales=len(An),
                note="threshold_multipliers in units of the plane's noise sigma (unused by gpu_pd)")
    np.savez(path, analysis=An.astype(np.float32), synthesis=Sy.astype(np.float32),
             threshold_multipliers=thr.astype(np.float32), meta=json.dumps(meta))
    # checks: gpu_pd's analysis planes == the blocks' analysis; W^-1 W = 1
    fb = gpu_pd.FilterBank2D(A.scales + 1, path)
    n = max(96, 2 * 2 ** A.scales + 8)                                 # reflect padding of the coarsest scale fits
    x = torch.randn(2, n, n, dtype=torch.float32)
    with torch.no_grad():
        p = fb.forward(x.to(gpu_pd.DEV)).cpu()
        err_inv = float((fb.inverse(p.to(gpu_pd.DEV)).cpu() - x).abs().max())
        c, i, err_a = x[:, None], 0, 0.0
        for j, b in enumerate(m.blocks):
            s = b._coarsen(c); w, c = c - s, s
            err_a = max(err_a, float((p[i + 1:i + 1 + A.filters].transpose(0, 1) - b._analyse(w)).abs().max()))
            i += 1 + A.filters
    return thr, err_inv, err_a


def evaluate(models, ds, n):
    """MSE per band (unit-noise units, so the noisy input scores ~1) for each model on n fixed draws"""
    names = [lab for lab, sl in band_slices(ds.b.shape[1], A.levels) for _ in range(sl.stop - sl.start)]
    res = {m: {} for m in ["noisy"] + list(models)}
    with torch.no_grad():
        eb = max(1, 64 * 256 ** 2 // A.patch ** 2)                    # 64 planes of 256^2 at a time, fewer if larger
        for i0 in range(0, n, eb):
            batch = [ds.draw(i) for i in range(i0, min(i0 + eb, n))]
            noisy = torch.from_numpy(np.stack([b[0] for b in batch]))[:, None].cuda()
            clean = torch.from_numpy(np.stack([b[1] for b in batch]))[:, None].cuda()
            outs = {"noisy": noisy} | {m: f(noisy) for m, f in models.items()}
            for m, o in outs.items():
                e = ((o - clean) ** 2).mean((1, 2, 3)).cpu().numpy()
                for (_, _, k, _), ei in zip(batch, e):
                    res[m].setdefault(names[k], []).append(float(ei))
    return {m: {b: float(np.mean(v)) for b, v in r.items()} | {"all": float(np.mean(sum(r.values(), [])))}
            for m, r in res.items()}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    setup_logging(os.path.join(OUT, "logs"), A.name)
    log.info(f"Command: {' '.join(sys.argv)}")
    for k, v in sorted(vars(A).items()):
        log.info(f"  {k:20s} {v}")
    seed_everything(A.seed)
    t0 = time.time()
    Wm = np.stack([cdf_fwd(torch.eye(32, dtype=torch.float64)[c].reshape(1, 32, 1, 1), A.levels)[0, :, 0, 0].numpy()
                   for c in range(32)], 1)                                      # (coefficient, channel)
    nk = np.sqrt((Wm ** 2).sum(1)).astype(np.float32)                           # each plane's noise / channel noise
    data = {s: load(g) for s, g in SPLIT.items()}
    log.info(f"data: {', '.join(f'{s} {len(d[2])} cubes' for s, d in data.items())}, planes {data['train'][0].shape[1:]}"
             f" ({time.time() - t0:.0f} s); band-plane noise / channel noise: {np.round(nk, 3).tolist()}")
    tr = Planes(data["train"][0], data["train"][1], A.samples_per_epoch, nk)
    va = Planes(data["valid"][0], data["valid"][1], A.valid_samples, nk, fixed=True, seed=1)
    te = Planes(data["test"][0], data["test"][1], A.test_samples, nk, fixed=True, seed=2)
    kw = dict(batch_size=A.batch_size, num_workers=A.workers, pin_memory=True, persistent_workers=True)
    trl = DataLoader(tr, shuffle=False, drop_last=True, **kw)
    val = DataLoader(va, shuffle=False, **kw)
    model = Learnlet2D(A.scales, A.kernel, A.filters, A.thresh, A.shared, A.thr_units).cuda()
    log.info(f"Learnlet2D: {A.scales - 1} learnt scales x {A.filters} filters of {A.kernel}x{A.kernel}, thresh "
             f"{A.thresh}{', shared across scales' if A.shared else ''}, {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    fit(A.epochs, model, get_loss("mse"), trl, val, torch.device("cuda"), OUT, scheduler_name="cosine_warmup",
        warmup_epochs=3, min_lr=1e-6, base_lr=A.lr, patience=A.patience)

    # test (B1) against the noisy input and the ImageNet learnlet, then export
    model.load_state_dict(torch.load(os.path.join(OUT, "weights_best.pt"), map_location="cuda")); model.eval()
    models = {"fire": model}
    try:
        sys.path.insert(0, A.imagenet)
        from learnlet.learnlet import Learnlet
        net = Learnlet(pretrained=False).cuda()
        net.load_state_dict(torch.load(os.path.join(A.imagenet, "weights/LEARNLET_FINAL_64_5_sc5_True_hard.pth"),
                                       map_location="cuda", weights_only=True)); net.eval()
        models["imagenet"] = lambda x: net(x, 1.0)
    except Exception as e:
        log.info(f"ImageNet learnlet baseline unavailable: {e}")
    res = evaluate(models, te, A.test_samples)
    json.dump(res, open(os.path.join(OUT, "test.json"), "w"), indent=1)
    for m, r in res.items():
        log.info(f"test (B1) MSE / noise^2, {m:9s}: all {r['all']:.4f}  " + "  ".join(f"{b} {r[b]:.4f}" for b in r if b != "all"))
    path = os.path.join(REPO, "dictionaries", f"{A.name}.npz")
    thr, e_inv, e_a = export(model, path)
    log.info(f"wrote {path}: thresholds (sigma) {np.round(thr, 3).tolist()}; gpu_pd check: |W^-1 W x - x| {e_inv:.1e}, "
             f"|analysis - model| {e_a:.1e}; total {(time.time() - t0) / 60:.1f} min")
