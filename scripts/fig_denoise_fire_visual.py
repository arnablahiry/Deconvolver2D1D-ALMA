#!/usr/bin/env python
"""
Visual denoising comparison, learnlet (FIRE-trained, scripts/train_learnlet_fire.py) vs starlet hard thresholding,
on whole held-out FIRE cubes (valid A8, test B1): white noise in the channels (brightest beam-smoothed channel at
the S/N given per row, the real cubes' range), CDF 9/7 along velocity -> every band plane denoised in units of its
own noise -> inverse CDF. Per cube: its brightest channel and its moment 0; columns clean, noisy, starlet 3 sigma,
starlet 5 sigma, learnlet; the rms error (units of the noise) in each panel. Every row its own colour scale (power
stretch), set by the clean image.

    CUDA_VISIBLE_DEVICES=3 python scripts/fig_denoise_fire_visual.py [--name learnlet_fire]
    -> figures/learnlet_fire/<name>_visual.{png,pdf}
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

ap = argparse.ArgumentParser(); ap.add_argument("--name", default="learnlet_fire")
ap.add_argument("--patch", type=int, default=256); E = ap.parse_args()
argv, sys.argv = sys.argv, [sys.argv[0], "--name", E.name]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_learnlet_fire as T  # noqa: E402
sys.argv = argv
from gpu_pd import cdf_fwd, cdf_inv  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NZ, L = 32, T.A.levels
Wm = np.stack([cdf_fwd(torch.eye(NZ, dtype=torch.float64)[c].reshape(1, NZ, 1, 1), L)[0, :, 0, 0].numpy() for c in range(NZ)], 1)
nk = np.sqrt((Wm ** 2).sum(1)).astype(np.float32)
fire = T.Learnlet2D(T.A.scales, T.A.kernel, T.A.filters, T.A.thresh).to(dev)
fire.load_state_dict(torch.load(os.path.join(T.OUT, "weights_best.pt"), map_location=dev)); fire.eval()
K2 = (lambda b: torch.outer(b, b).view(1, 1, 5, 5))(torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32, device=dev) / 16)


def planes(x, J):
    out, c = [], x
    for j in range(J):
        s = F.conv2d(F.pad(c, [2 * 2 ** j] * 4, mode="reflect"), K2, dilation=2 ** j); out.append(c - s); c = s
    return out, c


with torch.no_grad():
    SIG = [float(w.std()) for w in planes(torch.randn(32, 1, 256, 256, device=dev), T.A.scales - 1)[0]]
star = lambda k: (lambda x: (lambda ws, c: sum(w * (w.abs() > k * s) for w, s in zip(ws, SIG)) + c)(*planes(x, T.A.scales - 1)))
DEN = {"starlet, hard 3σ": star(3.0), "starlet, hard 5σ": star(5.0), "learnlet (FIRE)": fire}

# (split, cube name, S/N of the brightest channel)
ROWS = [("test", "B1_s80_los0", 20), ("test", "B1_s85_los2", 50), ("valid", "A8_s75_los1", 100), ("test", "B1_s60_los1", 200)]
data = {s: T.load(T.SPLIT[s]) for s in {r[0] for r in ROWS}}


def denoise(split, name, snr, seed=3):
    b, peak, names = data[split]; ci = names.index(name)
    n0 = (b.shape[2] - E.patch) // 2
    clean = cdf_inv(torch.from_numpy(b[ci, :, n0:n0 + E.patch, n0:n0 + E.patch].copy())[None].double(), L)[0].float()
    sc = peak[ci] / snr
    noisy = clean + sc * torch.from_numpy(np.random.default_rng(seed).standard_normal(clean.shape).astype(np.float32))
    bn = cdf_fwd(noisy[None].double(), L)[0].float(); s = torch.from_numpy(sc * nk)[:, None, None]
    out = {}
    with torch.no_grad():
        for m, f in DEN.items():
            out[m] = cdf_inv((f((bn / s)[:, None].to(dev))[:, 0].cpu() * s)[None].double(), L)[0].float().numpy()
    return clean.numpy(), noisy.numpy(), out, sc


plt.rcParams.update({"font.family": "serif", "font.size": 9})
cols = ["clean", "noisy"] + list(DEN)
fig, axs = plt.subplots(2 * len(ROWS), len(cols), figsize=(2.3 * len(cols), 2.3 * 2 * len(ROWS)),
                        gridspec_kw=dict(wspace=0.03, hspace=0.06))
for r, (split, name, snr) in enumerate(ROWS):
    cl, no, out, sc = denoise(split, name, snr)
    ch = int(np.argmax(cl.sum((1, 2))))
    for rr, (what, f_, norm_) in enumerate([(f"channel {ch}", lambda x: x[ch], 1.0), ("moment 0", lambda x: x.sum(0), np.sqrt(NZ))]):
        ref = f_(cl); vmax = np.percentile(ref, 99.8)
        for c, m in enumerate(cols):
            im = {"clean": cl, "noisy": no}.get(m, out.get(m))
            a = axs[2 * r + rr, c]; a.set_xticks([]); a.set_yticks([])
            a.imshow(f_(im), origin="lower", cmap="inferno", norm=PowerNorm(0.4, vmin=0, vmax=vmax), interpolation="nearest")
            if c:
                a.text(0.03, 0.03, f"{np.sqrt(((f_(im) - ref) ** 2).mean()) / (sc * norm_):.3f}", transform=a.transAxes,
                       color="w", fontsize=8)
            if r == 0 and rr == 0:
                a.set_title(m, fontsize=10, fontweight="bold")
        axs[2 * r + rr, 0].set_ylabel(f"{name} ({split})\nS/N {snr}, {what}" if rr == 0 else what, fontsize=8.5)
fig.subplots_adjust(top=0.965)
fig.suptitle("Denoising held-out FIRE [CII] cubes: band planes of the CDF along velocity denoised, then inverted; "
             "number = rms error in units of the noise", fontsize=11, y=0.985)
out = os.path.join(T.REPO, "figures", "learnlet_fire", f"{E.name}_visual.png")
fig.savefig(out, dpi=120, bbox_inches="tight"); fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", out)
