#!/usr/bin/env python
"""
Denoising performance of the FIRE-trained learnlet (scripts/train_learnlet_fire.py) on its own kind of data: the CDF
9/7 band planes of the FIRE [CII] cubes with white noise as in the real cubes (the training script's Planes: S/N of
the brightest beam-smoothed channel log-uniform in [20, 250]; each plane in units of its own noise), for the train
galaxies (A1 A2 A4 B2 C2), valid (A8) and test (B1), against
  * noisy     the input (MSE = 1 by construction)
  * imagenet  the ImageNet learnlet (vicbonj/learnlet, 64 filters), same unit-noise input
  * starlet<K> hard thresholding of the starlet detail planes at K x their noise (4 scales as the learnlet, coarse
              kept), for every K of --k
Metrics: MSE / noise^2 per band and as a denoising gain in dB, vs the sample's S/N; on planes carrying signal, the
normalised error ||x - clean||^2 / ||clean||^2. Then whole cubes: the test cubes' 32 band planes, noisy (the noise
added to the channels, so correlated across planes as in reality) -> each plane denoised -> inverse CDF, compared
in channel space and in moment 0.

    CUDA_VISIBLE_DEVICES=3 python scripts/eval_learnlet_fire.py [--name learnlet_fire] [--n 4096] [--k 3 5]
    -> results/learnlet_fire/<name>/eval[_<tag>].json, figures/learnlet_fire/<name>[_<tag>]_{metrics,examples,cube}.png
    --tag: e.g. a common test set for every model: --tag n600 --train-args "--data data/fire_cii_1x_n600_19.5pc
    --patch 256 [the model's own --shared --scales N]"
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="learnlet_fire"); ap.add_argument("--n", type=int, default=4096)
ap.add_argument("--k", type=float, nargs="+", default=[3.0, 5.0],
                help="starlet baselines: hard thresholds in units of each plane's noise")
ap.add_argument("--cube-snr", type=float, default=30.0, help="whole-cube test: S/N of the brightest channel")
ap.add_argument("--tag", default="", help="suffix of the outputs (another test set)")
ap.add_argument("--train-args", default="", help="the model's own train_learnlet_fire.py options, e.g. '--shared --scales 7'")
E = ap.parse_args()
argv, sys.argv = sys.argv, [sys.argv[0], "--name", E.name] + E.train_args.split()                 # import the training script's pieces
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_learnlet_fire as T  # noqa: E402
sys.argv = argv
from gpu_pd import band_slices, cdf_fwd, cdf_inv  # noqa: E402

REPO, OUT = T.REPO, T.OUT
FIG = os.path.join(REPO, "figures", "learnlet_fire"); os.makedirs(FIG, exist_ok=True)
SUF = f"_{E.tag}" if E.tag else ""
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NZ = 32
Wm = np.stack([cdf_fwd(torch.eye(NZ, dtype=torch.float64)[c].reshape(1, NZ, 1, 1), T.A.levels)[0, :, 0, 0].numpy()
               for c in range(NZ)], 1)
nk = np.sqrt((Wm ** 2).sum(1)).astype(np.float32)
BAND = [lab for lab, sl in band_slices(NZ, T.A.levels) for _ in range(sl.stop - sl.start)]
BANDS = [lab for lab, _ in band_slices(NZ, T.A.levels)][::-1]                # d1 ... d5, a

# ------------------------------------------------------------------ denoisers (unit-noise input)
fire = T.Learnlet2D(T.A.scales, T.A.kernel, T.A.filters, T.A.thresh, T.A.shared, T.A.thr_units).to(dev)
fire.load_state_dict(torch.load(os.path.join(OUT, "weights_best.pt"), map_location=dev)); fire.eval()
sys.path.insert(0, T.A.imagenet)
from learnlet.learnlet import Learnlet  # noqa: E402
inet = Learnlet(pretrained=False).to(dev)
inet.load_state_dict(torch.load(os.path.join(T.A.imagenet, "weights/LEARNLET_FINAL_64_5_sc5_True_hard.pth"),
                                map_location=dev, weights_only=True)); inet.eval()
B3 = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32, device=dev) / 16
K2 = torch.outer(B3, B3).view(1, 1, 5, 5)


def starlet_planes(x, J):
    out, c = [], x
    for j in range(J):
        s = F.conv2d(F.pad(c, [2 * 2 ** j] * 4, mode="reflect"), K2, dilation=2 ** j)
        out.append(c - s); c = s
    return out, c


with torch.no_grad():                                         # each starlet plane's noise for unit white noise
    z = torch.randn(64, 1, 256, 256, device=dev)
    SIG_J = [float(w.std()) for w in starlet_planes(z, T.A.scales - 1)[0]]


def starlet_hard(k):
    def f(x):
        ws, c = starlet_planes(x, T.A.scales - 1)
        return sum(w * (w.abs() > k * s) for w, s in zip(ws, SIG_J)) + c
    return f


DEN = {"fire": fire, "imagenet": lambda x: inet(x, 1.0)} | {f"starlet{k:g}": starlet_hard(k) for k in E.k}
GREENS = ["#1baf7a", "#0b6b45", "#7fd8b0", "#04402a"]
COL = {"noisy": "0.6", "fire": "#9b3fb5", "imagenet": "#2a78d6"} | {f"starlet{k:g}": GREENS[i % 4] for i, k in enumerate(E.k)}
LAB = {"noisy": "noisy input", "fire": "learnlet (FIRE, 16)", "imagenet": "learnlet (ImageNet, 64)"} | \
      {f"starlet{k:g}": f"starlet, hard {k:g}σ" for k in E.k}


# ------------------------------------------------------------------ band-plane metrics
def run(split, seed):
    b, peak, _ = T.load(T.SPLIT[split])
    ds = T.Planes(b, peak, E.n, nk, fixed=True, seed=seed)
    rec = []
    with torch.no_grad():
        for i0 in range(0, E.n, 64):
            batch = [ds.draw(i) for i in range(i0, min(i0 + 64, E.n))]
            noisy = torch.from_numpy(np.stack([x[0] for x in batch]))[:, None].to(dev)
            clean = torch.from_numpy(np.stack([x[1] for x in batch]))[:, None].to(dev)
            outs = {"noisy": noisy} | {m: f(noisy) for m, f in DEN.items()}
            ce = (clean ** 2).mean((1, 2, 3)).cpu().numpy()
            errs = {m: ((o - clean) ** 2).mean((1, 2, 3)).cpu().numpy() for m, o in outs.items()}
            for j, (_, _, k, S) in enumerate(batch):
                rec.append(dict(band=BAND[k], snr=S, signal=float(ce[j]), **{m: float(e[j]) for m, e in errs.items()}))
    return rec, (ds, b, peak)


res, keep = {}, {}
for split, seed in (("train", 11), ("valid", 1), ("test", 2)):
    res[split], keep[split] = run(split, seed)
    print(split, "done", flush=True)


def summary(rec):
    out = {}
    for m in ["noisy"] + list(DEN):
        e = np.array([r[m] for r in rec])
        sig = np.array([r["signal"] for r in rec]); has = sig > 0.05          # planes with signal above 0.05 noise^2
        out[m] = dict(mse=float(e.mean()), gain_db=float(10 * np.log10(1 / e.mean())),
                      nmse_signal=float(np.median(e[has] / sig[has])) if has.any() else None,
                      **{f"mse_{b}": float(np.mean([r[m] for r in rec if r["band"] == b])) for b in BANDS})
    return out


S = {s: summary(r) for s, r in res.items()}
for s, d in S.items():
    for m, v in d.items():
        print(f"{s:5s} {m:9s} MSE {v['mse']:.4f}  gain {v['gain_db']:5.1f} dB  median NMSE (signal planes) "
              f"{v['nmse_signal'] if v['nmse_signal'] is None else round(v['nmse_signal'], 3)}  "
              + " ".join(f"{b} {v['mse_' + b]:.4f}" for b in BANDS))

# ------------------------------------------------------------------ whole test cubes
def cube_test(ds, b, peak, ci, snr, seed=5):
    """noise in the channels of cube ci's 256 px centre crop -> CDF -> each plane denoised in units of its noise
    -> inverse; returns clean, noisy, {model: denoised} channel cubes"""
    n0 = (b.shape[2] - 256) // 2
    bc = torch.from_numpy(b[ci, :, n0:n0 + 256, n0:n0 + 256].copy())
    clean = cdf_inv(bc[None].double(), T.A.levels)[0].float()                # channels (the CDF is exact)
    sig_c = peak[ci] / snr
    noisy = clean + sig_c * torch.from_numpy(np.random.default_rng(seed).standard_normal(clean.shape).astype(np.float32))
    bn = cdf_fwd(noisy[None].double(), T.A.levels)[0].float()
    s = torch.from_numpy(sig_c * nk)[:, None, None]
    outs = {}
    with torch.no_grad():
        for m, f in DEN.items():
            d = f((bn / s)[:, None].to(dev))[:, 0].cpu() * s
            outs[m] = cdf_inv(d[None].double(), T.A.levels)[0].float()
    return clean.numpy(), noisy.numpy(), {m: o.numpy() for m, o in outs.items()}, sig_c


ds, b, peak = keep["test"]
names = T.load(T.SPLIT["test"])[2]
cub = []
for ci in range(len(b)):
    cl, no, ou, sc = cube_test(ds, b, peak, ci, E.cube_snr)
    rms = lambda x: float(np.sqrt(((x - cl) ** 2).mean()) / sc)
    m0 = lambda x: x.sum(0)
    cub.append(dict(cube=names[ci], noisy=rms(no), **{m: rms(o) for m, o in ou.items()},
                    **{f"m0_{m}": float(np.sqrt(((m0(o) - m0(cl)) ** 2).mean()) / (sc * np.sqrt(NZ))) for m, o in
                       (("noisy", no), *ou.items())}))
C = {k: float(np.mean([c[k] for c in cub])) for k in cub[0] if k != "cube"}
print("whole test cubes (S/N", E.cube_snr, "): channel rms / sigma", {k: round(v, 3) for k, v in C.items() if not k.startswith("m0")},
      "moment-0 rms / (sigma sqrt(nz))", {k: round(v, 3) for k, v in C.items() if k.startswith("m0")})
json.dump(dict(bands=S, cubes=dict(per_cube=cub, mean=C, snr=E.cube_snr), k_starlet=E.k, n=E.n),
          open(os.path.join(OUT, f"eval{'_' + E.tag if E.tag else ''}.json"), "w"), indent=1)

# ------------------------------------------------------------------ figures
plt.rcParams.update({"font.family": "serif", "font.size": 10, "xtick.direction": "in", "ytick.direction": "in"})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
rec = res["test"]; snr = np.array([r["snr"] for r in rec]); edges = np.geomspace(T.A.snr_min, T.A.snr_max, 8)
xc = np.sqrt(edges[:-1] * edges[1:])
for m in ["noisy"] + list(DEN):
    e = np.array([r[m] for r in rec])
    g = [10 * np.log10(1 / e[(snr >= lo) & (snr < hi)].mean()) for lo, hi in zip(edges[:-1], edges[1:])]
    ax[0].plot(xc, g, "o-", color=COL[m], label=LAB[m])
    ax[1].plot(range(len(BANDS)), [10 * np.log10(1 / S["test"][m]["mse_" + bb]) for bb in BANDS], "o-", color=COL[m])
w = 0.8 / len(DEN)
for j, m in enumerate(DEN):
    ax[2].bar(np.arange(3) + (j - (len(DEN) - 1) / 2) * w, [S[s][m]["gain_db"] for s in ("train", "valid", "test")], w, color=COL[m], label=LAB[m])
ax[0].set_xscale("log"); ax[0].set_xlabel("S/N of the brightest channel"); ax[0].set_ylabel("denoising gain (dB)")
ax[0].set_title("test galaxy (B1): gain vs S/N"); ax[0].legend(fontsize=8)
ax[1].set_xticks(range(len(BANDS))); ax[1].set_xticklabels([("a$_{%d}$" % T.A.levels) if bb == "a" else f"d$_{bb[1:]}$" for bb in BANDS])
ax[1].set_ylabel("denoising gain (dB)"); ax[1].set_title("test galaxy (B1): gain per CDF band")
ax[2].set_xticks(range(3)); ax[2].set_xticklabels(["train (A1 A2 A4 B2 C2)", "valid (A8)", "test (B1)"], fontsize=9)
ax[2].set_ylabel("denoising gain (dB)"); ax[2].set_title("all bands, by split"); ax[2].legend(fontsize=8)
ax[2].set_ylim(min(S[s][m]["gain_db"] for s in S for m in DEN) - 2, None)
fig.tight_layout(); fig.savefig(os.path.join(FIG, f"{E.name}{SUF}_metrics.png"), dpi=130); plt.close(fig)

# examples: test planes with signal, at three S/N
cands = [i for i, r in enumerate(rec) if r["signal"] > 0.5]
pick = []
for b_ in ("d1", "d3", "a"):
    ii = [i for i in cands if rec[i]["band"] == b_]
    if ii:
        pick.append(sorted(ii, key=lambda i: rec[i]["snr"])[len(ii) // 2])
NC = 3 + len(DEN)
fig, axs = plt.subplots(len(pick), NC, figsize=(2.65 * NC, 2.8 * len(pick)))
axs = np.atleast_2d(axs)
for r_, i in enumerate(pick):
    noisy, clean, k, Sn = ds.draw(i)
    x = torch.from_numpy(noisy)[None, None].to(dev)
    with torch.no_grad():
        outs = {m: f(x)[0, 0].cpu().numpy() for m, f in DEN.items()}
    m_ = np.percentile(np.abs(clean), 99.7)
    for c_, (t, im) in enumerate([("clean", clean), ("noisy", noisy)] + [(LAB[m], outs[m]) for m in DEN]
                                 + [("FIRE − clean", outs["fire"] - clean)]):
        a = axs[r_, c_]; a.set_xticks([]); a.set_yticks([])
        a.imshow(im, origin="lower", cmap="RdBu_r", vmin=-m_, vmax=m_, interpolation="nearest")
        a.set_title(t if r_ == 0 else "", fontsize=9)
        if 2 <= c_ < 2 + len(DEN):
            mm = list(DEN)[c_ - 2]; a.text(0.03, 0.03, f"MSE {((outs[mm] - clean) ** 2).mean():.3f}", transform=a.transAxes, fontsize=8)
    axs[r_, 0].set_ylabel(f"{BAND[k]}, S/N {Sn:.0f}", fontsize=9)
fig.suptitle("test galaxy B1: band planes in units of their noise (noisy MSE = 1)", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, f"{E.name}{SUF}_examples.png"), dpi=110); plt.close(fig)

# whole cube: the test cube with most flux, a bright channel and moment 0
ci = int(np.argmax(peak))
cl, no, ou, sc = cube_test(ds, b, peak, ci, E.cube_snr)
ch = int(np.argmax(cl.sum((1, 2))))
fig, axs = plt.subplots(2, 2 + len(DEN), figsize=(3.2 * (2 + len(DEN)), 6.6))
for r_, (t, f_) in enumerate([(f"channel {ch}", lambda x: x[ch]), ("moment 0 (sum over channels)", lambda x: x.sum(0))]):
    ref = f_(cl); vmax = np.percentile(ref, 99.9)
    for c_, (lab, im) in enumerate([("clean", cl), ("noisy", no)] + [(LAB[m], ou[m]) for m in DEN]):
        a = axs[r_, c_]; a.set_xticks([]); a.set_yticks([])
        a.imshow(f_(im), origin="lower", cmap="inferno", norm=PowerNorm(0.4, vmin=0, vmax=vmax), interpolation="nearest")
        a.set_title(f"{lab}" if r_ == 0 else "", fontsize=9)
        if c_:
            a.text(0.03, 0.03, f"rms err {np.sqrt(((f_(im) - ref) ** 2).mean()) / (sc * (1 if r_ == 0 else np.sqrt(NZ))):.3f}",
                   transform=a.transAxes, color="w", fontsize=8)
    axs[r_, 0].set_ylabel(t, fontsize=10)
fig.suptitle(f"whole test cube {names[ci]} (256 px, 32 channels), white noise in the channels, brightest channel "
             f"S/N {E.cube_snr:g}: CDF -> each band plane denoised -> inverse CDF (errors in units of the noise)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, f"{E.name}{SUF}_cube.png"), dpi=110); plt.close(fig)
print("figures in", FIG)
