#!/usr/bin/env python
"""
Experiment 16b figures (scripts/experiments/toy_floor.py runs): the low-level floor and the threshold
calibrations, on the GUI's synthetic disc, on its noise alone, and on the real compact cube.

    figures/experiments/16_floor/16b_synthetic.png   truth, dirty, models (moment 0) / model - truth
    figures/experiments/16_floor/16b_noise_only.png  models of the disc's noise alone (common scale)
    figures/experiments/16_floor/16b_compact.png     models (moment 0, line channels) / line-free channels
Panel titles carry each run's numbers. Moment 0 = sum over channels x channel width.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "gui"))
T = os.path.join(REPO, "data/experiments/16_floor/toy")
F = os.path.join(REPO, "figures/experiments/16_floor")
os.makedirs(F, exist_ok=True)
asinh = lambda t: np.arcsinh(t / 0.1) / np.arcsinh(10)
plt.rcParams.update({"font.family": "serif", "font.size": 10})


def run(name):
    p = os.path.join(T, name + ".npy")
    if not os.path.exists(p):
        return None, None
    return np.load(p).astype(np.float64), json.load(open(p.replace(".npy", ".json")))


def show(a, img, lo, hi, title, div=False):
    if div:
        a.imshow(img, origin="lower", cmap="RdBu_r", vmin=-hi, vmax=hi)
    else:
        a.imshow(asinh(np.clip((img - lo) / (hi - lo), 0, 1)), origin="lower", cmap="inferno", vmin=0, vmax=1)
    a.set_title(title, fontsize=9); a.set_xticks([]); a.set_yticks([])


def blank(a, title):
    a.set_title(title + "\n(not run yet)", fontsize=9); a.axis("off")


# ---------------------------------------------------------------- synthetic disc
from deconvgui.toy import make_toy  # noqa: E402
dirty, psf, truth, meta = make_toy()
dv = meta["width_kms"]
tm0 = truth.sum(0) * dv
hi = np.percentile(tm0, 99.5)
models = [("pdtau, noise-model λ (GUI)", "toy_pdtau_noise_k5"), ("pdtau, true-noise λ", "toy_pdtau_true_k5"),
          ("pdtau, certificate λ", "toy_pdtau_cert_k5"), ("pdtau, data-MAD λ", "toy_pdtau_datamad_k5"),
          ("FISTA, noise-model λ", "toy_fista_noise_k5"), ("FISTA, data-MAD λ", "toy_fista_datamad_k5")]
fig, ax = plt.subplots(2, 2 + len(models), figsize=(2.5 * (2 + len(models)), 5.6), constrained_layout=True)
show(ax[0, 0], tm0, 0, hi, f"true sky\n{truth.sum():.0f} Jy")
show(ax[0, 1], dirty.sum(0) * dv, *np.percentile(dirty.sum(0) * dv, [0.5, 99.5]), "dirty")
ax[1, 0].axis("off"); ax[1, 1].axis("off")
for c, (lab, name) in enumerate(models, start=2):
    x, s = run(name)
    if x is None:
        blank(ax[0, c], lab); ax[1, c].axis("off"); continue
    m0 = x.sum(0) * dv
    show(ax[0, c], m0, 0, hi, f"{lab}, k = 5\n{s['flux']:.0f} Jy; outside truth {s['flux_out']:.1f} Jy")
    show(ax[1, c], m0 - tm0, 0, hi * 0.5, f"model − truth\nerror {s['err']:.2f} (mom0 {s['err_mom0']:.2f})", div=True)
fig.suptitle("GUI synthetic disc (128² × 40, truth known): moment 0, asinh on the true sky's scale; bottom: model − truth", fontsize=11)
fig.savefig(os.path.join(F, "16b_synthetic.png"), dpi=70); plt.close(fig)
print("wrote 16b_synthetic.png")

# ---------------------------------------------------------------- noise only
nulls = [("pdtau, true-noise λ", "toy_null_pdtau_true_k5"), ("pdtau, certificate λ", "toy_null_pdtau_cert_k5"),
         ("FISTA, data-MAD λ", "toy_null_fista_datamad_k5")]
loaded = [(lab, *run(name)) for lab, name in nulls]
vmax = max([np.percentile(x.sum(0) * dv, 99.9) for _, x, _ in loaded if x is not None and x.sum() > 0] + [1e-6])
fig, ax = plt.subplots(1, len(nulls) + 1, figsize=(3.3 * (len(nulls) + 1), 3.8), constrained_layout=True)
sys.path.insert(0, os.path.join(REPO, "simple"))
from uv_to_image import ShiftInvariantOperator  # noqa: E402
noise = dirty - ShiftInvariantOperator(psf.astype(np.float64), dirty.astype(np.float64)).apply(truth.astype(np.float64))
nm0 = noise.sum(0) * dv
v = np.percentile(np.abs(nm0), 99.5)
ax[0].imshow(nm0, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
ax[0].set_title("input: the disc's noise only\n(dirty-cube noise, moment 0)", fontsize=9); ax[0].set_xticks([]); ax[0].set_yticks([])
for a, (lab, x, s) in zip(ax[1:], loaded):
    if x is None:
        blank(a, lab); continue
    a.imshow(x.sum(0) * dv, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
    a.set_title(f"{lab}, k = 5\nmodel {s['flux']:.2f} Jy", fontsize=9); a.set_xticks([]); a.set_yticks([])
fig.suptitle("Noise only: what each method makes of pure noise (moment 0, common linear scale; zero = black)", fontsize=11)
fig.savefig(os.path.join(F, "16b_noise_only.png"), dpi=70); plt.close(fig)
print("wrote 16b_noise_only.png")

# ---------------------------------------------------------------- real compact
L = list(range(3, 24)); FREE = [0, 1, 2] + list(range(24, 32))
E = os.path.join(REPO, "data/experiments")
compact = [("pdtau, noise-model λ", os.path.join(E, "04_pd_tau_reweighting/compact/pdtau_k5.npy"), True),
           ("pdtau, certificate λ", os.path.join(T, "compact_pdtau_cert_k5.npy"), False),
           ("pdtau, data-MAD λ", os.path.join(E, "16_floor/real_datamad_k5.npy"), False),
           ("FISTA, noise-model λ", os.path.join(T, "compact_fista_noise_k5.npy"), False),
           ("FISTA, data-MAD λ", os.path.join(T, "compact_fista_datamad_k5.npy"), False)]
fig, ax = plt.subplots(2, len(compact), figsize=(3.2 * len(compact), 6.6), constrained_layout=True)
ref = None
for c, (lab, p, fine) in enumerate(compact):
    if not os.path.exists(p):
        blank(ax[0, c], lab); ax[1, c].axis("off"); continue
    x = np.load(p).astype(np.float64)
    if fine:
        x = x.reshape(32, 400, 2, 400, 2).sum((2, 4))
    m_line, m_free = x[L].sum(0) * 25, x[FREE].sum(0) * 25
    if ref is None:
        ref = (np.percentile(m_line, 99.5), np.percentile(m_free, 99.5))
    show(ax[0, c], m_line, 0, ref[0], f"{lab}, k = 5\nline channels: {x[L].sum():.1f} Jy")
    show(ax[1, c], m_free, 0, ref[1], f"line-free channels (floor): {x[FREE].sum():.2f} Jy")
fig.suptitle("NGC 3110 compact (0.04″ grid): top, line moment 0; bottom, line-free channels (only floor can be there); "
             "asinh, each row on the first column's scale", fontsize=11)
fig.savefig(os.path.join(F, "16b_compact.png"), dpi=70); plt.close(fig)
print("wrote 16b_compact.png")
