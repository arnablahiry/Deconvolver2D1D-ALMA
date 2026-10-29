#!/usr/bin/env python
"""
The learnt 2D atoms of a learnlet dictionary (dictionaries/<name>.npz, gpu_pd.FilterBank2D), per detail scale j:

  detects   the impulse response of coefficient plane k: A_jk (dilated 2**j) applied to the starlet plane w_j, i.e.
            the starlet wavelet psi_j filtered by A_jk -- the pattern the coefficient responds to (as a correlation)
  atom      what coefficient k puts back into the image, W^-1 of a delta in plane k = S_jk dilated 2**j, seen through
            the scale-j wavelet psi_j (S_jk (*) psi_j): coefficient planes are band-limited to scale j, so a lone
            dilated 5 x 5 tap pattern (holes between the taps) is never what is synthesised. First tile "w_j": the
            starlet coefficient's own atom, (delta - sum_k S_jk (*) A_jk) (*) psi_j, i.e. plain starlet thresholding
            (whose atom is psi_j, first tile of the "detects" row) reshaped by the learnt correction
Filters ordered by their share of the reconstruction ||S_jk (*) A_jk|| (largest first); --n per scale. Each tile has
its own symmetric colour scale; tiles are cropped to the atom's support at that scale (side given per row).

    python scripts/fig_learnlet_atoms.py [--name learnlet] [--n 16] [--scales 4]
    -> figures/learnlet_atoms/<name>.{png,pdf}
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "simple"))
import gpu_pd as G  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="learnlet"); ap.add_argument("--n", type=int, default=16)
ap.add_argument("--scales", type=int, help="detail scales shown (default: all in the file, at most 6)")
a = ap.parse_args()
path = os.path.join(G.DICT_DIR, f"{a.name}.npz")
z = np.load(path)
meta = json.loads(str(z["meta"]))
nsc = a.scales or min(len(z["analysis"]), 6)
fb = G.FilterBank2D(nsc, path)
F, k = z["analysis"].shape[1], z["analysis"].shape[-1]
n = min(a.n, F)
N = 2 * 6 * 2 ** (nsc - 1) + 9                                      # field holding the coarsest analysis response
c0 = N // 2
delta = torch.zeros((1, N, N), device=G.DEV, dtype=G.DT); delta[0, c0, c0] = 1
planes = fb.forward(delta)                                          # (P, 1, N, N)
start = [i for i, j in enumerate(fb.scale) if j < nsc and (i == 0 or fb.scale[i - 1] != j)]

rows = []
for j in range(nsc):
    i0 = start[j]
    A_, S_ = fb.A[j][:, 0], fb.S[j][:, 0]
    share = torch.nn.functional.conv2d(A_[None], S_[:, None].flip(-1, -2), padding=k - 1, groups=F)[0].flatten(1).norm(dim=1)
    order = torch.argsort(-share).cpu().numpy()[:n]
    det = [planes[i0, 0].cpu().numpy()] + [planes[i0 + 1 + o, 0].cpu().numpy() for o in order]    # psi_j first
    atoms = []
    for o in [-1] + list(order):                                    # -1: the starlet coefficient w_j itself
        e = torch.zeros_like(planes); e[i0 + 1 + o if o >= 0 else i0] = delta
        atoms.append(fb.forward(fb.inverse(e))[i0, 0].cpu().numpy())                # band-limited: (*) psi_j
    hd = ha = 6 * 2 ** j                                            # half-side of psi_j (*) a 5-tap filter at 2**j
    rows.append(dict(j=j, det=[d[c0 - hd:c0 + hd + 1, c0 - hd:c0 + hd + 1] for d in det],
                     atoms=[x[c0 - ha:c0 + ha + 1, c0 - ha:c0 + ha + 1] for x in atoms],
                     share=share[order].cpu().numpy(), hd=hd, ha=ha))

cols = n + 2
plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, ax = plt.subplots(2 * nsc, cols, figsize=(0.62 * cols + 1.2, 0.68 * 2 * nsc + 0.9),
                       gridspec_kw=dict(wspace=0.06, hspace=0.32))
for r in rows:
    for t, (key, label) in enumerate((("det", "detects"), ("atoms", "atom"))):
        row = ax[2 * r["j"] + t]
        tiles = [None] + (r["det"] if key == "det" else r["atoms"])
        for c, (axc, im) in enumerate(zip(row, tiles)):
            axc.set_xticks([]); axc.set_yticks([])
            if im is None:
                axc.axis("off")
                axc.text(1.0, 0.5, f"scale {r['j']}\n{label}\n{2 * r['hd'] + 1} px", transform=axc.transAxes,
                         ha="right", va="center", fontsize=8.5, fontweight="bold")
                continue
            v = np.abs(im).max() or 1
            axc.imshow(im, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
            if c == 1:
                axc.set_title("ψ$_j$" if key == "det" else "w$_j$", fontsize=8.5, pad=2)
                for sp in axc.spines.values():
                    sp.set_linewidth(1.6)
src = meta.get("source", "").replace(REPO + "/", "").replace("data/", "")
fig.suptitle(f"{a.name}: {F} filters/scale{' (shared across scales)' if meta.get('shared') else ''}, top {n} by "
             f"share of the reconstruction — red +, blue −, each tile its own scale\n{src[:110]}", fontsize=9.5, y=0.995)
os.makedirs(os.path.join(REPO, "figures", "learnlet_atoms"), exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(REPO, "figures", "learnlet_atoms", f"{a.name}.{ext}"), dpi=160, bbox_inches="tight")
print("->", os.path.join(REPO, "figures", "learnlet_atoms", f"{a.name}.png"))
print("share of the reconstruction (top 3 per scale):", [np.round(r["share"][:3], 3).tolist() for r in rows])
