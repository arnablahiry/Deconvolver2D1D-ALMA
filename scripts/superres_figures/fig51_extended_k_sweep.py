"""
51: the EXTENDED-configuration pdtau model without a support mask, for several thresholds k, side by side.
Moment 0 over the line channels, Jy km/s arcsec^-2, full 16" field. Top: linear scale (99.9th percentile of
the k = 3 map); bottom: the same maps stretched to 5% of that, to show the low-level noise left in the model.
Prints, per k, the total and line-channel flux, the flux in the line-free channels and outside the reference
extended support mask (data/ngc3110/extended/pd/support_mask.npy; not used by these solves): both noise.

    MODELS=k3.npy,k5.npy,k7.npy LABELS=3,5,7 OUT=... python scripts/superres_figures/fig51_extended_k_sweep.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E = os.path.join(REPO, "data/experiments/13_extended_nomask")
KS = os.environ.get("LABELS", "3,5,7").split(",")
MODELS = os.environ.get("MODELS", ",".join(os.path.join(E, f"pdtau_k{k}.npy") for k in KS)).split(",")
OUT = os.environ.get("OUT", os.path.join(REPO, "figures/experiments/13_extended_nomask/51_extended_pdtau_k_sweep.png"))
CELL = 0.02
vel = 5073.9 + 25.0 * (np.arange(32) - 15.5)
L = [i for i, v in enumerate(vel) if 4750 <= v <= 5280]
FREE = [i for i, v in enumerate(vel) if v < 4780 or v > 5180]       # the extended noise window (common.noise_channels)
ref_mask = np.load(os.path.join(REPO, "data/ngc3110/extended/pd/support_mask.npy")).astype(bool)

maps = []
for k, p in zip(KS, MODELS):
    if not os.path.exists(p):
        print("missing", p); continue
    x = np.load(p).astype(np.float64)
    m = x[L].sum(0) * 25.0 / CELL ** 2
    maps.append((k, m))
    print(f"k = {k}: flux {x.sum():.2f} Jy (line channels {x[L].sum():.2f}); line-free channels {x[FREE].sum():.3f} Jy; "
          f"outside the reference mask {x[~ref_mask].sum():.2f} Jy ({x[~ref_mask].sum() / x.sum():.0%}); "
          f"line moment 0 {m.sum() * CELL ** 2:.1f} Jy km/s")

plt.rcParams.update({"font.family": "serif", "font.size": 12, "xtick.direction": "in", "ytick.direction": "in"})
n = len(maps)
fig, ax = plt.subplots(2, n, figsize=(5.6 * n + 1, 10.4), constrained_layout=True, squeeze=False)
vm = np.percentile(maps[0][1], 99.9)
ny = maps[0][1].shape[0]; half = ny * CELL / 2; ext = [half, -half, -half, half]
for c, (k, m) in enumerate(maps):
    for r, (top, title) in enumerate(((vm, f"pdtau, k = {k}, no mask"), (0.05 * vm, f"k = {k}: stretched to 5%"))):
        a = ax[r, c]
        im = a.imshow(m, origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=top, interpolation="nearest")
        a.set_title(title, fontsize=13)
        a.set_xlabel(r'$\Delta$RA ["]'); a.set_ylabel(r'$\Delta$Dec ["]')
        if c == n - 1:
            fig.colorbar(im, ax=ax[r, :], shrink=0.85, label=r"Jy km s$^{-1}$ arcsec$^{-2}$")
fig.suptitle("NGC 3110 extended configuration, raw pdtau model (moment 0, 4750-5280 km/s, 0.02″ pixels), "
             "no support mask, 20,000 iterations", fontsize=14)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=60)
print("saved", OUT)
