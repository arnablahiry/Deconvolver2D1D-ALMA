#!/usr/bin/env python
"""
Experiment 16 figure: the low-level floor of the compact pdtau models (scripts/experiments/floor_study.py).
Moment 0 (all channels, like the GUI), asinh display of each panel's own 0.5-99.5% range, 16" field:
  top:    null tests -- pdtau k = 5 on cubes with NO emission: synthetic noise, real line-free channels,
          the same without continuum
  bottom: real data -- pdtau k = 5 line-free channels only (the floor), no positivity (signed, diverging),
          data-MAD thresholds k = 5 (line-free channels)
and prints the summary table of every run.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(REPO, "data/experiments/16_floor")
OUT = os.path.join(REPO, "figures/experiments/16_floor/16_floor.png")
FREE = [0, 1, 2] + list(range(24, 32))

print(f"{'run':26s} {'data':22s} {'lam':8s} {'k':>3s} {'pos':>4s} {'rw':>4s} {'flux':>8s} {'line':>8s} {'linefree':>9s} {'+':>8s} {'-':>8s}")
for p in sorted(glob.glob(os.path.join(D, "*.json"))):
    s = json.load(open(p))
    if not s.get("positivity", True):
        continue                                  # positivity is part of the method: no-positivity runs are not used
    print(f"{os.path.basename(p)[:-5]:26s} {s['data']:22s} {s.get('lam', 'noise'):8s} {s['k']:3g} {str(s['positivity'])[0]:>4s} "
          f"{str(s['reweight'])[0]:>4s} {s['flux']:8.3f} {s['flux_line']:8.3f} {s['flux_linefree']:9.3f} {s['pos']:8.2f} {s['neg']:8.2f}")
ref = os.path.join(REPO, "data/experiments/04_pd_tau_reweighting/compact/pdtau_k5.npy")


def load(name):
    p = os.path.join(D, name + ".npy")
    return np.load(p).astype(np.float64) if os.path.exists(p) else None


panels = [("null: synthetic noise\n(no emission)", load("null_synth_k5"), None, "seq"),
          ("null: real line-free channels\n(no line emission)", load("null_linefree_k5"), None, "seq"),
          ("null: real line-free,\ncontinuum removed", load("null_linefree_nocont_k5"), None, "seq"),
          ("real data, line-free channels only\n(the floor)", np.load(ref).astype(np.float64).reshape(32, 400, 2, 400, 2).sum((2, 4)), FREE, "seq"),
          ("real data, 60,000 iterations\n(line-free channels)", load("real_long_k5"), FREE, "seq"),
          ("real data, data-MAD thresholds\n(line-free channels)", load("real_datamad_k5"), FREE, "seq")]
asinh = lambda t: np.arcsinh(t / 0.1) / np.arcsinh(10)
plt.rcParams.update({"font.family": "serif", "font.size": 11})
fig, ax = plt.subplots(2, 3, figsize=(15, 10.4), constrained_layout=True)
for a, (title, x, ch, kind) in zip(ax.ravel(), panels):
    if x is None:
        a.set_title(title + "\n(not run)"); a.axis("off"); continue
    m = (x[ch] if ch else x).sum(0) * 25
    if kind == "div":
        v = np.percentile(np.abs(m), 99.5)
        im = a.imshow(m, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, extent=[8, -8, -8, 8])
    else:
        lo, hi = np.percentile(m, [0.5, 99.5])
        im = a.imshow(asinh(np.clip((m - lo) / (hi - lo), 0, 1)), origin="lower", cmap="inferno", extent=[8, -8, -8, 8])
    a.set_title(f"{title}\nflux {m.sum() / 25:.2f} Jy (in these channels)", fontsize=11)
    a.set_xlabel(r'$\Delta$RA ["]'); a.set_ylabel(r'$\Delta$Dec ["]')
fig.suptitle("NGC 3110 compact, pdtau k = 5: the low-level floor (moment 0, asinh display, each panel its own range)", fontsize=13)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=60)
print("saved", OUT)
