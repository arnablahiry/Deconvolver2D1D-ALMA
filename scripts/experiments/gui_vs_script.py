#!/usr/bin/env python
"""
Why does the GUI's pdtau (k = 5) on NGC 3110 compact look so much smoother than the scripts' pdtau
(data/experiments/04_pd_tau_reweighting/compact/pdtau_k5)? Controlled runs, one setting changed at a time;
all: pdtau (primal-dual, reweighting reference tau*lambda), noise-model lambda, k = 5, 20,000 iterations,
reweighting every 2000 from 6000 (eps 1), positivity, 2x2 rebinning to 0.04".

  gui cube    = the GUI dataset (CASA briggs, 60 x 10 km/s, 10" crop): /scratch/alahiry/deconvgui/datasets/...
  script cube = data/ngc3110/compact/cube (natural, 32 x 25 km/s, 16")

    gui_vs_script.py --run gui_J5_L2      the GUI's own setup (auto scales J 5, L 2; auto line-free channels)
    gui_vs_script.py --run gui_J5_L5      + spectral axis zero-padded 60 -> 64 channels so that L = 5 fits
    gui_vs_script.py --run gui_J7_L5      + 7 starlet scales (the scripts' J)
    gui_vs_script.py --run script_J5_L2   the script cube with the GUI's dictionary depth (J 5, L 2)
    gui_vs_script.py --run script_J7_L2   the script cube, J 7, only L reduced to 2
    gui_vs_script.py --run script_crop10  the script cube cut to the central 10" (PSF to 2x that), as the GUI's crop
    gui_vs_script.py --run gui_bin3       the GUI cube averaged 3 channels -> 20 x 30 km/s (PSF likewise)
    gui_vs_script.py --run gui_natural    the GUI imaging redone with natural weighting (else identical: 60 x 10 km/s, 10")
    gui_vs_script.py --plot               -> figures/experiments/17_gui_vs_script/

Each run: data/experiments/17_gui_vs_script/<run>.npy (line moment 0, central 10", 0.02" grid,
Jy km/s arcsec^-2) + .json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(REPO, "simple")); sys.path.insert(0, os.path.join(REPO, "gui"))
GUI_DS = "/scratch/alahiry/deconvgui/datasets/20261005-212918-NGC3110-crop-500px-e003"
OUT = os.path.join(REPO, "data/experiments/17_gui_vs_script")
FIG = os.path.join(REPO, "figures/experiments/17_gui_vs_script")
RUNS = {"gui_J5_L2": ("gui", 5, 2, 0), "gui_J5_L5": ("gui", 5, 5, 2), "gui_J7_L5": ("gui", 7, 5, 2),
        "script_J5_L2": ("script", 5, 2, 0), "script_J7_L2": ("script", 7, 2, 0),
        "script_crop10": ("script_crop10", 5, 5, 0), "gui_bin3": ("gui_bin3", 5, 2, 0),
        "gui_natural": ("gui_natural", 5, 2, 0)}
NATURAL = "/scratch/alahiry/gui_vs_script_natural"     # the GUI imaging with weighting natural (image.py there)
K, N_ITER, CELL = 5.0, 20000, 0.02


def load(cube):
    """(dirty, psf, dv, line channels, line-free channels) at 0.02", the central 10" (500 px) for the script cube's
    output (the solve itself uses the cube's whole field)"""
    if cube == "gui_bin3":
        d, p, _, _, _, _ = load("gui")
        d = d.reshape(20, 3, *d.shape[1:]).mean(1); p = p.reshape(20, 3, *p.shape[1:]).mean(1)
        vel = 4752.0 + 10.0 * (3 * np.arange(20) + 1)
        from deconvgui.worker import _auto_line_free, _auto_scales
        import gpu_pd as G
        db, _ = G.rebin(d, p, 2)
        free = _auto_line_free(G, G.to_t(db), _auto_scales(*db.shape[1:], db.shape[0])[0])
        return d, p, 30.0, [i for i, v in enumerate(vel) if 4750 <= v <= 5280], free, None
    if cube == "script_crop10":
        d, p, dv, line, free, _ = load("script")
        py, px = np.unravel_index(np.argmax(p[0]), p.shape[1:])
        return d[:, 150:650, 150:650], p[:, py - 500:py + 500, px - 500:px + 500], dv, line, free, None
    if cube in ("gui", "gui_natural"):
        from deconvgui.worker import _auto_line_free, _auto_scales
        import gpu_pd as G
        src = GUI_DS if cube == "gui" else NATURAL
        d = np.load(os.path.join(src, "dirty.npy")).astype(np.float64)
        p = np.load(os.path.join(src, "psf.npy")).astype(np.float64)
        vel = 4752.0 + 10.0 * np.arange(d.shape[0])
        db, _ = G.rebin(d, p, 2)
        J, _ = _auto_scales(*db.shape[1:], db.shape[0])
        free = _auto_line_free(G, G.to_t(db), J)          # as the GUI: [0, 1, 51..59]
        return d, p, 10.0, [i for i, v in enumerate(vel) if 4750 <= v <= 5280], free, None
    from common import read, line_channels, noise_channels
    d, p = read("ngc3110", "compact", "dirty_cube.fits"), read("ngc3110", "compact", "dirty_beam_cube.fits")
    return d, p, 25.0, line_channels("ngc3110"), noise_channels("ngc3110", "compact"), np.s_[150:650, 150:650]


def run(name):
    import torch
    import gpu_pd as G
    from common import RW
    cube, J, L, pad = RUNS[name]
    d, p, dv, line, free, crop = load(cube)
    db, pb = G.rebin(d, p, 2)
    if pad:                                   # zero-pad the spectral axis (no data there: the model stays 0)
        db = np.concatenate([np.zeros((pad,) + db.shape[1:]), db, np.zeros((pad,) + db.shape[1:])])
        pb = np.concatenate([pb[:1].repeat(pad, 0), pb, pb[-1:].repeat(pad, 0)])
        free = [c + pad for c in free]
    op = G.Op(pb, db)
    D = G.Dict2D1D(op.d.shape, J, L)
    lam = G.noise_lambdas(D, op.d, free, k=K)[0]
    beta = G.field_lipschitz(op)
    t0 = time.time()
    x, _, _, _ = G.solve_pd(op, D, lam, n_iter=N_ITER, beta=beta, reweight=dict(RW, ref=lam / beta), print_every=4000,
                            log=lambda m: print(f"[{name}] {m}", flush=True))
    x = x.cpu().numpy().astype(np.float64)
    if pad:
        x = x[pad:-pad]
    x = G.upsample(x, 2)                                       # 0.02", Jy/pixel
    m0 = x[line].sum(0) * dv / CELL ** 2                       # line moment 0, Jy km/s arcsec^-2
    if crop is not None:
        m0 = m0[crop]
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, name + ".npy"), m0.astype(np.float32))
    s = dict(run=name, cube=cube, J=J, L=L, pad=pad, k=K, n_iter=N_ITER, line_free=free, flux=float(x.sum()),
             flux_line=float(x[line].sum()), seconds=round(time.time() - t0))
    json.dump(s, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    print(json.dumps(s), flush=True)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    os.makedirs(FIG, exist_ok=True)
    panels = []
    # the scripts' pdtau k = 5 (J 7, L 5, script cube): line moment 0, central 10"
    x = np.load(os.path.join(REPO, "data/experiments/04_pd_tau_reweighting/compact/pdtau_k5.npy")).astype(np.float64)
    from common import line_channels
    panels.append(("script cube, J 7, L 5\n(the scripts' pdtau)", x[line_channels("ngc3110")].sum(0)[150:650, 150:650] * 25 / CELL ** 2))
    # the GUI result itself (r002): model.npy, line channels
    g = np.load(os.path.join(GUI_DS, "results/r002-gpu_pd/model.npy")).astype(np.float64)
    vel = 4752.0 + 10.0 * np.arange(g.shape[0])
    panels.append(("GUI run r002\n(GUI cube, J 5, L 2)", g[(vel >= 4750) & (vel <= 5280)].sum(0) * 10 / CELL ** 2))
    labels = {"gui_J5_L2": "GUI cube, J 5, L 2\n(rerun of the GUI setup)", "gui_J5_L5": "GUI cube, J 5, L 5\n(60 -> 64 ch padded)",
              "gui_J7_L5": "GUI cube, J 7, L 5\n(the scripts' dictionary)", "script_J5_L2": "script cube, J 5, L 2\n(the GUI's dictionary)",
              "script_J7_L2": "script cube, J 7, L 2\n(only L reduced)",
              "script_crop10": "script cube cut to 10\"\n(as the GUI crop)", "gui_bin3": "GUI cube, 30 km/s channels\n(3 channels averaged)",
              "gui_natural": "GUI imaging, NATURAL weighting\n(10 km/s, 10\" crop)"}
    for r in RUNS:
        f = os.path.join(OUT, r + ".npy")
        if os.path.exists(f):
            panels.append((labels[r], np.load(f).astype(np.float64)))
    vmax = np.percentile(panels[0][1], 99.9)
    asinh = lambda t: np.arcsinh(np.clip(t, 0, None) / 0.05) / np.arcsinh(1 / 0.05)
    ext = [5, -5, -5, 5]
    rows = []
    n = len(panels)
    fig, ax = plt.subplots(2, n, figsize=(3.3 * n, 7.0), constrained_layout=True)
    for i, (lab, m0) in enumerate(panels):
        hp = m0 - gaussian_filter(m0, 0.3 / CELL / 2.355)            # structure finer than 0.3"
        stats = dict(total=float(m0.sum() * CELL ** 2), peak=float(m0.max()), p999=float(np.percentile(m0, 99.9)),
                     fine_rms=float(hp[m0 > 0.05 * vmax].std()) if (m0 > 0.05 * vmax).any() else 0.0,
                     frac_zero=float((m0 <= 1e-6 * vmax).mean()))
        rows.append((lab.replace("\n", " "), stats))
        ax[0, i].imshow(np.clip(m0 / vmax, 0, 1), origin="lower", cmap="inferno", extent=ext, vmin=0, vmax=1)
        ax[0, i].set_title(f"{lab}\n{stats['total']:.0f} Jy km/s; peak {stats['peak']:.0f}", fontsize=8.5)
        ax[1, i].imshow(asinh(m0 / vmax), origin="lower", cmap="inferno", extent=ext, vmin=0, vmax=1)
        ax[1, i].set_title(f"asinh; fine-scale rms {stats['fine_rms']:.2f}", fontsize=8.5)
        for a in ax[:, i]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("NGC 3110 compact, pdtau k = 5, line moment 0, central 10\" (Jy km/s arcsec$^{-2}$); "
                 "top linear, bottom asinh; all on the scripts' 99.9th percentile", fontsize=10)
    fig.savefig(os.path.join(FIG, "17_gui_vs_script.png"), dpi=90); plt.close(fig)
    for lab, s in rows:
        print(f"{lab:55s} total {s['total']:7.1f}  peak {s['peak']:7.1f}  p99.9 {s['p999']:6.1f}  "
              f"fine-rms {s['fine_rms']:6.2f}  zero-frac {s['frac_zero']:.2f}")
    print("wrote", os.path.join(FIG, "17_gui_vs_script.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=list(RUNS))
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.run:
        run(a.run)
    if a.plot:
        plot()
