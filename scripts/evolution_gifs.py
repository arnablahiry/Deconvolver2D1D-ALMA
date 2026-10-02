"""
Shared rendering for the deconvolution-evolution GIFs: a frame grabber to pass as
gpu_pd.solve_pd(callback=...), and render() to write

    <out>/<prefix>.gif               raw model moment 0 | residual moment 0 | convergence
    <out>/<prefix>_model_hires.gif   the model panel alone, high resolution
    <out>/frames/<prefix>_frame_NNN.png, <prefix>_hires_NNN.png
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


class FrameGrabber:
    """callback(it, x, g) for gpu_pd.solve_pd: moment-0 frames of the model (Jy km/s arcsec^-2)
    and residual (Jy/beam km/s) at the iterations `marks`, cropped to `crop`, plus the
    per-iteration relative change every 25 iterations."""

    def __init__(self, marks, line, cell, crop, dv=25.0):
        self.marks, self.line, self.cell, self.crop, self.dv = set(marks), line, cell, crop, dv
        self.frames, self.trace, self._prev = [], [], None

    def __call__(self, it, x, g):
        if it % 25 == 24:
            self._prev = x.clone()
        elif it % 25 == 0 and self._prev is not None:
            self.trace.append((it, float((x - self._prev).norm() / max(float(x.norm()), 1e-30))))
        if it in self.marks:
            m = (x[self.line].sum(0) * self.dv).cpu().numpy()[self.crop] / self.cell ** 2
            r = (-g[self.line].sum(0) * self.dv).cpu().numpy()[self.crop]
            self.frames.append((it, m, r, float(x.sum())))


def _gif(files, path):
    imgs = [Image.open(p).convert("RGB") for p in files]
    durs = [120] * len(imgs); durs[-1] = 2500
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=durs, loop=0, optimize=True)
    print("wrote", path, flush=True)


def render(frames, trace, out, prefix, title, label, n_iter, reweight_iters=(), extent=(5, -5, -5, 5)):
    """frames: list of (iteration, model mom0, residual mom0, flux); label(it) -> short settings text"""
    fdir = os.path.join(out, "frames")
    os.makedirs(fdir, exist_ok=True)
    for f in glob.glob(os.path.join(fdir, f"{prefix}_*.png")):
        os.remove(f)
    vm = np.percentile(frames[-1][1], 99.9)
    vr = 5 * frames[-1][2].std()
    ti, tc = np.array(trace).T
    plt.rcParams.update({"font.family": "serif", "font.size": 13, "xtick.direction": "in", "ytick.direction": "in"})
    paths = []
    for k, (it, m, r, flux) in enumerate(frames):
        fig, ax = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw=dict(width_ratios=[1, 1, 1.1]), constrained_layout=True)
        im = ax[0].imshow(m, origin="lower", extent=extent, cmap="inferno", vmin=0, vmax=vm)
        ax[0].set_title("raw model, moment 0 (no convolution)"); plt.colorbar(im, ax=ax[0], shrink=0.85, label=r"Jy km s$^{-1}$ arcsec$^{-2}$")
        im = ax[1].imshow(r, origin="lower", extent=extent, cmap="RdBu_r", vmin=-vr, vmax=vr)
        ax[1].set_title("residual d $-$ N x, moment 0"); plt.colorbar(im, ax=ax[1], shrink=0.85, label=r"Jy beam$^{-1}$ km s$^{-1}$")
        for a in ax[:2]:
            a.set_xlabel(r'$\Delta$RA ["]'); a.set_ylabel(r'$\Delta$Dec ["]')
        ax[2].loglog(ti, tc, color="0.8", lw=1.2)
        done = ti <= it
        ax[2].loglog(ti[done], tc[done], color="#2a78d6", lw=1.8)
        for rw in reweight_iters:
            ax[2].axvline(rw, color="0.85", lw=0.8, zorder=0)
        ax[2].axvline(it, color="#eb6834", lw=1.5)
        ax[2].set_xlabel("iteration"); ax[2].set_ylabel("relative change per iteration"); ax[2].set_xlim(20, n_iter)
        ax[2].set_title("convergence" + (" (thin grey lines: reweighting)" if len(reweight_iters) else ""))
        fig.suptitle(f"{title}  --  iteration {it:,}  |  {label(it)}  --  model flux {flux:.1f} Jy", fontsize=14)
        p = os.path.join(fdir, f"{prefix}_frame_{k:03d}.png"); fig.savefig(p, dpi=70); plt.close(fig); paths.append(p)
    _gif(paths, os.path.join(out, f"{prefix}.gif"))

    plt.rcParams.update({"font.size": 15})
    hi = []
    for k, (it, m, r, flux) in enumerate(frames):
        fig = plt.figure(figsize=(9.2, 8.6)); ax = fig.add_axes([0.1, 0.08, 0.72, 0.8]); cax = fig.add_axes([0.84, 0.08, 0.03, 0.8])
        im = ax.imshow(m, origin="lower", extent=extent, cmap="inferno", vmin=0, vmax=vm, interpolation="bicubic")
        cb = fig.colorbar(im, cax=cax); cb.set_label(r"Jy km s$^{-1}$ arcsec$^{-2}$")
        for sp in ax.spines.values():
            sp.set_visible(True)
        ax.set_xlabel(r'$\Delta$RA ["]'); ax.set_ylabel(r'$\Delta$Dec ["]')
        ax.text(0.03, 0.97, f"iteration {it:,}", transform=ax.transAxes, color="w", fontsize=20, fontweight="bold", va="top")
        ax.text(0.03, 0.905, label(it), transform=ax.transAxes, color="w", fontsize=14, fontweight="bold", va="top")
        ax.text(0.97, 0.03, f"model flux {flux:.1f} Jy", transform=ax.transAxes, color="w", fontsize=14, fontweight="bold", ha="right")
        fig.suptitle(f"{title}: raw model (moment 0, no convolution)", fontsize=16, y=0.955)
        p = os.path.join(fdir, f"{prefix}_hires_{k:03d}.png"); fig.savefig(p, dpi=130); plt.close(fig); hi.append(p)
    _gif(hi, os.path.join(out, f"{prefix}_model_hires.gif"))
