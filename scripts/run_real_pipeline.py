#!/usr/bin/env python
"""
Resumable runner for the real-data (TW Hya) CLEAN + 2D-1D wavelet
deconvolution pipeline, checkpointing to disk between calls.

Why this exists (as opposed to just running everything inline in the demo
notebook, like the toy-cube notebook does): the real cube is ~10x more
pixels than the toy demo (128x128x40 vs 80x80x41 after cropping, see
`io_fits.load_twhya_data`), and each FISTA iteration involves a full 2D-1D
wavelet transform of the whole cube, so a fully-converged run (~60-80
iterations, soft and hard) takes a few minutes of wall time -- fine for a
user actually running the notebook locally, but each *build* of this repo's
pre-computed notebook output needed many separate short tool invocations, so
progress is checkpointed to `scripts/.checkpoints/` between calls rather
than needing one single long-running process.

Usage:
    python3 scripts/run_real_pipeline.py clean
    python3 scripts/run_real_pipeline.py soft --n_iter_chunk 20 --total_iter 60
    python3 scripts/run_real_pipeline.py hard --n_iter_chunk 20 --total_iter 60
Re-running the same command resumes from the last checkpoint until
`total_iter` is reached (soft/hard) or just runs once (clean, which is fast
enough not to need chunking).
"""

import argparse
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
DATA_DIR = os.path.join(REPO_ROOT, "data")
CKPT_DIR = os.path.join(REPO_ROOT, "scripts", ".checkpoints")

sys.path.insert(0, SRC_DIR)

from io_fits import load_twhya_data, taper_psf, crop_psf_support  # noqa: E402
from deconvolver import Deconvolver2D1D, convolve_cube, lipschitz_constant  # noqa: E402
from clean import hogbom_clean_cube, restore_clean_cube  # noqa: E402

K_START, K_END = 6.0, 1.5  # overall (global) threshold-decay schedule
NUM_SCALES_2D, NUM_SCALES_1D = 4, 3
PSF_SUPPORT = 41  # see crop_psf_support docstring


def _load_common():
    d = load_twhya_data(DATA_DIR, crop=128, load_benchmark=True, load_residual=False)
    # Crop-to-support + taper for the wavelet path only -- see
    # crop_psf_support/taper_psf docstrings: real ALMA dirty beams don't
    # decay to ~0 within the imaged field, and convolving with that abrupt
    # truncation via FFT gives an enormous Lipschitz constant that makes
    # ISTA/FISTA converge impractically slowly. Hogbom CLEAN (clean.py) has
    # no such dependence and keeps using the full d['psf'] untouched.
    d["psf_tapered"] = taper_psf(crop_psf_support(d["psf"], PSF_SUPPORT), flat_frac=0.0)
    return d


def _ckpt_path(stage):
    return os.path.join(CKPT_DIR, f"{stage}.npz")


def run_clean(d, gain, threshold_sigma, n_iter_max):
    t0 = time.time()
    model, residual, n_comp = hogbom_clean_cube(
        d["dirty"], d["psf"], d["sigma_noise"], gain=gain,
        threshold_sigma=threshold_sigma, n_iter_max=n_iter_max, verbose=True,
    )
    restored, clean_beam = restore_clean_cube(model, residual, d["psf"])
    print(f"[clean] done in {time.time() - t0:.1f}s")
    os.makedirs(CKPT_DIR, exist_ok=True)
    np.savez_compressed(_ckpt_path("clean"), model=model, residual=residual,
                         restored=restored, n_components=np.array(n_comp))
    print(f"[clean] flux={restored.sum():.4g}")


def run_wavelet(stage, d, n_iter_chunk, total_iter):
    assert stage in ("soft", "hard")
    # _v2: fresh checkpoint namespace using the cropped+tapered PSF (smaller,
    # much better-conditioned Lipschitz constant) -- the original soft.npz/
    # hard.npz checkpoints were built against the full 128-pixel-support
    # PSF's much larger L and shouldn't be resumed from under this schedule.
    ckpt = _ckpt_path(stage + "_v2")
    if os.path.exists(ckpt):
        z = np.load(ckpt)
        x = z["x"]
        done = int(z["iters_done"])
        residual_std_history = list(z["residual_std_history"])
        print(f"[{stage}] resuming from checkpoint: {done}/{total_iter} iterations done")
    else:
        x = None
        done = 0
        residual_std_history = []
        print(f"[{stage}] starting fresh: 0/{total_iter} iterations")

    if done >= total_iter:
        print(f"[{stage}] already complete ({done} >= {total_iter}); nothing to do")
        return

    chunk_n = min(n_iter_chunk, total_iter - done)
    # Continue the *global* [K_START, K_END] linear threshold-decay schedule
    # across chunks, rather than restarting it at K_START every call.
    def k_at(i):
        frac = i / max(total_iter - 1, 1)
        return K_START + (K_END - K_START) * frac
    chunk_k_start = k_at(done)
    chunk_k_end = k_at(done + chunk_n - 1)

    dec = Deconvolver2D1D(num_scales_2d=NUM_SCALES_2D, num_scales_1d=NUM_SCALES_1D,
                           threshold_type=stage, positivity=True, verbose=True)
    t0 = time.time()
    x_new, hist = dec.deconvolve(
        d["dirty"], d["psf_tapered"], d["sigma_noise"], n_iter=chunk_n,
        k_start=chunk_k_start, k_end=chunk_k_end, fista=True, x0=x,
    )
    dt = time.time() - t0
    print(f"[{stage}] chunk of {chunk_n} iterations took {dt:.1f}s "
          f"({dt / chunk_n:.2f}s/iter)")

    done += chunk_n
    residual_std_history += hist["residual_std"]
    os.makedirs(CKPT_DIR, exist_ok=True)
    np.savez_compressed(ckpt, x=x_new, iters_done=done,
                         residual_std_history=np.array(residual_std_history))
    print(f"[{stage}] checkpoint saved: {done}/{total_iter} iterations, "
          f"flux={x_new.sum():.4g}, residual_std={residual_std_history[-1]:.4g}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["clean", "soft", "hard"])
    p.add_argument("--n_iter_chunk", type=int, default=20)
    p.add_argument("--total_iter", type=int, default=60)
    p.add_argument("--gain", type=float, default=0.15)
    p.add_argument("--threshold_sigma", type=float, default=3.0)
    p.add_argument("--n_iter_max_clean", type=int, default=3000)
    args = p.parse_args()

    d = _load_common()
    print(f"[data] dirty {d['dirty'].shape}, psf {d['psf'].shape}, "
          f"sigma_noise={d['sigma_noise']:.4g}")

    if args.stage == "clean":
        run_clean(d, args.gain, args.threshold_sigma, args.n_iter_max_clean)
    else:
        run_wavelet(args.stage, d, args.n_iter_chunk, args.total_iter)


if __name__ == "__main__":
    main()
