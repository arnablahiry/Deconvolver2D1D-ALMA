#!/usr/bin/env python
"""
Runs the simple 2D-1D wavelet deconvolution (`legacy/simple/deconvolve.py`,
unmodified) on the simulated FIRE [CII] dirty cube built by
`scripts/build_fire_cii_moment1_notebook.py`'s last cell.

Inputs (from simple_results/data/):
    fire_cii_true_cube.npy   (204, 204, 401)  Jy/pixel   -- ground truth
    fire_cii_dirty_cube.npy  (204, 204, 401)  Jy/beam    -- truth (*) dirty beam
    fire_cii_dirty_beam.npy  (204, 204)                  -- 5.0 x 3.79 px lobe

Two shape changes are needed before the solver will take it:

  * Spectral rebinning. The native cube is 401 channels at 5 km/s; this
    trims to the +/-500 km/s window that holds 99.75% of the line flux and
    rebins x4 to **20 km/s**, giving 50 channels. Values are averaged, not
    summed, because each channel is a flux density -- the moment-0
    (sum * dv) is unchanged. Spectral rebinning commutes exactly with the
    spatial convolution that made the dirty cube, so rebinning truth and
    dirty identically stays self-consistent.

  * Axis order and a PSF *cube*. The solver wants (nchan, ny, nx) and
    requires `psf.shape == dirty.shape`, so the single 2-D beam is
    broadcast across channels (it is the same beam for every channel here).

The deconvolver's own module-level paths are patched rather than its code
being copied, so this runs the identical algorithm.
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "simple_results", "data")
SIMPLE = os.path.join(REPO, "legacy", "simple")

V_WINDOW_KMS = 500.0     # keep |v| <= this (99.75% of the line flux)
REBIN = 4                # 5 km/s -> 20 km/s
DV_NATIVE = 5.0


def prepare():
    true_cube = np.load(f"{DATA}/fire_cii_true_cube.npy")     # (ny, nx, nchan)
    dirty_cube = np.load(f"{DATA}/fire_cii_dirty_cube.npy")
    beam = np.load(f"{DATA}/fire_cii_dirty_beam.npy")
    velocity = np.linspace(-1000.0, 1000.0, true_cube.shape[2])

    keep = np.abs(velocity) <= V_WINDOW_KMS
    lo, hi = np.argmax(keep), len(keep) - np.argmax(keep[::-1])
    hi = lo + ((hi - lo) // REBIN) * REBIN                    # make it divide evenly

    def rebin(cube):                                          # mean: these are flux densities
        sub = cube[:, :, lo:hi]
        return sub.reshape(sub.shape[0], sub.shape[1], -1, REBIN).mean(axis=3)

    true_r, dirty_r = rebin(true_cube), rebin(dirty_cube)
    vel_r = velocity[lo:hi].reshape(-1, REBIN).mean(axis=1)
    dv_r = DV_NATIVE * REBIN

    # (ny, nx, nchan) -> (nchan, ny, nx), and one PSF plane per channel.
    true_r = np.ascontiguousarray(np.moveaxis(true_r, 2, 0), dtype=np.float64)
    dirty_r = np.ascontiguousarray(np.moveaxis(dirty_r, 2, 0), dtype=np.float64)
    psf_cube = np.ascontiguousarray(np.broadcast_to(beam, dirty_r.shape), dtype=np.float64)

    print(f"[prep] {true_cube.shape[2]} channels @ {DV_NATIVE:g} km/s "
          f"-> {true_r.shape[0]} channels @ {dv_r:g} km/s "
          f"(|v| <= {V_WINDOW_KMS:g}, v = {vel_r[0]:+.0f} .. {vel_r[-1]:+.0f})")
    flux_kept = true_r.sum() * dv_r / (np.load(f"{DATA}/fire_cii_true_cube.npy").sum() * DV_NATIVE)
    print(f"[prep] flux retained after trim+rebin: {100 * flux_kept:.2f}%")
    print(f"[prep] cube {dirty_r.shape}, psf {psf_cube.shape}, "
          f"{dirty_r.nbytes / 1e6:.0f} MB each (float64)")

    np.save(f"{DATA}/fire_cii_deconv_psf.npy", psf_cube)
    np.save(f"{DATA}/fire_cii_deconv_dirty.npy", dirty_r)
    np.save(f"{DATA}/fire_cii_deconv_true.npy", true_r)
    np.save(f"{DATA}/fire_cii_deconv_velocity.npy", vel_r)
    return true_r, dirty_r, psf_cube, vel_r, dv_r


def run_deconvolution():
    sys.path.insert(0, SIMPLE)
    import deconvolve

    deconvolve.PSF_PATH = f"{DATA}/fire_cii_deconv_psf.npy"
    deconvolve.DIRTY_PATH = f"{DATA}/fire_cii_deconv_dirty.npy"
    deconvolve.OUT_PATH = f"{DATA}/fire_cii_model.npy"
    print(f"[run] N_ITER={deconvolve.N_ITER}, K_SIGMA={deconvolve.K_SIGMA}, "
          f"scales2d={deconvolve.NUM_SCALES_2D}, levels1d={deconvolve.NUM_LEVELS_1D}")
    deconvolve.main()
    return np.load(f"{DATA}/fire_cii_model.npy")


def clean_beam_from(beam):
    """Gaussian fitted to the dirty beam's main lobe, area-normalized -- the
    CLEAN restoring beam, used to compare model and truth at one resolution."""
    from scipy.optimize import curve_fit
    half = 16
    cy, cx = np.unravel_index(np.argmax(beam), beam.shape)
    sub = beam[cy - half:cy + half + 1, cx - half:cx + half + 1]
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]

    def gauss2d(coords, amp, x0, y0, sx, sy, th):
        x, y = coords
        a = np.cos(th) ** 2 / (2 * sx ** 2) + np.sin(th) ** 2 / (2 * sy ** 2)
        b = -np.sin(2 * th) / (4 * sx ** 2) + np.sin(2 * th) / (4 * sy ** 2)
        c = np.sin(th) ** 2 / (2 * sx ** 2) + np.cos(th) ** 2 / (2 * sy ** 2)
        return amp * np.exp(-(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0)
                              + c * (y - y0) ** 2))

    lobe = sub >= 0.2
    p, _ = curve_fit(gauss2d, (xx[lobe], yy[lobe]), sub[lobe],
                     p0=[1, 0, 0, 2, 1.5, 0], maxfev=40000)
    n = beam.shape[0]
    grid_y, grid_x = np.mgrid[:n, :n] - n // 2
    clean = gauss2d((grid_x, grid_y), 1.0, 0, 0, p[3], p[4], p[5])
    return clean / clean.sum()


def report(true_r, dirty_r, psf_cube, model, dv_r):
    from scipy.signal import fftconvolve
    beam = psf_cube[0]
    beam_sum = beam.sum()
    true_mom0 = true_r.sum(axis=0) * dv_r          # Jy km/s per pixel
    model_mom0 = model.sum(axis=0) * dv_r
    dirty_mom0 = dirty_r.sum(axis=0) * dv_r        # Jy km/s per beam

    print("\n=== flux ===")
    print(f"total  true {true_mom0.sum():.4e} | model {model_mom0.sum():.4e} Jy km/s "
          f"({100 * model_mom0.sum() / true_mom0.sum():.1f}%)")
    print(f"dirty total / sum(beam) = {dirty_mom0.sum() / beam_sum:.4e} "
          f"(flux implied by the beam-convolved data)")
    print(f"model is {100 * (model > 0).mean():.1f}% non-zero voxels")

    # Comparing a sparse, super-resolved model with the smooth truth pixel by
    # pixel punishes sub-pixel position errors and understates the fit. The
    # standard check is at one common resolution: restore the model with the
    # CLEAN beam and smooth the truth by the same kernel.
    clean = clean_beam_from(beam)
    smooth = lambda cube: np.stack([fftconvolve(pl, clean, mode="same") for pl in cube])
    true_s = smooth(true_r).sum(axis=0) * dv_r
    model_s = smooth(model).sum(axis=0) * dv_r
    dirty_s = dirty_mom0 / beam_sum                 # Jy/beam -> Jy/px-equivalent

    print("\n=== matched resolution (model restored, truth smoothed) ===")
    for name, img in [("restored model", model_s), ("dirty image  ", dirty_s)]:
        corr = np.corrcoef(img.ravel(), true_s.ravel())[0, 1]
        rmse = np.sqrt(((img - true_s) ** 2).mean()) / true_s.std()
        print(f"{name} vs truth: corr {corr:.4f}, rel. RMSE {rmse:.3f}")
    print(f"(raw, unsmoothed model vs truth: corr "
          f"{np.corrcoef(model_mom0.ravel(), true_mom0.ravel())[0, 1]:.4f})")
    return true_mom0, dirty_mom0, model_mom0, true_s, model_s


def make_figure(true_mom0, dirty_mom0, model_mom0, true_s, model_s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(19, 5), constrained_layout=True)
    panels = [
        (true_mom0, "Truth (Jy km/s /px)"),
        (dirty_mom0, "Dirty (Jy km/s /beam)"),
        (model_mom0, "Deconvolved model (Jy km/s /px)"),
        (model_s, "Model restored w/ clean beam"),
    ]
    for ax, (img, title) in zip(axes, panels):
        im = ax.imshow(img, origin="lower", cmap="magma", vmin=0, vmax=img.max(),
                       interpolation="nearest")
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_box_aspect(1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    out = os.path.join(REPO, "simple_results", "fire_cii_deconvolution.png")
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    true_r, dirty_r, psf_cube, vel_r, dv_r = prepare()
    model_path = f"{DATA}/fire_cii_model.npy"
    if "--reuse" in sys.argv and os.path.exists(model_path):
        print(f"[run] reusing existing {model_path}")
        model = np.load(model_path)
    else:
        model = run_deconvolution()
    make_figure(*report(true_r, dirty_r, psf_cube, model, dv_r))
