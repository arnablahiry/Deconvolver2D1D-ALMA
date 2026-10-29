#!/usr/bin/env python
"""
Does the extended data hold the emission the compact model has outside the extended CLEAN's mask? One test in data
space, no mask choice in the judgement:

  footprint C   the compact deconvolved model, smoothed to the extended beam, > FRAC of its per-channel peak
  region R      C minus the extended CLEAN's mask (line channels): where the compact model has flux and the CLEAN does not
  predicted     extended dirty beam * (compact model - CLEAN model): the extended residual if the compact model were right
  actual        the CLEAN's residual

Prints, in R and at a few smoothings, the actual and predicted residual in sigma of the line-free channels, the same
footprint in line-free channels as a null, and the fit actual = s x predicted. s ~ 1: the extended data hold what the
compact model has there; actual ~ 0: the CLEAN has modelled it (or the compact model over-predicts it); actual ~ s x
predicted with s < 1: the compact model over-predicts.

    python scripts/experiments/compact_only_residual.py NGC3110 [msclean_finished msclean_cmask_n3 ...]
"""
import json
import os
import sys

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRAC = 0.01


def sq(p):
    return np.squeeze(fits.getdata(p)).astype(np.float64)


def compact_footprint(gal_dir, frac=FRAC):
    """(nchan, ny, nx) bool: the compact model smoothed to the extended natural beam, > frac of its channel's peak"""
    cfg = json.load(open(os.path.join(gal_dir, "config.json")))
    bmaj, bmin, _ = cfg["beams_natural"]["extended"]
    s = np.sqrt(bmaj * bmin) / cfg["cell_arcsec"] / 2.355
    cs = np.stack([gaussian_filter(c, s) for c in np.load(os.path.join(gal_dir, "compact", "deconvolved", "model.npy"))
                   .astype(np.float64)])
    return cs > frac * cs.max(axis=(1, 2), keepdims=True)


def main():
    T = sys.argv[1]
    cleans = sys.argv[2:] or ["msclean_finished"]
    R = os.path.join(REPO, "results", T)
    cfg = json.load(open(os.path.join(R, "config.json"))); cell, dv = cfg["cell_arcsec"], cfg["width_kms"]
    L = json.load(open(os.path.join(R, "line_channels.json")))["line"]
    free = [c for c in range(cfg["nchan"]) if c not in L and 1 < c < cfg["nchan"] - 2]
    comp = np.load(os.path.join(R, "compact", "deconvolved", "model.npy")).astype(np.float64)
    psf = sq(os.path.join(R, "extended", "cube", "dirty_beam_cube.fits"))
    bmaj, bmin, _ = cfg["beams_natural"]["extended"]; bpx = np.pi * bmaj * bmin / (4 * np.log(2)) / cell ** 2
    E0 = np.load(os.path.join(R, "extended", "msclean_finished", "mask_final.npy")) > 0
    C = compact_footprint(R)
    Rg = C & ~E0                                   # the same region for every CLEAN: the original mask's gap
    ny = comp.shape[1]
    print(f"{T}: extended beam {bmaj:.3f}x{bmin:.3f}\" ({bpx:.0f} px); {len(L)} line channels; compact-only region "
          f"{Rg[L].sum() / bpx:.0f} beams x channels (original mask {E0[L].sum() / bpx:.0f}); compact model flux: "
          f"total {comp[L].sum() * dv:.1f}, in region {(comp * Rg)[L].sum() * dv:.1f} Jy km/s")
    for name in cleans:
        D = os.path.join(R, "extended", name)
        cm, res = sq(os.path.join(D, "model.fits")), sq(os.path.join(D, "residual.fits"))
        pred = np.zeros_like(res)
        for c in L:
            p = psf[c]; cy, cx = np.unravel_index(np.argmax(p), p.shape)
            pred[c] = fftconvolve(comp[c] - cm[c], p[cy - ny: cy + ny, cx - ny: cx + ny] / p.max(), mode="same")
        print(f"\n{name}: CLEAN components {cm[L].sum() * dv:.1f} Jy km/s, of which in region {(cm * Rg)[L].sum() * dv:.1f}; "
              f"line-free channels {cm[free].sum() * dv:+.2f} Jy km/s (sum |.| {np.abs(cm[free]).sum() * dv:.2f})")
        for sm in (0.0, 0.3, 0.6, 1.2):           # smoothing FWHM, arcsec
            f = (lambda a: gaussian_filter(a, sm / cell / 2.355)) if sm else (lambda a: a)
            r = np.stack([f(x) for x in res]); pr = np.stack([f(x) for x in pred])
            sig = np.median([1.4826 * np.median(np.abs(r[c] - np.median(r[c]))) for c in free])
            a, b = r[L][Rg[L]], pr[L][Rg[L]]
            nul = np.concatenate([r[c][Rg[L].any(0)] for c in free])
            print(f"  smooth {sm:.1f}\": noise {1e3 * sig:.3f} mJy/bm | actual {a.mean() / sig:+.2f} sigma "
                  f"(null {nul.mean() / sig:+.2f}), predicted {b.mean() / sig:+.2f} sigma; actual = "
                  f"{(a * b).sum() / (b * b).sum():.2f} x predicted, corr {np.corrcoef(a, b)[0, 1]:.2f}")


if __name__ == "__main__":
    main()
