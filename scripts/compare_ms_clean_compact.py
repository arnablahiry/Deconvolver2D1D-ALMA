#!/usr/bin/env python
"""
Extended multiscale CLEAN vs compact 2D-1D, both as model (*) extended clean
beam -- no residual added to either -- on the shared 800 x 800 x 32 grid.

    python scripts/compare_ms_clean_compact.py ngc3110 [--compact cube_pd_floor3]

The MS-CLEAN model (data/<g>/extended/ms_clean/model.fits, Jy/pixel) is
convolved here with the beam of data/<g>/compact/<compact>/restored_extended_beam.fits
(the compact 2D-1D model restored with the extended beam, by the same
simple/restore.convolve_with_beam), so the two cubes differ only in the model.
<compact> is the compact 2D-1D run: cube_pd_floor3 (default; primal-dual, 1000 it,
lambda = 3 x max(real, white) noise per sub-band) or cube (FISTA 60 it).
Writes results/<g>_msclean_extended_vs_compact_2d1d[_<compact>].{png,pdf}.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import Ellipse
from astropy.io import fits

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "simple"))
from restore import convolve_with_beam

CELL_ARCSEC = 0.02
C_KMS = 2.99792458e5
ZOOM = 5.0                                  # half-width ["] of the map panels
CHANNEL_MAPS = (5, 8, 11, 14, 17, 20)
COMPACT_RUNS = {"cube_pd_floor3": "primal-dual, floored real-noise lambda",
                "cube": "FISTA 60 it"}
BLUE, ORANGE, MUTED, INK2 = "#2a78d6", "#eb6834", "#898781", "#52514e"


def load(path):
    with fits.open(path) as h:
        return np.squeeze(h[0].data).astype(np.float64), h[0].header


def velocities(h):
    """Radio LSRK velocity [km/s] per channel from a VRAD or FREQ axis."""
    x = h["CRVAL3"] + h["CDELT3"] * (np.arange(h["NAXIS3"]) + 1 - h["CRPIX3"])
    if h["CTYPE3"] == "VRAD":
        return x / 1e3
    assert h["CTYPE3"] == "FREQ", h["CTYPE3"]
    return C_KMS * (1 - x / h["RESTFRQ"])


def extent(n):
    half = n * CELL_ARCSEC / 2
    return (half, -half, -half, half)


def map_panel(ax, img, title, norm, cmap="inferno", beam=None):
    im = ax.imshow(img, origin="lower", extent=extent(img.shape[-1]), cmap=cmap, norm=norm,
                   interpolation="nearest")
    ax.set_xlim(ZOOM, -ZOOM)
    ax.set_ylim(-ZOOM, ZOOM)
    if beam is not None:
        ax.add_patch(Ellipse((0.85 * ZOOM, -0.85 * ZOOM), beam[1], beam[0], angle=-beam[2],
                             fill=False, ec="white", lw=1.2))
    ax.set_title(title, fontsize=10)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("galaxy", nargs="?", default="ngc3110")
    ap.add_argument("--compact", default="cube_pd_floor3", choices=sorted(COMPACT_RUNS))
    args = ap.parse_args()
    g = args.galaxy
    base = os.path.join(REPO_ROOT, "data", g)
    label = g.upper().replace("NGC", "NGC ")

    ms_model, ms_hdr = load(os.path.join(base, "extended", "ms_clean", "model.fits"))
    ms_mask, _ = load(os.path.join(base, "extended", "ms_clean", "mask.fits"))
    cp, cp_hdr = load(os.path.join(base, "compact", args.compact, "restored_extended_beam.fits"))
    run = COMPACT_RUNS[args.compact]
    assert ms_hdr["BUNIT"] == "Jy/pixel" and cp_hdr["BUNIT"] == "Jy/beam"
    assert ms_model.shape == cp.shape, (ms_model.shape, cp.shape)
    for k in ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CDELT1", "CDELT2"):
        assert np.isclose(ms_hdr[k], cp_hdr[k]), k
    vel = velocities(ms_hdr)
    assert np.allclose(vel, velocities(cp_hdr), atol=0.05)
    dv = abs(vel[1] - vel[0])

    beam = (cp_hdr["BMAJ"] * 3600, cp_hdr["BMIN"] * 3600, cp_hdr["BPA"])
    ms = convolve_with_beam(ms_model, CELL_ARCSEC, *beam)       # Jy/beam, no residual
    beam_px = np.pi * beam[0] * beam[1] / (4 * np.log(2)) / CELL_ARCSEC ** 2

    # Channels with emission: any MS-CLEAN mask pixel (wider than the notebook's line window).
    mask = ms_mask > 0.5
    chans = np.where(mask.reshape(len(vel), -1).any(1))[0]
    m0_ms, m0_cp = ms[chans].sum(0) * dv, cp[chans].sum(0) * dv
    flux_ms, flux_cp = ms.sum((1, 2)) / beam_px, cp.sum((1, 2)) / beam_px   # Jy per channel
    mask_any = mask.any(0)

    fig = plt.figure(figsize=(21, 17.5), constrained_layout=True)
    top, mid, bot = fig.subfigures(3, 1, height_ratios=(0.95, 1.3, 0.75))

    # Row 1: moment 0 of both, difference, pixel scatter ----------------------
    a = top.subplots(1, 4)
    vmax = np.percentile(np.concatenate([m0_ms.ravel(), m0_cp.ravel()]), 99.9)
    norm = plt.Normalize(0, vmax)
    im = map_panel(a[0], m0_ms, f"(a) extended MS-CLEAN model * beam\n"
                   f"{m0_ms.sum() / beam_px:.0f} Jy km/s in field", norm, beam=beam)
    map_panel(a[1], m0_cp, f"(b) compact 2D-1D ({run}) * extended beam\n"
              f"{m0_cp.sum() / beam_px:.0f} Jy km/s in field", norm, beam=beam)
    top.colorbar(im, ax=a[:2], fraction=0.025, pad=0.01, label="Jy/beam km/s")
    diff = m0_ms - m0_cp
    r = np.percentile(np.abs(diff), 99.5)
    im = map_panel(a[2], diff, "(c) (a) - (b)", TwoSlopeNorm(0, -r, r), cmap="RdBu_r", beam=beam)
    a[2].contour(mask_any.astype(float), levels=[0.5], colors="0.25", linewidths=0.6,
                 extent=extent(mask_any.shape[-1]), origin="lower")
    top.colorbar(im, ax=a[2], fraction=0.046, pad=0.02, label="Jy/beam km/s")
    for ax in a[:3]:
        ax.set_xlabel('RA offset ["]')
        ax.set_ylabel('Dec offset ["]')
    # pixel-by-pixel moment 0 inside the MS-CLEAN mask union (1 px of every beam area ~ fine)
    sel = mask_any & (m0_ms > 0.01 * vmax) & (m0_cp > 0.01 * vmax)
    lo, hi = 0.01 * vmax, 1.5 * max(m0_ms[sel].max(), m0_cp[sel].max())
    bins = np.geomspace(lo, hi, 90)
    h = a[3].hist2d(m0_cp[sel], m0_ms[sel], bins=(bins, bins), cmap="Blues", norm=LogNorm(),
                    cmin=1)
    a[3].plot([lo, hi], [lo, hi], color=MUTED, lw=1.2, ls="--")
    a[3].text(0.95 * hi, 0.8 * hi, "1:1", ha="right", va="top", fontsize=9, color=INK2)
    a[3].set_xscale("log")
    a[3].set_yscale("log")
    a[3].set_aspect("equal")
    med = np.median(m0_ms[sel] / m0_cp[sel])
    a[3].set_title(f"(d) pixel moment 0 inside the clean mask\nmedian (a)/(b) = {med:.2f}",
                   fontsize=10)
    a[3].set_xlabel("compact 2D-1D [Jy/beam km/s]")
    a[3].set_ylabel("extended MS-CLEAN [Jy/beam km/s]")
    top.colorbar(h[3], ax=a[3], fraction=0.046, pad=0.02, label="pixels")
    top.suptitle(f"{label}: extended multiscale CLEAN vs compact 2D-1D, both model * extended "
                 f"clean beam {beam[0]:.3f}\" x {beam[1]:.3f}\" @ {beam[2]:.0f} deg, no residual; "
                 f"moment 0 over {vel[chans[0]]:.0f}-{vel[chans[-1]]:.0f} km/s ({len(chans)} ch)",
                 fontsize=12)

    # Rows 2-3: channel maps, one colour scale -------------------------------
    b = mid.subplots(2, len(CHANNEL_MAPS), sharex=True, sharey=True)
    cmax = np.percentile(np.concatenate([ms[list(CHANNEL_MAPS)].ravel(),
                                         cp[list(CHANNEL_MAPS)].ravel()]), 99.95)
    cnorm = plt.Normalize(0, cmax)
    for j, c in enumerate(CHANNEL_MAPS):
        im = map_panel(b[0, j], ms[c], f"ch {c}: {vel[c]:.0f} km/s", cnorm,
                       beam=beam if j == 0 else None)
        map_panel(b[1, j], cp[c], "", cnorm)
        b[1, j].set_xlabel('RA offset ["]')
    b[0, 0].set_ylabel('extended MS-CLEAN\nDec offset ["]')
    b[1, 0].set_ylabel('compact 2D-1D\nDec offset ["]')
    mid.colorbar(im, ax=b, fraction=0.015, pad=0.01, label="Jy/beam")
    mid.suptitle("(e) channel maps, one colour scale: top extended MS-CLEAN, bottom compact 2D-1D",
                 fontsize=11)

    # Row 4: flux spectra and single-pixel spectra ----------------------------
    d = bot.subplots(1, 4)
    for y, col, lab in ((flux_ms, BLUE, "extended MS-CLEAN"), (flux_cp, ORANGE, "compact 2D-1D")):
        d[0].step(vel, y, where="mid", color=col, lw=2,
                  label=f"{lab}: {y[chans].sum() * dv:.0f} Jy km/s")
    d[0].set_title("(f) total flux per channel", fontsize=10)
    d[0].set_ylabel("Jy")
    d[0].legend(frameon=False, fontsize=8)
    peak = np.unravel_index(np.argmax(m0_ms), m0_ms.shape)
    c0 = ms.shape[-1] // 2
    spots = [("peak", peak), ("centre", (c0, c0)),
             ("southern arm", (c0 - int(4.0 / CELL_ARCSEC), c0 + int(1.0 / CELL_ARCSEC)))]
    for ax, (name, (y, x)), tag in zip(d[1:], spots, "ghi"):
        ax.step(vel, ms[:, y, x] * 1e3, where="mid", color=BLUE, lw=2, label="extended MS-CLEAN")
        ax.step(vel, cp[:, y, x] * 1e3, where="mid", color=ORANGE, lw=2, label="compact 2D-1D")
        ax.set_title(f"({tag}) {name} ({(c0 - x) * CELL_ARCSEC:+.1f}\", "
                     f"{(y - c0) * CELL_ARCSEC:+.1f}\")", fontsize=10)
        ax.set_ylabel("mJy/beam")
        for aa in (a[0], a[1]):
            aa.plot((c0 - x) * CELL_ARCSEC, (y - c0) * CELL_ARCSEC, "+", color="white", ms=9, mew=1.2)
            aa.annotate(tag, ((c0 - x) * CELL_ARCSEC, (y - c0) * CELL_ARCSEC), xytext=(5, 4),
                        textcoords="offset points", color="white", fontsize=9)
    d[1].legend(frameon=False, fontsize=8)
    for ax in d:
        ax.set_xlabel("v (LSRK, radio) [km/s]")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.grid(alpha=0.3)

    out = os.path.join(REPO_ROOT, "results")
    os.makedirs(out, exist_ok=True)
    suffix = "" if args.compact == "cube_pd_floor3" else f"_{args.compact}"
    stem = os.path.join(out, f"{g}_msclean_extended_vs_compact_2d1d{suffix}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=130)
    print(f"wrote {os.path.relpath(stem, REPO_ROOT)}.{{png,pdf}}; flux in field "
          f"MS-CLEAN {flux_ms[chans].sum() * dv:.1f}, compact 2D-1D {flux_cp[chans].sum() * dv:.1f} "
          f"Jy km/s; median pixel ratio {med:.2f}")


if __name__ == "__main__":
    main()
