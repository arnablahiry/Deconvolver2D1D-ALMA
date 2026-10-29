#!/usr/bin/env python
"""
Compact vs extended Briggs CO(2-1) images of the sample, two sets of
(compact | extended | spectrum) columns, one galaxy per row, in the order of MAIN.

Each row shares one field of view, centred on and sized to the galaxy's
>5 sigma emission in the compact (restored) image; image panels show the clean
model (x) clean beam (image - residual), with no axes, only a scalebar. The spectrum panel overlays both configurations' integrated CO(2-1)
spectra of the clean model (x) clean beam (no residual), summed over one aperture (that emission, grown by one compact beam).
Needs the FITS from scripts/sample/image_co21_briggs.py and image_co21_cube.py.

    python scripts/sample/fig_compact_vs_extended.py

Writes figures/sample/compact_vs_extended_co21_briggs.{png,pdf}, and
compact_vs_extended_co21_briggs_IC4518_split.{png,pdf} with IC 4518 W and E
imaged as separate single pointings, and
compact_vs_extended_co21_briggs_MCG-02-33-098_split.{png,pdf} with the two
members of that pair cropped from its mosaic, and
compact_vs_extended_co21_briggs_dirty.{png,pdf}: the main figure from the dirty
(niter=0) images and cubes.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_co21_briggs import CONFIGS, OUT_ROOT, REPO_ROOT, SAMPLE, SPLIT, TARGETS

FIG_DIR = os.path.join(REPO_ROOT, "figures", "sample")
CMAP = plt.get_cmap("magma").copy()
CMAP.set_bad(CMAP(0.0))      # outside mosaic / primary-beam coverage
DETECT_SIGMA = 5.0
PAD = 0.25                 # fractional padding around the detected emission
# scalebar in axes fractions: same length on every panel, labelled with the arcsec it spans
BAR_FRAC = 0.25
BAR_RIGHT = 0.92
BAR_Y = 0.07
BAR_CAP = 0.018
# spectra: reference categorical slots 1-2 (blue/orange, validated CVD-safe pair)
SPEC_COLOR = dict(compact="#eb6834", extended="#2a78d6")
SPEC_XLIM = (-500, 500)
C_KMS = 299792.458
# Members of a pair inside one mosaic image: a view centred on one of the two brightest
# emission components of the compact image (no re-imaging -- same data and weights).
CROPS = {
    "MCG-02-33-098W": dict(label="MCG-02-33-098 W", data="MCG-02-33-098", side="west"),
    "MCG-02-33-098E": dict(label="MCG-02-33-098 E", data="MCG-02-33-098", side="east"),
}
# Main figure rows: SAMPLE order, with the two pairs shown as their separate members.
MAIN = []
for _g in SAMPLE:
    MAIN += {"MCG-02-33-098": ["MCG-02-33-098W", "MCG-02-33-098E"], "IC4518": list(SPLIT)}.get(_g, [_g])


def load(galaxy, config, dirty=False):
    d = os.path.join(OUT_ROOT, galaxy, config, "co21_briggs" + ("_dirty" if dirty else ""))
    if not os.path.exists(os.path.join(d, "image.fits")):
        return None
    with fits.open(os.path.join(d, "image.fits")) as h:
        img, hdr = np.squeeze(h[0].data).astype(float), h[0].header
    with fits.open(os.path.join(d, "pb.fits")) as h:
        pb = np.squeeze(h[0].data).astype(float)
    # displayed: clean model (x) clean beam = image - residual; dirty products have no model
    disp = img
    if not dirty:
        with fits.open(os.path.join(d, "residual.fits")) as h:
            disp = img - np.squeeze(h[0].data).astype(float)
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    ny, nx = img.shape
    # offsets from the phase centre in arcsec; x increases to the east (left on the sky)
    x = (np.arange(nx) + 1 - hdr["CRPIX1"]) * hdr["CDELT1"] * 3600
    y = (np.arange(ny) + 1 - hdr["CRPIX2"]) * hdr["CDELT2"] * 3600
    return dict(img=img, disp=disp, pb=pb, x=x, y=y, rms=meta["rms_jy_beam"], beam=meta["beam_arcsec"],
                mosaic=meta["mosaic"])


def emission(d, side=None):
    """Mask of the significant emission of one image;
    side="east"/"west": only that one of its two brightest components."""
    beam_px = np.pi / (4 * np.log(2)) * d["beam"][0] * d["beam"][1] / abs(d["x"][1] - d["x"][0]) ** 2
    det = (d["img"] > DETECT_SIGMA * d["rms"]) & (d["pb"] > 0.5)
    lab, n = ndimage.label(det)
    comps = [lab == i for i in range(1, n + 1)]
    comps = [c for c in comps if c.sum() >= beam_px and d["img"][c].max() > 2 * DETECT_SIGMA * d["rms"]]
    if side is not None:
        pair = sorted(comps, key=lambda c: d["img"][c].sum())[-2:]
        east_x = [d["x"][np.nonzero(c)[1]].mean() for c in pair]
        comps = [pair[int(np.argmax(east_x)) if side == "east" else int(np.argmin(east_x))]]
    keep = np.zeros_like(det)
    for c in comps:
        keep |= c
    if not keep.any():
        keep = d["img"] >= np.nanmax(d["img"])
    return keep


def field_of_view(d, side=None):
    """(xc, yc, half-size) arcsec around emission(d, side)."""
    iy, ix = np.nonzero(emission(d, side))
    x0, x1 = d["x"][ix].min(), d["x"][ix].max()
    y0, y1 = d["y"][iy].min(), d["y"][iy].max()
    half = 0.5 * max(x1 - x0, y1 - y0) * (1 + 2 * PAD) + 2 * d["beam"][0]
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1), half


def aperture(d, side=None):
    """emission(d, side) grown by one beam FWHM: the spectrum aperture, on d's grid."""
    r = d["beam"][0] / abs(d["x"][1] - d["x"][0])
    yy, xx = np.mgrid[-int(r) - 1:int(r) + 2, -int(r) - 1:int(r) + 2]
    return ndimage.binary_dilation(emission(d, side), structure=xx ** 2 + yy ** 2 <= r ** 2)


def spectrum(galaxy, config, ap, ref, dirty=False):
    """(v - v_sys km/s, flux Jy) of the CO(2-1) clean model (x) clean beam, summed over ap
    (a mask on ref's grid). image - residual is that convolution, as CASA restores it;
    leaving the residual out avoids its dirty-beam units (Jy / dirty beam) in the flux.
    dirty=True: the dirty cube itself, divided by the area of the beam fitted to its PSF."""
    d = os.path.join(OUT_ROOT, galaxy, config, "co21_cube" + ("_dirty" if dirty else ""))
    need = ("image",) if dirty else ("image", "residual")
    if not all(os.path.exists(os.path.join(d, f"{e}.fits")) for e in need):
        return None
    with fits.open(os.path.join(d, "image.fits")) as h:
        cube, hdr = np.squeeze(h[0].data).astype(float), h[0].header
    if not dirty:
        with fits.open(os.path.join(d, "residual.fits")) as h:
            cube -= np.squeeze(h[0].data).astype(float)
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    nchan, ny, nx = cube.shape
    x = (np.arange(nx) + 1 - hdr["CRPIX1"]) * hdr["CDELT1"] * 3600
    y = (np.arange(ny) + 1 - hdr["CRPIX2"]) * hdr["CDELT2"] * 3600
    # aperture onto the cube grid (same phase centre): nearest pixel of ref's grid
    ix = np.rint((x - ref["x"][0]) / (ref["x"][1] - ref["x"][0])).astype(int)
    iy = np.rint((y - ref["y"][0]) / (ref["y"][1] - ref["y"][0])).astype(int)
    ok_x, ok_y = (ix >= 0) & (ix < ap.shape[1]), (iy >= 0) & (iy < ap.shape[0])
    m = np.zeros((ny, nx), bool)
    m[np.ix_(ok_y, ok_x)] = ap[np.ix_(iy[ok_y], ix[ok_x])]
    bmaj, bmin = meta["beam_arcsec"][:2]
    beam_px = np.pi / (4 * np.log(2)) * bmaj * bmin / abs(x[1] - x[0]) ** 2
    flux = np.array([np.nansum(cube[k][m]) for k in range(nchan)]) / beam_px
    f = (np.arange(nchan) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"] + hdr["CRVAL3"]
    f0 = meta["f_line_ghz"] * 1e9
    return C_KMS * (f0 - f) / f0, flux


def show_spectrum(ax, spectra, bottom):
    lo, hi = 0.0, 0.0
    for c, sp in spectra.items():
        if sp is not None:
            ax.plot(*sp, drawstyle="steps-mid", lw=0.9, color=SPEC_COLOR[c], label=c.capitalize())
            lo, hi = min(lo, np.nanmin(sp[1])), max(hi, np.nanmax(sp[1]))
    if hi > lo:
        ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.15 * (hi - lo))
    # few y ticks, the top one dropped, so labels of neighbouring rows never collide
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="upper"))
    ax.axhline(0, color="0.6", lw=0.5, zorder=0)
    ax.set_xlim(*SPEC_XLIM)
    ax.set_xticks([-400, 0, 400])
    ax.yaxis.tick_right()
    ax.tick_params(labelsize=5.5, length=2, pad=1.5, width=0.5)
    if not bottom:
        ax.tick_params(labelbottom=False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("0.4")


def fit_inside(view, images):
    """Shrink a view until it holds no blanked (pb < pblimit) pixels in any of the images."""
    xc, yc, half = view
    for d in images:
        while half > 0:
            sx, sy = np.abs(d["x"] - xc) <= half, np.abs(d["y"] - yc) <= half
            if np.isfinite(d["img"][np.ix_(sy, sx)]).all():
                break
            half *= 0.98
    return xc, yc, half


def show(ax, d, view):
    xc, yc, half = view
    dx, dy = d["x"][1] - d["x"][0], d["y"][1] - d["y"][0]
    sx = (np.abs(d["x"] - xc) <= half)
    sy = (np.abs(d["y"] - yc) <= half)
    shown = d["disp"][np.ix_(sy, sx)]
    ax.set_facecolor(CMAP(0.0))
    # per-panel 100% clip: min to max of the pixels in view
    norm = ImageNormalize(vmin=np.nanmin(shown), vmax=np.nanmax(shown), stretch=AsinhStretch(0.1))
    ax.imshow(d["disp"], origin="lower", cmap=CMAP, norm=norm, interpolation="nearest",
              extent=[d["x"][0] - dx / 2, d["x"][-1] + dx / 2, d["y"][0] - dy / 2, d["y"][-1] + dy / 2])
    ax.set_xlim(xc + half, xc - half)          # east to the left
    ax.set_ylim(yc - half, yc + half)


def scalebar(ax, half):
    arcsec = BAR_FRAC * 2 * half
    label = f"{arcsec:.0f}\u2033" if arcsec >= 10 else f"{arcsec:.1f}\u2033"
    x0, x1 = BAR_RIGHT - BAR_FRAC, BAR_RIGHT
    kw = dict(color="white", lw=1.0, solid_capstyle="butt", transform=ax.transAxes)
    ax.plot([x0, x1], [BAR_Y, BAR_Y], **kw)
    for x in (x0, x1):
        ax.plot([x, x], [BAR_Y - BAR_CAP, BAR_Y + BAR_CAP], **kw)
    ax.text(0.5 * (x0 + x1), BAR_Y + BAR_CAP + 0.012, label, color="white", fontsize=7,
            ha="center", va="bottom", transform=ax.transAxes)


def blank(ax, text=None):
    ax.set_facecolor(CMAP(0.0))
    if text:
        ax.text(0.5, 0.5, text, color="0.75", ha="center", va="center", fontsize=7, transform=ax.transAxes)


def make_figure(galaxies, out, dirty=False):
    nrows = (len(galaxies) + 1) // 2
    sets = [galaxies[:nrows], galaxies[nrows:]] if len(galaxies) > 1 else [galaxies]

    panel = 1.55
    gap = 0.34                 # holds the spectrum's right-hand tick labels too
    top_in, bottom_in = 0.22, 0.3
    ratios = [1, 1, 1, gap, 1, 1, 1, 0.2][:4 * len(sets)]
    height = nrows * panel + top_in + bottom_in
    fig = plt.figure(figsize=(sum(ratios) * panel, height))
    gs = GridSpec(nrows, len(ratios), figure=fig, width_ratios=ratios,
                  left=0.004, right=0.996, bottom=bottom_in / height, top=1 - top_in / height,
                  wspace=0.025, hspace=0.025)

    for s, members in enumerate(sets):
        for row in range(nrows):
            axes = [fig.add_subplot(gs[row, 4 * s + k]) for k in range(3)]
            for ax in axes[:2]:
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
            if row == 0:
                titles = [("Dirty " if dirty else "") + c for c in CONFIGS] + ["Spectrum (Jy)"]
                for ax, title in zip(axes, [t[0].upper() + t[1:] for t in titles]):
                    ax.set_title(title, fontsize=8, pad=3)
            if row >= len(members):
                for ax in axes:
                    ax.set_visible(False)
                continue
            last = row == len(members) - 1
            g = members[row]
            crop = CROPS.get(g)
            key = crop["data"] if crop else g
            data = {c: load(key, c, dirty) for c in CONFIGS}
            ref = data["compact"] or data["extended"]
            if ref is None:
                for ax in axes[:2]:
                    blank(ax, "not imaged")
                axes[2].set_visible(False)
                continue
            side = crop and crop["side"]
            # view and aperture always from the clean compact image, so clean and dirty figures match
            clean_ref = (load(key, "compact") or load(key, "extended")) if dirty else ref
            ref = clean_ref or ref
            view = field_of_view(ref, side)
            if crop or not ref["mosaic"]:      # a mosaic's outline is part of the picture; single pointings stay inside the pb
                view = fit_inside(view, [d for d in data.values() if d is not None])
            for ax, c in zip(axes, CONFIGS):
                if data[c] is None:
                    blank(ax, "not available")
                    continue
                show(ax, data[c], view)
                scalebar(ax, view[2])
            axes[0].text(0.04, 0.95, (crop or TARGETS[g])["label"], color="white", fontsize=7, ha="left", va="top",
                         transform=axes[0].transAxes)
            ap = aperture(ref, side)
            show_spectrum(axes[2], {c: spectrum(key, c, ap, ref, dirty) for c in CONFIGS}, bottom=last)
            if last:
                axes[2].set_xlabel("v \u2212 v$_{sys}$ (km s$^{-1}$)", fontsize=6, labelpad=1)
            if s == 0 and row == 0:
                axes[2].legend(fontsize=5.5, frameon=False, loc="upper left", handlelength=1.2,
                               borderaxespad=0.3, labelspacing=0.2)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=250)
    plt.close(fig)
    print(f"wrote {os.path.relpath(out, REPO_ROOT)}.{{png,pdf}}")


def has_both(g, dirty=False):
    key = CROPS[g]["data"] if g in CROPS else g
    return all(os.path.exists(os.path.join(OUT_ROOT, key, c, "co21_briggs" + ("_dirty" if dirty else ""), "image.fits"))
               for c in CONFIGS)


def main():
    # rows with a missing configuration (ESO320-G030 extended, X485 not staged yet) are left out
    make_figure([g for g in MAIN if has_both(g)], os.path.join(FIG_DIR, "compact_vs_extended_co21_briggs"))
    # same rows, from the niter=0 products (image_co21_*.py --dirty)
    make_figure([g for g in MAIN if has_both(g)], os.path.join(FIG_DIR, "compact_vs_extended_co21_briggs_dirty"),
                dirty=True)
    make_figure(list(SPLIT), os.path.join(FIG_DIR, "compact_vs_extended_co21_briggs_IC4518_split"))
    make_figure(list(CROPS), os.path.join(FIG_DIR, "compact_vs_extended_co21_briggs_MCG-02-33-098_split"))


if __name__ == "__main__":
    main()
