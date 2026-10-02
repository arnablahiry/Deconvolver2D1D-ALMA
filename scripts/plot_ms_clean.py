#!/usr/bin/env python
"""
Diagnostic plots of a multiscale CLEAN run (scripts/msclean_cube.py), next to
the dirty cube and the 2D-1D model of the same grid.

    python scripts/plot_ms_clean.py ngc3110 extended

Reads data/<galaxy>/<config>/ms_clean/*.fits and data/<galaxy>/<config>/cube/
{dirty_cube,model_cube,restored_own_beam}.fits; writes numbered PNGs + INDEX.md
into data/<galaxy>/<config>/ms_clean/.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm, TwoSlopeNorm
from matplotlib.patches import Ellipse
from astropy.io import fits

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CELL_ARCSEC = 0.02
LINE_KMS = dict(ngc3110=(4780.0, 5180.0), ngc2369=(2950.0, 3520.0))
ZOOM_ARCSEC = 5.0                      # half-width of the zoomed panels
CHANNELS = range(3, 23)                # channel maps shown (20 panels)

BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
SEQ, DIV = "inferno", "RdBu_r"


def load(path):
    """FITS cube -> (nchan, ny, nx) float64, header."""
    with fits.open(path) as h:
        return np.squeeze(h[0].data).astype(np.float64), h[0].header


def beam_arcsec(hdr):
    return hdr["BMAJ"] * 3600, hdr["BMIN"] * 3600, hdr["BPA"]


def extent(n):
    half = n * CELL_ARCSEC / 2
    return (half, -half, -half, half)          # RA increases to the left


def show(ax, img, title, cmap=SEQ, norm=None, zoom=True, beam=None, cbar=""):
    im = ax.imshow(img, origin="lower", extent=extent(img.shape[-1]), cmap=cmap, norm=norm,
                   interpolation="nearest")
    if zoom:
        ax.set_xlim(ZOOM_ARCSEC, -ZOOM_ARCSEC)
        ax.set_ylim(-ZOOM_ARCSEC, ZOOM_ARCSEC)
    if beam is not None:
        lim = ZOOM_ARCSEC if zoom else img.shape[-1] * CELL_ARCSEC / 2
        ax.add_patch(Ellipse((0.88 * lim, -0.88 * lim), beam[1], beam[0], angle=-beam[2],
                             fill=False, ec="white", lw=1.2))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('RA offset ["]')
    ax.set_ylabel('Dec offset ["]')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=cbar)
    return im


def mask_outline(ax, mask2d, color="cyan"):
    ax.contour(mask2d.astype(float), levels=[0.5], colors=color, linewidths=0.7,
               extent=extent(mask2d.shape[-1]), origin="lower")


def channel_grid(cubes, vel, title, path, cmap, norm, masks=None, zoom=True, cbar=""):
    fig, axes = plt.subplots(4, 5, figsize=(16, 13.5), constrained_layout=True,
                             sharex=True, sharey=True)
    for ax, c in zip(axes.flat, CHANNELS):
        im = ax.imshow(cubes[c], origin="lower", extent=extent(cubes.shape[-1]), cmap=cmap,
                       norm=norm, interpolation="nearest")
        if masks is not None and masks[c].any():
            mask_outline(ax, masks[c])
        if zoom:
            ax.set_xlim(ZOOM_ARCSEC, -ZOOM_ARCSEC)
            ax.set_ylim(-ZOOM_ARCSEC, ZOOM_ARCSEC)
        ax.text(0.04, 0.95, f"ch {c}  {vel[c]:.0f} km/s", transform=ax.transAxes, va="top",
                fontsize=8, color="white" if cmap == SEQ else "black",
                bbox=None if cmap == SEQ else dict(fc="white", ec="none", alpha=0.7, pad=1.5))
    for ax in axes[-1]:
        ax.set_xlabel('RA offset ["]')
    for ax in axes[:, 0]:
        ax.set_ylabel('Dec offset ["]')
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label=cbar)
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("galaxy")
    ap.add_argument("config")
    args = ap.parse_args()
    base = os.path.join(REPO_ROOT, "data", args.galaxy, args.config)
    d, cube_dir = os.path.join(base, "ms_clean"), os.path.join(base, "cube")
    label = f"{args.galaxy.upper().replace('NGC', 'NGC ')} {args.config}"

    image, hdr = load(os.path.join(d, "image.fits"))
    model, _ = load(os.path.join(d, "model.fits"))
    resid, _ = load(os.path.join(d, "residual.fits"))
    mask, _ = load(os.path.join(d, "mask.fits"))
    mask = mask > 0.5
    psf, _ = load(os.path.join(d, "psf.fits"))
    dirty, _ = load(os.path.join(cube_dir, "dirty_cube.fits"))
    pd_model, _ = load(os.path.join(cube_dir, "model_cube.fits"))
    pd_restored, pd_hdr = load(os.path.join(cube_dir, "restored_own_beam.fits"))
    assert image.shape == dirty.shape == pd_restored.shape, (image.shape, dirty.shape)
    nchan = image.shape[0]

    assert hdr["CTYPE3"] == "VRAD", hdr["CTYPE3"]
    vel = (hdr["CRVAL3"] + hdr["CDELT3"] * (np.arange(nchan) + 1 - hdr["CRPIX3"])) / 1e3
    lo, hi = LINE_KMS[args.galaxy]
    line = (vel >= lo) & (vel <= hi)
    dv = abs(hdr["CDELT3"]) / 1e3
    beam = beam_arcsec(hdr)
    pd_beam = beam_arcsec(pd_hdr)
    # noise from channels with an empty clean mask: the notebook's line window is
    # narrower than the emission, so its "line-free" channels are not all empty
    empty = ~mask.reshape(nchan, -1).any(1)
    sigma = np.std(dirty[empty], axis=(1, 2)).mean()          # Jy/beam per channel
    beam_px = np.pi * beam[0] * beam[1] / (4 * np.log(2)) / CELL_ARCSEC ** 2
    mask_any = mask.any(axis=0)
    index = []

    def save(fig, name, text):
        fig.savefig(os.path.join(d, name), dpi=110)
        plt.close(fig)
        index.append(f"- `{name}`: {text}")
        print("wrote", name)

    # 01: moment-0 maps -------------------------------------------------------
    m0 = {k: v[line].sum(0) * dv for k, v in
          dict(dirty=dirty, image=image, model=model, resid=resid, pd=pd_restored).items()}
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5), constrained_layout=True)
    vmax = np.percentile(m0["image"], 99.9)
    for row, zoom in enumerate((True, False)):
        a = axes[row]
        norm = None if zoom else AsinhNorm(linear_width=0.01 * vmax, vmin=0, vmax=vmax)
        show(a[0], m0["dirty"], "dirty" + ("" if zoom else ", full 16\""), norm=norm, zoom=zoom,
             beam=beam, cbar="Jy/beam km/s")
        show(a[1], m0["image"], "MS-CLEAN restored (mask outline cyan)", norm=norm, zoom=zoom,
             beam=beam, cbar="Jy/beam km/s")
        mask_outline(a[1], mask_any)
        mv = np.percentile(m0["model"][m0["model"] > 0], 99.5) if (m0["model"] > 0).any() else 1
        show(a[2], m0["model"], f"MS-CLEAN model; flux {m0['model'].sum():.1f} Jy km/s",
             norm=AsinhNorm(linear_width=0.05 * mv, vmin=0, vmax=mv), zoom=zoom, cbar="Jy/px km/s")
        r = np.percentile(np.abs(m0["resid"]), 99.5)
        show(a[3], m0["resid"], "MS-CLEAN residual", cmap=DIV,
             norm=TwoSlopeNorm(0, -r, r), zoom=zoom, beam=beam, cbar="Jy/beam km/s")
        mask_outline(a[3], mask_any, color="black")
    fig.suptitle(f"{label}: moment 0 over {lo:.0f}-{hi:.0f} km/s ({line.sum()} channels)")
    save(fig, "01_mom0.png", "Moment-0 of dirty, restored, model and residual: 10\" zoom (linear, "
         "top) and full field (asinh, bottom). Cyan/black outline: union of the auto-multithresh "
         "mask over channels.")

    # 02/03: channel maps -----------------------------------------------------
    vmax_ch = np.percentile(image[list(CHANNELS)], 99.95)
    channel_grid(image, vel, f"{label}: MS-CLEAN restored channels (cyan: clean mask)",
                 os.path.join(d, "02_channels_image.png"), SEQ, plt.Normalize(-3 * sigma, vmax_ch),
                 masks=mask, cbar="Jy/beam")
    index.append("- `02_channels_image.png`: Restored channel maps 3-22, 10\" zoom, one colour "
                 "scale; cyan = that channel's clean mask.")
    r = 8 * sigma
    channel_grid(resid, vel, f"{label}: MS-CLEAN residual channels, +/-8 sigma (cyan: clean mask)",
                 os.path.join(d, "03_channels_residual.png"), DIV, TwoSlopeNorm(0, -r, r),
                 masks=mask, zoom=False, cbar="Jy/beam")
    index.append("- `03_channels_residual.png`: Residual channel maps 3-22, full field, "
                 "+/-8 sigma; emission left outside the mask shows as coherent red.")
    print("wrote 02_channels_image.png, 03_channels_residual.png")

    # 04: residual statistics per channel -------------------------------------
    rms = resid.reshape(nchan, -1).std(1)
    out_peak = np.array([resid[c][~mask[c]].max() for c in range(nchan)])
    in_peak = np.array([resid[c][mask[c]].max() if mask[c].any() else np.nan for c in range(nchan)])
    area = mask.reshape(nchan, -1).sum(1) / beam_px
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True, constrained_layout=True,
                                 gridspec_kw=dict(height_ratios=(2, 1)))
    for y, c, lab in ((out_peak / sigma, ORANGE, "peak outside mask"),
                      (in_peak / sigma, AQUA, "peak inside mask"),
                      (rms / sigma, BLUE, "rms, whole field")):
        a1.plot(vel, y, "-o", color=c, lw=2, ms=4, label=lab)
        k = np.nanargmax(y)
        a1.annotate(lab, (vel[k], y[k]), xytext=(6, 4), textcoords="offset points", fontsize=8,
                    color="#52514e")
    a1.axhline(1, color=MUTED, ls="--", lw=1)
    a1.axhline(3, color=MUTED, ls=":", lw=1)
    a1.text(vel[0], 3, "3 sigma stop ", va="top", ha="left", fontsize=8, color=MUTED)
    a1.set_ylabel(f"residual / sigma  (sigma = {sigma * 1e3:.2f} mJy/beam)")
    a1.legend(frameon=False, fontsize=8)
    a1.grid(alpha=0.3)
    a2.bar(vel, area, width=0.8 * dv, color=BLUE)
    a2.set_ylabel("mask area [beams]")
    a2.set_xlabel("v (LSRK, radio) [km/s]")
    for a in (a1, a2):
        a.axvspan(lo, hi, color="0.92", zorder=0)
    fig.suptitle(f"{label}: MS-CLEAN residual per channel (grey: line window)")
    save(fig, "04_residual_per_channel.png", "Per-channel residual rms and peaks (inside / outside "
         "the mask) in units of the line-free noise, and the mask area per channel.")

    # 05: flux spectra --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for y, c, lab in ((model.sum((1, 2)), BLUE, "MS-CLEAN model"),
                      (pd_model.sum((1, 2)), ORANGE, "2D-1D model"),
                      (image[:, mask_any].sum(1) / beam_px, AQUA, "MS-CLEAN restored, in mask")):
        ax.step(vel, y * 1e3, where="mid", color=c, lw=2, label=f"{lab}: {y[line].sum() * dv:.1f} Jy km/s")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvspan(lo, hi, color="0.92", zorder=0)
    ax.set_xlabel("v (LSRK, radio) [km/s]")
    ax.set_ylabel("flux density [mJy]")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title(f"{label}: total flux per channel (legend: integrated over the line window)")
    save(fig, "05_flux_spectra.png", "Flux per channel of the MS-CLEAN model, the 2D-1D model "
         "(cube/model_cube.fits), and the restored image summed inside the mask.")

    # 06: MS-CLEAN vs 2D-1D, both restored with the extended beam -------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    vmax = max(np.percentile(m0["image"], 99.9), np.percentile(m0["pd"], 99.9))
    show(axes[0], m0["image"], f"MS-CLEAN restored ({beam[0]:.3f}\" x {beam[1]:.3f}\")",
         norm=plt.Normalize(0, vmax), beam=beam, cbar="Jy/beam km/s")
    show(axes[1], m0["pd"], f"2D-1D restored ({pd_beam[0]:.3f}\" x {pd_beam[1]:.3f}\")",
         norm=plt.Normalize(0, vmax), beam=pd_beam, cbar="Jy/beam km/s")
    diff = m0["image"] - m0["pd"]
    r = np.percentile(np.abs(diff), 99.5)
    show(axes[2], diff, "MS-CLEAN - 2D-1D", cmap=DIV, norm=TwoSlopeNorm(0, -r, r), beam=beam,
         cbar="Jy/beam km/s")
    fig.suptitle(f"{label}: moment 0, MS-CLEAN vs 2D-1D restored (cube/restored_own_beam.fits)")
    save(fig, "06_vs_2d1d_mom0.png", "Moment-0 of the MS-CLEAN and 2D-1D restored cubes (same "
         "beam) and their difference, 10\" zoom.")

    # 07: spectra at a few positions ------------------------------------------
    peak = np.unravel_index(np.argmax(m0["image"]), m0["image"].shape)
    c0 = image.shape[-1] // 2
    spots = dict(peak=peak, centre=(c0, c0), arm=(c0 - int(4.0 / CELL_ARCSEC), c0 + int(1.0 / CELL_ARCSEC)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3), constrained_layout=True, sharey=False)
    for ax, (name, (y, x)) in zip(axes, spots.items()):
        for cube, c, lab in ((dirty, MUTED, "dirty"), (image, BLUE, "MS-CLEAN restored"),
                             (pd_restored, ORANGE, "2D-1D restored")):
            ax.step(vel, cube[:, y, x] * 1e3, where="mid", color=c, lw=2, label=lab)
        ax.axvspan(lo, hi, color="0.92", zorder=0)
        off = ((c0 - x) * CELL_ARCSEC, (y - c0) * CELL_ARCSEC)
        ax.set_title(f"{name}: ({off[0]:+.1f}\", {off[1]:+.1f}\")", fontsize=9)
        ax.set_xlabel("v [km/s]")
        ax.set_ylabel("mJy/beam")
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{label}: single-pixel spectra (RA, Dec offsets)")
    save(fig, "07_spectra_positions.png", "Single-pixel spectra of dirty, MS-CLEAN restored and "
         "2D-1D restored at the mom-0 peak, the phase centre and a point on the southern arm.")

    # 08: PSF -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    p = psf[nchan // 2]
    show(axes[0], p, f"PSF, channel {nchan // 2} (central 4\")", cmap="RdBu_r", norm=TwoSlopeNorm(0, -0.2, 1),
         zoom=False, cbar="")
    axes[0].set_xlim(2, -2)
    axes[0].set_ylim(-2, 2)
    cut = p[p.shape[0] // 2]
    x = (np.arange(cut.size) - cut.size // 2) * CELL_ARCSEC
    axes[1].plot(x, cut, color=BLUE, lw=2)
    axes[1].axhline(0, color=MUTED, lw=0.8)
    axes[1].set_xlim(-4, 4)
    axes[1].set_xlabel('RA offset ["]')
    axes[1].set_title("PSF cut through centre (RA)", fontsize=9)
    axes[1].grid(alpha=0.3)
    save(fig, "08_psf.png", "The tclean PSF (800 px, central channel) and its RA cut.")

    with open(os.path.join(d, "INDEX.md"), "w") as f:
        f.write(f"# {label}: multiscale CLEAN plots\n\n"
                f"Run: scripts/msclean_cube.py (tclean multiscale, scales "
                f"[0, 10, 30, 60] px, nsigma=3, auto-multithresh long-baseline settings, common beam "
                f"{beam[0]:.3f}\" x {beam[1]:.3f}\" @ {beam[2]:.1f} deg). Plots: scripts/plot_ms_clean.py.\n"
                f"Noise sigma = {sigma * 1e3:.3f} mJy/beam (dirty-cube channels with an empty clean mask: {int(empty.sum())}).\n\n"
                + "\n".join(index) + "\n")
    print("wrote INDEX.md")


if __name__ == "__main__":
    main()
