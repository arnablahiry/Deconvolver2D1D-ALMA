"""
Animates simple_results/mom0_halfbeam_vs_fullbeam.ipynb's first figure (UV
coverage / dirty image / dirty beam) across all NCHAN channels, cycling the
dirty image and dirty beam panels through the cube while the UV coverage
(channel-independent) stays fixed. Same dark/serif aesthetic, same
sidelobe-revealing asinh stretch on the dirty beam, but computed globally
over the whole cube here so brightness is comparable frame to frame.

Run from the repo root:
    python simple_results/make_channel_gif.py
"""

import io

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import AsinhNorm
from casatools import table
from PIL import Image

DATA_DIR = "simple_results/data"
OUT_GIF = "simple_results/dirty_channels.gif"
CELL_ARCSEC = 0.04
WIDTH_KMS = 5.0
START_KMS = 4610.0

SURFACE = "#121212"
INK = "#e6e6e6"
MUTED = "#9a9a9a"
COLOR_UV = "#56B4E9"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": MUTED, "font.size": 15, "font.family": "serif",
    "xtick.direction": "in", "ytick.direction": "in",
})


def main():
    t = table()
    t.open("data/ngc7469_co21.ms")
    uvw = t.getcol("UVW")
    flag_row = t.getcol("FLAG_ROW")
    t.close()
    good = ~flag_row
    u_m, v_m = uvw[0][good], uvw[1][good]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(u_m), size=min(40000, len(u_m)), replace=False)
    u_s, v_s = u_m[idx], v_m[idx]

    psf = np.load(f"{DATA_DIR}/psf.npy")
    dirty = np.load(f"{DATA_DIR}/dirty.npy")
    bmaj, bmin, bpa = np.load(f"{DATA_DIR}/beam_arcsec_deg.npy")
    nchan = dirty.shape[0]
    velocity = START_KMS + np.arange(nchan) * WIDTH_KMS

    # Normalize the dirty cube by its own global peak (over ALL channels,
    # not per-frame) to percent-of-peak -- same convention as the beam
    # comparison GIF/notebook figure -- so brightness is directly
    # comparable channel to channel instead of each frame rescaling itself.
    dirty = dirty / dirty.max() * 100.0
    dirty_vmin, dirty_vmax = 0.0, 100.0

    # The PSF is already peak-1 normalized per channel by CASA itself (not
    # a display choice), so its own asinh sidelobe stretch stays as is --
    # computed globally over the whole cube either way.
    sidelobe_scale = np.percentile(np.abs(psf[psf < 0.5]), 99.5)
    psf_norm = AsinhNorm(linear_width=sidelobe_scale, vmin=-3 * sidelobe_scale, vmax=1.0)

    ny, nx = dirty.shape[1:]
    extent = np.array([nx, -nx, -ny, ny]) / 2.0 * CELL_ARCSEC
    beam_center = (2.0, -2.0)

    fig, (ax_uv, ax_dirty, ax_psf) = plt.subplots(1, 3, figsize=(16.5, 5.5), dpi=110,
                                                  constrained_layout=True)

    # --- UV coverage (static across all frames) ---
    ax_uv.scatter(u_s, v_s, s=2, alpha=0.25, color=COLOR_UV, linewidths=0)
    ax_uv.scatter(-u_s, -v_s, s=2, alpha=0.25, color=COLOR_UV, linewidths=0)
    ax_uv.set_box_aspect(1)
    ax_uv.set_aspect("equal", adjustable="box")
    ax_uv.set_xlabel("u (m)", fontsize=18, labelpad=12)
    ax_uv.set_ylabel("v (m)", fontsize=18, labelpad=12)
    ax_uv.grid(True, color=MUTED, alpha=0.25, linewidth=0.8)
    ax_uv.set_axisbelow(True)
    ax_uv.tick_params(which="both", direction="in", top=True, bottom=True,
                      left=True, right=True, length=8, width=1.2, labelsize=14,
                      pad=8)
    ax_uv.minorticks_on()
    ax_uv.tick_params(which="minor", length=4)
    for spine in ax_uv.spines.values():
        spine.set_color(MUTED)
    ax_uv.text(0.05, 0.95, "UV Coverage", transform=ax_uv.transAxes, ha="left",
              va="top", color=INK, fontsize=18)

    # --- dirty image / dirty beam (updated per frame) ---
    im_dirty = ax_dirty.imshow(dirty[0], origin="lower", extent=extent, cmap="magma",
                               vmin=dirty_vmin, vmax=dirty_vmax, interpolation="bilinear")
    im_psf = ax_psf.imshow(psf[0], origin="lower", extent=extent, cmap="magma",
                           norm=psf_norm, interpolation="bilinear")

    labels = {}
    for ax, im, name in [(ax_dirty, im_dirty, "Dirty Image"), (ax_psf, im_psf, "Dirty Beam")]:
        ax.set_xlim(2.5, -2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)
        for spine in ax.spines.values():
            spine.set_color(MUTED)
        labels[name] = ax.text(0.05, 0.95, name, transform=ax.transAxes, ha="left",
                               va="top", color=INK, fontsize=18)
        beam_patch = Ellipse(beam_center, width=bmin, height=bmaj, angle=-bpa,
                             facecolor=INK, edgecolor="none", alpha=0.85)
        ax.add_patch(beam_patch)

    bar_x_center, bar_y = -1.4, -2.05
    bar_half_len, cap_half_h, label_pad = 0.5, 0.10, 0.22
    x0, x1 = bar_x_center - bar_half_len, bar_x_center + bar_half_len
    ax_dirty.plot([x0, x1], [bar_y, bar_y], color=INK, linewidth=2, solid_capstyle="butt")
    ax_dirty.plot([x0, x0], [bar_y - cap_half_h, bar_y + cap_half_h], color=INK, linewidth=2)
    ax_dirty.plot([x1, x1], [bar_y - cap_half_h, bar_y + cap_half_h], color=INK, linewidth=2)
    ax_dirty.text(bar_x_center, bar_y + label_pad, '1"', color=INK,
                 ha="center", va="bottom", fontsize=16)

    frames = []
    for chan in range(nchan):
        im_dirty.set_data(dirty[chan])
        im_psf.set_data(psf[chan])
        chan_str = f"chan {chan}, v={velocity[chan]:.0f} km/s"
        labels["Dirty Image"].set_text(f"Dirty Image\n{chan_str}")
        labels["Dirty Beam"].set_text(f"Dirty Beam\n{chan_str}")

        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))

    plt.close(fig)

    frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:],
                   duration=90, loop=0)
    print(f"wrote {OUT_GIF} ({nchan} frames)")


if __name__ == "__main__":
    main()
