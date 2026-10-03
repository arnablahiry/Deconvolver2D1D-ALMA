"""
Animates the 2x2 figure in simple_results/mom0_halfbeam_vs_fullbeam.ipynb
(full / half / quarter beam images + spectrum) across all NCHAN channels,
instead of the notebook's static moment-0 maps: each image panel cycles
through its own per-channel slice of the model convolved with that beam
(no residual, no noise), on a fixed (own) color scale per panel so
brightness is comparable frame to frame. The spectrum panel is unchanged
except for a vertical dashed line marking the current channel's velocity,
which moves as the animation advances.

Run from the repo root:
    python simple_results/make_beam_channels_gif.py
"""

import io
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from scipy.signal import fftconvolve
from PIL import Image

sys.path.insert(0, "src")
from fista_2d1d import gaussian_beam

DATA_DIR = "simple_results/data"
OUT_GIF = "simple_results/beam_channels.gif"
CELL_ARCSEC = 0.04
WIDTH_KMS = 5.0
START_KMS = 4610.0

# Same four regions marked on the Half Beam panel in the notebook's static
# 2x2 figure and used by the region-cutout figures.
REGIONS = [
    ("Nucleus", 0.00, 0.00),
    ("Upper Arm", 0.95, 1.10),
    ("Left Clump", 1.50, -0.85),
    ("Right Clump", -1.40, -0.75),
]
BOX_HALF = 0.5  # arcsec

SURFACE = "#121212"
INK = "#e6e6e6"
MUTED = "#9a9a9a"
COLOR_DIRTY, COLOR_FULL, COLOR_HALF, COLOR_QUARTER = (
    "#999999", "#E69F00", "#56B4E9", "#009E73",
)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": MUTED, "font.size": 15, "font.family": "serif",
    "xtick.direction": "in", "ytick.direction": "in",
})


def convolve_beam(model, bmaj_arcsec, bmin_arcsec, bpa_deg):
    beam_pix = gaussian_beam(model.shape[1:], bmaj_arcsec / CELL_ARCSEC,
                             bmin_arcsec / CELL_ARCSEC, bpa_deg)
    out = np.empty_like(model)
    for k in range(model.shape[0]):
        out[k] = fftconvolve(model[k], beam_pix, mode="same")
    return out


def main():
    model = np.load(f"{DATA_DIR}/model.npy")
    dirty = np.load(f"{DATA_DIR}/dirty.npy")
    bmaj, bmin, bpa = np.load(f"{DATA_DIR}/beam_arcsec_deg.npy")
    nchan = model.shape[0]
    velocity = START_KMS + np.arange(nchan) * WIDTH_KMS

    beams = {
        "full": (bmaj, bmin),
        "half": (bmaj / 2.0, bmin / 2.0),
        "quarter": (bmaj / 4.0, bmin / 4.0),
    }
    raw_cubes = {name: convolve_beam(model, bm, bn, bpa) for name, (bm, bn) in beams.items()}

    # Normalize each cube by its own global peak (over ALL channels, not
    # per-frame) to percent-of-peak -- same convention as the notebook's
    # static moment-0 figure, so a channel's brightness is directly
    # comparable to every other channel and to the other two beams.
    cubes = {name: cube / cube.max() * 100.0 for name, cube in raw_cubes.items()}
    vranges = {name: (0.0, 100.0) for name in cubes}

    spectra = {name: cube.sum(axis=(1, 2)) for name, cube in cubes.items()}
    spectra["dirty"] = dirty.sum(axis=(1, 2))
    norm_spectra = {name: s / np.abs(s).max() for name, s in spectra.items()}

    ny, nx = model.shape[1:]
    extent = np.array([nx, -nx, -ny, ny]) / 2.0 * CELL_ARCSEC
    beam_center = (2.0, -2.0)

    fig, axes = plt.subplots(2, 2, figsize=(11, 11), dpi=110, constrained_layout=True)
    ax_full, ax_half = axes[0, 0], axes[0, 1]
    ax_quarter, ax_spec = axes[1, 0], axes[1, 1]

    image_panels = [
        (ax_full, "full", "Full Beam"),
        (ax_half, "half", "Half Beam"),
        (ax_quarter, "quarter", "Quarter Beam"),
    ]

    ims, labels = {}, {}
    for ax, name, title in image_panels:
        vmin, vmax = vranges[name]
        ims[name] = ax.imshow(cubes[name][0], origin="lower", extent=extent, cmap="magma",
                              vmin=vmin, vmax=vmax, interpolation="bilinear")
        ax.set_xlim(2.5, -2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)
        for spine in ax.spines.values():
            spine.set_color(MUTED)
        labels[name] = ax.text(0.05, 0.95, title, transform=ax.transAxes, ha="left",
                               va="top", color=INK, fontsize=18)
        bm, bn = beams[name]
        beam_patch = Ellipse(beam_center, width=bn, height=bm, angle=-bpa,
                             facecolor=INK, edgecolor="none", alpha=0.85)
        ax.add_patch(beam_patch)

    for _, cx, cy in REGIONS:
        ax_half.add_patch(Rectangle((cx - BOX_HALF, cy - BOX_HALF), 2 * BOX_HALF, 2 * BOX_HALF,
                                    edgecolor=INK, facecolor="none", linewidth=1.8))

    ax_full.sharex(ax_full)
    ax_half.sharex(ax_full)
    ax_quarter.sharex(ax_full)
    ax_half.sharey(ax_full)
    ax_quarter.sharey(ax_full)

    bar_x_center, bar_y = -1.4, -2.05
    bar_half_len, cap_half_h, label_pad = 0.5, 0.10, 0.22
    x0, x1 = bar_x_center - bar_half_len, bar_x_center + bar_half_len
    ax_full.plot([x0, x1], [bar_y, bar_y], color=INK, linewidth=2, solid_capstyle="butt")
    ax_full.plot([x0, x0], [bar_y - cap_half_h, bar_y + cap_half_h], color=INK, linewidth=2)
    ax_full.plot([x1, x1], [bar_y - cap_half_h, bar_y + cap_half_h], color=INK, linewidth=2)
    ax_full.text(bar_x_center, bar_y + label_pad, '1"', color=INK,
                ha="center", va="bottom", fontsize=16)

    series = [
        ("dirty", "Dirty", COLOR_DIRTY, ":"),
        ("full", "Full beam", COLOR_FULL, "-"),
        ("half", "Half beam", COLOR_HALF, "--"),
        ("quarter", "Quarter beam", COLOR_QUARTER, "-."),
    ]
    for name, label, color, style in series:
        ax_spec.plot(velocity, norm_spectra[name], linestyle=style, color=color,
                    linewidth=2, label=label)

    chan_line = ax_spec.axvline(velocity[0], color="white", linestyle="--", linewidth=1.5)

    ax_spec.set_box_aspect(1)
    ax_spec.set_xlabel("Velocity (km/s)", fontsize=18, labelpad=12)
    ax_spec.set_ylabel("Normalized flux", fontsize=18, labelpad=16)
    ax_spec.yaxis.set_label_position("right")
    ax_spec.grid(True, color=MUTED, alpha=0.25, linewidth=0.8)
    ax_spec.set_axisbelow(True)
    ax_spec.tick_params(which="both", direction="in", top=True, bottom=True,
                        left=True, right=True, labelleft=False, labelright=True,
                        length=8, width=1.2, labelsize=14, pad=8)
    ax_spec.tick_params(which="minor", length=4)
    ax_spec.minorticks_on()
    for spine in ax_spec.spines.values():
        spine.set_color(MUTED)

    legend = ax_spec.legend(loc="upper left", frameon=True, fontsize=15,
                            facecolor=SURFACE, edgecolor=MUTED, framealpha=0.9)
    plt.setp(legend.get_texts(), color=INK)

    titles = {"full": "Full Beam", "half": "Half Beam", "quarter": "Quarter Beam"}

    frames = []
    for chan in range(nchan):
        chan_str = f"chan {chan}, v={velocity[chan]:.0f} km/s"
        for name in ("full", "half", "quarter"):
            ims[name].set_data(cubes[name][chan])
            labels[name].set_text(f"{titles[name]}\n{chan_str}")
        chan_line.set_xdata([velocity[chan], velocity[chan]])

        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))

    plt.close(fig)

    frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:], duration=90, loop=0)
    print(f"wrote {OUT_GIF} ({nchan} frames)")


if __name__ == "__main__":
    main()
