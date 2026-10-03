"""
Animates the 4x3 region-cutout figure in
simple_results/mom0_halfbeam_vs_fullbeam.ipynb (nucleus + 3 off-nuclear
clumps, x full/half/quarter beam) across all NCHAN channels: each cutout
cycles through its own per-channel slice of the model convolved with that
beam (no residual, no noise), normalized by each beam-cube's own global
peak (over ALL channels, not per-frame) to percent-of-peak -- same
convention as make_beam_channels_gif.py.

Run from the repo root:
    python simple_results/make_region_beam_gif.py
"""

import io
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from PIL import Image

sys.path.insert(0, "src")
from fista_2d1d import gaussian_beam

DATA_DIR = "simple_results/data"
OUT_GIF = "simple_results/region_beam_channels.gif"
CELL_ARCSEC = 0.04
WIDTH_KMS = 5.0
START_KMS = 4610.0

SURFACE = "#121212"
INK = "#e6e6e6"
MUTED = "#9a9a9a"

REGIONS = [
    ("Nucleus", 0.00, 0.00),
    ("Upper Arm", 0.95, 1.10),
    ("Left Clump", 1.50, -0.85),
    ("Right Clump", -1.40, -0.75),
]
BOX_HALF = 0.5  # arcsec

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
    bmaj, bmin, bpa = np.load(f"{DATA_DIR}/beam_arcsec_deg.npy")
    nchan = model.shape[0]
    velocity = START_KMS + np.arange(nchan) * WIDTH_KMS

    beams = {
        "full": (bmaj, bmin),
        "half": (bmaj / 2.0, bmin / 2.0),
        "quarter": (bmaj / 4.0, bmin / 4.0),
    }
    raw_cubes = {name: convolve_beam(model, bm, bn, bpa) for name, (bm, bn) in beams.items()}
    # Percent-of-peak, normalized by each cube's own global (all-channel) max.
    cubes = {name: cube / cube.max() * 100.0 for name, cube in raw_cubes.items()}

    ny, nx = model.shape[1:]
    extent = np.array([nx, -nx, -ny, ny]) / 2.0 * CELL_ARCSEC

    beam_cols = [("full", "Full Beam"), ("half", "Half Beam"), ("quarter", "Quarter Beam")]

    fig, axes = plt.subplots(4, 3, figsize=(9, 12), dpi=110, constrained_layout=True)

    ims = {}
    for row, (region_name, cx, cy) in enumerate(REGIONS):
        for col, (key, title) in enumerate(beam_cols):
            ax = axes[row, col]
            ims[(row, col)] = ax.imshow(cubes[key][0], origin="lower", extent=extent,
                                        cmap="magma", vmin=0, vmax=100,
                                        interpolation="bilinear")
            ax.set_xlim(cx + BOX_HALF, cx - BOX_HALF)
            ax.set_ylim(cy - BOX_HALF, cy + BOX_HALF)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_box_aspect(1)
            for spine in ax.spines.values():
                spine.set_color(MUTED)
            if row == 0:
                ax.set_title(title, fontsize=18, pad=10)
            if col == 0:
                ax.set_ylabel(region_name, fontsize=16)

    suptitle = fig.suptitle("", fontsize=16, color=INK)

    frames = []
    for chan in range(nchan):
        for row, _ in enumerate(REGIONS):
            for col, (key, _) in enumerate(beam_cols):
                ims[(row, col)].set_data(cubes[key][chan])
        suptitle.set_text(f"chan {chan}, v={velocity[chan]:.0f} km/s")

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
