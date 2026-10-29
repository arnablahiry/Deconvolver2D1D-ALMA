"""Make the GUI's animated background: a wide lognormal density field in
plasma (the same recipe as Albedo's background) and a heavily beam-smeared
copy of it. The page cross-fades between the two, from blurred to sharp and
back, as a nod to what the app does.

Both images are built and smoothed in Fourier space, so they are periodic:
they tile seamlessly left to right, which lets the page pan them in an
endless loop.

    python gui/assets/make_background.py
"""
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "deconvgui" / "static"

W, H = 3840, 1920            # 2:1, tiles horizontally
BOX = 2.5                    # field spans this many "readme boxes" vertically
SMOOTH = 1.5                 # Gaussian smoothing in pixels, removes pixel grain
BEAM = 45                    # Gaussian sigma in pixels of the blurred copy
QUALITY = 82

fx = np.fft.fftfreq(W)[None, :]
fy = np.fft.fftfreq(H)[:, None]
gauss = lambda sigma: np.exp(-0.5 * (np.hypot(fx, fy) * 2 * np.pi * sigma) ** 2)

rng = np.random.default_rng(7)
kk = np.hypot(fx * W / H * BOX, fy * BOX)
kk[0, 0] = 1
spec = kk ** -1.4 * gauss(SMOOTH)
spec[0, 0] = 0
field = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(H, W))) * spec))
dens = 2 + np.exp(field / field.std()) - 1             # 1 + delta > 0, linear "sky"
smeared = np.real(np.fft.ifft2(np.fft.fft2(dens) * gauss(BEAM)))


def paint(lin):
    img = np.log10(lin)
    lo, hi = np.percentile(img, [0.5, 99.5])            # each image on its own stretch, as a viewer would
    rgb = colormaps["plasma"](np.clip((img - lo) / (hi - lo), 0, 1))[..., :3]
    return np.round(rgb * 255).astype(np.uint8)


for name, lin in (("bg-sharp", dens), ("bg-blurred", smeared)):
    out = OUT / f"{name}.webp"
    arr = paint(lin)
    Image.fromarray(arr).save(out, quality=QUALITY, method=6)
    print(f"{out.name}: {arr.shape[1]}x{arr.shape[0]}, {out.stat().st_size / 1e6:.2f} MB")
