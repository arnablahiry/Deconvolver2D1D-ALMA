"""Server-side rendering: colormap tables for the browser, and the evolution GIF."""

import numpy as np


def colormap_lut(name="inferno", n=256):
    from matplotlib import colormaps
    cmap = colormaps[name]
    return (cmap(np.linspace(0, 1, n))[:, :3] * 255).round().astype(np.uint8)


def stretch(img, lo, hi, mode="asinh"):
    t = np.clip((img - lo) / max(hi - lo, 1e-30), 0, 1)
    if mode == "asinh":
        a = 0.1
        t = np.arcsinh(t / a) / np.arcsinh(1 / a)
    elif mode == "sqrt":
        t = np.sqrt(t)
    return t


def make_gif(frames, iters, path, cmap="inferno", size=420, duration_ms=90):
    """Moment-0 frames -> animated GIF, one fixed stretch for all frames (the
    final frame's 0.5-99.8 percentile) so the build-up of flux is visible."""
    from PIL import Image, ImageDraw

    frames = np.asarray(frames, dtype=np.float32)
    final = frames[-1]
    lo = 0.0
    hi = float(np.percentile(final, 99.8)) or float(final.max()) or 1.0
    lut = colormap_lut(cmap)
    images = []
    for f, it in zip(frames, iters):
        t = stretch(f, lo, hi)
        rgb = lut[(t * 255).astype(np.uint8)][::-1]      # row 0 = south -> bottom
        im = Image.fromarray(rgb, "RGB")
        scale = max(1, size // max(im.size))
        if scale > 1:
            im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 112, 16], fill=(0, 0, 0))
        d.text((4, 3), f"iteration {it}", fill=(255, 255, 255))
        images.append(im.convert("P", palette=Image.ADAPTIVE, colors=255))
    durations = [duration_ms] * (len(images) - 1) + [1500]
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0)
    return path
