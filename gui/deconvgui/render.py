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


EDGE_FRAC = 0.02          # fraction of the field cropped on each side in displays (edge artifacts)


def edge_px(ny, nx):
    """pixels to crop on each side for display"""
    return max(2, int(round(EDGE_FRAC * min(ny, nx))))


GIF_VERSION = 4          # bump when the GIF's look changes: the server re-renders older ones from frames.npy

# the GUI's dark theme and font (static/app.css), so the GIF looks like the app's snapshot panel
_SURFACE, _LINE, _FG, _MUTED = (26, 25, 20), (46, 44, 37), (235, 233, 227), (154, 149, 135)
_STATIC = __import__("os").path.join(__import__("os").path.dirname(__file__), "static")


def _font(size, bold=False):
    from PIL import ImageFont
    import os
    try:
        return ImageFont.truetype(os.path.join(_STATIC, "abaddon-bold.ttf" if bold else "abaddon-light.ttf"), size)
    except OSError:
        return ImageFont.load_default()


def nice_ticks(lo, hi, n=5):
    """as niceTicks in app.js"""
    if not hi > lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** np.floor(np.log10(raw))
    step = next((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), raw)
    v, out = np.ceil(lo / step) * step, []
    while v <= hi + step * 1e-9:
        out.append(0.0 if abs(v) < step * 1e-6 else float(f"{v:.12g}"))
        v += step
    return out


def tick_label(v):
    """as tick() in app.js"""
    if v == 0:
        return "0"
    if abs(v) >= 1e5 or abs(v) < 1e-3:
        return f"{v:.1e}".replace("e+", "e")
    return f"{float(f'{v:.4g}'):g}"


def log_subset(iters, n=140):
    """indices of frames at ~log-spaced iterations (all of them when they already are)"""
    it = np.maximum(np.asarray(iters, dtype=float), 1)
    want = np.geomspace(it[0], it[-1], n) if it[-1] > it[0] else it[-1:]
    idx = sorted({int(np.argmin(np.abs(np.log(it) - np.log(w)))) for w in want} | {len(it) - 1})
    return idx


def _unit_label(unit, font, small):
    """the unit as an RGBA image, read bottom to top; "⁻²" drawn as a raised small "-2" (not in the font)"""
    from PIL import Image, ImageDraw
    base, sup = (unit.split("\u207b\u00b2")[0], "-2") if "\u207b\u00b2" in unit else (unit, "")
    w1 = int(font.getlength(base)); w2 = int(small.getlength(sup)) + (2 if sup else 0)
    hgt = font.size + 6
    im = Image.new("RGBA", (w1 + w2 + 2, hgt), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.text((0, 3), base, font=font, fill=_FG)
    if sup:
        d.text((w1 + 2, 0), sup, font=small, fill=_FG)
    return im.rotate(90, expand=True)


def make_gif(frames, iters, path, cmap="inferno", unit="", cell=None, duration_ms=120, size=480):
    """Moment-0 frames (surface brightness) -> animated GIF drawn like the app's snapshot panel (dark card,
    "iteration N" title, rounded image, colour bar on the right with ticks outside and the unit as its
    label, the app's font): edges cropped, frames at log-spaced iterations, one linear scale for all frames
    from 0 to the final frame's maximum (clip 100%)."""
    from PIL import Image, ImageDraw

    keep = log_subset(iters)
    frames = np.asarray(frames, dtype=np.float32)[keep]
    iters = [int(iters[i]) for i in keep]
    ny, nx = frames.shape[1:]
    m = edge_px(ny, nx)
    frames = frames[:, m:ny - m, m:nx - m]
    ny, nx = frames.shape[1:]
    vmax = float(frames[-1].max()) or 1.0
    lut = colormap_lut(cmap)

    # geometry (px): image scaled to ~size (integer nearest-neighbour up, smooth down), as the panel
    k = size / max(nx, ny)
    k = max(1, int(k)) if k >= 1 else k
    W, H = max(1, round(nx * k)), max(1, round(ny * k))
    f_title, f_tick, f_unit, f_sup = _font(17, True), _font(14), _font(15), _font(11)
    pad, top, gap, strip, tl = 16, 34, 12, 14, 6
    ticks = [v for v in nice_ticks(0, vmax, 5) if 0 <= v <= vmax * (1 + 1e-6)]
    labels = [tick_label(v) for v in ticks]
    lw = max(int(f_tick.getlength(t)) for t in labels)
    ulab = _unit_label(unit, f_unit, f_sup) if unit else None
    x_img, y_img = pad, pad + top
    x_bar = x_img + W + gap
    x_unit = x_bar + strip + tl + 4 + lw + 10
    width = x_unit + (ulab.width if ulab else 0) + pad
    height = y_img + H + pad

    # the static parts: card, colour bar, unit
    card = Image.new("RGB", (width, height), _SURFACE)
    d = ImageDraw.Draw(card)
    grad = lut[np.linspace(255, 0, H).astype(int)][:, None, :].repeat(strip, axis=1)
    smask = Image.new("L", (strip, H), 0)                      # rounded and outlined as the image
    ImageDraw.Draw(smask).rounded_rectangle([0, 0, strip - 1, H - 1], radius=7, fill=255)
    card.paste(Image.fromarray(grad.astype(np.uint8), "RGB"), (x_bar, y_img), smask)
    d.rounded_rectangle([x_bar, y_img, x_bar + strip - 1, y_img + H - 1], radius=7, outline=_LINE, width=1)
    for v, t in zip(ticks, labels):
        y = round(y_img + H - 1 - v / vmax * (H - 1))
        d.line([x_bar + strip, y, x_bar + strip + tl - 1, y], fill=_MUTED, width=1)
        d.text((x_bar + strip + tl + 4, y), t, font=f_tick, fill=_FG, anchor="lm")
    if ulab:
        card.paste(ulab, (x_unit, y_img + (H - ulab.height) // 2), ulab)
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, W - 1, H - 1], radius=7, fill=255)

    images = []
    for f, it in zip(frames, iters):
        t = np.clip(f / vmax, 0, 1)
        im = Image.fromarray(lut[(t * 255).astype(np.uint8)][::-1], "RGB")      # row 0 = south -> bottom
        im = im.resize((W, H), Image.NEAREST if k >= 1 else Image.LANCZOS)
        fr = card.copy()
        fr.paste(im, (x_img, y_img), mask)
        dd = ImageDraw.Draw(fr)
        dd.rounded_rectangle([x_img, y_img, x_img + W - 1, y_img + H - 1], radius=7, outline=_LINE, width=1)
        dd.text((x_img, pad + top // 2 - 2), f"iteration {it:,}", font=f_title, fill=_FG, anchor="lm")
        images.append(fr.convert("P", palette=Image.ADAPTIVE, colors=255))
    durations = [duration_ms] * (len(images) - 1) + [2500]
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    with open(path + ".v", "w") as fh:
        fh.write(str(GIF_VERSION))
    return path
