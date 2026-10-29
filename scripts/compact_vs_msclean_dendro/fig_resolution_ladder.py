#!/usr/bin/env python
"""
Main result: the compact 2D-1D deconvolution against the extended MS-CLEAN over a ladder of comparison beams, from the
compact clean beam to the extended clean beam, with the compact array's resolution limit (lambda/B_max, its longest
baseline) in the middle; at each beam inside that beam's own astrodendro mask of the extended MS-CLEAN.

Beams: the compact clean beam (elliptical, from compact/msclean/image.fits), NSIDE circular Gaussians log-spaced in FWHM
down to lambda/B_max,compact (circular), NSIDE more down to the extended clean beam (elliptical, from the CLEAN's
image.fits). The circular FWHMs run from the compact beam's minor axis to the extended beam's major axis (ends
excluded), so each lies inside the compact beam and contains the extended one: a strictly finer, resp. coarser,
resolution in every direction.

At each beam b (line channels):
  C         the compact deconvolved model (unmasked) * b, Jy/beam of b
  K         the extended MS-CLEAN components (unmasked) * b
  I         the extended restored image at b, the reference: K + the CLEAN's residual brought from the extended clean
            beam to b (beams.to_beam: the Gaussian taking one beam to the other, then x omega_b / omega_ext), i.e.
            everything the extended data hold at that resolution, including the faint emission CLEAN left in its
            residual (about 55 Jy km/s inside the mask at the compact beam for NGC 3110)
  M_b       msclean_complete.dendro_mask of I (the recipe of the extended CLEAN's own mask_final.npy, at b:
            per-channel sigma of the smoothed residual, trunks >= 3 sigma, >= 2 beams b, peak above the negated image's)
and over the voxels of M_b, against I (solid curves) and, dashed, against K alone (the CLEAN model only):
  RMSE              100 ||C - I|| / ||I||                                                        (%)
  total flux        100 (sum C - sum I) / sum I                                                   (%)
  median deviation  median of 100 (C - I) / I over the voxels with I > 5% of I's maximum in M_b  (%)
                    (signed: the typical voxel's bias; the median of |.| is in the JSON)
The residual also carries noise and sidelobes, so the RMSE against I does not go to zero even for a perfect model.
The mask grows with the beam, as the convolved emission does, so nothing of either model spills out of its own region.

Maps (above), five of the beams (the two clean beams, lambda/B_max in the middle, one beam on each side of it):
moment 0 (line channels) of C (top) and I (middle), unmasked, Jy arcsec^-2 km/s on one linear scale per column,
with M_b's footprint as a thin contour; bottom 100 (C - I) / I of the moment-0 maps of C x M_b and I x M_b, +-50%, grey
outside M_b's footprint or where I < 5% of its peak. No colorbars (the ranges are in the row labels).

Ladder (below): the x axis is the beam index (evenly spaced, lambda/B_max at the centre); the tick labels give the FWHM
(major x minor on one line for the two clean beams), and the beams are drawn below to scale as outlines (one arcsec
scale for all; RA increases left).

    [GAL_DIR=results/NGC3110] [NSIDE=4] [CLEAN=msclean_finished] [OUT_DIR=...] \
        python scripts/compact_vs_msclean_dendro/fig_resolution_ladder.py
    -> <GAL_DIR>/figures/resolution_ladder/resolution_ladder.{png,pdf,json}
"""
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in ("simple", "scripts", "scripts/compact_vs_msclean_dendro"):
    sys.path.insert(0, os.path.join(REPO, _p))
from restore import convolve_with_beam  # noqa: E402
from msclean_complete import dendro_mask  # noqa: E402
import beams  # noqa: E402

R = os.environ.get("GAL_DIR", os.path.join(REPO, "results/NGC3110"))
NSIDE = int(os.environ.get("NSIDE", 4))
CLEAN = os.environ.get("CLEAN", "msclean_finished")
G20 = os.environ.get("GRID") == "20kms"   # REF=joint on the 64 x 20 km/s, 0.08" grid of the pixel-scale runs (51" field)
OUT = os.environ.get("OUT_DIR", os.path.join(R, "figures", "resolution_ladder" + ("_joint" if os.environ.get("REF") == "joint" else "")
                                             + ("20" if G20 else "")))
os.makedirs(OUT, exist_ok=True)
GC = json.load(open(os.path.join(R, "config.json")))
CELL, DV = GC["cell_arcsec"], GC["width_kms"]
L = json.load(open(os.path.join(R, "line_channels.json")))["line"]
T3110 = os.path.basename(os.path.normpath(R)) == "NGC3110"      # other targets: the 20 km/s gallery products (<R>/g20)
if G20:
    _g = json.load(open(os.path.join(R, "consistency_20kms_0.08/grid.json" if T3110 else "g20/grid.json")))
    L, DV = _g["line"], _g["width_kms"]
BLUE, ORANGE, GREEN, MUTED = "#2a6fdb", "#e07b00", "#1a9e77", "#8a8a8a"
TICK, LAB = 13, 15
plt.rcParams.update({"font.family": "serif", "font.size": 13, "xtick.direction": "in", "ytick.direction": "in"})
rd = lambda p: np.nan_to_num(fits.getdata(p).astype(np.float64)).squeeze()
geo = lambda b: float(np.sqrt(b[0] * b[1]))
beam = lambda p: (lambda h: (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"]))(fits.getheader(p))
omega = lambda b: np.pi * b[0] * b[1] / (4 * np.log(2))                     # beam solid angle, arcsec^2

BC = beam(os.path.join(R, "compact/msclean/image.fits"))
BE = beam(os.path.join(R, "extended", CLEAN, "image.fits"))
LB = float(GC["lbmax_compact_arcsec"])
circ = lambda a, b: [(f, f, 0.0) for f in np.geomspace(a, b, NSIDE + 2)[1:-1]]
# lambda/B_max can reach the compact beam's minor axis (NGC 1614: 1.145" vs 1.14"): then nothing fits between them;
# the joint ladder (ladder_joint.py) images the four (nearly equal) circular beams anyway, so that every target has 11
left = circ(BC[1], LB) if (LB < BC[1] or os.environ.get("REF") == "joint") else []
if not left:
    print(f"lambda/B_max {LB:.3f}\" >= the compact beam's minor axis {BC[1]:.3f}\": no circular beams between them")
BEAMS = [BC] + left + [(LB, LB, 0.0)] + circ(LB, BE[0]) + [BE]
ILB = len(left) + 1                                                      # lambda/B_max's position
NAMES = {0: "compact\nclean beam", ILB: "λ/B$_{\\rm max}$\n(compact)", len(BEAMS) - 1: "extended\nclean beam"}

REF = os.environ.get("REF", "extended")                # "joint": each beam's own joint (compact + extended) image
xc = (np.load(os.path.join(R, "pixel_scale/model_0.08_support.npy" if T3110 else "g20/model_0.08_support.npy")) if G20 else
      np.load(os.path.join(R, "compact/deconvolved/model.npy"))).astype(np.float64)[L]
if REF == "joint":
    # scripts/experiments/ladder_joint.py: one MS-CLEAN of the joint visibilities per ladder beam (its own robust and
    # uv taper, restored with that beam), 0.04" pixels, 51" field. The compact model (0.02", 16") is summed onto 0.04"
    # pixels (exactly the grid it was solved on), the joint images cropped to its 16"
    from scipy.ndimage import label as _label
    JD = os.path.join(R, "ladder_joint_20kms" if G20 else "ladder_joint")
    if G20:
        # the 2D-1D model of the pixel-scale run: 0.08" pixels, 640 px (51.2"), the same 64 channels as the joint cubes;
        # the joint cubes (0.04", 1280 px) brought onto its grid: 0.08" pixel j covers 0.04" pixels 2j - 1 (half), 2j,
        # 2j + 1 (half) on each axis (pixel 640 at the phase centre maps to 320), averaged for Jy/beam, summed for Jy/pixel
        from scipy.ndimage import convolve1d as _c1
        CELL = 0.08
        def crop(p):
            x = rd(p)[L]; k = np.array([0.5, 1.0, 0.5]); y = _c1(_c1(x, k, axis=-1, mode="constant"), k, axis=-2, mode="constant")
            return y[..., ::2, ::2] / (1.0 if p.endswith("model.fits") else 4.0)
    else:
        xc = xc.reshape(len(L), xc.shape[1] // 2, 2, xc.shape[2] // 2, 2).sum((2, 4)); CELL = 2 * CELL
        _n = xc.shape[-1]; _o = (fits.getheader(os.path.join(JD, "b00/clean_finished/image.fits"))["NAXIS1"] - _n) // 2
        crop = lambda p: rd(p)[L][:, _o:_o + _n, _o:_o + _n]
    jdir = lambda i: os.path.join(JD, f"b{i:02d}", "clean_finished")
    for i, b in enumerate(BEAMS):
        assert np.allclose(json.load(open(os.path.join(JD, f"b{i:02d}", "info.json")))["beam"], b, rtol=1e-3), i
    SB_NSIG = float(os.environ.get("SB_NSIG", 5.0))

    def sb_mask(I, res, b, S0):
        """the tight mask at beam b: surface brightness I / omega_b >= S0 (Jy arcsec^-2, one level for every beam)
        and I >= SB_NSIG x the channel's residual sigma; connected regions (per channel) of >= 2 beams"""
        sig = 1.4826 * np.median(np.abs(res - np.median(res, axis=(1, 2), keepdims=True)), axis=(1, 2))
        m = (I / omega(b) >= S0) & (I >= SB_NSIG * sig[:, None, None])
        out = np.zeros_like(m); nmin = 2 * omega(b) / CELL ** 2; nst = 0
        for c in range(len(m)):
            lab, n = _label(m[c])
            if n:
                sz = np.bincount(lab.ravel()); keep = np.nonzero(sz >= nmin)[0]; keep = keep[keep > 0]
                out[c] = np.isin(lab, keep); nst += keep.size
        return out, dict(peak_bar=float(S0), structures=int(nst), sigma_line_mJy=(1e3 * sig).tolist())

    # S0: chosen so that at the compact clean beam the mask covers the footprint of the tight (25 sigma) astrodendro
    # mask of the tapered extended MS-CLEAN (the "galaxy structures only" region)
    I0, r0 = crop(os.path.join(jdir(0), "image.fits")), crop(os.path.join(jdir(0), "residual.fits"))
    if T3110:
        _tm = os.environ.get("TARGET_MASK", os.path.join(R, "consistency_20kms_0.16/clean_extended_taper_cmask/dendro_mask_min25.npy"))
        _tg = json.load(open(os.path.join(os.path.dirname(os.path.dirname(_tm)), "grid.json")))
        TARGET = float(np.load(_tm).astype(bool)[_tg["line"]].any(0).sum() * _tg["cell"] ** 2)
        lo_, hi_ = 1e-4, float((I0 / omega(BEAMS[0])).max())
        for _ in range(40):                                              # bisection in log S0 on the footprint area
            mid = np.sqrt(lo_ * hi_)
            area = sb_mask(I0, r0, BEAMS[0], mid)[0].any(0).sum() * CELL ** 2
            lo_, hi_ = (mid, hi_) if area > TARGET else (lo_, mid)
        S0 = float(np.sqrt(lo_ * hi_))
    else:                                                                # the gallery's rule: SB_FRAC x the peak surface
        _ib = int(np.argmin([abs(b[0] - LB) + abs(b[1] - LB) for b in BEAMS]))   # brightness of the joint image at lambda/B_max
        _Il = crop(os.path.join(jdir(_ib), "image.fits"))
        S0 = float(os.environ.get("SB_FRAC", 0.511)) * float(np.percentile((_Il / omega(BEAMS[_ib])), 99.9)); del _Il
        TARGET = float("nan")
    M0 = sb_mask(I0, r0, BEAMS[0], S0)[0]; del I0, r0                       # every finer mask stays inside this one
    print(f"joint reference; mask level S0 = {S0:.4f} Jy/arcsec^2 (footprint {TARGET:.1f} arcsec^2 at the compact beam), "
          f">= {SB_NSIG:g} sigma", flush=True)
else:
    xk = rd(os.path.join(R, "extended", CLEAN, "model.fits"))[L]
    res = rd(os.path.join(R, "extended", CLEAN, "residual.fits"))[L]     # Jy per (dirty ~ extended clean) beam
    to_beam = lambda cube, b: beams.to_beam(cube, BE, b, CELL)            # extended clean beam -> b
    print(f"{GC['label']}: {len(BEAMS)} beams; models: compact {xc.sum() * DV:.1f}, CLEAN {xk.sum() * DV:.1f} Jy km/s "
          f"(line channels, whole field)", flush=True)
rows, MAPS, MASKS = [], [], {}
for ib, b in enumerate(BEAMS):
    t0 = time.time()
    C = convolve_with_beam(xc, CELL, *b)
    if REF == "joint":
        I = crop(os.path.join(jdir(ib), "image.fits"))                   # the joint restored image at b
        K = convolve_with_beam(crop(os.path.join(jdir(ib), "model.fits")), CELL, *b)
        M, info = sb_mask(I, crop(os.path.join(jdir(ib), "residual.fits")), b, S0)
        M &= M0
    else:
        K = convolve_with_beam(xk, CELL, *b)
        rb = to_beam(res, b)
        M, info = dendro_mask(K + rb, rb, omega(b) / CELL ** 2, list(range(len(L))))
        I = K + rb                                                       # the extended restored image at b
    c, i, k = C[M], I[M], K[M]
    bpx = omega(b) / CELL ** 2
    MASKS[f"b{ib:02d}"] = M

    def stats(ref, tag=""):
        sel = ref > 0.05 * ref.max()
        dev = 100 * (c[sel] - ref[sel]) / ref[sel]
        return {"rmse" + tag: float(100 * np.linalg.norm(c - ref) / np.linalg.norm(ref)),
                "flux" + tag: float(100 * (c.sum() - ref.sum()) / ref.sum()),
                "median_dev" + tag: float(np.median(dev)), "median_abs_dev" + tag: float(np.median(np.abs(dev)))}
    rows.append(dict(bmaj=float(b[0]), bmin=float(b[1]), bpa=float(b[2]), fwhm_geo=geo(b), **stats(i),
                     **stats(k, "_model"), mask_voxels=int(M.sum()), mask_area_arcsec2=float(M.any(0).sum() * CELL ** 2),
                     flux_in_mask_Jy_kms=dict(compact=float(c.sum() / bpx * DV), image=float(i.sum() / bpx * DV),
                                              clean_model=float(k.sum() / bpx * DV)),
                     mask_peak_bar=info["peak_bar"], mask_structures=info["structures"],
                     sigma_median_mJy=float(np.median(info["sigma_line_mJy"])), seconds=round(time.time() - t0)))
    MAPS.append((C.sum(0) * DV / omega(b), I.sum(0) * DV / omega(b),          # moment 0, Jy arcsec^-2 km/s
                 (C * M).sum(0) * DV / omega(b), (I * M).sum(0) * DV / omega(b), M.any(0)))
    print("  " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in rows[-1].items()), flush=True)
REFN = "joint" if REF == "joint" else "extended"
np.savez_compressed(os.path.join(OUT, "masks.npz"), cell=CELL, line=np.array(L), **MASKS)   # the masks (line channels)
json.dump(dict(galaxy=GC["name"], clean=CLEAN, reference_data=REFN,
               mask=(f"surface brightness >= {S0:.4f} Jy/arcsec^2 and >= {SB_NSIG:g} sigma, regions >= 2 beams, at each beam, "
                     "inside the compact-beam mask"
                     if REF == "joint" else "astrodendro of the extended CLEAN image at each beam"),
               reference=f"{REFN} restored image (model + residual) at each beam; *_model: CLEAN model only",
               compact_beam=BC, extended_beam=BE, lbmax_compact=LB, rows=rows),
          open(os.path.join(OUT, "resolution_ladder.json"), "w"), indent=1)

# ======================================================================== figure (inches, bottom up: label, beam strip, main panel, maps)
n = len(BEAMS); x = np.arange(n); XR = n + 0.9                        # x range in columns (padded ends)
W, H, HS, ML, MR, MB, MT, GAP = 12.6, 5.2, 1.25, 1.0, 0.25, 0.55, 0.15, 0.42
PG, P_TT, P_GT = 0.05, 0.62, 0.3                                       # map gap, column titles, gap to the main panel
P = (W - 4 * PG) / 5                                                    # map panel size
Y_MAPS = MB + HS + GAP + H + P_GT                                       # bottom of the maps' block
FW, FH = ML + W + MR, Y_MAPS + 3 * P + 2 * PG + P_TT + MT
fig = plt.figure(figsize=(FW, FH))
a = fig.add_axes([ML / FW, (MB + HS + GAP) / FH, W / FW, H / FH])
for key, mk, col, lab in (("rmse", "o", BLUE, "RMSE"), ("flux", "s", ORANGE, "total flux"),
                          ("median_dev", "^", GREEN, "median deviation")):
    a.plot(x, [r[key + "_model"] for r in rows], "--" + mk, color=col, lw=1.3, ms=5, alpha=0.55, mfc="none")
    a.plot(x, [r[key] for r in rows], "-" + mk, color=col, lw=2.2, ms=8, label=lab)
a.plot([], [], "-", color="0.3", lw=2.2, label=f"vs {REFN} restored image")
a.plot([], [], "--", color="0.3", lw=1.3, alpha=0.7, label=f"vs {REFN} CLEAN model only")
a.axhline(0, color=MUTED, lw=1)
a.axvline(ILB, color=MUTED, lw=1, ls=":")
a.set_xlim(-0.95, n - 0.05)
a.set_xticks(x)
a.set_xticklabels([f"[{b[0]:.2f}″ × {b[1]:.2f}″]" if b[0] != b[1] else f"{b[0]:.2f}″" for b in BEAMS],
                  fontsize=TICK)
for i in NAMES:                                                          # the two clean beams and lambda/B_max in bold
    a.get_xticklabels()[i].set_fontweight("bold")
a.tick_params(axis="x", top=True, bottom=True, length=6, pad=8)
a.tick_params(axis="y", labelsize=TICK, right=True, length=6)
a.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
a.set_ylabel(f"compact 2D-1D − {REFN} MS-CLEAN [%]", fontsize=LAB, labelpad=8)
a.legend(frameon=False, fontsize=LAB - 1, loc="best")

# the beams to scale below their tick labels, on one arcsec scale: the widest at most 0.8 of a column and the tallest
# (as rotated) at most BH inches, above the names (NB inches) at the bottom of the strip
s = a.inset_axes([0, -(GAP + HS) / H, 1, HS / H]); s.set_xlim(-0.95, n - 0.05)
BH, NB = 0.72, 0.42
ang = lambda b: np.radians(90 + b[2])                                    # PA east of north, RA increasing left
ext = lambda b, c, sn: np.hypot(b[0] / 2 * c, b[1] / 2 * sn)             # half-extent along x (c, sn = cos, sin) or y
wide = max(2 * ext(b, np.cos(ang(b)), np.sin(ang(b))) for b in BEAMS)
tall = max(2 * ext(b, np.sin(ang(b)), np.cos(ang(b))) for b in BEAMS)
ipc = W / XR                                                             # inches per column
k = min(0.8 * ipc / wide, BH / tall) / ipc                               # columns per arcsec
du = XR / W                                                              # data units per inch, both axes: true shapes
s.set_ylim(0, HS * du)
for i, b in enumerate(BEAMS):
    s.add_patch(Ellipse((i, (HS - 0.04 - BH / 2) * du), width=b[0] * k, height=b[1] * k, angle=90 + b[2],
                        facecolor="none", edgecolor="k" if i in NAMES else MUTED, lw=1.6))
    if i in NAMES:
        s.text(i, (HS - 0.04 - BH - 0.06 - NB) * du, NAMES[i], ha="center", va="bottom", fontsize=TICK - 1, color="0.3")
s.axis("off")
fig.text((ML + W / 2) / FW, 0.15 / FH, "comparison beam (FWHM)", ha="center", va="bottom", fontsize=LAB)

# ======================================================================== maps: 3 rows x 5 beams
lo, hi = list(range(1, ILB)), list(range(ILB + 1, n - 1))              # the circular beams on each side of lambda/B_max
d = min(len(lo), len(hi)) // 2 + 1 if lo and hi else 0
mid = [ILB - d, ILB + d] if d else [hi[len(hi) // 3], hi[2 * len(hi) // 3]]   # symmetric about it; else both finer
COLS = sorted({0, ILB, n - 1, *mid})
N = MAPS[0][0].shape[0]; C0 = N // 2
if "half_arcsec" in GC:
    HALF = float(GC["half_arcsec"])
else:                                                                   # the largest mask's footprint plus 0.5"
    fy, fx = np.nonzero(np.any([m[4] for m in MAPS], axis=0))
    HALF = float(min(N * CELL / 2 - 0.1, max(np.abs(fx - C0).max(), np.abs(fy - C0).max()) * CELL + 0.5))
hp = int(round(HALF / CELL)); sl = np.s_[C0 - hp:C0 + hp, C0 - hp:C0 + hp]; EXT = [HALF, -HALF, -HALF, HALF]
BAR = 2 if HALF > 3 else 1                                              # scale bar, arcsec
PCT = 50
cm_pct = plt.get_cmap("RdBu_r").copy(); cm_pct.set_bad("0.9")
ROWLAB = ["Compact 2D-1D", f"{REFN.capitalize()} MS-CLEAN\n(restored)", f"(C − I) / I\n[±{PCT}%]"]
for j, i in enumerate(COLS):
    b = BEAMS[i]; c0, k0, c0m, k0m, foot = MAPS[i]
    norm = Normalize(vmin=0, vmax=np.percentile(np.maximum(c0, k0)[foot], 99.9))          # linear
    ok = foot & (k0m > 0.05 * k0m[foot].max())
    pct = np.where(ok, 100 * (c0m - k0m) / np.where(k0m > 0, k0m, 1), np.nan)
    for r, (im2, kw) in enumerate(((c0, dict(cmap="inferno", norm=norm)), (k0, dict(cmap="inferno", norm=norm)),
                                   (pct, dict(cmap=cm_pct, vmin=-PCT, vmax=PCT)))):
        ax = fig.add_axes([(ML + j * (P + PG)) / FW, (Y_MAPS + (2 - r) * (P + PG)) / FH, P / FW, P / FH])
        ax.imshow(im2[sl], origin="lower", extent=EXT, interpolation="nearest", **kw)
        ax.set_xticks([]); ax.set_yticks([])
        if r < 2:
            ax.contour(foot[sl].astype(float), levels=[0.5], colors="w", linewidths=0.5, extent=EXT, origin="lower")
        pad = 0.06 * HALF                                               # the beam, bottom left (white on the maps)
        ax.add_patch(Ellipse((HALF - pad - b[0] / 2, -HALF + pad + b[0] / 2), width=b[0], height=b[1], angle=90 - b[2],
                             facecolor="w" if r < 2 else "none", edgecolor="w" if r < 2 else "k", lw=1.0))
        if j == 0:
            ax.set_ylabel(ROWLAB[r], fontsize=TICK, labelpad=6)
        if r == 0 and j == 0:                                           # scale bar, top right
            xs, ys, cap = -HALF + 0.08 * HALF, HALF - 0.1 * HALF, 0.03 * HALF
            ax.plot([xs + BAR, xs], [ys, ys], color="w", lw=2.0, solid_capstyle="butt")
            for xe in (xs, xs + BAR):
                ax.plot([xe, xe], [ys - cap, ys + cap], color="w", lw=1.4, solid_capstyle="butt")
            ax.annotate(f"{BAR}″", (xs + BAR / 2, ys - cap), xytext=(0, -2.5), textcoords="offset points", color="w",
                        ha="center", va="top", fontsize=TICK - 1, fontweight="bold")
        if r == 0:
            t = a.get_xticklabels()[i].get_text() + ("\n" + NAMES[i].replace("\n", " ") if i in NAMES else "")
            ax.set_title(t, fontsize=TICK, fontweight="bold" if i in NAMES else "normal", pad=5)

fig.savefig(os.path.join(OUT, "resolution_ladder.png"), dpi=150)
fig.savefig(os.path.join(OUT, "resolution_ladder.pdf")); plt.close(fig)
print("saved", os.path.join(OUT, "resolution_ladder.png"))
