#!/usr/bin/env python
"""
Gallery of the pipeline's targets (results/<target>/, scripts/pipeline/run_target.py), in the style of figure 40:
one row per galaxy, columns compact MS-CLEAN (its components x mask * compact clean beam), compact deconvolved *
extended clean beam, extended MS-CLEAN (finished; components x mask * extended clean beam), compact deconvolved -
extended MS-CLEAN, and the integrated spectra inside the mask (velocity relative to the cube centre). Maps in surface
brightness (Jy arcsec^-2 km/s), everything x the target's astrodendro mask (mask_final.npy); outside it dark grey
(light grey on the difference). Rows: the N targets whose moment-0 maps agree best (lowest relative rms difference,
figures/stats.json -> mom0.rel_rms_diff), or TARGETS=a,b,c.

COMPARE=lbmax: compact deconvolved and extended MS-CLEAN both at the compact array's lambda/B_max (circular) instead of
the extended clean beam (same rows). The difference panels give the total error (flux in the 3D mask) and the RMSE of
the whole cube (every masked line voxel, both cubes * the comparison beam), relative to the extended MS-CLEAN.

    N=7 [TARGETS=...] [COMPARE=extended|lbmax] [PAPER=1] python scripts/compact_vs_msclean_dendro/fig_gallery.py

PAPER=1: the paper's version, drawn at its final size for one A&A page (panels 0.9", fonts 5.5-7 pt), no title, no
beam-size text (the beams are in the paper's Table 1), capped scale bars; -> gallery_top<N>_paper.{png,pdf}.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.patches import Ellipse
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
from restore import convolve_with_beam  # noqa: E402

RES = os.path.join(REPO, "results")
ALL = ["NGC3110", "NGC2369", "ESO297-G011", "NGC1614", "IC4518E", "IC4518W", "IRASF17138-1017", "NGC7130", "IC5179",
       "NGC5135", "IC4687", "MCG-02-33-098", "NGC3256"]
N = int(os.environ.get("N", 7))
rel = {t: json.load(open(os.path.join(RES, t, "figures/extended_clean_beam_comparison/stats.json")))["mom0"]["rel_rms_diff"] for t in ALL}
TARGETS = os.environ["TARGETS"].split(",") if os.environ.get("TARGETS") else sorted(ALL, key=rel.get)[:N]
COMPARE = os.environ.get("COMPARE", "extended")
PAPER = os.environ.get("PAPER", "0") == "1"
OUT = os.environ.get("OUT", os.path.join(RES, "gallery", f"gallery_top{len(TARGETS)}" + ("_20kms_joint" if os.environ.get("GRID") == "20kms" else "")
                                         + ("_lbmax_compact" if COMPARE == "lbmax" else "")
                                         + ("_paper" if PAPER else "") + ".png"))
BN = "extended clean beam" if COMPARE == "extended" else "λ/B$_{\\rm max,\\,compact}$ beam"
BLUE, GREEN, MUTED = "#2a6fdb", "#1a9e77", "#8a8a8a"
plt.rcParams.update({"font.family": "serif", "xtick.direction": "in", "ytick.direction": "in"})
rd = lambda p: np.nan_to_num(fits.getdata(p).astype(np.float64)).squeeze()
beam_of = lambda h: (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"])
omega = lambda b: np.pi * b[0] * b[1] / (4 * np.log(2))


G20 = os.environ.get("GRID") == "20kms"


def _to008(x, model):
    """a 0.04" cube (1280 px) onto the 0.08" grid (640 px), pixel centres aligned (0.08" pixel j = 0.04" pixels 2j - 1
    (half), 2j, 2j + 1 (half)): summed for Jy/pixel models, OR-ed for masks"""
    from scipy.ndimage import convolve1d, maximum_filter
    if x.dtype == bool:
        return maximum_filter(x, size=(1, 3, 3))[..., ::2, ::2]
    k = np.array([0.5, 1.0, 0.5])
    return convolve1d(convolve1d(x, k, axis=-1, mode="constant"), k, axis=-2, mode="constant")[..., ::2, ::2]


def _beam_any(p):
    """the clean beam of an image: its single beam, or the median of its per-channel beams (BEAMS table, arcsec)"""
    with fits.open(p) as h:
        if "BMAJ" in h[0].header:
            return beam_of(h[0].header)
        b = h["BEAMS"].data
        return (float(np.median(b["BMAJ"])), float(np.median(b["BMIN"])), float(np.median(b["BPA"])))


def load20(T):
    """the 20 km/s gallery: compact MS-CLEAN, compact 2D-1D and joint (compact + extended) MS-CLEAN at the extended
    clean beam, all models (components) on the 0.08", 64 x 20 km/s grid, inside the joint image's astrodendro mask"""
    R = os.path.join(RES, T); c = json.load(open(os.path.join(R, "config.json")))
    if T == "NGC3110":                                                  # the products of the accuracy tests (Sect. 5.1)
        gd = os.path.join(R, "consistency_20kms_0.08"); cc = os.path.join(gd, "clean_compact_finished")
        cmp = os.path.join(R, "pixel_scale/model_0.08_support.npy"); jd = os.path.join(R, "ladder_joint_20kms/b10/clean_finished")
    else:
        gd = os.path.join(R, "g20"); cc = os.path.join(gd, "clean_compact_finished")
        cmp = os.path.join(gd, "model_0.08_support.npy"); jd = os.path.join(gd, "joint_ext")
    g = json.load(open(os.path.join(gd, "grid.json"))); L, dv, nch, cell = g["line"], g["width_kms"], g["nchan"], 0.08
    n = 640; c0 = n // 2
    BE = beam_of(fits.getheader(os.path.join(jd, "image.fits"))); BC = _beam_any(os.path.join(cc, "image.fits"))
    if COMPARE == "lbmax":                       # the compact array's lambda/B_max (circular): models convolved with it
        BE = (c["lbmax_compact_arcsec"], c["lbmax_compact_arcsec"], 0.0)
    # image-space comparison, as in the resolution ladder: the 2D-1D model * B against the joint RESTORED image at B
    # (components * B + residual; at a beam coarser than the extended one the residual is brought to B with
    # beams.to_beam), both Jy/beam, both x the joint image's 3D mask; fluxes and spectra from the images (/ beam area).
    # Comparing components alone would be biased: CLEAN puts every component inside its own mask by construction,
    # while a smooth model spreads beyond its voxels, and the components leave out the flux left in the residual.
    sys.path.insert(0, os.path.join(REPO, "scripts", "compact_vs_msclean_dendro")); import beams as _bm
    BJ = beam_of(fits.getheader(os.path.join(jd, "image.fits")))
    jimg = _to008(rd(os.path.join(jd, "image.fits")), True) / 4.0          # averaged onto 0.08" (Jy/beam)
    jres = _to008(rd(os.path.join(jd, "residual.fits")), True) / 4.0
    if np.allclose(BE, BJ, rtol=1e-3):
        I, Ir = jimg, jres
    else:
        Ir = _bm.to_beam(jres, BJ, BE, cell)
        I = convolve_with_beam(_to008(rd(os.path.join(jd, "model.fits")), True), cell, *BE) + Ir
    del jimg, jres
    # the comparison mask, as in the consistency overview: astrodendro (msclean_complete.dendro_mask) of the joint
    # restored image at the comparison beam, trunks >= MASKSIG sigma (per-channel sigma from its residual), >= 2 beams
    if os.environ.get("MASK20", "sb") == "dendro":
        sys.path.insert(0, os.path.join(REPO, "scripts")); from msclean_complete import dendro_mask as _dm
        M, _mi = _dm(I, Ir, omega(BE) / cell ** 2, L, min_value=float(os.environ.get("MASKSIG", 16)))
    else:
        # the resolution ladder's rule (fig_resolution_ladder.py): surface brightness I / omega >= S0 per channel and
        # >= 5 sigma (per-channel sigma from the residual), connected regions of >= 2 beams; S0 = SB_FRAC x the 99.9th
        # percentile of the target's line surface brightness, SB_FRAC fixed so that NGC 3110 gets the ladder's level
        # ... and, as in the ladder, inside the mask made by the same rule at the compact clean beam
        from scipy.ndimage import label as _label

        def sbmask(img, res, b, S0):
            sig = 1.4826 * np.median(np.abs(res - np.median(res, axis=(1, 2), keepdims=True)), axis=(1, 2))
            m = (img / omega(b) >= S0) & (img >= 5 * sig[:, None, None]); out = np.zeros_like(m); nmin = 2 * omega(b) / cell ** 2
            for ch in L:
                lab, nl = _label(m[ch])
                if nl:
                    sz = np.bincount(lab.ravel()); keep = np.nonzero(sz >= nmin)[0]; keep = keep[keep > 0]
                    out[ch] = np.isin(lab, keep)
            return out
        peak = float(np.percentile((I / omega(BE))[L], 99.9))
        S0 = float(os.environ.get("SB_FRAC", 0.511)) * peak              # 0.511: NGC 3110 -> 0.0457, the ladder level
        jres_ = _to008(rd(os.path.join(jd, "residual.fits")), True) / 4.0
        rC = _bm.to_beam(jres_, BJ, BC, cell); del jres_
        IC_ = convolve_with_beam(_to008(rd(os.path.join(jd, "model.fits")), True), cell, *BC) + rC
        M0 = sbmask(IC_, rC, BC, S0); del IC_, rC
        M = sbmask(I, Ir, BE, S0) & M0
        print(f"{T}: peak SB {peak:.4f}, S0 {S0:.4f} Jy/arcsec^2, footprint {M[L].any(0).sum() * cell ** 2:.1f} arcsec^2 "
              f"(compact-beam mask {M0[L].any(0).sum() * cell ** 2:.1f})", flush=True)
    del Ir
    foot = M[L].any(0)
    C = convolve_with_beam(np.load(cmp).astype(np.float64), cell, *BE)
    bpx = omega(BE) / cell ** 2
    xcc = rd(os.path.join(cc, "model.fits")) * M                        # display only (its own beam)
    sb = dict(cc=convolve_with_beam((xcc[L].sum(0) * dv)[None], cell, *BC)[0] / omega(BC),
              cmp=(C * M)[L].sum(0) * dv / omega(BE), cln=(I * M)[L].sum(0) * dv / omega(BE))
    sb["d"] = sb["cmp"] - sb["cln"]
    mv = M[L]
    err = dict(total=float(100 * ((C * M)[L].sum() - (I * M)[L].sum()) / (I * M)[L].sum()),
               rmse=float(100 * np.linalg.norm((C[L] - I[L])[mv]) / np.linalg.norm(I[L][mv])))
    spec = {"cmp": (C * M).sum((1, 2)) / bpx, "cln": (I * M).sum((1, 2)) / bpx}
    flux = {k: float(spec[k][L].sum() * dv) for k in spec}
    del C, I, xcc
    fy, fx = np.nonzero(foot)
    half = float(c["half_arcsec"]) if "half_arcsec" in c else \
        float(min(n * cell / 2 - 0.1, max(np.abs(fx - c0).max(), np.abs(fy - c0).max()) * cell + 0.5))
    vrel = dv * (np.arange(nch) - (nch - 1) / 2)
    return dict(err=err, T=T, label=c["label"], cell=cell, n=n, c0=c0, foot=foot, BE=BE, BC=BC, sb=sb, half=half, vrel=vrel,
                L=L, dv=dv, spec=spec, flux=flux, mosaic=c.get("mosaic", False))


def load(T):
    if G20:
        return load20(T)
    R = os.path.join(RES, T); c = json.load(open(os.path.join(R, "config.json")))
    L = json.load(open(os.path.join(R, "line_channels.json")))["line"]
    cell, dv, nch = c["cell_arcsec"], c["width_kms"], c["nchan"]
    M = np.load(os.path.join(R, "extended/msclean_finished/mask_final.npy")).astype(bool)
    foot = M[L].any(0); n = foot.shape[0]; c0 = n // 2
    BE = beam_of(fits.getheader(os.path.join(R, "extended/msclean_finished/image.fits")))
    if COMPARE == "lbmax":
        BE = (c["lbmax_compact_arcsec"], c["lbmax_compact_arcsec"], 0.0)
    BC = beam_of(fits.getheader(os.path.join(R, "compact/msclean/image.fits")))
    x = dict(cc=rd(os.path.join(R, "compact/msclean/model.fits")) * M,
             cmp=np.load(os.path.join(R, "compact/deconvolved/model.npy")).astype(np.float64) * M,
             cln=rd(os.path.join(R, "extended/msclean_finished/model.fits")) * M)
    sb = {}                                           # moment 0 of the masked model, then the beam (linear: same thing)
    for k, b in (("cc", BC), ("cmp", BE), ("cln", BE)):
        sb[k] = convolve_with_beam((x[k][L].sum(0) * dv)[None], cell, *b)[0] / omega(b)
    sb["d"] = sb["cmp"] - sb["cln"]
    # whole-cube errors: both cubes * the comparison beam, over every masked line voxel; total = flux in the 3D mask
    cc_, cl_ = convolve_with_beam(x["cmp"][L], cell, *BE), convolve_with_beam(x["cln"][L], cell, *BE)
    mv = M[L]
    err = dict(total=float(100 * (x["cmp"][L].sum() - x["cln"][L].sum()) / x["cln"][L].sum()),
               rmse=float(100 * np.linalg.norm((cc_ - cl_)[mv]) / np.linalg.norm(cl_[mv])))
    del cc_, cl_
    if "half_arcsec" in c:
        half = float(c["half_arcsec"])
    else:
        fy, fx = np.nonzero(foot)
        half = float(min(n * cell / 2 - 0.1, max(np.abs(fx - c0).max(), np.abs(fy - c0).max()) * cell + 0.5))
    vrel = dv * (np.arange(nch) - (nch - 1) / 2)
    spec = {k: x[k].sum((1, 2)) for k in ("cmp", "cln")}
    flux = {k: float(x[k][L].sum() * dv) for k in ("cmp", "cln")}
    return dict(err=err, T=T, label=c["label"], cell=cell, n=n, c0=c0, foot=foot, BE=BE, BC=BC, sb=sb, half=half, vrel=vrel,
                L=L, dv=dv, spec=spec, flux=flux, mosaic=c.get("mosaic", False))


gals = [load(t) for t in TARGETS]
print("rows:", [(g["T"], round(rel[g["T"]], 3), {k: round(v, 1) for k, v in g["err"].items()}) for g in gals])

# ======================================================================== layout (inches)
if PAPER:                                                        # final size, one A&A page (text 523.5 x 705 pt)
    TW = 523.5307 / 72.27                                          # A&A \textwidth: the figure is full width at 1:1
    S, CBH, CBP, ROWGAP = 1.15, 0.05, 0.025, 0.13
    TICK, LAB = 5.5, 6.5
    ML, MR, MB, MT = 0.20, 0.42, 0.60, 0.34
    GAP = (TW - ML - MR - 5 * S) / 4                               # columns spread to fill the width
    # name box, its second line, scale-bar label, heading line 1 / 2, error box, spectrum text, legend
    F = dict(box=6, box2=5, sbar=6, head=6.5, head2=6.5, err=5.5, err2=5, spec=5.5, leg=5.5)
    LW = 0.45
    PCT_FLOOR = 0.05                                    # difference shown (in %) where the extended map >= 5% of its peak
else:
    S, GAP, CBH, CBP, ROWGAP = 3.4, 0.22, 0.13, 0.08, 0.62           # panel, column gap, colorbar height/pad, gap below a row's colorbars
    TICK, LAB = 12, 13
    ML, MR, MB, MT = 0.2, 1.05, 0.75, 1.45
    F = dict(box=15, box2=12, sbar=12, head=15, head2=14, err=15, err2=12, spec=11.5, leg=10.5)
    LW = 1.0
nr = len(gals)
FW, FH = ML + 5 * S + 4 * GAP + MR, MB + nr * S + nr * (CBP + CBH) + (nr - 1) * ROWGAP + MT
fig = plt.figure(figsize=(FW, FH))
inch = lambda x, y, w, h: fig.add_axes([x / FW, y / FH, w / FW, h / FH])
PX = lambda c: ML + c * (S + GAP)
PY = lambda r: FH - MT - S - r * (S + CBP + CBH + ROWGAP)        # bottom edge of row r's panels


def textbox(a, text, light, size=15, size2=12):
    """first line bold, the rest smaller"""
    col = "k" if light else "w"
    head, *rest = text.split("\n")
    lines = [TextArea(head, textprops=dict(color=col, fontsize=size, fontweight="bold"))]
    lines += [TextArea(t, textprops=dict(color=col, fontsize=size2)) for t in rest]
    ab = AnchoredOffsetbox(loc="upper left", child=VPacker(children=lines, align="left", pad=0, sep=2 * LW),
                           pad=0.35 * LW, borderpad=0.5 * LW if not PAPER else 0.3, frameon=True)
    ab.patch.set_boxstyle("round,pad=0.25")
    ab.patch.set(facecolor="white" if light else "black", edgecolor="0.6" if light else "0.35",
                 alpha=0.9 if light else 0.92, linewidth=1.0 * LW)
    a.add_artist(ab)


def panel(g, r, c, img2, beam, cmap="inferno", lo=0.0, hi=None, light=False, outline=True, label=None, sizes=None):
    a = inch(PX(c), PY(r), S, S)
    half, cell, c0 = g["half"], g["cell"], g["c0"]
    hp = int(round(half / cell)); sl = np.s_[c0 - hp:c0 + hp, c0 - hp:c0 + hp]; ext = [half, -half, -half, half]
    cm = plt.get_cmap(cmap).copy()
    if outline:
        img2 = np.where(g["foot"], img2, np.nan)
        cm.set_bad("0.93" if light else "0.12")
    im = a.imshow(img2[sl], origin="lower", extent=ext, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
    if outline:
        a.contour(g["foot"][sl].astype(float), levels=[0.5], colors="0.55", linewidths=0.5 * max(LW, 0.6), extent=ext, origin="lower")
    a.set_xticks([]); a.set_yticks([])
    ink = "k" if light else "w"
    sb_len = 1.0 if half < 7 else 2.0                                  # scale bar
    x0, y0 = -half + 0.08 * half, half - 0.1 * half
    if PAPER:                                                          # first column only; capped bar, bold label below it
        if c == 0:
            a.plot([x0 + sb_len, x0], [y0, y0], color=ink, lw=2.2 * LW, solid_capstyle="butt")
            cap = 0.035 * half
            for xe in (x0, x0 + sb_len):
                a.plot([xe, xe], [y0 - cap, y0 + cap], color=ink, lw=1.5 * LW, solid_capstyle="butt")
            a.annotate(f"{sb_len:g}″", (x0 + sb_len / 2, y0 - cap), xytext=(0, -1.5), textcoords="offset points",
                       color=ink, ha="center", va="top", fontsize=F["sbar"], fontweight="bold")
    else:
        a.plot([x0 + sb_len, x0], [y0, y0], color=ink, lw=2.2 * LW, solid_capstyle="butt")
        a.text(x0 + sb_len / 2, y0 - 0.025 * half, f"{sb_len:g}″", color=ink, ha="center", va="top", fontsize=12)
    pad = 0.06 * half
    bx, by = half - pad - beam[0] / 2, -half + pad + beam[0] / 2
    a.add_patch(Ellipse((bx, by), width=beam[0], height=beam[1], angle=90 - beam[2], facecolor="0.15" if light else "w",
                        edgecolor=ink, lw=1.0 * LW))
    f = ".3f" if beam[0] < 1 else ".2f"
    btxt = f"{beam[0]:{f}}″ (circular)" if beam[0] == beam[1] else f"{beam[0]:{f}}″ × {beam[1]:{f}}″ @ {beam[2]:.0f}°"
    if not PAPER:                                                      # PAPER: beams in the paper's Table 1
        a.text(bx - beam[0] / 2 - 0.04 * half, by, btxt, color=ink, ha="left", va="center", fontsize=10.5)
    if label:
        textbox(a, label, light, *(sizes or (F["box"], F["box2"])))
    cax = inch(PX(c), PY(r) - CBP - CBH, S, CBH)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cax.tick_params(labelsize=TICK, pad=3 * LW if not PAPER else 1.2, direction="out", length=4 * LW if not PAPER else 1.8,
                    width=0.5 if PAPER else 0.8)
    cb.outline.set_linewidth(0.4 if PAPER else 0.8)
    cb.locator = matplotlib.ticker.MaxNLocator(3 if PAPER else 5); cb.update_ticks()
    return a


REFN, REFS = ("Joint", "joint") if G20 else ("Extended", "ext.")      # the reference: joint (compact + extended) or extended
HEAD = (["Compact\nMS-CLEAN", "Compact\n2D–1D", f"{REFN}\nMS-CLEAN", f"Difference (%)\n(2D–1D − {REFS}) / {REFS}", "Integrated\nspectra"] if PAPER else
        ["Compact MS-CLEAN", "Compact deconvolved\n∗ " + BN, "Extended MS-CLEAN\n∗ " + BN,
         "Difference\ncompact deconv. − ext. MS-CLEAN", "Integrated spectra\ninside the mask"])
HY = FH - MT + (0.20 if PAPER else 0.62)               # top of every heading's first line, inches
for c, t in enumerate(HEAD):                          # first line bold; second line smaller, regular
    if PAPER:                                         # both lines bold, just above the panels
        fig.text((PX(c) + S / 2) / FW, (FH - MT + 0.03) / FH, t, ha="center", va="bottom", fontsize=F["head"],
                 linespacing=1.1)
        continue
    head, *rest = t.split("\n")
    fig.text((PX(c) + S / 2) / FW, HY / FH, head, ha="center", va="top", fontsize=F["head"], fontweight="bold")
    for k, line in enumerate(rest):
        fig.text((PX(c) + S / 2) / FW, (HY - 0.27 * (k + 1)) / FH, line, ha="center", va="top", fontsize=F["head2"])
if not PAPER:
    fig.text(0.5, 1 - 0.15 / FH, "Compact deconvolution vs extended MS-CLEAN"
             + (" at λ/B$_{\\rm max,\\,compact}$" if COMPARE == "lbmax" else ""), ha="center", va="top", fontsize=22,
             fontweight="bold")

for r, g in enumerate(gals):
    sb = g["sb"]; ve = np.percentile(sb["cln"][g["foot"]], 99.9)
    name = g["label"] + ("\ncentral pointing" if g["mosaic"] and not PAPER else "")   # PAPER: in the caption
    panel(g, r, 0, sb["cc"], g["BC"], hi=np.percentile(sb["cc"][g["foot"]], 99.9), outline=False,
          label=None if PAPER else name)
    if PAPER:                                                     # row label, left of the row
        fig.text((ML - 0.06) / FW, (PY(r) + S / 2) / FH, g["label"], rotation=90, ha="right", va="center",
                 fontsize=F["head"])
    panel(g, r, 1, sb["cmp"], g["BE"], hi=ve)
    panel(g, r, 2, sb["cln"], g["BE"], hi=ve)
    if PAPER:                                         # percentage of the extended map, where it is >= 5% of its peak
        dpct = np.where(sb["cln"] >= PCT_FLOOR * ve, 100 * sb["d"] / np.maximum(sb["cln"], 1e-30), np.nan)
    DLIM = (25 if COMPARE == "lbmax" else 60) if G20 else 100
    panel(g, r, 3, dpct if PAPER else sb["d"], g["BE"], cmap="RdBu_r", lo=-DLIM if PAPER else -ve,
          hi=DLIM if PAPER else ve, light=True, sizes=(F["err"], F["err2"]),
          label=(f"ΔF {g['err']['total']:+.1f}%   |Δ| {np.nanmedian(np.abs(dpct)):.0f}%" if PAPER else
                 f"total error {g['err']['total']:+.1f}%\nRMSE (cube) {g['err']['rmse']:.1f}%"))
    a = inch(PX(4), PY(r), S, S)
    v = g["vrel"]; L = g["L"]
    a.axvspan(v[L[0]] - g["dv"] / 2, v[L[-1]] + g["dv"] / 2, color=MUTED, alpha=0.1, lw=0)
    a.step(v, g["spec"]["cmp"], where="mid", color=BLUE, lw=1.8 * LW if not PAPER else 0.9,
           label="compact 2D–1D" if PAPER else "compact deconvolved")
    a.step(v, g["spec"]["cln"], where="mid", color=GREEN, lw=1.8 * LW if not PAPER else 0.9, label=f"{REFN.lower()} MS-CLEAN")
    a.step(v, g["spec"]["cmp"] - g["spec"]["cln"], where="mid", color=BLUE, lw=1.2 * LW if not PAPER else 0.6, ls="--",
           label="difference")
    a.axhline(0, color=MUTED, lw=0.7 * LW if not PAPER else 0.4)
    for sp in a.spines.values():
        sp.set_linewidth(0.5 if PAPER else 0.8)
    a.set_xlim(v[0] - g["dv"] / 2, v[-1] + g["dv"] / 2)
    a.set_ylim(top=1.75 * max(g["spec"]["cmp"].max(), g["spec"]["cln"].max()))
    a.set_xticks([-200, 0, 200])
    a.yaxis.tick_right(); a.yaxis.set_label_position("right")
    a.tick_params(which="both", direction="in", top=True, bottom=True, left=True, right=True, labelleft=False, labelright=True)
    a.tick_params(which="major", labelsize=TICK, pad=5 if not PAPER else 1.5, length=6 if not PAPER else 2, width=0.8 if not PAPER else 0.5)
    a.tick_params(which="minor", length=3 if not PAPER else 1, width=0.6 if not PAPER else 0.4)
    a.minorticks_on()
    a.set_ylabel(r"$S_\nu$ (Jy)" if PAPER else "flux density [Jy]", fontsize=LAB, labelpad=8 if not PAPER else 2)
    a.text(0.04, 0.95, f"{g['flux']['cmp']:.0f} / {g['flux']['cln']:.0f}" if PAPER else
           f"{g['flux']['cmp']:.0f} vs {g['flux']['cln']:.0f} Jy km/s", transform=a.transAxes, ha="left", va="top",
           fontsize=F["spec"])
    if r == 0 and not PAPER:
        a.legend(frameon=False, fontsize=F["leg"], loc="upper right", bbox_to_anchor=(1.0, 0.86 if not PAPER else 0.84),
                 handlelength=1.6 if PAPER else 2.0, borderaxespad=0.2 if PAPER else 0.5)
    if r == nr - 1:
        a.set_xlabel(r"$v - v_\mathrm{centre}$ (km s$^{-1}$)" if PAPER else r"velocity − v$_{\rm centre}$ [km s$^{-1}$]",
                     fontsize=LAB, labelpad=6 if not PAPER else 1.5)
    else:
        a.tick_params(labelbottom=False)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
if PAPER:                                             # common colorbar labels, then the legend, below the bottom row
    yl = PY(nr - 1) - CBP - CBH - 0.13
    fig.text((PX(0) + PX(2) + S) / 2 / FW, yl / FH, r"Jy arcsec$^{-2}$ km s$^{-1}$", ha="center", va="top", fontsize=LAB)
    fig.text((PX(3) + S / 2) / FW, yl / FH, "difference (%)", ha="center", va="top", fontsize=LAB)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.17 / FH),
               ncol=3, frameon=False, fontsize=F["leg"], handlelength=1.6, columnspacing=1.0, borderaxespad=0)
fig.savefig(OUT, dpi=300 if PAPER else 65)
if PAPER:
    fig.savefig(OUT.replace(".png", ".pdf"), dpi=400)
print("saved", OUT, f"({FW:.2f} x {FH:.2f} in)")
