#!/usr/bin/env python
"""
Transposed 20 km/s gallery (paper version): one column per galaxy, the compact 2D-1D deconvolution against the joint
compact + extended data at two comparison beams, the compact array's lambda/B_max (circular) and the extended clean
beam (the joint data uv-tapered to it).

  rows  1 compact MS-CLEAN, restored (its clean beam)
        2 compact 2D-1D model * lambda/B_max       3 joint restored at lambda/B_max      4 difference (% of joint)
        5 compact 2D-1D model * extended beam      6 joint restored at the extended beam 7 difference (% of joint)
        8 integrated spectra inside the masks: 2D-1D and joint at lambda/B_max (solid) and the extended beam (dashed),
          compact MS-CLEAN at its clean beam (dotted, inside the compact-beam mask)

Inputs as fig_gallery.py GRID=20kms (results/<T>/g20/, and the accuracy-test products for NGC 3110), on 64 x 20 km/s
and 0.08" pixels. The joint restored image at a beam B coarser than the extended one is its CLEAN components * B plus
its residual brought to B (beams.to_beam). Masks: the resolution ladder's rule (fig_resolution_ladder.py) -- line voxels
>= 5 sigma and >= S0 in surface brightness, in connected regions of >= 2 beams, inside the mask made by the same rule
at the compact clean beam; S0 = 0.511 x the 99.9th percentile of the target's line surface brightness at lambda/B_max
(0.0457 Jy arcsec^-2 for NGC 3110, the ladder's level), the same S0 at every beam.

    python scripts/compact_vs_msclean_dendro/fig_gallery_transposed.py
    -> results/gallery/gallery_top6_20kms_transposed_paper.{png,pdf,json}
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
from scipy.ndimage import convolve1d, label

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("simple", "scripts/compact_vs_msclean_dendro"):
    sys.path.insert(0, os.path.join(REPO, p))
from restore import convolve_with_beam  # noqa: E402
import beams  # noqa: E402

RES = os.path.join(REPO, "results")
TARGETS = os.environ.get("TARGETS", "IC5179,MCG-02-33-098,NGC3110,IC4518W,ESO297-G011,NGC3256").split(",")
SB_FRAC = float(os.environ.get("SB_FRAC", 0.511))
OUT = os.path.join(RES, "gallery", "gallery_top6_20kms_transposed_paper")
rd = lambda p: np.nan_to_num(fits.getdata(p)).astype(np.float64).squeeze()
omega = lambda b: np.pi * b[0] * b[1] / (4 * np.log(2))
CELL = 0.08


def beam_any(p):
    with fits.open(p) as h:
        if "BMAJ" in h[0].header:
            x = h[0].header; return (x["BMAJ"] * 3600, x["BMIN"] * 3600, x["BPA"])
        b = h["BEAMS"].data
        return (float(np.median(b["BMAJ"])), float(np.median(b["BMIN"])), float(np.median(b["BPA"])))


def to008(x, avg):
    """0.04" (1280 px) -> 0.08" (640 px), pixel centres aligned; summed (Jy/pixel) or averaged (Jy/beam)"""
    k = np.array([0.5, 1.0, 0.5])
    y = convolve1d(convolve1d(x, k, axis=-1, mode="constant"), k, axis=-2, mode="constant")[..., ::2, ::2]
    return y / 4.0 if avg else y


def sbmask(img, res, b, S0, L):
    sig = 1.4826 * np.median(np.abs(res - np.median(res, axis=(1, 2), keepdims=True)), axis=(1, 2))
    m = (img / omega(b) >= S0) & (img >= 5 * sig[:, None, None]); out = np.zeros_like(m); nmin = 2 * omega(b) / CELL ** 2
    for ch in L:
        lab, n = label(m[ch])
        if n:
            sz = np.bincount(lab.ravel()); keep = np.nonzero(sz >= nmin)[0]; keep = keep[keep > 0]
            out[ch] = np.isin(lab, keep)
    return out


def crange(img, half, cen=None):
    """min and max of the displayed box (+-half arcsec about cen, default the image centre)"""
    cy, cx = cen or (img.shape[0] // 2,) * 2; hp = int(round(half / CELL)); b = img[cy - hp:cy + hp, cx - hp:cx + hp]
    return float(b.min()), float(b.max())


ZOOM = set(os.environ.get("ZOOM", "MCG-02-33-098,NGC7130").split(","))  # rows layout: zoom onto the main structure


def zoom(g, margin=0.5):
    """centre and half-width of the box around the largest connected region of the two comparison masks"""
    n = g["cc"].shape[0]
    if g["T"] not in ZOOM:
        return (n // 2, n // 2), g["half"]
    lab, _ = label(g["l"]["foot"] | g["e"]["foot"]); big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    yy, xx = np.nonzero(lab == big); cy, cx = (yy.min() + yy.max()) // 2, (xx.min() + xx.max()) // 2
    h = max(np.ptp(yy), np.ptp(xx)) * CELL / 2 + margin; hp = int(round(h / CELL))
    cy, cx = (int(np.clip(c_, hp, n - hp)) for c_ in (cy, cx))
    return (cy, cx), h


def compute(T):
    R = os.path.join(RES, T); c = json.load(open(os.path.join(R, "config.json")))
    if T == "NGC3110":
        gd = os.path.join(R, "consistency_20kms_0.08"); jd = os.path.join(R, "ladder_joint_20kms/b10/clean_finished")
        cmp = os.path.join(R, "pixel_scale/model_0.08_support.npy")
    else:
        gd = os.path.join(R, "g20"); jd = os.path.join(gd, "joint_ext"); cmp = os.path.join(gd, "model_0.08_support.npy")
    cc = os.path.join(gd, "clean_compact_finished")
    g = json.load(open(os.path.join(gd, "grid.json"))); L, dv, nch = g["line"], g["width_kms"], g["nchan"]
    BJ = beam_any(os.path.join(jd, "image.fits")); BC = beam_any(os.path.join(cc, "image.fits"))
    lb = float(c["lbmax_compact_arcsec"]); BL = (lb, lb, 0.0)
    jm = to008(rd(os.path.join(jd, "model.fits")), False); jr = to008(rd(os.path.join(jd, "residual.fits")), True)
    x = np.load(cmp).astype(np.float64)
    ref = {}
    for key, b in (("c", BC), ("l", BL)):
        r_ = beams.to_beam(jr, BJ, b, CELL); ref[key] = (convolve_with_beam(jm, CELL, *b) + r_, r_)
    ref["e"] = (to008(rd(os.path.join(jd, "image.fits")), True), jr)
    S0 = SB_FRAC * float(np.percentile((ref["l"][0] / omega(BL))[L], 99.9))
    M0 = sbmask(*ref["c"], BC, S0, L)
    out = dict(T=T, label=c["label"], mosaic=c.get("mosaic", False), L=L, dv=dv, vrel=dv * (np.arange(nch) - (nch - 1) / 2),
               BC=BC, BL=BL, BE=BJ, S0=S0)
    for key, b in (("l", BL), ("e", BJ)):
        I, r_ = ref[key]; M = sbmask(I, r_, b, S0, L) & M0
        C = convolve_with_beam(x, CELL, *b); bpx = omega(b) / CELL ** 2
        mv = M[L]
        out[key] = dict(
            cmp=(C * M)[L].sum(0) * dv / omega(b), ref=(I * M)[L].sum(0) * dv / omega(b), foot=M[L].any(0),
            spec_c=(C * M).sum((1, 2)) / bpx, spec_r=(I * M).sum((1, 2)) / bpx,
            dF=float(100 * ((C * M)[L].sum() - (I * M)[L].sum()) / (I * M)[L].sum()),
            rmse=float(100 * np.linalg.norm((C[L] - I[L])[mv]) / np.linalg.norm(I[L][mv])))
        out[key]["flux_c"], out[key]["flux_r"] = (float(out[key][s][L].sum() * dv) for s in ("spec_c", "spec_r"))
        d = out[key]
        d["pct"] = np.where(d["foot"] & (d["ref"] >= 0.05 * d["ref"][d["foot"]].max()),
                            100 * (d["cmp"] - d["ref"]) / np.where(d["ref"] > 0, d["ref"], 1), np.nan)
        d["mad"] = float(np.nanmedian(np.abs(d["pct"])))
    ci = rd(os.path.join(cc, "image.fits"))
    out["cc"] = ci[L].sum(0) * dv / omega(BC)                                # compact MS-CLEAN, restored, its beam
    out["spec_cc"] = (ci * M0).sum((1, 2)) / (omega(BC) / CELL ** 2)        # its spectrum inside the compact-beam mask
    foot = M0[L].any(0); fy, fx = np.nonzero(foot); n = foot.shape[0]; c0 = n // 2
    out["half"] = float(c["half_arcsec"]) if "half_arcsec" in c else \
        float(min(n * CELL / 2 - 0.1, max(np.abs(fx - c0).max(), np.abs(fy - c0).max()) * CELL + 0.5))
    print(f"{T}: lambda/B_max dF {out['l']['dF']:+.1f}% RMSE {out['l']['rmse']:.1f}% | extended dF {out['e']['dF']:+.1f}% "
          f"RMSE {out['e']['rmse']:.1f}%", flush=True)
    return out


import pickle  # noqa: E402
CACHE = OUT.replace("_transposed_paper", "") + f"_cache_{'_'.join(TARGETS)}.pkl"     # the computed maps (both layouts)
if os.path.exists(CACHE) and os.environ.get("RECOMPUTE") != "1":
    G = pickle.load(open(CACHE, "rb"))
else:
    G = [compute(t) for t in TARGETS]; pickle.dump(G, open(CACHE, "wb"))
json.dump({g["T"]: {k: {q: g[k][q] for q in ("dF", "rmse", "mad", "flux_c", "flux_r")} for k in ("l", "e")} for g in G},
          open(OUT + ".json", "w"), indent=1)

if os.environ.get("LAYOUT") == "rows":
    # ==================================================================== one galaxy per row (A&A text width)
    # columns: compact MS-CLEAN | 2D-1D, joint, difference at lambda/B_max | the same at the extended beam | spectra.
    # Maps normalised per galaxy (compact MS-CLEAN to its own peak, 2D-1D and joint at each beam to the joint peak),
    # so that one colorbar per column group, at the bottom, serves every row
    FS, FT, FL = 7, 6, 5.5
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": FS, "xtick.direction": "in",
                         "ytick.direction": "in", "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5})
    FW = 184 / 25.4; nr = len(G)
    ML, MR, GX, GB, GY = 0.2, 0.08, 0.03, 0.1, 0.03                    # margins, gap, gap between groups, row gap
    P = (FW - ML - MR - 6 * GX - 2 * GB - 0.22) / 8                     # 8 equal columns; 0.22: spectra's right ticks
    MT, MB, CBH = 0.42, 0.64, 0.05
    FH = MT + nr * P + (nr - 1) * GY + MB
    fig = plt.figure(figsize=(FW, FH))
    inch = lambda x_, y_, w, h: fig.add_axes([x_ / FW, y_ / FH, w / FW, h / FH])
    XC = lambda c: ML + c * (P + GX) + (GB - GX) * ((c >= 1) + (c >= 4) + (c >= 7))
    YR = lambda r: FH - MT - (r + 1) * P - r * GY
    cm_d = plt.get_cmap("RdBu_r").copy(); cm_d.set_bad("0.93")
    DL = dict(l=50, e=50)                                               # one range for both difference columns

    def m(c, r, img, beam, lo, hi, cmap="inferno", foot=None, light=False, half=5.5, cen=None):
        a = inch(XC(c), YR(r), P, P)
        cy, cx = cen or (img.shape[0] // 2,) * 2; hp = int(round(half / CELL)); sl = np.s_[cy - hp:cy + hp, cx - hp:cx + hp]
        cm = plt.get_cmap(cmap).copy(); cm.set_bad("0.93" if light else "0.12")
        show = np.where(foot, img, np.nan) if foot is not None else img
        im = a.imshow(show[sl], origin="lower", extent=[half, -half, -half, half], cmap=cm, vmin=lo, vmax=hi,
                      interpolation="nearest")
        if foot is not None:
            a.contour(foot[sl].astype(float), levels=[0.5], colors="0.55", linewidths=0.3, extent=[half, -half, -half, half],
                      origin="lower")
        a.set_xticks([]); a.set_yticks([])
        pad = 0.07 * half
        a.add_patch(Ellipse((half - pad - beam[0] / 2, -half + pad + beam[0] / 2), width=beam[0], height=beam[1],
                            angle=90 - beam[2], facecolor="0.15" if light else "w", edgecolor="0.15" if light else "w", lw=0.3))
        return a, im

    IM = {}
    for r, g in enumerate(G):
        ce, h = zoom(g)
        lo_, hi_ = crange(g["cc"], h, ce)                                 # 100% linear: min to max of the panel
        a, IM[0] = m(0, r, g["cc"] / hi_, g["BC"], lo_ / hi_, 1, half=h, cen=ce)
        sb = 2.0 if h > 3 else 1.0; xs, ys, cap = -h + 0.1 * h, h - 0.14 * h, 0.035 * h     # scale bar, top right
        a.plot([xs, xs + sb], [ys, ys], color="w", lw=0.8, solid_capstyle="butt")
        for xe in (xs, xs + sb):
            a.plot([xe, xe], [ys - cap, ys + cap], color="w", lw=0.6)
        a.text(xs + sb / 2, ys - 2.2 * cap, f"{sb:g}″", color="w", ha="center", va="top", fontsize=FL - 0.5, fontweight="bold")
        fig.text((ML - 0.04) / FW, (YR(r) + P / 2) / FH, g["label"], rotation=90, ha="right", va="center", fontsize=FS - 0.5)
        for k, c0_ in (("l", 1), ("e", 4)):
            d = g[k]; nm = np.percentile(d["ref"][d["foot"]], 99.9); b = g["BL"] if k == "l" else g["BE"]
            m(c0_, r, d["cmp"] / nm, b, 0, 1, foot=d["foot"], half=h, cen=ce)
            a, IM[c0_ + 1] = m(c0_ + 1, r, d["ref"] / nm, b, 0, 1, foot=d["foot"], half=h, cen=ce)
            a, IM[c0_ + 2] = m(c0_ + 2, r, d["pct"], b, -DL[k], DL[k], cmap=cm_d, foot=d["foot"], light=True, half=h,
                               cen=ce)
            a.text(0.04, 0.955, f"ΔF {d['dF']:+.1f}%\nRMSE {d['rmse']:.1f}%", transform=a.transAxes, ha="left", va="top",
                   fontsize=FL - 0.5, linespacing=1.25,
                   bbox=dict(boxstyle="square,pad=0.25", facecolor="w", edgecolor="0.6", linewidth=0.4, alpha=0.95))
        a = inch(XC(7), YR(r), P, P)                                     # spectra, normalised to the joint lambda/B_max peak
        v, L = g["vrel"], g["L"]; nrm = g["l"]["spec_r"].max()
        a.axvspan(v[L[0]] - g["dv"] / 2, v[L[-1]] + g["dv"] / 2, color="0.5", alpha=0.08, lw=0)
        a.step(v, g["spec_cc"] / nrm, where="mid", color="#e07b00", ls=":", lw=0.9, zorder=3)
        for k, ls, lw in (("l", "-", 0.75), ("e", "--", 0.65)):
            a.step(v, g[k]["spec_c"] / nrm, where="mid", color="#2a6fdb", ls=ls, lw=lw)
            a.step(v, g[k]["spec_r"] / nrm, where="mid", color="#1a9e77", ls=ls, lw=lw)
        a.axhline(0, color="0.6", lw=0.3)
        a.set_xlim(v[L[0]] - 6 * g["dv"], v[L[-1]] + 6 * g["dv"]); a.set_ylim(-0.08, 1.6); a.set_yticks([0, 0.5, 1])
        a.set_xticks([-300, 0, 300])
        a.yaxis.tick_right()
        a.tick_params(which="both", top=True, left=True, labelsize=FL - 0.5, length=1.5, width=0.4, pad=1,
                      labelbottom=(r == nr - 1))
        a.text(0.04, 0.96, f"{g['l']['flux_c']:.0f} / {g['l']['flux_r']:.0f}", transform=a.transAxes, ha="left", va="top",
               fontsize=FL - 0.5)
        if r == nr - 1:
            a.set_xlabel(r"$v - v_\mathrm{c}$ (km s$^{-1}$)", fontsize=FL, labelpad=1)
    # headings: block labels above the column titles
    TIT = ["compact\nMS-CLEAN", "2D–1D", "joint", "difference", "2D–1D", "joint", "difference", "spectra"]
    for c, t in enumerate(TIT):
        fig.text((XC(c) + P / 2) / FW, (FH - MT + 0.03) / FH, t, ha="center", va="bottom", fontsize=FS - 0.5, linespacing=1.0)
    for c0_, t in ((1, r"at λ/B$_{\mathbf{max}}$ (compact)"), (4, "at the extended clean beam")):
        x0, x1 = XC(c0_), XC(c0_ + 2) + P; yb = FH - MT + 0.2
        fig.add_artist(matplotlib.lines.Line2D([x0 / FW, x1 / FW], [yb / FH, yb / FH], color="0.3", lw=0.5))
        fig.text((x0 + x1) / 2 / FW, (yb + 0.03) / FH, t, ha="center", va="bottom", fontsize=FS, fontweight="bold")
    # colorbars at the bottom: normalised maps (per group), differences
    def bcb(im, c0_, c1_, lab, ticks):
        ins = 0.1                                                        # inset: no clash between neighbouring bars
        cax = inch(XC(c0_) + ins, MB - 0.17 - CBH, XC(c1_) + P - XC(c0_) - 2 * ins, CBH)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.outline.set_linewidth(0.35); cb.set_ticks(ticks)
        cax.tick_params(labelsize=FL - 0.5, length=1.5, width=0.4, pad=1); cb.set_label(lab, fontsize=FL, labelpad=1)
    bcb(IM[0], 0, 0, "/ peak", [0, 0.5, 1])                          # starts at each panel's (negative) minimum
    bcb(IM[2], 1, 2, "/ joint peak", [0, 0.5, 1])
    bcb(IM[3], 3, 3, "%", [-50, 0, 50])
    bcb(IM[5], 4, 5, "/ joint peak", [0, 0.5, 1])
    bcb(IM[6], 6, 6, "%", [-50, 0, 50])
    HL = [(plt.plot([], [], color=c_, lw=lw_, ls=ls_)[0], t_) for c_, ls_, lw_, t_ in (
        ("#2a6fdb", "-", 0.8, r"2D–1D, $\lambda/B_\mathrm{max}$"), ("#2a6fdb", "--", 0.7, "2D–1D, extended beam"),
        ("#1a9e77", "-", 0.8, r"joint MS-CLEAN, $\lambda/B_\mathrm{max}$"),
        ("#1a9e77", "--", 0.7, "joint MS-CLEAN, extended beam"), ("#e07b00", ":", 0.9, "compact MS-CLEAN, compact beam"))]
    fig.legend(*zip(*HL), loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False, fontsize=FL - 0.5, handlelength=1.6,
               columnspacing=0.6, labelspacing=0.2)
    OR = OUT.replace("_transposed_paper", "_rows_paper")
    for ext in ("png", "pdf"):
        fig.savefig(OR + "." + ext, dpi=300 if ext == "png" else None)
    print("saved", OR + ".png", f"({FW * 25.4:.0f} x {FH * 25.4:.0f} mm)"); sys.exit(0)

# ======================================================================== figure (A&A text width, one page)
FS, FT, FL = 7, 6, 5.5
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": FS, "xtick.direction": "in",
                     "ytick.direction": "in", "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.minor.width": 0.4, "ytick.minor.width": 0.4})
FW = 184 / 25.4; nc = len(G)
GX = 0.05; P = min(0.95, (FW - 0.85 - (nc - 1) * GX) / nc)                # gap, panel side (one page with its caption)
ML = MR = (FW - nc * P - (nc - 1) * GX) / 2                              # centred; row labels left, % colorbars right
MT, MB = 0.24, 0.42
CBH, CBP, CBL = 0.045, 0.025, 0.16                                       # colorbar height / pad / tick-label room
GY = 0.025; GB = 0.06                                                     # gap between rows, extra gap between blocks
HS = 0.62 * P                                                            # spectra row height
ROWS = ["cc", "l_cmp", "l_ref", "l_d", "e_cmp", "e_ref", "e_d", "spec"]
CBUNDER = {"cc", "l_ref", "e_ref"}                                       # per-column colorbars under these rows
y = {}; cur = 0.0
for r in ROWS:                                                           # top-down positions (inches from the top)
    if r in ("l_cmp", "e_cmp", "spec"):
        cur += GB
    y[r] = cur; cur += (HS if r == "spec" else P) + GY + ((CBP + CBH + CBL) if r in CBUNDER else 0)
FH = MT + cur + MB
fig = plt.figure(figsize=(FW, FH))
inch = lambda x_, y_, w, h: fig.add_axes([x_ / FW, y_ / FH, w / FW, h / FH])
X = lambda j: ML + j * (P + GX)
Y = lambda r, h=None: FH - MT - y[r] - (h if h is not None else P)    # bottom edge of row r
cm_d = plt.get_cmap("RdBu_r").copy(); cm_d.set_bad("0.93")
LAB = dict(cc="compact\nMS-CLEAN", l_cmp="2D–1D", l_ref="joint", l_d="difference", e_cmp="2D–1D", e_ref="joint",
           e_d="difference", spec="spectra")
DL = dict(l_d=25, e_d=60)


def mapax(j, r, img, beam, lo, hi, cmap="inferno", foot=None, light=False, half=5.5):
    a = inch(X(j), Y(r), P, P)
    n = img.shape[0]; c0 = n // 2; hp = int(round(half / CELL)); sl = np.s_[c0 - hp:c0 + hp, c0 - hp:c0 + hp]
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("0.93" if light else "0.12")
    show = np.where(foot, img, np.nan) if foot is not None else img
    im = a.imshow(show[sl], origin="lower", extent=[half, -half, -half, half], cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
    if foot is not None:
        a.contour(foot[sl].astype(float), levels=[0.5], colors="0.55", linewidths=0.35, extent=[half, -half, -half, half],
                  origin="lower")
    a.set_xticks([]); a.set_yticks([])
    pad = 0.07 * half
    a.add_patch(Ellipse((half - pad - beam[0] / 2, -half + pad + beam[0] / 2), width=beam[0], height=beam[1],
                        angle=90 - beam[2], facecolor="0.15" if light else "w", edgecolor="0.15" if light else "w", lw=0.3))
    return a, im


def cbar(im, j, r, nt=3):
    cax = inch(X(j), Y(r) - CBP - CBH, P, CBH)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.outline.set_linewidth(0.35)
    cb.locator = matplotlib.ticker.MaxNLocator(nt); cb.update_ticks()
    cax.tick_params(labelsize=FL, length=1.5, width=0.4, pad=1)


for j, g in enumerate(G):
    h = g["half"]
    a, im = mapax(j, "cc", g["cc"], g["BC"], *crange(g["cc"], h), half=h)          # 100% linear: min to max of the panel
    cbar(im, j, "cc")
    a.set_title(g["label"], fontsize=FS, pad=3)
    if j == 0:                                                           # 1" or 2" scale bar, top left, end caps
        sb = 2.0 if h > 3 else 1.0; xs, ys, cap = h - 0.1 * h - sb, h - 0.14 * h, 0.035 * h
        xs = -h + 0.1 * h
        a.plot([xs, xs + sb], [ys, ys], color="w", lw=0.9, solid_capstyle="butt")
        for xe in (xs, xs + sb):
            a.plot([xe, xe], [ys - cap, ys + cap], color="w", lw=0.7)
        a.text(xs + sb / 2, ys - 2.2 * cap, f"{sb:g}″", color="w", ha="center", va="top", fontsize=FL, fontweight="bold")
    for key in ("l", "e"):
        d = g[key]; hi = np.percentile(d["ref"][d["foot"]], 99.9)
        b = g["BL"] if key == "l" else g["BE"]
        mapax(j, f"{key}_cmp", d["cmp"], b, 0, hi, foot=d["foot"], half=h)
        a, im = mapax(j, f"{key}_ref", d["ref"], b, 0, hi, foot=d["foot"], half=h)
        cbar(im, j, f"{key}_ref")
        a, imd = mapax(j, f"{key}_d", d["pct"], b, -DL[f"{key}_d"], DL[f"{key}_d"], cmap=cm_d, foot=d["foot"], light=True, half=h)
        a.text(0.04, 0.96, f"ΔF {d['dF']:+.1f}%  |Δ| {d['mad']:.0f}%", transform=a.transAxes, ha="left", va="top",
               fontsize=FL - 0.5)
        if j == nc - 1:                                                  # one difference colorbar per beam, at the right
            cax = inch(X(nc - 1) + P + 0.08, Y(f"{key}_d"), 0.07, P)
            cb = fig.colorbar(imd, cax=cax); cb.outline.set_linewidth(0.35); cb.set_ticks([-DL[f"{key}_d"], 0, DL[f"{key}_d"]])
            cax.tick_params(labelsize=FL, length=1.5, width=0.4, pad=1); cb.set_label("%", fontsize=FL, labelpad=1)
    # spectra: both beams
    a = inch(X(j), Y("spec", HS), P, HS)
    v, L = g["vrel"], g["L"]
    a.axvspan(v[L[0]] - g["dv"] / 2, v[L[-1]] + g["dv"] / 2, color="0.5", alpha=0.08, lw=0)
    nrm = g["l"]["spec_r"].max()                                         # normalised to the joint lambda/B_max peak
    for key, ls, lw in (("l", "-", 0.8), ("e", "--", 0.7)):
        a.step(v, g[key]["spec_c"] / nrm, where="mid", color="#2a6fdb", ls=ls, lw=lw)
        a.step(v, g[key]["spec_r"] / nrm, where="mid", color="#1a9e77", ls=ls, lw=lw)
    a.axhline(0, color="0.6", lw=0.3)
    a.set_xlim(v[0] - g["dv"] / 2, v[-1] + g["dv"] / 2); a.set_xticks([-400, 0, 400])
    a.set_ylim(-0.08, 1.55); a.set_yticks([0, 0.5, 1])
    a.tick_params(which="both", top=True, right=True, labelsize=FL, length=1.5, width=0.4, pad=1, labelleft=(j == 0))
    a.text(0.04, 0.96, f"{g['l']['flux_c']:.0f} / {g['l']['flux_r']:.0f}\n{g['e']['flux_c']:.0f} / {g['e']['flux_r']:.0f}",
           transform=a.transAxes, ha="left", va="top", fontsize=FL - 0.5, linespacing=1.05)
    if j == 0:
        a.set_xlabel(r"$v - v_\mathrm{c}$ (km s$^{-1}$)", fontsize=FL, labelpad=1)
    if j == 0:
        a.set_ylabel(r"$S_\nu$ / peak", fontsize=FL, labelpad=1)
for r in ROWS[:-1]:                                                      # row labels, left
    h_ = HS if r == "spec" else P
    fig.text((ML - 0.05) / FW, (Y(r, h_) + h_ / 2) / FH, LAB[r], rotation=90, ha="right", va="center", fontsize=FS - 0.5,
             linespacing=1.0)
for key, top, bot, t in (("l", "l_cmp", "l_d", r"at λ/B$_{\mathbf{max}}$ (compact)"), ("e", "e_cmp", "e_d", "at the extended beam")):
    yt, yb = Y(top) + P, Y(bot)                                          # block bracket label, far left
    fig.text((ML - 0.33) / FW, (yt + yb) / 2 / FH, t, rotation=90, ha="left", va="center", fontsize=FS, fontweight="bold")
h1, = plt.plot([], [], color="#2a6fdb", lw=0.8); h2, = plt.plot([], [], color="#1a9e77", lw=0.8)
h3, = plt.plot([], [], color="0.3", lw=0.8); h4, = plt.plot([], [], color="0.3", lw=0.7, ls="--")
fig.legend([h1, h2, h3, h4], ["compact 2D–1D", "joint MS-CLEAN", r"$\lambda/B_\mathrm{max}$", "extended beam"],
           loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, fontsize=FL, handlelength=2.0,
           columnspacing=1.2)
fig.text((ML + (FW - ML - MR) / 2) / FW, 0.2 / FH, "", fontsize=1)
for ext in ("png", "pdf"):
    fig.savefig(OUT + "." + ext, dpi=300 if ext == "png" else None)
print("saved", OUT + ".png", f"({FW * 25.4:.0f} x {FH * 25.4:.0f} mm)")
