#!/usr/bin/env python
"""
Three-row version of fig_consistency_overview.py: compact, uv-tapered extended and uv-tapered joint (compact +
extended) data of NGC 3110, all at the compact beam, compared inside ONE mask (the tapered extended MS-CLEAN's
astrodendro mask at MASKSIG sigma), continuum kept (CONTSUB=0 behaviour).

  row 1 compact               uv coverage | dirty | MS-CLEAN | spectra of all three (all channels, mask footprint)
  row 2 extended, tapered     uv (weight x taper) | dirty | MS-CLEAN | (compact - extended) / extended
  row 3 joint, tapered        uv (weight x taper) | dirty | MS-CLEAN | (compact - joint) / joint

Dirty images: each its own colour scale; MS-CLEAN images: one scale (common min / max of the three).

    SET=results/NGC3110/consistency_20kms_0.08 MASKSIG=16 python scripts/experiments/fig_consistency_overview3.py
    -> <SET>/figures/fig5_overview3_<MASKSIG>.{png,pdf}

BEAM=lbmax (SET=results/NGC3110/consistency_20kms_lbmax, made by lbmax_extended_taper.py and consistency_target.py
TAPER_JSON=<SET>/ext_taper.json TARGET_RESTORE=1): the same at the compact lambda/B_max beam (0.60" circular), where the
compact data alone cannot be imaged: two rows, the joint data (the resolution ladder's lambda/B_max image,
ladder_joint_20kms/b05: Briggs robust 0.5 + taper) as the reference and the extended data alone, tapered.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in ("simple", "scripts", "scripts/compact_vs_msclean_dendro"):
    sys.path.insert(0, os.path.join(REPO, _p))
import beams  # noqa: E402

SET = os.path.join(REPO, os.environ["SET"]); MS = os.environ.get("MASKSIG", "16")
D_ = os.path.join(REPO, "data/ngc3110"); OUT = os.path.join(SET, "figures"); os.makedirs(OUT, exist_ok=True)
G_ = json.load(open(os.path.join(SET, "grid.json")))
L, DV, CELL, NCH = G_["line"], G_["width_kms"], G_["cell"], G_["nchan"]
vel = G_["start_kms"] + DV * np.arange(NCH)
LBM = os.environ.get("BEAM") == "lbmax"
if LBM:
    RUNS = dict(joint=dict(img=os.path.join(REPO, "results/NGC3110/ladder_joint_20kms/b05/clean_finished/image.fits"),
                           dirty=f"{SET}/dirty_joint_lbmax", taper=f"{SET}/dirty_joint_lbmax/setup.json", robust=0.5),
                extended=dict(img=f"{SET}/clean_extended_taper_cmask/image.fits", dirty=f"{SET}/dirty_extended_taper",
                              taper=f"{SET}/ext_taper.json"))
else:
    RUNS = dict(compact=dict(img=f"{SET}/clean_compact_finished/image.fits", dirty=f"{SET}/dirty_compact"),
                extended=dict(img=f"{SET}/clean_extended_taper_cmask/image.fits", dirty=f"{SET}/dirty_extended_taper",
                              taper=f"{D_}/extended/msclean_taper/taper.json"),
                joint=dict(img=f"{SET}/clean_joint_taper_finished/image.fits", dirty=f"{SET}/dirty_joint_taper",
                           taper=f"{SET}/joint_taper.json"))
REF = next(iter(RUNS)); NR = len(RUNS)                                    # the reference: the first row
COL = dict(compact="#0b3c6e", extended="#e07b00", joint="#1a9e77"); MUTED = "#8a8a8a"
TICK, LAB = 13, 15
plt.rcParams.update({"font.family": "serif", "font.size": 13, "xtick.direction": "in", "ytick.direction": "in"})
rd = lambda p: np.nan_to_num(fits.getdata(p)).astype(np.float64).squeeze()
bm = lambda p: (lambda h: (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"]))(fits.getheader(p))
BC = bm(RUNS[REF]["img"]); bpx = beams.omega(BC) / CELL ** 2
MF = f"{SET}/clean_extended_taper_cmask/dendro_mask_min{MS}.npy"
if not os.path.exists(MF):                                                # the astrodendro mask at MASKSIG sigma
    from msclean_complete import dendro_mask  # noqa: E402
    _d = f"{SET}/clean_extended_taper_cmask"
    np.save(MF, dendro_mask(rd(f"{_d}/image.fits"), rd(f"{_d}/residual.fits"), bpx, L, min_value=float(MS))[0].astype(np.uint8))
M = np.load(MF).astype(bool)
FOOT = M[L].any(0)
m0 = lambda c: c[L].sum(0) * DV

# ======================================================================== data: every cube at the compact clean beam
IMG, DIRTY, BD, SPEC, FLUX = {}, {}, {}, {}, {}
def to008(x):
    """0.04" (1280 px) -> 0.08" (640 px), pixel centres aligned, averaged (Jy/beam)"""
    from scipy.ndimage import convolve1d
    k_ = np.array([0.25, 0.5, 0.25])
    return convolve1d(convolve1d(x, k_, axis=-1, mode="constant"), k_, axis=-2, mode="constant")[..., ::2, ::2]


for k, r in RUNS.items():
    c = rd(r["img"]); b = bm(r["img"])
    if c.shape[-1] == 2 * int(G_["imsize"]):                             # the ladder's 0.04" grid
        c = to008(c)
    if not np.allclose(b, BC, rtol=0.005):
        c = beams.to_beam(c, b, BC, CELL)
    IMG[k] = c
    SPEC[k] = (c * FOOT).sum((1, 2)) / bpx                              # all channels, over the mask's footprint
    FLUX[k] = float((c * M)[L].sum() / bpx * DV)                         # line channels, inside the mask
    d = rd(f"{r['dirty']}/dirty.fits"); DIRTY[k] = m0(d)
    hp = fits.getheader(f"{r['dirty']}/psf.fits"); BD[k] = bm(f"{r['dirty']}/psf.fits") if "BMAJ" in hp else BC
STAT = {}
for k in [q for q in RUNS if q != REF]:
    a, b = IMG[REF][L][M[L]], IMG[k][L][M[L]]
    STAT[k] = (100 * (FLUX[REF] - FLUX[k]) / FLUX[k], 100 * np.linalg.norm(a - b) / np.linalg.norm(b))
    print(f"mask {MS} sigma: {REF} {FLUX[REF]:.1f} vs {k} {FLUX[k]:.1f} Jy km/s -> total {STAT[k][0]:+.1f}%, "
          f"RMSE (cube) {STAT[k][1]:.1f}%")

# uv: target-field baselines, the tapers' uv-plane weights
z = np.load(f"{D_}/raw_visibilities.npz"); cols = list(z["columns"])
UV = {k: tuple(z[k][cols.index(q)] for q in ("u_kl", "v_kl", "weight")) for k in ("compact", "extended")}
UV["joint"] = tuple(np.r_[UV["compact"][i], UV["extended"][i]] for i in range(3))


def briggs(u, v, w, robust):
    """Briggs weights on the image's uv grid (cell 1 / field), the visibilities and their conjugates gridded"""
    du = 1 / (G_["imsize"] * CELL * np.pi / 180 / 3600) / 1e3              # k-lambda
    iu, iv = np.round(np.r_[u, -u] / du).astype(int), np.round(np.r_[v, -v] / du).astype(int)
    key = (iu - iu.min()) * (iv.max() - iv.min() + 1) + (iv - iv.min()); ww = np.r_[w, w]
    _, inv = np.unique(key, return_inverse=True); Wk = np.bincount(inv, ww)
    f2 = (5 * 10 ** (-robust)) ** 2 / ((Wk ** 2).sum() / ww.sum())
    return w / (1 + Wk[inv[:u.size]] * f2)


for k, r in RUNS.items():                                                # imaging weights of each row
    if "robust" in r:
        UV[k] = UV[k][:2] + (briggs(*UV[k], r["robust"]),)
asr = np.pi / 180 / 3600
NAMES = dict(compact="Compact", extended="Extended", joint="Joint")


def taper_fn(path):
    t = json.load(open(path))["uvtaper"]
    ta, tb, pa = float(t[0].rstrip("arcsec")), float(t[1].rstrip("arcsec")), float(t[2].rstrip("deg"))

    def w(u, v):
        p = np.deg2rad(pa); up, vp = u * np.sin(p) + v * np.cos(p), u * np.cos(p) - v * np.sin(p)
        return np.exp(-np.pi ** 2 / (4 * np.log(2)) * ((ta * asr * up * 1e3) ** 2 + (tb * asr * vp * 1e3) ** 2))
    return w, (ta, tb)


# ======================================================================== PAPER=1: tight paper layout, then stop
if os.environ.get("PAPER") == "1":
    import matplotlib.patheffects as pe
    FS = 11.5
    plt.rcParams.update({"font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS - 1.5, "ytick.labelsize": FS - 1.5,
                         "legend.fontsize": FS - 2.5, "axes.linewidth": 1.0})
    OUTL = [pe.withStroke(linewidth=2.3, foreground="k")]
    FW, ML, MR, MB, MT = 12.0, 0.62, 0.62, 0.78, 0.5
    G01, GD, G23, CBW, GR = 0.3, 0.88, 0.1, 0.09, 0.08                    # gaps; dirty colorbar width; row gap
    S = (FW - ML - MR - G01 - GD - G23) / 4
    FH = MB + NR * S + (NR - 1) * GR + MT
    fig = plt.figure(figsize=(FW, FH))
    inch = lambda x, y, w, h: fig.add_axes([x / FW, y / FH, w / FW, h / FH])
    X0 = [ML, ML + S + G01]; X0 += [X0[1] + S + GD]; X0 += [X0[2] + S + G23]
    YR = lambda r: MB + (NR - 1 - r) * (S + GR)
    H = 8.0; N = IMG[REF].shape[-1]; hp = int(H / CELL)
    sl = np.s_[N // 2 - hp:N // 2 + hp, N // 2 - hp:N // 2 + hp]; EXT = [H, -H, -H, H]
    ALL = [m0(IMG[k]) for k in RUNS]
    clo, chi = min(x[sl].min() for x in ALL), max(x[sl].max() for x in ALL)

    def plabel(a, t, light=False):
        a.text(0.04, 0.96, t, transform=a.transAxes, ha="left", va="top", fontsize=FS, fontweight="bold",
               color="k" if light else "w", zorder=10, path_effects=None if light else OUTL)

    def pmap(x, y, img, beam, t, cmap="inferno", lo=None, hi=None, light=False, outline=False, bar=False, oa=0.45):
        a = inch(x, y, S, S)
        cm = plt.get_cmap(cmap).copy(); cm.set_bad("0.9")
        im = a.imshow(img[sl], origin="lower", extent=EXT, cmap=cm, norm=Normalize(lo, hi), interpolation="nearest")
        if outline:
            a.contour(FOOT[sl].astype(float), levels=[0.5], colors="k" if light else "w", linewidths=0.6, extent=EXT,
                      origin="lower", alpha=oa)
        a.set_xticks([]); a.set_yticks([])
        ink = "k" if light else "w"
        if beam is not None:
            pad = 0.06 * H
            a.add_patch(Ellipse((H - pad - beam[0] / 2, -H + pad + beam[0] / 2), width=beam[0], height=beam[1],
                                angle=90 - beam[2], facecolor="0.2" if light else "w", edgecolor=ink))
        if bar:                                                          # 2" scale bar, end caps, bold label below
            xs, ys, cap = H - 0.08 * H - 2, -H + 0.12 * H, 0.03 * H
            xs = -H + 0.08 * H
            a.plot([xs + 2, xs], [ys, ys], color=ink, lw=1.8, solid_capstyle="butt")
            for xe in (xs, xs + 2):
                a.plot([xe, xe], [ys - cap, ys + cap], color=ink, lw=1.3, solid_capstyle="butt")
            a.text(xs + 1, ys + 2 * cap, "2″", color=ink, ha="center", va="bottom", fontsize=FS - 1, fontweight="bold",
                   path_effects=None if light else OUTL)
        plabel(a, t, light)
        return a, im

    def puv(x, y, k, t, taper=None, xlabels=False):
        a = inch(x, y, S, S); R = 200.0; u, v, w = UV[k]
        sel = np.nonzero((np.abs(u) < R) & (np.abs(v) < R))[0]
        pick = np.random.default_rng(1).choice(sel, min(sel.size, 40000), replace=False)
        if taper is None:
            a.scatter(np.r_[u[pick], -u[pick]], np.r_[v[pick], -v[pick]], s=0.5, color=COL["compact"], lw=0,
                      alpha=0.6, rasterized=True)
        else:
            e = w * taper(u, v); e = e / e.max(); cc = np.r_[e[pick], e[pick]]; o_ = np.argsort(cc)
            sc = a.scatter(np.r_[u[pick], -u[pick]][o_], np.r_[v[pick], -v[pick]][o_], s=1.0, c=cc[o_],
                           cmap="viridis", vmin=0, vmax=1, lw=0, rasterized=True)
            gu, gv = np.meshgrid(np.linspace(-R, R, 400), np.linspace(-R, R, 400))
            a.contour(gu, gv, taper(gu, gv), levels=[0.5], colors="w", linewidths=1.1)
            cax = a.inset_axes([0.45, 0.07, 0.5, 0.035])
            cb = fig.colorbar(sc, cax=cax, orientation="horizontal"); cb.set_ticks([0, 0.5, 1])
            cax.tick_params(labelsize=FS - 4, colors="w", direction="in", length=2.5, pad=1.5); cb.outline.set_edgecolor("w")
            cax.set_title("weight × taper", color="w", fontsize=FS - 4, pad=1.5)
        a.set_xlim(R, -R); a.set_ylim(-R, R); a.set_aspect("equal"); a.set_facecolor("0.12" if taper else "white")
        a.tick_params(which="both", top=True, right=True, color="0.6" if taper else "k", pad=2, length=3)
        a.set_xticks([-100, 0, 100]); a.set_yticks([-100, 0, 100])
        a.set_ylabel(r"$v$ (k$\lambda$)", labelpad=1)
        if xlabels:
            a.set_xlabel(r"$u$ (k$\lambda$)", labelpad=1)
        else:
            a.tick_params(labelbottom=False)
        plabel(a, t, light=taper is None)
        return a

    L3 = "abcdefghijkl"
    for r, k in enumerate(RUNS):
        y = YR(r); tl = "" if "taper" not in RUNS[k] else ", tapered"
        tw, tt = (None, None) if "taper" not in RUNS[k] else taper_fn(RUNS[k]["taper"])
        au = puv(X0[0], y, k, f"({L3[4 * r]}) {k}{tl}", taper=tw, xlabels=(r == NR - 1))
        if tt:                                                            # the taper (image-plane FWHM)
            wt = f"robust {RUNS[k]['robust']:g}, " if "robust" in RUNS[k] else ""
            au.text(0.04, 0.86, f"{wt}uvtaper {tt[0]:.2f}″ × {tt[1]:.2f}″", transform=au.transAxes, ha="left", va="top",
                    color="w", fontsize=FS - 2, path_effects=OUTL)
        a, im = pmap(X0[1], y, DIRTY[k], BD[k], f"({L3[4 * r + 1]}) dirty", lo=DIRTY[k][sl].min(), hi=DIRTY[k][sl].max(),
                     bar=(r == 0))
        cax = inch(X0[1] + S + 0.05, y, CBW, S)                          # this dirty map's own scale
        cb = fig.colorbar(im, cax=cax); cb.locator = matplotlib.ticker.MaxNLocator(5); cb.update_ticks()
        cax.tick_params(labelsize=FS - 3, length=2.5, pad=1.5)
        if r == (NR - 1) // 2:
            cb.set_label(r"Jy (dirty beam)$^{-1}$ km s$^{-1}$", fontsize=FS - 1.5, labelpad=3)
        a, imc = pmap(X0[2], y, ALL[r], BC, f"({L3[4 * r + 2]}) MS-CLEAN", lo=clo, hi=chi, outline=True)
        if k != REF:
            Ia, Ib = (IMG[REF] * M)[L].sum(0), (IMG[k] * M)[L].sum(0)
            pct = np.where(FOOT & (Ib > 0.05 * Ib[FOOT].max()), 100 * (Ia - Ib) / np.where(Ib > 0, Ib, 1), np.nan)
            ad, imd = pmap(X0[3], y, pct, None, f"({L3[4 * r + 3]}) {REF} vs {k}", cmap="RdBu_r", lo=-50, hi=50,
                           light=True, outline=True, oa=1.0)
            ad.text(0.04, 0.84, f"total {STAT[k][0]:+.1f}%\nRMSE {STAT[k][1]:.1f}%", transform=ad.transAxes, ha="left",
                    va="top", fontsize=FS - 1)
    for col, im_, lab in ((2, imc, r"Jy beam$^{-1}$ km s$^{-1}$"), (3, imd, f"({REF} − other) / other (%)")):
        cax = inch(X0[col], MB - 0.12 - 0.1, S, 0.1)                     # shared colorbar under the column
        cb = fig.colorbar(im_, cax=cax, orientation="horizontal"); cb.locator = matplotlib.ticker.MaxNLocator(5)
        cb.update_ticks(); cax.tick_params(labelsize=FS - 2, length=3, pad=1.5); cb.set_label(lab, fontsize=FS - 1, labelpad=2)
    # (d) spectra, velocity axis on top, flux axis on the right
    a = inch(X0[3], YR(0), S, S)
    a.axvspan(vel[L[0]] - DV / 2, vel[L[-1]] + DV / 2, color=MUTED, alpha=0.1, lw=0)
    for k in RUNS:
        a.step(vel, SPEC[k], where="mid", color=COL[k], lw=1.5, label=f"{k}: {SPEC[k][L].sum() * DV:.0f}")
    a.axhline(0, color=MUTED, lw=0.8)
    a.set_xlim(vel[0] - DV / 2, vel[-1] + DV / 2); a.set_ylim(top=1.95 * max(s_.max() for s_ in SPEC.values()))
    a.xaxis.tick_top(); a.xaxis.set_label_position("top"); a.yaxis.tick_right(); a.yaxis.set_label_position("right")
    a.tick_params(which="both", top=True, bottom=True, left=True, right=True, labelbottom=False, labeltop=True,
                  labelleft=False, labelright=True, length=3)
    a.minorticks_on(); a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    a.set_xlabel(r"$v_\mathrm{LSRK}$ (km s$^{-1}$)", labelpad=3); a.set_ylabel(r"$S_\nu$ (Jy)", labelpad=3)
    plabel(a, "(d) spectra", light=True)
    a.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), frameon=True, framealpha=0.95, edgecolor="0.6", fancybox=False,
             title="Jy km s$^{-1}$", title_fontsize=FS - 2.5)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig5_overview3_{MS}_paper.{ext}"), dpi=200 if ext == "png" else 300)
    print("saved", os.path.join(OUT, f"fig5_overview3_{MS}_paper.png"), json.dumps({k: [round(v_, 2) for v_ in s_]
                                                                                   for k, s_ in STAT.items()})); sys.exit(0)

# ======================================================================== layout (inches): colorbar under every map
S, G, CBH, CBP, CBL = 3.4, 0.5, 0.14, 0.1, 0.62
ML, MR, MB, MT = 0.95, 1.05, 0.35, 1.05
ROW = S + CBP + CBH + CBL                                                # one row's height incl. its colorbar
FW, FH = ML + 4 * S + 3 * G + 0.25 + MR, MB + 3 * ROW + 0.25 + MT
fig = plt.figure(figsize=(FW, FH))
inch = lambda x, y, w, h: fig.add_axes([x / FW, y / FH, w / FW, h / FH])
PX = lambda c: ML + c * (S + G) + (0.25 if c > 0 else 0)
PY = lambda r: MB + (2 - r) * (ROW + 0.12) + CBP + CBH + CBL          # bottom of row r's panels
H = 8.0; N = IMG["compact"].shape[-1]; hp = int(H / CELL)
sl = np.s_[N // 2 - hp:N // 2 + hp, N // 2 - hp:N // 2 + hp]; EXT = [H, -H, -H, H]


def label(a, text, light=False, box=False):
    a.text(0.04, 0.96, text, transform=a.transAxes, ha="left", va="top", fontsize=LAB, fontweight="bold",
           color="k" if light else "w", zorder=10,
           bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85) if box else None)


def cbar(im, x, y, unit):
    cax = inch(x, y - CBP - CBH, S, CBH)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cax.tick_params(labelsize=TICK - 1, direction="out", length=4, pad=2)
    cb.locator = matplotlib.ticker.MaxNLocator(5); cb.update_ticks(); cb.set_label(unit, fontsize=LAB - 2, labelpad=3)


def image(x, y, img, beam, text, unit, cmap="inferno", lo=None, hi=None, light=False, outline=False, bar=True, oa=0.4):
    a = inch(x, y, S, S)
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("0.9")
    im = a.imshow(img[sl], origin="lower", extent=EXT, cmap=cm, norm=Normalize(lo, hi), interpolation="nearest")
    if outline:
        a.contour(FOOT[sl].astype(float), levels=[0.5], colors="k" if light else "w", linewidths=0.6, extent=EXT,
                  origin="lower", alpha=oa)
    a.set_xticks([]); a.set_yticks([])
    ink = "k" if light else "w"
    if beam is not None:
        pad = 0.06 * H
        a.add_patch(Ellipse((H - pad - beam[0] / 2, -H + pad + beam[0] / 2), width=beam[0], height=beam[1],
                            angle=90 - beam[2], facecolor="0.2" if light else "w", edgecolor=ink))
    if bar:
        xs, ys, cap = -H + 0.08 * H, H - 0.1 * H, 0.03 * H
        a.plot([xs + 2, xs], [ys, ys], color=ink, lw=2.0, solid_capstyle="butt")
        for xe in (xs, xs + 2):
            a.plot([xe, xe], [ys - cap, ys + cap], color=ink, lw=1.4, solid_capstyle="butt")
        a.text(xs + 1, ys - 2.2 * cap, "2″", color=ink, ha="center", va="top", fontsize=12, fontweight="bold")
    label(a, text, light)
    cbar(im, x, y, unit)
    return a


def uv_panel(x, y, u, v, w, text, taper=None):
    a = inch(x, y, S, S); R = 200.0
    sel = np.nonzero((np.abs(u) < R) & (np.abs(v) < R))[0]
    pick = np.random.default_rng(1).choice(sel, min(sel.size, 40000), replace=False)
    if taper is None:
        a.scatter(np.r_[u[pick], -u[pick]], np.r_[v[pick], -v[pick]], s=0.6, color=COL["compact"], lw=0, alpha=0.6,
                  rasterized=True)
    else:
        e = w * taper(u, v); e = e / e.max(); cc = np.r_[e[pick], e[pick]]; o_ = np.argsort(cc)
        sc = a.scatter(np.r_[u[pick], -u[pick]][o_], np.r_[v[pick], -v[pick]][o_], s=1.2, c=cc[o_], cmap="viridis",
                       vmin=0, vmax=1, lw=0, rasterized=True)
        gu, gv = np.meshgrid(np.linspace(-R, R, 400), np.linspace(-R, R, 400))
        a.contour(gu, gv, taper(gu, gv), levels=[0.5], colors="w", linewidths=1.2)
        cax = a.inset_axes([0.42, 0.07, 0.52, 0.035])                     # inside the dark panel, bottom right
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cax.tick_params(labelsize=TICK - 3, colors="w", direction="in", length=3, pad=2)
        cb.set_ticks([0, 0.5, 1]); cb.outline.set_edgecolor("w")
        cax.set_title("weight × taper", color="w", fontsize=TICK - 3, pad=2)
    a.set_xlim(R, -R); a.set_ylim(-R, R); a.set_aspect("equal"); a.set_facecolor("0.12" if taper else "white")
    a.tick_params(which="both", top=True, right=True, labelsize=TICK, color="0.6" if taper else "k", pad=2)
    a.set_ylabel(r"$v$ (k$\lambda$)", fontsize=LAB, labelpad=2)
    a.set_xlabel(r"$u$ (k$\lambda$)", fontsize=LAB - 1, labelpad=1)
    label(a, text, light=taper is None, box=taper is None)


ALL = [m0(IMG[k]) for k in RUNS]
clo, chi = min(x[sl].min() for x in ALL), max(x[sl].max() for x in ALL)
PARAM = "multiscale (0, 1.2″, 3.6″)\nnatural, 51″ field\nnsigma 3, auto-multithresh\n→ nsigma 1.5, dendrogram mask"
for r, k in enumerate(RUNS):
    y = PY(r)
    if k == "compact":
        uv_panel(ML, y, *UV[k], "Compact visibilities")
    else:
        tw, (ta, tb) = taper_fn(RUNS[k]["taper"])
        uv_panel(ML, y, *UV[k], f"{NAMES[k]} visibilities,\ntapered", taper=tw)
    t = "" if k == "compact" else "\ntapered"
    image(PX(1), y, DIRTY[k], BD[k], f"{NAMES[k]} dirty{t}", r"Jy (dirty beam)$^{-1}$ km s$^{-1}$",
          lo=DIRTY[k][sl].min(), hi=DIRTY[k][sl].max())
    a = image(PX(2), y, ALL[r], BC, f"{NAMES[k]} MS-CLEAN{t}", r"Jy beam$^{-1}$ km s$^{-1}$", lo=clo, hi=chi, outline=True)
    txt = PARAM if k == "compact" else f"uvtaper {ta:.2f}″ × {tb:.2f}″\n" + (
        PARAM if k == "joint" else "multiscale (0, 1.2″, 3.6″)\nnatural, 51″ field\nin the compact mask, 1.5σ")
    a.text(0.96, 0.04, txt, transform=a.transAxes, ha="right", va="bottom", fontsize=10, color="w", linespacing=1.15)
    if k != "compact":                                                   # (compact - k) / k inside the mask
        Ia, Ib = (IMG["compact"] * M)[L].sum(0), (IMG[k] * M)[L].sum(0)
        pct = np.where(FOOT & (Ib > 0.05 * Ib[FOOT].max()), 100 * (Ia - Ib) / np.where(Ib > 0, Ib, 1), np.nan)
        ar = image(PX(3), y, pct, None, f"Compact vs {k}", f"(compact − {k}) / {k} (%)", cmap="RdBu_r", lo=-50, hi=50,
                   light=True, outline=True, bar=False, oa=1.0)
        ar.text(0.04, 0.86, f"total {STAT[k][0]:+.1f}%\nRMSE (cube) {STAT[k][1]:.1f}%", transform=ar.transAxes,
                ha="left", va="top", fontsize=13)

# spectra (row 1, last column)
a = inch(PX(3), PY(0), S, S)
a.axvspan(vel[L[0]] - DV / 2, vel[L[-1]] + DV / 2, color=MUTED, alpha=0.1, lw=0)
for k in RUNS:
    a.step(vel, SPEC[k], where="mid", color=COL[k], lw=1.8, label=f"{k}{'' if k == 'compact' else ', tapered'}: "
           f"{SPEC[k][L].sum() * DV:.0f}")
a.axhline(0, color=MUTED, lw=0.8)
a.set_xlim(vel[0] - DV / 2, vel[-1] + DV / 2); a.set_ylim(top=1.95 * max(s.max() for s in SPEC.values()))
a.yaxis.tick_right(); a.yaxis.set_label_position("right")
a.tick_params(which="both", top=True, bottom=True, left=True, right=True, labelleft=False, labelright=True, labelsize=TICK)
a.minorticks_on(); a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
a.set_xlabel(r"$v_\mathrm{LSRK}$ (km s$^{-1}$)", fontsize=LAB - 1, labelpad=2)
a.set_ylabel(r"$S_\nu$ in the mask (Jy)", fontsize=LAB, labelpad=6)
label(a, "MS-CLEAN spectra\nall channels, mask footprint", light=True)
a.legend(frameon=False, fontsize=11, loc="upper right", bbox_to_anchor=(1.0, 0.84), title="Jy km s$^{-1}$", title_fontsize=11)
fig.text(0.5, 1 - 0.15 / FH, "NGC 3110: compact, extended and joint, all at the compact beam", ha="center", va="top",
         fontsize=21, fontweight="bold")
fig.text(0.5, 1 - 0.55 / FH, f"{CELL:g}″ pixels, {NCH} × {DV:g} km s$^{{-1}}$; compared inside the {MS}σ astrodendro mask "
         "of the tapered extended MS-CLEAN", ha="center", va="top", fontsize=14)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig5_overview3_{MS}.{ext}"), dpi=110 if ext == "png" else 300)
print("saved", os.path.join(OUT, f"fig5_overview3_{MS}.png"))
