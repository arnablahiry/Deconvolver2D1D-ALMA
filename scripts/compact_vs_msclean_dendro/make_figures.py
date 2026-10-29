#!/usr/bin/env python
"""
A galaxy's CO(2-1) (NGC 3110 by default; GAL_DIR=results/<galaxy> for the pipeline): the compact deconvolved model (raw, Jy/pixel at 0.02") vs the extended multiscale CLEAN, both
inside the astrodendro mask of the extended CLEAN image (scripts/superres_figures/fig61_dendro_mask_extended.py)
and both at the extended clean beam:

  compact   compact pdtau k = 5 model x mask, convolved with the extended clean beam (Jy/beam); in figure 40 also
            with the lambda/B_max beam (the extended array's longest-baseline resolution, 0.1134" geometric mean, the
            clean beam's shape: scripts/superres_figures/fig45_beam_sequence.py)
  CLEAN     extended CLEAN components x mask, convolved with the same beam (like for like: no residual)
  CLEAN residual: tclean's residual x mask (Jy per dirty beam), as the scale against which compact - CLEAN is judged

Writes figures 40-49, INDEX.md and stats.json to figures/compact_vs_msclean_dendro/:

  40 overview (Jy arcsec^-2 km/s): compact-configuration MS-CLEAN at its beam; compact deconvolved at lambda/B_max
     and at the ext. beam; extended CLEAN; compact - CLEAN (full +- range); integrated spectra
  41 zooms: four boxes (moment 0, compact vs CLEAN) and their spectra
  42 channel maps: compact, CLEAN and compact - CLEAN, eight line channels
  43 brightness cuts through the bar and across both arms
  44 moment 1: compact, CLEAN, difference
  45 pixel-by-pixel: moment 0 and voxel scatter, compact vs CLEAN
  46 residual on the extended dirty cube (one operator, gpu_pd.Op): maps and rms per channel
  47 one-beam spectra at the brightest peaks
  48 radial profile and cumulative flux of moment 0
  49 summary of the numbers

    MASK=...|none COMPACT=... CLEAN=ms_clean OUT_DIR=... python scripts/compact_vs_msclean_dendro/make_figures.py
    GAL_DIR=results/<galaxy> [CLEAN=msclean_finished] [MASK=...] [MASK_DESC="..."] [OUT_DIR=...] python ...
        (MASK_DESC: how the titles and INDEX.md name the mask)
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from astropy.io import fits
from scipy.ndimage import maximum_filter, map_coordinates

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts", "experiments"))
from common import G, read  # noqa: E402
from restore import convolve_with_beam  # noqa: E402

GAL_DIR = os.environ.get("GAL_DIR")                  # results/<galaxy>/ of scripts/pipeline/run_target.py
GC = {}
if GAL_DIR:
    GC = json.load(open(os.path.join(GAL_DIR, "config.json")))
    NAME, LABEL = GC["name"], GC["label"]
    CLEAN = os.environ.get("CLEAN", "msclean_finished")
    CDIR = os.path.join(GAL_DIR, "extended", CLEAN)
    MASK = os.environ.get("MASK", os.path.join(CDIR, "mask_final.npy"))
    COMPACT = os.path.join(GAL_DIR, "compact", "deconvolved", "model.npy")
    CC = os.path.join(GAL_DIR, "compact", "msclean")
    EXT_CUBE = os.path.join(GAL_DIR, "extended", "cube")
    OUT_DIR = os.environ.get("OUT_DIR", os.path.join(GAL_DIR, "figures"))
    CELL, DV, NCH = GC["cell_arcsec"], GC["width_kms"], GC["nchan"]
    vel = GC["vsys_kms"] + DV * (np.arange(NCH) - (NCH - 1) / 2)
    L = json.load(open(os.path.join(GAL_DIR, "line_channels.json")))["line"]
    LBMAX = GC["lbmax_arcsec"]
    LBMAX_C = GC.get("lbmax_compact_arcsec")
else:                                                # the original NGC3110 layout
    NAME, LABEL = "NGC3110", "NGC 3110"
    CLEAN = os.environ.get("CLEAN", "ms_clean")
    CDIR = os.path.join(REPO, "data/ngc3110/extended", CLEAN)
    MASK = os.environ.get("MASK", os.path.join(REPO, f"data/ngc3110/extended/ms_clean_masks/dendro_{CLEAN}.npy"))
    COMPACT = os.environ.get("COMPACT", os.path.join(REPO, "data/experiments/04_pd_tau_reweighting/compact/pdtau_k5.npy"))
    CC = os.path.join(REPO, "data/ngc3110/compact", os.environ.get("COMPACT_CLEAN", "ms_clean"))
    EXT_CUBE = os.path.join(REPO, "data/ngc3110/extended/cube")
    OUT_DIR = os.environ.get("OUT_DIR", os.path.join(REPO, "figures/compact_vs_msclean_dendro"))
    CELL, DV, NCH = 0.02, 25.0, 32
    vel = 5073.9 + 25.0 * (np.arange(32) - 15.5)
    L = [i for i, v in enumerate(vel) if 4750 <= v <= 5280]
    LBMAX = 0.1134                                   # lambda/B_max of the extended array, arcsec
    LBMAX_C = 0.604                                  # ... and of the compact array (B_max 451 m)
FREE = [i for i in range(NCH) if i not in L]
MASK_DESC = os.environ.get("MASK_DESC", "the astrodendro mask of the extended MS-CLEAN cube")
REF = os.environ.get("REF", "Extended MS-CLEAN")                     # figure 40: the reference CLEAN, its short name
REF_S = os.environ.get("REF_S", "ext. MS-CLEAN")
REF_BEAM = os.environ.get("REF_BEAM", "extended clean beam")              # ... and the comparison beam
BLUE, GREEN, MUTED = "#2a6fdb", "#1a9e77", "#8a8a8a"
UNIT = r"Jy beam$^{-1}$ km s$^{-1}$"
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 12, "xtick.direction": "in", "ytick.direction": "in"})
INDEX = []

rd = lambda p: np.nan_to_num(fits.getdata(p).astype(np.float64)).squeeze()
h = fits.getheader(os.path.join(CDIR, "image.fits"))
N = int(h["NAXIS1"]); C0 = N // 2                                          # image size, centre pixel
BE = (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"])
BEAM_PX = np.pi * BE[0] * BE[1] / (4 * np.log(2)) / CELL ** 2
BL = f"{BE[0]:.3f}″×{BE[1]:.3f}″, PA {BE[2]:.1f}°"
# COMPARE=lbmax: figures 40, 41, 42, 44 and 45 compare at the compact array's lambda/B_max (circular) instead of the extended
# clean beam; they go to figures/largest_compact_baseline_comparison/ (the extended-beam set to
# figures/extended_clean_beam_comparison/), and nothing else is made in that mode
BE_EXT = BE
COMPARE = os.environ.get("COMPARE", "extended")
SUB = {"extended": "extended_clean_beam_comparison", "lbmax": "largest_compact_baseline_comparison"}[COMPARE]
BN = "extended clean beam" if COMPARE == "extended" else "λ/B$_{\\rm max,\\,compact}$ beam"
if COMPARE == "lbmax":
    BE = (LBMAX_C, LBMAX_C, 0.0)
    BEAM_PX = np.pi * BE[0] * BE[1] / (4 * np.log(2)) / CELL ** 2
cmask = rd(os.path.join(CDIR, "mask.fits")) > 0
NOMASK = MASK.lower() == "none"                    # nothing applied; CLEAN's own mask only outlines and bounds the stats
M = cmask if NOMASK else np.load(MASK).astype(bool)
FOOT = M[L].any(0)                                                          # the mask's footprint on the sky
W = 1.0 if NOMASK else M
x_cmp = np.load(COMPACT).astype(np.float64) * W                            # Jy/pixel
x_cln = rd(os.path.join(CDIR, "model.fits")) * W
IN = "total" if NOMASK else "in the mask"
c_res = rd(os.path.join(CDIR, "residual.fits"))
res_m = c_res * W                                                          # tclean residual in the mask, Jy/dirty beam
s0 = LBMAX / np.sqrt(BE_EXT[0] * BE_EXT[1]); B1 = (BE_EXT[0] * s0, BE_EXT[1] * s0, BE_EXT[2])   # clean-beam shape, geometric mean LBMAX
r_cmp = convolve_with_beam(x_cmp, CELL, *BE)                               # Jy/beam (extended)
r_hi = convolve_with_beam(x_cmp, CELL, *B1)                                # Jy/beam (lambda/B_max)
r_cln = convolve_with_beam(x_cln, CELL, *BE)
m0 = lambda c: c[L].sum(0) * DV
M0 = dict(cmp=m0(r_cmp), cln=m0(r_cln), res=m0(res_m), hi=m0(r_hi))
print('no mask applied' if NOMASK else f'mask {MASK}')
TOP = (f"{LABEL} CO(2-1), NO mask applied (grey outline / 'mask' statistics: CLEAN's own mask), at the extended clean beam ({BL})"
       if NOMASK else f"{LABEL} CO(2-1), inside {MASK_DESC}, at the extended clean beam ({BL})")
LAB = dict(cmp="compact deconvolved ∗ ext. beam", cln="extended MS-CLEAN components ∗ beam", res="extended MS-CLEAN residual")
print(f"mask {M.mean():.2%} of voxels, footprint {FOOT.sum() * CELL ** 2:.1f} arcsec^2")
yy, xx = np.indices((N, N))
if GAL_DIR and "half_arcsec" not in GC:             # map half-size: the mask footprint plus 0.5", inside the field
    fy, fx = np.nonzero(FOOT)
    HALF = float(min(N * CELL / 2 - 0.1, max(np.abs(fx - C0).max(), np.abs(fy - C0).max()) * CELL + 0.5)) if fy.size else N * CELL / 2 - 0.1
else:
    HALF = float(GC.get("half_arcsec", 5.5))


def save(fig, name, text):
    if name == "_skip":
        plt.close(fig); return
    if name[:2] not in ("40", "41", "42", "44", "45"):   # 43, 46-49 are no longer made
        plt.close(fig); return
    if True:          # the beam-dependent comparison set: its own folder per beam
        os.makedirs(os.path.join(OUT_DIR, SUB), exist_ok=True)
        name = os.path.join(SUB, name)
    elif COMPARE != "extended":
        plt.close(fig); return
    fig.savefig(os.path.join(OUT_DIR, name), dpi=65); plt.close(fig)
    INDEX.append(f"- `{name}`: {text}"); print("saved", name)


def show(a, im2, half, title, cmap="inferno", lo=0, hi=None, center=(0.0, 0.0), label=UNIT, foot=True, cb=True):
    """map of a 2D array on the 0.02" grid, box of half-size `half` arcsec centred at (dRA, dDec) = center"""
    ci, cj = int(round(C0 - center[0] / CELL)), int(round(C0 + center[1] / CELL)); hp = int(round(half / CELL))
    sl = np.s_[cj - hp:cj + hp, ci - hp:ci + hp]
    ext = [center[0] + half, center[0] - half, center[1] - half, center[1] + half]
    im = a.imshow(im2[sl], origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    if foot:
        a.contour(FOOT[sl].astype(float), levels=[0.5], colors="0.55", linewidths=0.6, extent=ext, origin="lower")
    a.set_title(title, fontsize=11); a.set_xlabel(r'$\Delta$RA ["]'); a.set_ylabel(r'$\Delta$Dec ["]')
    if cb:
        plt.colorbar(im, ax=a, shrink=0.85, label=label)
    return im


# ======================================================================== numbers
R10 = FOOT
flux = dict(cmp=float(x_cmp[L].sum() * DV), cln=float(x_cln[L].sum() * DV), res=float(res_m[L].sum() / BEAM_PX * DV),
            cmp_unmasked=float(np.load(COMPACT)[L].sum() * DV))
spec = dict(cmp=x_cmp.sum((1, 2)), cln=x_cln.sum((1, 2)), res=res_m.sum((1, 2)) / BEAM_PX)
d0 = M0["cmp"] - M0["cln"]
stats = dict(mask=MASK, compact=COMPACT, clean=CLEAN, beam=BE, footprint_arcsec2=float(FOOT.sum() * CELL ** 2),
             flux_line_Jy_kms=flux, flux_ratio_compact_over_clean=flux["cmp"] / flux["cln"],
             mom0=dict(pearson=float(np.corrcoef(M0["cmp"][FOOT], M0["cln"][FOOT])[0, 1]),
                       rel_rms_diff=float(np.linalg.norm(d0[FOOT]) / np.linalg.norm(M0["cln"][FOOT])),
                       peak_cmp=float(M0["cmp"].max()), peak_cln=float(M0["cln"].max()), peak_cmp_lbmax=float(M0["hi"].max())),
             lbmax_beam=B1)

# whole-cube relative RMSE at the comparison beam (BE): ||C - K|| / ||K|| over every masked line voxel, C and K the
# compact deconvolved and extended MS-CLEAN cubes, both x mask and * BE (the same convolution as the maps)
MV = M[L]
RMSE_CUBE = float(100 * np.linalg.norm((r_cmp[L] - r_cln[L])[MV]) / np.linalg.norm(r_cln[L][MV]))
stats["rmse_cube_pct"] = RMSE_CUBE

# ======================================================================== 40 overview
PCT_LIM, PCT_FLOOR = 50.0, 0.05                       # % difference: colour range +-50%, shown where CLEAN > 5% of its peak


def pct_diff(c, k):
    """100 (c - k) / k, blanked (NaN -> light grey) where k is below PCT_FLOOR of its peak inside the mask footprint"""
    return np.where(k > PCT_FLOOR * k[FOOT].max(), 100 * (c - k) / np.where(k > 0, k, 1), np.nan)


# surface brightness, Jy arcsec^-2 km/s (moment 0 in Jy/beam km/s over the beam's solid angle), so maps at different
# resolutions share units; everything x mask. The compact configuration's own MS-CLEAN (its components x mask,
# convolved with its own clean beam) comes first.
omega = lambda b: np.pi * b[0] * b[1] / (4 * np.log(2))                     # beam solid angle, arcsec^2
hc = fits.getheader(os.path.join(CC, "image.fits")); BC = (hc["BMAJ"] * 3600, hc["BMIN"] * 3600, hc["BPA"])
x_cc = rd(os.path.join(CC, "model.fits")) * W
SB = dict(cc=m0(convolve_with_beam(x_cc, CELL, *BC)) / omega(BC), hi=M0["hi"] / omega(B1),
          cmp=M0["cmp"] / omega(BE), cln=M0["cln"] / omega(BE))
# the compact array's own resolution limit: a circular Gaussian of FWHM lambda/B_max (B_max: its longest baseline)
B2 = (LBMAX_C, LBMAX_C, 0.0)
SB["hic"] = m0(convolve_with_beam(x_cmp, CELL, *B2)) / omega(B2)
stats["lbmax_compact_beam"] = B2
SB["d"] = SB["cmp"] - SB["cln"]
flux["cc"] = float(x_cc[L].sum() * DV); spec["cc"] = x_cc.sum((1, 2))
stats["compact_msclean"] = dict(dir=CC, beam=BC, flux_line_Jy_kms=flux["cc"])
SBU = r"Jy arcsec$^{-2}$ km s$^{-1}$"
hp = int(round(HALF / CELL)); box = np.s_[C0 - hp:C0 + hp, C0 - hp:C0 + hp]
from matplotlib.patches import Ellipse
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from mpl_toolkits.axes_grid1 import make_axes_locatable
# layout in inches: six square panels of side S with the same gap GAP between them horizontally and vertically (the
# spectrum is one of them); each map's colorbar in its own axes, as wide as the map, above the top row / below the bottom
S, GAP, CBH, CBP = 5.2, 0.45, 0.22, 0.16
TICK, LAB40 = 16, 17                                  # tick-label and axis-label sizes
ML, MR, MB, MT = 0.45, 1.45, 1.35, 1.95               # margins: right = the spectrum's y axis, top = colorbar + title
FW, FH = ML + 3 * S + 2 * GAP + MR, MB + 2 * S + GAP + MT
fig = plt.figure(figsize=(FW, FH))
inch = lambda x, y, w, h: fig.add_axes([x / FW, y / FH, w / FW, h / FH])
PX = lambda c: ML + c * (S + GAP)                       # left edge of column c
PY = lambda r: MB + (1 - r) * (S + GAP)                 # bottom edge of row r
ax = np.array([[inch(PX(c), PY(r), S, S) for c in range(3)] for r in range(2)])
ve = np.percentile(SB["cln"][FOOT], 99.9)
vm = np.percentile(M0["cln"][FOOT], 99.9)                                # Jy/beam km/s scale of figs 41-49


def textbox40(a, text, light):
    """top-left label: first line bold at 17 pt, the rest at 16 pt; black box (dark grey border) on dark maps,
    white box on light panels"""
    col = "k" if light else "w"
    head, *rest = text.split("\n")
    lines = [TextArea(head, textprops=dict(color=col, fontsize=17, fontweight="bold"))]
    lines += [TextArea(t, textprops=dict(color=col, fontsize=16)) for t in rest]
    ab = AnchoredOffsetbox(loc="upper left", child=VPacker(children=lines, align="left", pad=0, sep=4),
                           pad=0.45, borderpad=0.6, frameon=True)
    ab.patch.set_boxstyle("round,pad=0.3")
    ab.patch.set(facecolor="white" if light else "black", edgecolor="0.6" if light else "0.35",
                 alpha=0.9 if light else 0.92, linewidth=1.2)
    a.add_artist(ab)


def panel40(a, img2, beam, text, row, col, cmap="inferno", lo=0.0, hi=None, light=False, outline=True, cblabel=None):
    """map without ticks; colorbar in its own axes above (row 0) or below (row 1), as wide as the map; 1" scale bar
    top right, beam bottom left, text box top left"""
    hp = int(round(HALF / CELL)); sl = np.s_[C0 - hp:C0 + hp, C0 - hp:C0 + hp]
    ext = [HALF, -HALF, -HALF, HALF]
    cm = plt.get_cmap(cmap).copy()
    if outline:                                       # outside the mask: dark grey on the maps, light grey on the difference
        img2 = np.where(FOOT, img2, np.nan)
        cm.set_bad("0.93" if light else "0.12")
    im = a.imshow(img2[sl], origin="lower", extent=ext, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
    if outline:
        a.contour(FOOT[sl].astype(float), levels=[0.5], colors="0.55", linewidths=0.6, extent=ext, origin="lower")
    a.set_xticks([]); a.set_yticks([])
    ink = "k" if light else "w"
    u = HALF / 5.5                                    # offsets scale with the field (tuned on NGC 3110's 5.5" half-width)
    sbl = 1.0 if HALF < 7 else 2.0                    # scale bar length, arcsec
    x0, y0 = -HALF + 0.6 * u, HALF - 0.7 * u          # scale bar: right end 0.6 u from the edge
    a.plot([x0 + sbl, x0], [y0, y0], color=ink, lw=2.5, solid_capstyle="butt")
    a.text(x0 + sbl / 2, y0 - 0.15 * u, f"{sbl:g}″", color=ink, ha="center", va="top", fontsize=17)
    bx, by = (HALF - 0.35 * u - beam[0] / 2, -HALF + 0.35 * u + beam[0] / 2) if beam is not None else (0, 0)
    if beam is not None:                              # beam=None: a raw model, no beam drawn
        a.add_patch(Ellipse((bx, by), width=beam[0], height=beam[1], angle=90 - beam[2], facecolor="0.15" if light else "w",
                            edgecolor=ink, lw=1.2))
        f = ".3f" if beam[0] < 1 else ".2f"
        btxt = f"{beam[0]:{f}}″ (circular)" if beam[0] == beam[1] else f"{beam[0]:{f}}″ × {beam[1]:{f}}″ @ {beam[2]:.0f}°"
        a.text(bx - beam[0] / 2 - 0.2 * u, by, btxt, color=ink, ha="left", va="center", fontsize=16)
    textbox40(a, text, light)
    cax = inch(PX(col), PY(row) + S + CBP if row == 0 else PY(row) - CBP - CBH, S, CBH)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    if row == 0:
        cax.xaxis.set_ticks_position("top"); cax.xaxis.set_label_position("top")
    cax.tick_params(labelsize=TICK, pad=6, direction="out", length=6)
    cb.set_label(cblabel or SBU, fontsize=LAB40, labelpad=8)
    return im


panel40(ax[0, 0], SB["cc"], BC, "Compact MS-CLEAN", 0, 0, hi=np.percentile(SB["cc"][FOOT], 99.9), outline=False)
SB["raw"] = m0(x_cmp) / CELL ** 2                   # the raw model's moment 0, Jy arcsec^-2 km/s (no beam)
panel40(ax[0, 1], SB["raw"], None, "Compact deconvolved\nraw model (no beam)", 0, 1, hi=np.percentile(SB["raw"][FOOT], 99.5))
panel40(ax[0, 2], SB["cmp"], BE, "Compact deconvolved\n∗ " + REF_BEAM, 0, 2, hi=ve)
panel40(ax[1, 0], SB["cln"], BE, REF, 1, 0, hi=ve)
dmax = float(np.abs(SB["d"][box]).max())
panel40(ax[1, 1], pct_diff(SB["cmp"], SB["cln"]), BE, "Difference / " + REF_S + "\n"
        f"total error {100 * (flux['cmp'] - flux['cln']) / flux['cln']:+.1f}%, "
        f"RMSE (cube) {RMSE_CUBE:.1f}%", 1, 1, cmap="RdBu_r", lo=-PCT_LIM, hi=PCT_LIM, light=True,
        cblabel=f"difference [% of {REF_S}]")
a = ax[1, 2]
a.axvspan(vel[L[0]] - DV / 2, vel[L[-1]] + DV / 2, color=MUTED, alpha=0.1, lw=0)
a.step(vel, spec["cmp"], where="mid", color=BLUE, lw=2, label=f"compact deconvolved: {flux['cmp']:.0f}")
a.step(vel, spec["cln"], where="mid", color=GREEN, lw=2, label=f"{REF} model: {flux['cln']:.0f}")
a.step(vel, spec["cmp"] - spec["cln"], where="mid", color=BLUE, lw=1.5, ls="--",
       label=f"compact deconv. − {REF_S}: {flux['cmp'] - flux['cln']:+.0f}")
a.axhline(0, color=MUTED, lw=0.8)
a.set_xlabel(r"velocity [km s$^{-1}$]", fontsize=LAB40, labelpad=8)
a.set_ylabel(f"flux density {IN} [Jy]", fontsize=LAB40, labelpad=10)
a.yaxis.tick_right(); a.yaxis.set_label_position("right")
a.tick_params(which="both", direction="in", top=True, bottom=True, left=True, right=True, labelleft=False, labelright=True)
a.tick_params(which="major", labelsize=TICK, pad=6, length=7); a.tick_params(which="minor", length=4)
a.minorticks_on()
a.set_ylim(top=1.8 * max(spec["cmp"].max(), spec["cln"].max()))
textbox40(a, "Integrated spectra", light=True)
a.legend(frameon=False, fontsize=13, loc="upper right", bbox_to_anchor=(0.995, 0.86), title="Jy km/s", title_fontsize=13)
fig.text(0.5, 1 - 0.32 / FH, LABEL, ha="center", va="top", fontsize=28, fontweight="bold")
save(fig, "40_overview.png" if COMPARE == "extended" else "_skip", "Moment 0 (Jy arcsec⁻² km/s, all × mask) of the compact configuration's MS-CLEAN model at its clean beam, the "
     "compact deconvolved model at λ/B_max and at the extended clean beam, the extended MS-CLEAN model at the extended clean beam, "
     "compact − extended CLEAN (full ± range), and the integrated spectra inside the mask.")

# 40b: the same, everything compared at the compact array's lambda/B_max (circular): compact MS-CLEAN; compact
# deconvolved and extended MS-CLEAN components, both * that beam (one scale); extended MS-CLEAN at its own clean beam;
# their difference at lambda/B_max,compact; spectra. All x the same mask.
SB["clnc"] = m0(convolve_with_beam(x_cln, CELL, *B2)) / omega(B2)
SB["dc"] = SB["hic"] - SB["clnc"]
fig = plt.figure(figsize=(FW, FH))
ax = np.array([[inch(PX(c), PY(r), S, S) for c in range(3)] for r in range(2)])
vc = max(np.percentile(SB["hic"][FOOT], 99.9), np.percentile(SB["clnc"][FOOT], 99.9))
lbl = "λ/B$_{\\rm max,\\,compact}$ beam"
panel40(ax[0, 0], SB["cc"], BC, "Compact MS-CLEAN", 0, 0, hi=np.percentile(SB["cc"][FOOT], 99.9), outline=False)
panel40(ax[0, 1], SB["hic"], B2, "Compact deconvolved\n∗ " + lbl, 0, 1, hi=vc)
SB["cln_ext"] = m0(convolve_with_beam(x_cln, CELL, *BE_EXT)) / omega(BE_EXT)
panel40(ax[0, 2], SB["cln_ext"], BE_EXT, "Extended MS-CLEAN\n∗ extended clean beam", 0, 2,
        hi=np.percentile(SB["cln_ext"][FOOT], 99.9))
panel40(ax[1, 0], SB["clnc"], B2, "Extended MS-CLEAN\n∗ " + lbl, 1, 0, hi=vc)
rms_c = RMSE_CUBE if COMPARE == "lbmax" else float("nan")   # 40b is only saved in lbmax mode, where r_* are * B2
stats["lbmax_compact_comparison"] = dict(beam=B2, rms_pct=float(rms_c), total_pct=float(100 * (flux["cmp"] - flux["cln"]) / flux["cln"]))
panel40(ax[1, 1], pct_diff(SB["hic"], SB["clnc"]), B2, "Difference / ext. MS-CLEAN\n"
        f"total error {100 * (flux['cmp'] - flux['cln']) / flux['cln']:+.1f}%, RMSE (cube) {rms_c:.1f}%", 1, 1, cmap="RdBu_r",
        lo=-PCT_LIM, hi=PCT_LIM, light=True, cblabel="difference [% of ext. MS-CLEAN]")
a = ax[1, 2]
a.axvspan(vel[L[0]] - DV / 2, vel[L[-1]] + DV / 2, color=MUTED, alpha=0.1, lw=0)
a.step(vel, spec["cmp"], where="mid", color=BLUE, lw=2, label=f"compact deconvolved: {flux['cmp']:.0f}")
a.step(vel, spec["cln"], where="mid", color=GREEN, lw=2, label=f"extended MS-CLEAN model: {flux['cln']:.0f}")
a.step(vel, spec["cmp"] - spec["cln"], where="mid", color=BLUE, lw=1.5, ls="--",
       label=f"compact deconv. − ext. MS-CLEAN: {flux['cmp'] - flux['cln']:+.0f}")
a.axhline(0, color=MUTED, lw=0.8)
a.set_xlabel(r"velocity [km s$^{-1}$]", fontsize=LAB40, labelpad=8)
a.set_ylabel(f"flux density {IN} [Jy]", fontsize=LAB40, labelpad=10)
a.yaxis.tick_right(); a.yaxis.set_label_position("right")
a.tick_params(which="both", direction="in", top=True, bottom=True, left=True, right=True, labelleft=False, labelright=True)
a.tick_params(which="major", labelsize=TICK, pad=6, length=7); a.tick_params(which="minor", length=4)
a.minorticks_on()
a.set_ylim(top=1.8 * max(spec["cmp"].max(), spec["cln"].max()))
textbox40(a, "Integrated spectra", light=True)
a.legend(frameon=False, fontsize=13, loc="upper right", bbox_to_anchor=(0.995, 0.86), title="Jy km/s", title_fontsize=13)
fig.text(0.5, 1 - 0.32 / FH, LABEL + " at λ/B$_{\\rm max,\\,compact}$", ha="center", va="top", fontsize=28, fontweight="bold")
save(fig, "40_overview.png" if COMPARE == "lbmax" else "_skip", "As 40, compared at the compact array's λ/B_max (circular): compact MS-CLEAN; compact "
     "deconvolved and extended MS-CLEAN both * that beam (one scale); extended MS-CLEAN at its clean beam; their difference; spectra.")

# ======================================================================== 41 zooms
def peaks(img2, n, sep_arcsec):
    """the n brightest local maxima of img2 inside the footprint, at least sep_arcsec apart: [(row, col)]"""
    size = int(sep_arcsec / CELL) | 1
    loc = (img2 == maximum_filter(img2, size=size)) & (img2 > 0.1 * img2.max()) & FOOT
    return sorted(zip(*np.nonzero(loc)), key=lambda p: -img2[p])[:n]


off = lambda j, i: (float(-(i - C0) * CELL), float((j - C0) * CELL))       # (dRA, dDec) of pixel (row, col)
if NAME == "NGC3110":
    BOXES = [("A: nucleus / bar", (-0.4, 0.2), 1.1), ("B: bar, north end", (-0.4, 1.8), 0.8),
             ("C: east arm", (3.0, 1.0), 1.6), ("D: south arm", (-1.8, -4.0), 1.6)]
else:                                               # the four brightest separated CLEAN peaks
    hb_ = max(0.8, min(1.6, HALF / 3.5))
    inside = lambda v: float(np.clip(v, -(N * CELL / 2 - hb_ - 2 * CELL), N * CELL / 2 - hb_ - 2 * CELL))   # box within the field
    BOXES = [(f"{'ABCD'[k]}: peak at ({off(*p)[0]:+.1f}″, {off(*p)[1]:+.1f}″)", (inside(off(*p)[0]), inside(off(*p)[1])), hb_)
             for k, p in enumerate(peaks(M0["cln"], 4, 2 * hb_))]
# same look as figure 40. Left: the full field in three stacked panels (compact MS-CLEAN, compact deconvolved and
# extended MS-CLEAN, each * its clean beam) with the boxes drawn; then one row per box: a colorbar shared by the compact
# and CLEAN zooms (to their left), compact deconvolved, extended MS-CLEAN, their difference with its colorbar (to its
# right), and the box spectra. Square panels on an inch grid, equal gaps; outside the mask grey.
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import HPacker


def make41(common):
    """common=False: each box has its own scale (colorbar left of the compact zoom, the difference's to its right), the
    two overview scales to the left of the overview. common=True: one scale for every inferno map (0 .. the extended
    CLEAN's 99.9th percentile) with one colorbar under the overview and the two zoom columns, and one for the
    difference (+- the same) under the difference column."""
    nb = len(BOXES)
    ZS, ZG = 4.0, 0.45                                # zoom panel side, gap between panels (both directions)
    ZRH = nb * ZS + (nb - 1) * ZG                     # height of the rows
    OS = (ZRH - 2 * ZG) / 3                           # overview panel side: three stacked = the rows' height
    CBW, CBP = 0.22, 0.14                             # colorbar width and its gap to the panel
    if common:
        ZML, ZMR, ZMB, ZMT = 0.3, 1.45, 1.75, 2.0
        XO = ZML
        XC = [XO + OS + 0.75 + k * (ZS + ZG) for k in range(3)]
        XSP = XC[2] + ZS + ZG
    else:
        ZML, ZMR, ZMB, ZMT = 0.3, 1.45, 0.95, 2.0
        XOC = ZML + 1.3                               # overview colorbars (tick labels and units to their left)
        XO = XOC + CBW + CBP
        XCL = XO + OS + 1.35                          # left colorbar (its tick labels sit in the gap to its left)
        XC = [XCL + CBW + CBP + k * (ZS + ZG) for k in range(3)]   # compact, CLEAN, difference
        XCR = XC[2] + ZS + CBP                        # difference colorbar
        XSP = XCR + CBW + 1.55                        # spectra (after the colorbar's tick labels and label)
    ZW, ZH = XSP + ZS + ZMR, ZMB + ZRH + ZMT
    fig = plt.figure(figsize=(ZW, ZH))
    zin = lambda x, y, w, h: fig.add_axes([x / ZW, y / ZH, w / ZW, h / ZH])
    ZY = lambda r: ZMB + ZRH - ZS - r * (ZS + ZG)     # bottom of row r
    OY = lambda k: ZMB + ZRH - OS - k * (OS + ZG)     # bottom of overview panel k


    def scale_beam(a, cen, half, beam, ink, fill, zoom=False):
        """scale bar (top right; zooms: bottom right, ~25% of the width) and the beam (bottom left)"""
        u = half / 5.5                                    # offsets as figure 40, scaled to the panel
        if zoom:
            sbl = min((0.25, 0.5, 1.0, 2.0), key=lambda v: abs(v - 0.5 * half))
            x0, y0 = cen[0] - half + 0.6 * u, cen[1] - half + 0.75 * u  # RA runs right to left: x0 is the right end
            a.plot([x0 + sbl, x0], [y0, y0], color=ink, lw=3, solid_capstyle="butt")
            a.text(x0 + sbl / 2, y0 + 0.15 * u, f"{sbl:g}″", color=ink, ha="center", va="bottom", fontsize=20)
        else:
            sbl = min((0.2, 0.25, 0.5, 1.0, 2.0), key=lambda v: abs(v - 0.3 * half))   # ~15% of the width
            x0, y0 = cen[0] - half + 0.6 * u, cen[1] + half - 0.7 * u
            a.plot([x0 + sbl, x0], [y0, y0], color=ink, lw=3, solid_capstyle="butt")
            a.text(x0 + sbl / 2, y0 - 0.3 * u, f"{sbl:g}″", color=ink, ha="center", va="top", fontsize=20)
        bx, by = cen[0] + half - 0.35 * u - beam[0] / 2, cen[1] - half + 0.35 * u + beam[0] / 2
        if not zoom:                                  # zoom boxes: no beam; full-field panels: the beam and its size
            a.add_patch(Ellipse((bx, by), width=beam[0], height=beam[1], angle=90 - beam[2], facecolor=fill,
                                edgecolor=ink, lw=1.2))                                  # full-field panels: the beam's size beside it
            f = ".3f" if beam[0] < 1 else ".2f"
            a.text(bx - beam[0] / 2 - 0.2 * u, by, f"{beam[0]:{f}}″ (circular)" if beam[0] == beam[1] else
                   f"{beam[0]:{f}}″ × {beam[1]:{f}}″ @ {beam[2]:.0f}°", color=ink,
                   ha="left", va="center", fontsize=16, zorder=6)


    def zpanel(a, img2, cen, half, beam, hi, lo=0.0, cmap="inferno", light=False, text=None, mask=True, zoom=True):
        """img2 (surface brightness) centred on cen = (dRA, dDec), half-size `half` arcsec"""
        ci, cj, hp = int(round(C0 - cen[0] / CELL)), int(round(C0 + cen[1] / CELL)), int(round(half / CELL))
        sl = np.s_[cj - hp:cj + hp, ci - hp:ci + hp]
        ext = [cen[0] + half, cen[0] - half, cen[1] - half, cen[1] + half]
        cm = plt.get_cmap(cmap).copy(); cm.set_bad("0.93" if light else "0.12")
        im = a.imshow((np.where(FOOT, img2, np.nan) if mask else img2)[sl], origin="lower", extent=ext, cmap=cm, vmin=lo,
                      vmax=hi, interpolation="nearest")
        if mask:
            a.contour(FOOT[sl].astype(float), levels=[0.5], colors="0.55", linewidths=0.6, extent=ext, origin="lower")
        a.set_xticks([]); a.set_yticks([])
        scale_beam(a, cen, half, beam, "k" if light else "w", "0.15" if light else "w", zoom=zoom)
        if text:
            textbox40(a, text, light)
        return im


    def vbar(im, x, r, side, label, y=None, h=None):
        cax = zin(x, ZY(r) if y is None else y, CBW, ZS if h is None else h)
        cb = fig.colorbar(im, cax=cax, orientation="vertical")
        cax.yaxis.set_ticks_position(side); cax.yaxis.set_label_position(side)
        cax.tick_params(labelsize=TICK, pad=6, direction="out", length=6)
        cb.locator = matplotlib.ticker.MaxNLocator(5); cb.update_ticks()
        if label:
            cb.set_label(SBU, fontsize=LAB40, labelpad=10)


    # ---- overview column
    VC = ve                                           # the common scale: 0 .. the extended CLEAN's 99.9th percentile
    for k, (img2, beam, t, msk, hi) in enumerate(((SB["cc"], BC, "Compact MS-CLEAN", False,
                                                   VC if common else np.percentile(SB["cc"][FOOT], 99.9)),
                                                  (SB["cmp"], BE, "Compact deconvolved\n∗ " + BN, True, ve),
                                                  (SB["cln"], BE, "Extended MS-CLEAN", True, ve))):
        a = zin(XO, OY(k), OS, OS)
        imo = zpanel(a, img2, (0.0, 0.0), HALF, beam, hi, text=t, mask=msk, zoom=False)
        if common:
            ov_im = imo
        elif k == 0:                                      # compact MS-CLEAN: its own colorbar
            vbar(imo, XOC, None, "left", True, y=OY(0), h=OS)
        elif k == 2:                                      # the two at the extended beam share one, spanning both
            vbar(imo, XOC, None, "left", True, y=OY(2), h=2 * OS + ZG)
        for name, cen, half in BOXES:
            a.add_patch(Rectangle((cen[0] - half, cen[1] - half), 2 * half, 2 * half, fill=False, edgecolor="cyan", lw=1.6))
            a.text(cen[0] + half, cen[1] + half + 0.08, name.split(":")[0], color="cyan", ha="left", va="bottom",
                   fontsize=15, fontweight="bold", clip_on=True)
        a.set_xlim(HALF, -HALF); a.set_ylim(-HALF, HALF)   # boxes reaching past the field must not widen the axes

    # ---- zoom rows
    ZHEAD = [(XO, "Full field of view"), (XC[0], "Compact deconvolved\n∗ " + BN), (XC[1], "Extended MS-CLEAN\n∗ " + BN),
             (XC[2], "Difference\ncompact deconv. − ext. MS-CLEAN"), (XSP, "Box spectra")]
    ZHY = ZH - ZMT + 0.85
    for x, t in ZHEAD:                                    # first line bold; second line 1 pt smaller, regular
        head, *rest = t.split("\n")
        w_ = OS if x == XO else ZS
        fig.text((x + w_ / 2) / ZW, ZHY / ZH, head, ha="center", va="top", fontsize=18, fontweight="bold")
        for k, line in enumerate(rest):
            fig.text((x + w_ / 2) / ZW, (ZHY - 0.33 * (k + 1)) / ZH, line, ha="center", va="top", fontsize=17)
    for r, (name, cen, half) in enumerate(BOXES):
        ci, cj, hp = int(round(C0 - cen[0] / CELL)), int(round(C0 + cen[1] / CELL)), int(round(half / CELL))
        sl = np.s_[cj - hp:cj + hp, ci - hp:ci + hp]
        hi = VC if common else max(SB["cmp"][sl][FOOT[sl]].max(), SB["cln"][sl][FOOT[sl]].max())
        im = zpanel(zin(XC[0], ZY(r), ZS, ZS), SB["cmp"], cen, half, BE, hi, text=name)
        zpanel(zin(XC[1], ZY(r), ZS, ZS), SB["cln"], cen, half, BE, hi)
        bx3 = np.zeros((N, N), bool); bx3[sl] = True
        fc3, fl3 = x_cmp[L][:, bx3].sum() * DV, x_cln[L][:, bx3].sum() * DV          # box fluxes (3D mask)
        mv3 = M[L] & bx3[None]                                                      # masked line voxels in the box
        rms3 = 100 * np.linalg.norm((r_cmp[L] - r_cln[L])[mv3]) / np.linalg.norm(r_cln[L][mv3])   # whole-cube RMSE
        imd = zpanel(zin(XC[2], ZY(r), ZS, ZS), SB["d"], cen, half, BE, hi, lo=-hi, cmap="RdBu_r", light=True,
                     text=f"{name.split(':')[0]}\ntotal error {100 * (fc3 - fl3) / fl3:+.1f}%\nRMSE (cube) {rms3:.1f}%")
        stats.setdefault("boxes_err", {})[name] = dict(total_pct=float(100 * (fc3 - fl3) / fl3), rms_pct=float(rms3))
        if not common:
            vbar(im, XCL, r, "left", True)
            vbar(imd, XCR, r, "right", True)
        a = zin(XSP, ZY(r), ZS, ZS)
        box_ = np.zeros((N, N), bool); box_[sl] = True
        sc, sl_c = x_cmp[:, box_].sum(1), x_cln[:, box_].sum(1)
        a.axvspan(vel[L[0]] - DV / 2, vel[L[-1]] + DV / 2, color=MUTED, alpha=0.1, lw=0)
        a.step(vel, sc, where="mid", color=BLUE, lw=2, label="compact deconvolved")
        a.step(vel, sl_c, where="mid", color=GREEN, lw=2, label="extended MS-CLEAN")
        a.step(vel, sc - sl_c, where="mid", color=BLUE, lw=1.5, ls="--", label="difference")
        a.axhline(0, color=MUTED, lw=0.8)
        a.yaxis.tick_right(); a.yaxis.set_label_position("right")
        a.tick_params(which="both", direction="in", top=True, bottom=True, left=True, right=True, labelleft=False, labelright=True)
        a.tick_params(which="major", labelsize=TICK, pad=6, length=7); a.tick_params(which="minor", length=4)
        a.minorticks_on(); a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        a.set_ylabel("flux density in the box [Jy]", fontsize=LAB40, labelpad=10)
        if r == nb - 1:
            a.set_xlabel(r"velocity [km s$^{-1}$]", fontsize=LAB40, labelpad=8)
        else:
            a.tick_params(labelbottom=False)
        a.set_ylim(top=1.55 * max(sc.max(), sl_c.max()))
        fc, fl = x_cmp[L][:, box_].sum() * DV, x_cln[L][:, box_].sum() * DV
        from matplotlib.offsetbox import HPacker
        parts = [(f"{fc:.1f}", dict(color=BLUE, fontweight="bold")), (" vs ", {}), (f"{fl:.1f}", dict(color=GREEN, fontweight="bold")),
                 (f" Jy km/s ({fc / fl:.2f}×)", {})]
        a.add_artist(AnchoredOffsetbox(loc="upper left", frameon=False, pad=0.2, borderpad=0.4, child=HPacker(
            children=[TextArea(t, textprops=dict(fontsize=15, **kw)) for t, kw in parts], align="baseline", pad=0, sep=0)))
        if r == 0:
            a.legend(frameon=False, fontsize=13, loc="upper right", bbox_to_anchor=(0.995, 0.88))
        stats.setdefault("boxes", {})[name] = dict(center=cen, half=half, flux_cmp=float(fc), flux_cln=float(fl))
    fig.text(0.5, 1 - 0.3 / ZH, f"{LABEL}: zoomed boxes" + (" [common scale]" if common else ""), ha="center", va="top", fontsize=28, fontweight="bold")
    if common:                                        # horizontal colorbars under the bottom row
        for im_, x_, w_ in ((ov_im, XO, XC[1] + ZS - XO), (imd, XC[2], ZS)):
            cax = zin(x_, ZY(nb - 1) - CBP - 0.18 - CBW, w_, CBW)
            cb = fig.colorbar(im_, cax=cax, orientation="horizontal")
            cax.tick_params(labelsize=TICK, pad=6, direction="out", length=6)
            cb.locator = matplotlib.ticker.MaxNLocator(9 if w_ > ZS else 5); cb.update_ticks()
            cb.set_label(SBU, fontsize=LAB40, labelpad=8)
    out = "41_zooms_common_scale.png" if common else "41_zooms.png"
    save(fig, out, "Full field (compact MS-CLEAN, compact deconvolved and extended MS-CLEAN) with four boxes (NGC 3110: "
         "nucleus/bar, north end of the bar, east arm, south arm; else the four brightest separated CLEAN peaks), then per "
         "box compact deconvolved, extended MS-CLEAN, their difference and the box spectra; "
         + ("one common scale for all maps." if common else "a scale per box."))


make41(False)
make41(True)

# ======================================================================== 42 channel maps
# four line channels as columns; rows compact deconvolved, extended MS-CLEAN (both * extended clean beam) and their
# difference, in surface brightness. Each channel's own mask (grey outside); a scale per column: one colorbar above for
# the two maps, one below for the difference (+- the same). Square panels on an inch grid with equal gaps.
NCH42 = 5
chans = [L[int(round(t))] for t in np.linspace(0.1, 0.9, NCH42) * (len(L) - 1)]
CS, CG, CBH2, CBP2 = 4.4, 0.45, 0.22, 0.16
CML, CMR, CMB, CMT = 1.0, 1.55, 1.45, 2.75
CSPW = CS                                             # width of the sideways spectrum (its height: the three rows)
CW, CH = CML + NCH42 * CS + (NCH42 - 1) * CG + CG + CSPW + CMR, CMB + 3 * CS + 2 * CG + CMT
fig = plt.figure(figsize=(CW, CH))
cin = lambda x, y, w, h: fig.add_axes([x / CW, y / CH, w / CW, h / CH])
CX = lambda c: CML + c * (CS + CG)
CY = lambda r: CMB + (2 - r) * (CS + CG)
om = omega(BE)


def cpanel(a, img2, m2, hi, lo=0.0, cmap="inferno", light=False, scalebar=False):
    hp = int(round(HALF / CELL)); sl = np.s_[C0 - hp:C0 + hp, C0 - hp:C0 + hp]; ext = [HALF, -HALF, -HALF, HALF]
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("0.93" if light else "0.12")
    im = a.imshow(np.where(m2, img2, np.nan)[sl], origin="lower", extent=ext, cmap=cm, vmin=lo, vmax=hi,
                  interpolation="nearest")
    a.contour(m2[sl].astype(float), levels=[0.5], colors="0.55", linewidths=0.6, extent=ext, origin="lower")
    a.set_xticks([]); a.set_yticks([])
    ink = "k" if light else "w"; u = HALF / 5.5
    if scalebar:
        sbl = 1.0 if HALF < 7 else 2.0
        x0, y0 = -HALF + 0.6 * u, HALF - 0.7 * u
        a.plot([x0 + sbl, x0], [y0, y0], color=ink, lw=3, solid_capstyle="butt")
        a.text(x0 + sbl / 2, y0 - 0.3 * u, f"{sbl:g}″", color=ink, ha="center", va="top", fontsize=20)
    bx, by = HALF - 0.35 * u - BE[0] / 2, -HALF + 0.35 * u + BE[0] / 2
    a.add_patch(Ellipse((bx, by), width=BE[0], height=BE[1], angle=90 - BE[2], facecolor="0.15" if light else "w",
                        edgecolor=ink, lw=1.2))
    return im


def hbar(im, x, y, top):
    cax = cin(x, y, CS, CBH2)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    if top:
        cax.xaxis.set_ticks_position("top"); cax.xaxis.set_label_position("top")
    cax.tick_params(labelsize=TICK, pad=6, direction="out", length=6)
    cb.locator = matplotlib.ticker.MaxNLocator(5); cb.update_ticks()
    cb.set_label(r"Jy arcsec$^{-2}$", fontsize=LAB40, labelpad=8)


for k, c in enumerate(chans):
    m2 = M[c]
    sc, sl_ = r_cmp[c] / om, r_cln[c] / om                # Jy/beam -> Jy arcsec^-2
    hi = max(np.percentile(sc[m2], 99.9), np.percentile(sl_[m2], 99.9)) if m2.any() else None
    im = cpanel(cin(CX(k), CY(0), CS, CS), sc, m2, hi, scalebar=(k == 0))
    cpanel(cin(CX(k), CY(1), CS, CS), sl_, m2, hi)
    ad = cin(CX(k), CY(2), CS, CS)
    imd = cpanel(ad, sc - sl_, m2, hi, lo=-hi, cmap="RdBu_r", light=True)
    fcmp, fcln = float(x_cmp[c][m2].sum()), float(x_cln[c][m2].sum())     # channel flux in the mask, Jy
    from matplotlib.offsetbox import HPacker
    row1 = [(f"{fcmp:.3f}", dict(color=BLUE, fontweight="bold")), (" vs ", {}),
            (f"{fcln:.3f}", dict(color=GREEN, fontweight="bold")), (" Jy", {})]
    row2 = [("Δ = ", {}), (f"{fcmp - fcln:+.3f} Jy ({100 * (fcmp - fcln) / fcln:+.1f}%)", dict(fontweight="bold"))]
    box_ = VPacker(children=[HPacker(children=[TextArea(t, textprops=dict(fontsize=15, **kw)) for t, kw in row],
                                     align="baseline", pad=0, sep=0) for row in (row1, row2)], align="left", pad=0, sep=4)
    ab = AnchoredOffsetbox(loc="upper left", child=box_, pad=0.45, borderpad=0.6, frameon=True)
    ab.patch.set_boxstyle("round,pad=0.3"); ab.patch.set(facecolor="white", edgecolor="0.6", alpha=0.9, linewidth=1.2)
    ad.add_artist(ab)
    stats.setdefault("channels", {})[f"{vel[c]:.0f}"] = dict(channel=int(c), flux_cmp_Jy=fcmp, flux_cln_Jy=fcln)
    hbar(im, CX(k), CY(0) + CS + CBP2, True)
    hbar(imd, CX(k), CY(2) - CBP2 - CBH2, False)
    fig.text((CX(k) + CS / 2) / CW, (CY(0) + CS + 1.3) / CH, f"{vel[c]:.0f} km s$^{{-1}}$", ha="center", va="bottom",
             fontsize=20, fontweight="bold")
# sideways integrated spectra (flux along x, velocity up), as tall as the three rows; the channels shown marked
a = cin(CX(NCH42), CY(2), CSPW, 3 * CS + 2 * CG)
edges = np.append(vel - DV / 2, vel[-1] + DV / 2)
a.axhspan(vel[L[0]] - DV / 2, vel[L[-1]] + DV / 2, color=MUTED, alpha=0.1, lw=0)
a.stairs(x_cmp.sum((1, 2)), edges, orientation="horizontal", color=BLUE, lw=2, label="compact deconvolved")
a.stairs(x_cln.sum((1, 2)), edges, orientation="horizontal", color=GREEN, lw=2, label="extended MS-CLEAN")
a.axvline(0, color=MUTED, lw=0.8)
for c in chans:
    a.axhline(vel[c], color="0.15", lw=1.4, ls="--", zorder=0)
a.set_ylim(edges[0], edges[-1])
a.set_xlim(right=1.5 * max(x_cmp.sum((1, 2)).max(), x_cln.sum((1, 2)).max()))
a.yaxis.tick_right(); a.yaxis.set_label_position("right")
a.tick_params(which="both", direction="in", top=True, bottom=True, left=True, right=True, labelleft=False, labelright=True)
a.tick_params(which="major", labelsize=TICK, pad=6, length=7); a.tick_params(which="minor", length=4)
a.minorticks_on(); a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
a.set_xlabel(f"flux density {IN} [Jy]", fontsize=LAB40, labelpad=8)
a.set_ylabel(r"velocity [km s$^{-1}$]", fontsize=LAB40, labelpad=10)
a.legend(frameon=False, fontsize=13, loc="upper right")
fig.text((CX(NCH42) + CSPW / 2) / CW, (CY(0) + CS + 1.3) / CH, "Integrated spectra", ha="center", va="bottom",
         fontsize=20, fontweight="bold")
for r, t in enumerate(("Compact deconvolved", "Extended MS-CLEAN", "Difference")):
    fig.text((CML - 0.35) / CW, (CY(r) + CS / 2) / CH, t, rotation=90, ha="center", va="center", fontsize=20,
             fontweight="bold")
fig.text(0.5, 1 - 0.3 / CH, f"{LABEL}: channel maps", ha="center", va="top", fontsize=28, fontweight="bold")
save(fig, "42_channels.png", "Five line channels: compact deconvolved and extended MS-CLEAN (both * extended clean beam, "
     "Jy arcsec⁻², a common scale per channel) and their difference (± the same), each inside that channel's mask; "
     "the integrated spectra sideways on the right, the channels shown marked.")

# ======================================================================== 43 cuts
if NAME == "NGC3110":
    CUTS = [("bar, north-south at ΔRA = −0.40″", (-0.40, -2.0), (-0.40, 2.6)),
            ("east arm, north-south at ΔRA = +3.0″", (3.0, -1.0), (3.0, 3.0)),
            ("south arm, east-west at ΔDec = −4.0″", (1.0, -4.0), (-4.5, -4.0))]
else:                                               # through the brightest CLEAN peak, and N-S through the second
    pk2 = peaks(M0["cln"], 2, 1.5)
    (x1, y1) = off(*pk2[0]); (x2, y2) = off(*pk2[-1]); hc_ = 0.9 * HALF
    CUTS = [(f"north-south at ΔRA = {x1:+.2f}″", (x1, -hc_), (x1, hc_)),
            (f"east-west at ΔDec = {y1:+.2f}″", (hc_, y1), (-hc_, y1)),
            (f"north-south at ΔRA = {x2:+.2f}″", (x2, -hc_), (x2, hc_))]
fig = plt.figure(figsize=(26, 6.4), constrained_layout=True)
gs = fig.add_gridspec(1, 4)
a = fig.add_subplot(gs[0, 0])
show(a, M0["cln"], HALF, "CLEAN, with the cuts", hi=vm)
for n, p0, p1 in CUTS:
    a.plot([p0[0], p1[0]], [p0[1], p1[1]], color="cyan", lw=1.5)
for k, (n, p0, p1) in enumerate(CUTS):
    a = fig.add_subplot(gs[0, k + 1])
    npt = int(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / CELL)
    dxs, dys = np.linspace(p0[0], p1[0], npt), np.linspace(p0[1], p1[1], npt)
    rows, cols = C0 + dys / CELL, C0 - dxs / CELL
    s = np.hypot(dxs - p0[0], dys - p0[1])
    for key, col, ls in (("cmp", BLUE, "-"), ("cln", GREEN, "-")):
        a.plot(s, map_coordinates(M0[key], [rows, cols], order=1), color=col, lw=2 if ls == "-" else 1.5, ls=ls,
               label=LAB[key])
    a.axhline(0, color=MUTED, lw=0.8); a.set_xlabel("distance along the cut [″]"); a.set_ylabel(UNIT)
    a.set_title(n, fontsize=12); a.grid(alpha=0.25)
    if k == 0:
        a.legend(frameon=False, fontsize=10)
fig.suptitle(f"{TOP}: moment-0 brightness cuts", fontsize=14)
save(fig, "43_profiles.png", "Moment-0 cuts (NGC 3110: along the bar, across the east and south arms; else through the "
     "brightest CLEAN peaks): compact, CLEAN components.")

# ======================================================================== 44 moment 1
def mom1(cube, m0map, frac=0.05):
    w = np.clip(cube[L], 0, None)
    s = w.sum(0)
    v1 = (w * vel[L][:, None, None]).sum(0) / np.maximum(s, 1e-30)
    return np.where((m0map > frac * vm) & FOOT, v1, np.nan)


V1 = dict(cmp=mom1(r_cmp, M0["cmp"]), cln=mom1(r_cln, M0["cln"]))
dv1 = V1["cmp"] - V1["cln"]
ok = np.isfinite(dv1)
stats["mom1"] = dict(rms_diff_kms=float(np.sqrt(np.nanmean(dv1 ** 2))), median_diff_kms=float(np.nanmedian(dv1)),
                     pixels=int(ok.sum()))
# same look as figures 40-42: three square panels on an inch grid, no ticks, grey outside the shown pixels (moment 0 above
# 5% of CLEAN's peak, inside the mask); one colorbar above the two velocity fields, one above the difference
MS_, MG = 5.2, 0.45
MML, MMR, MMB, MMT = 0.3, 0.3, 0.45, 2.05
MW, MH = MML + 3 * MS_ + 2 * MG + MMR, MMB + MS_ + MMT
fig = plt.figure(figsize=(MW, MH))
min_ = lambda x, y, w, h: fig.add_axes([x / MW, y / MH, w / MW, h / MH])
MX = lambda c: MML + c * (MS_ + MG)
lo1, hi1 = np.nanpercentile(V1["cln"], 1), np.nanpercentile(V1["cln"], 99)
dlim = 50.0


def mpanel(a, img2, lo, hi, cmap, bad, ink, fill, scalebar=False):
    hp = int(round(HALF / CELL)); sl = np.s_[C0 - hp:C0 + hp, C0 - hp:C0 + hp]; ext = [HALF, -HALF, -HALF, HALF]
    cm = plt.get_cmap(cmap).copy(); cm.set_bad(bad)
    im = a.imshow(img2[sl], origin="lower", extent=ext, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
    a.contour(FOOT[sl].astype(float), levels=[0.5], colors="0.55", linewidths=0.6, extent=ext, origin="lower")
    a.set_xticks([]); a.set_yticks([])
    u = HALF / 5.5
    if scalebar:
        sbl = 1.0 if HALF < 7 else 2.0
        x0, y0 = -HALF + 0.6 * u, HALF - 0.7 * u
        a.plot([x0 + sbl, x0], [y0, y0], color=ink, lw=3, solid_capstyle="butt")
        a.text(x0 + sbl / 2, y0 - 0.3 * u, f"{sbl:g}″", color=ink, ha="center", va="top", fontsize=20)
    bx, by = HALF - 0.35 * u - BE[0] / 2, -HALF + 0.35 * u + BE[0] / 2
    a.add_patch(Ellipse((bx, by), width=BE[0], height=BE[1], angle=90 - BE[2], facecolor=fill, edgecolor=ink, lw=1.2))
    if scalebar:                                      # first panel: the beam's size beside it
        a.text(bx - BE[0] / 2 - 0.2 * u, by, f"{BE[0]:.3f}″ (circular)" if BE[0] == BE[1] else
               f"{BE[0]:.3f}″ × {BE[1]:.3f}″ @ {BE[2]:.0f}°", color=ink, ha="left",
               va="center", fontsize=16, bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.7))
    return im


a0_ = min_(MX(0), MMB, MS_, MS_); a1_ = min_(MX(1), MMB, MS_, MS_); a2_ = min_(MX(2), MMB, MS_, MS_)
im = mpanel(a0_, V1["cmp"], lo1, hi1, "RdBu_r", "0.93", "k", "0.15", scalebar=True)
mpanel(a1_, V1["cln"], lo1, hi1, "RdBu_r", "0.93", "k", "0.15")
imd = mpanel(a2_, dv1, -dlim, dlim, "PuOr_r", "0.93", "k", "0.15")
textbox40(a0_, "Compact deconvolved\n∗ " + BN, light=True)
textbox40(a1_, "Extended MS-CLEAN\n∗ " + BN, light=True)
textbox40(a2_, f"Difference\nrms {stats['mom1']['rms_diff_kms']:.1f}, median {stats['mom1']['median_diff_kms']:+.1f} km s$^{{-1}}$",
          light=True)
for im_, x_, w_ in ((im, MX(0), 2 * MS_ + MG), (imd, MX(2), MS_)):
    cax = min_(x_, MMB + MS_ + 0.16, w_, 0.22)
    cb = fig.colorbar(im_, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("top"); cax.xaxis.set_label_position("top")
    cax.tick_params(labelsize=TICK, pad=6, direction="out", length=6)
    cb.locator = matplotlib.ticker.MaxNLocator(9 if w_ > MS_ else 5); cb.update_ticks()
    cb.set_label(r"velocity [km s$^{-1}$]" if im_ is im else r"Δ velocity [km s$^{-1}$]", fontsize=LAB40, labelpad=8)
fig.text(0.5, 1 - 0.3 / MH, f"{LABEL}: moment 1", ha="center", va="top", fontsize=28, fontweight="bold")
save(fig, "44_moment1.png", "Intensity-weighted velocity of compact and CLEAN (where moment 0 > 5% of the peak, inside the mask; grey elsewhere) and their difference (±50 km/s).")

# ======================================================================== 45 pixel-by-pixel
# two square scatter panels (moment 0 per pixel; every masked voxel), colorbars above, the second with its y axis on the
# right; underneath, spanning both, a short panel of the median compact / CLEAN ratio against CLEAN's brightness.
# Surface brightness throughout (moment 0: Jy arcsec^-2 km/s, voxels: Jy arcsec^-2), both at the extended clean beam.
xa, ya = SB["cln"][FOOT], SB["cmp"][FOOT]
vm3 = M[L] & (r_cln[L] > 0)
xv, yv = r_cln[L][vm3] / omega(BE), r_cmp[L][vm3] / omega(BE)
sl_ = np.sum(xa * ya) / np.sum(xa * xa)
rv = float(np.corrcoef(xv, yv)[0, 1]); stats["voxel_pearson"] = rv
stats["mom0_slope_through_0"] = float(sl_)
slv = np.sum(xv * yv) / np.sum(xv * xv)               # least squares through 0, and R^2 of that fit
r2 = lambda x_, y_, b_: float(1 - np.sum((y_ - b_ * x_) ** 2) / np.sum((y_ - y_.mean()) ** 2))
R2a, R2v = r2(xa, ya, sl_), r2(xv, yv, slv)
# percentage error with respect to the extended MS-CLEAN: of the total (flux), and rms over the points (relative rms)
pct_tot = lambda x_, y_: float(100 * (y_.sum() - x_.sum()) / x_.sum())
pct_rms = lambda x_, y_: float(100 * np.linalg.norm(y_ - x_) / np.linalg.norm(x_))
E = dict(mom0_total_pct=pct_tot(xa, ya), mom0_rms_pct=pct_rms(xa, ya), voxel_total_pct=pct_tot(xv, yv), voxel_rms_pct=pct_rms(xv, yv))
stats.update(mom0_R2=R2a, voxel_slope_through_0=float(slv), voxel_R2=R2v, pct_error_wrt_clean=E)
PS, PG, PRH = 5.2, 0.45, 1.9                          # scatter side, gap, ratio-panel height
PML, PMR, PMB, PMT = 1.45, 1.45, 1.0, 2.2
PXG = 0.0                                             # scatter and ratio panel touch: one shared x axis line
PW, PH = PML + 2 * PS + PG + PMR, PMB + PRH + PXG + PS + PMT
fig = plt.figure(figsize=(PW, PH))
pin = lambda x, y, w, h: fig.add_axes([x / PW, y / PH, w / PW, h / PH])
PY0 = PMB + PRH + PXG                                 # bottom of the scatter panels


def style(a, right=False):
    a.tick_params(which="both", direction="in", top=True, bottom=True, left=True, right=True)
    a.tick_params(which="major", labelsize=TICK, pad=6, length=7); a.tick_params(which="minor", length=4)
    a.minorticks_on()
    if right:
        a.yaxis.tick_right(); a.yaxis.set_label_position("right")
        a.tick_params(which="both", left=True, labelleft=False, labelright=True)


SBM = r"Jy arcsec$^{-2}$ km s$^{-1}$"
SCAT, LIMS = [], []
for k, (xs, ys, lab_x, lab_y, cbl, head) in enumerate((
        (xa, ya, f"extended MS-CLEAN [{SBM}]", f"compact deconvolved [{SBM}]", r"$N_{\rm pixels}$",
         f"Moment 0 per pixel\n$R^2$ = {R2a:.3f}, slope {sl_:.3f}\n"
         f"total error {E['mom0_total_pct']:+.1f}%"),
        (xv, yv, r"extended MS-CLEAN [Jy arcsec$^{-2}$]", r"compact deconvolved [Jy arcsec$^{-2}$]", r"$N_{\rm voxels}$",
         f"Every masked voxel\n$R^2$ = {R2v:.3f}, slope {slv:.3f}\n"
         f"total error {E['voxel_total_pct']:+.1f}%"))):
    a = pin(PML + k * (PS + PG), PY0, PS, PS)
    SCAT.append(a)
    hb = a.hexbin(xs, ys, gridsize=70, bins="log", cmap="viridis", mincnt=1)
    lim = max(xs.max(), ys.max())
    a.plot([0, lim], [0, lim], color="k", lw=1.2, ls="--", label="1:1")
    a.plot([0, lim], [0, (sl_ if k == 0 else slv) * lim], color=BLUE, lw=1.8, label="best fit through 0")
    a.set_xlim(0, lim); a.set_ylim(0, lim); LIMS.append(lim)
    a.set_ylabel(lab_y, fontsize=LAB40, labelpad=10)
    style(a, right=(k == 1))
    a.tick_params(labelbottom=False)                  # x axis shared with the ratio panel below
    a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5)); a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    textbox40(a, head, light=True)
    a.legend(frameon=False, fontsize=13, loc="lower right")
    cax = pin(PML + k * (PS + PG), PY0 + PS + 0.16, PS, 0.22)
    cb = fig.colorbar(hb, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("top"); cax.xaxis.set_label_position("top")
    cax.tick_params(which="both", labelsize=TICK, pad=6, direction="out")
    cb.set_label(cbl, fontsize=LAB40, labelpad=8)
# ratio against brightness: a short panel under each scatter (its own brightness axis), every point faintly behind the
# median per brightness bin
for k, (xs, ys, xl, what) in enumerate(((xa, ya, f"extended MS-CLEAN [{SBM}]", "pixel"),
                                        (xv, yv, r"extended MS-CLEAN [Jy arcsec$^{-2}$]", "voxel"))):
    a = pin(PML + k * (PS + PG), PMB, PS, PRH)
    top = LIMS[k]                                     # same x range as the scatter above
    bins = np.linspace(0, top, 25); mid = 0.5 * (bins[1:] + bins[:-1])   # over the whole x range
    rat = ys / np.maximum(xs, 1e-12)
    med = [np.median(rat[(xs >= b0) & (xs < b1)]) if ((xs >= b0) & (xs < b1)).sum() >= 5 else np.nan
           for b0, b1 in zip(bins[:-1], bins[1:])]
    sel = np.random.default_rng(0).choice(xs.size, min(xs.size, 60000), replace=False)
    a.scatter(xs[sel], rat[sel], s=2, color=BLUE, alpha=0.03, lw=0, rasterized=True)
    a.axhline(1, color="k", lw=1.2, ls="--")
    a.plot(mid, med, "-o", color=BLUE, lw=2, ms=6)
    a.set_ylim(0.4, 1.6); a.set_xlim(0, top)
    a.set_xlabel(xl, fontsize=LAB40, labelpad=8)
    a.set_ylabel(r"$I_{\rm compact}\,/\,I_{\rm CLEAN}$", fontsize=LAB40, labelpad=10)
    style(a, right=(k == 1))
    # the shared x axis: one horizontal line with ticks both ways; its "0" (scatter) sits just above it and its "1.5"
    # (ratio) just below, so they don't overlap
    SCAT[k].spines["bottom"].set_visible(False)
    a.spines["top"].set_linewidth(1.0); a.spines["top"].set_color("k"); a.set_zorder(SCAT[k].get_zorder() + 1)
    a.patch.set_alpha(0)
    fig.canvas.draw()
    for t_ in SCAT[k].get_yticklabels():
        if t_.get_text() in ("0", "0.0", "0.00"):
            t_.set_va("bottom")
    for t_ in a.get_yticklabels():
        if t_.get_text() == "1.5":
            t_.set_va("top")
    a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    SCAT[k].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    SCAT[k].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    a.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 1.0, 1.5]))
    for ax_ in (SCAT[k], a):                          # axis titles aligned in x
        ax_.yaxis.set_label_coords(1.17 if k == 1 else -0.17, 0.5)
    a.text(0.02, 0.92, f"median per {what} brightness bin", transform=a.transAxes, ha="left", va="top", fontsize=14,
           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8))
fig.text(0.5, 1 - 0.3 / PH, f"{LABEL}: pixel-by-pixel", ha="center", va="top", fontsize=28, fontweight="bold")
save(fig, "45_pixel_scatter.png", "Compact deconvolved vs extended MS-CLEAN per moment-0 pixel and per voxel (log density; dashed "
     "1:1), and under each the compact/CLEAN ratio against CLEAN's brightness (points faint, median per bin).")

json.dump(stats, open(os.path.join(OUT_DIR, SUB, "stats.json"), "w"), indent=1)
with open(os.path.join(OUT_DIR, SUB, "INDEX.md"), "w") as fh:
    fh.write(f"# {LABEL}: compact deconvolved vs extended MS-CLEAN, compared at "
             + (f"the extended clean beam ({BL})" if COMPARE == "extended" else
                f"the compact array's λ/B_max ({LBMAX_C:.3f}″, circular)") + "\n\n" + "\n".join(INDEX) + "\n")
sys.exit(0)

# ======================================================================== 46 residuals on the extended data
op = G.Op(rd(os.path.join(EXT_CUBE, "dirty_beam_cube.fits")), rd(os.path.join(EXT_CUBE, "dirty_cube.fits")))
res = dict(cmp=op.residual_sky(G.to_t(x_cmp)).cpu().numpy().astype(np.float64),
           cln=op.residual_sky(G.to_t(x_cln)).cpu().numpy().astype(np.float64))
del op
sig = float(c_res[FREE].std())
F3 = np.broadcast_to(FOOT, (len(L), N, N))
stats["ext_residual_mJy_beam"] = {k: dict(rms_line=1e3 * float(r[L].std()), rms_line_in_footprint=1e3 * float(r[L][F3].std()),
                                          rms_line_in_mask=1e3 * float(r[L][M[L]].std()),
                                          mean_line_in_mask=1e3 * float(r[L][M[L]].mean()),
                                          rms_per_channel=[round(1e3 * float(r[c].std()), 4) for c in range(NCH)],
                                          rms_per_channel_in_footprint=[round(1e3 * float(r[c][FOOT].std()), 4) for c in range(NCH)])
                                  for k, r in res.items()}
stats["ext_residual_mJy_beam"]["tclean"] = dict(rms_line=1e3 * float(c_res[L].std()))
stats["ext_noise_sigma_mJy_beam"] = 1e3 * sig
fig = plt.figure(figsize=(22, 11), constrained_layout=True)
gs = fig.add_gridspec(2, 3)
rm = {k: m0(r) for k, r in res.items()}
lim = np.percentile(np.abs(np.concatenate([rm["cmp"][FOOT], rm["cln"][FOOT]])), 99.5)
for k, (key, t) in enumerate((("cmp", "compact × mask: d − N x"), ("cln", "CLEAN × mask: d − N x"))):
    show(fig.add_subplot(gs[0, k]), rm[key], HALF, t, cmap="RdBu_r", lo=-lim, hi=lim)
show(fig.add_subplot(gs[0, 2]), rm["cmp"] - rm["cln"], HALF, "compact − CLEAN residual (= CLEAN − compact model, through N)",
     cmap="RdBu_r", lo=-lim, hi=lim)
a = fig.add_subplot(gs[1, :])
a.axvspan(vel[L[0]] - DV / 2, vel[L[-1]] + DV / 2, color=MUTED, alpha=0.1, lw=0)
for key, col in (("cmp", BLUE), ("cln", GREEN)):
    s_ = stats["ext_residual_mJy_beam"][key]
    a.plot(vel, s_["rms_per_channel"], "-o", color=col, lw=2, ms=6, label=f"{LAB[key].split(' ∗')[0]}: whole field")
    a.plot(vel, s_["rms_per_channel_in_footprint"], "--s", color=col, lw=2, ms=5, label=f"{LAB[key].split(' ∗')[0]}: in the footprint")
a.axhline(1e3 * sig, color="0.3", lw=1, ls="-.")
a.text(vel[-1], 1e3 * sig, f" σ = {1e3 * sig:.3f}", va="bottom", ha="right", color="0.3", fontsize=11)
a.set_xlabel(r"velocity [km s$^{-1}$]"); a.set_ylabel("residual rms [mJy/beam, extended]"); a.legend(frameon=False, ncol=2)
a.set_title("residual rms per channel on the extended dirty cube", fontsize=12); a.grid(alpha=0.25)
fig.suptitle(f"{TOP}: residual d_ext − N_ext x on the EXTENDED dirty cube (same operator for both)", fontsize=14)
save(fig, "46_residuals_extended.png", "Residual of the masked compact model and the masked CLEAN components on the extended dirty cube "
     "(moment 0 and rms per channel, whole field and inside the footprint).")

# ======================================================================== 47 spectra at peaks
pk = peaks(M0["cln"], 8, 0.62)
fig, ax = plt.subplots(2, 4, figsize=(22, 9.5), constrained_layout=True)
stats["peaks"] = []
for a, (j, i) in zip(ax.ravel(), pk):
    dx, dy = off(j, i)
    a.step(vel, 1e3 * r_cmp[:, j, i], where="mid", color=BLUE, lw=2, label="compact")
    a.step(vel, 1e3 * r_cln[:, j, i], where="mid", color=GREEN, lw=2, label="CLEAN")
    a.axhline(0, color=MUTED, lw=0.8); a.grid(alpha=0.25)
    a.set_title(f"({dx:+.2f}″, {dy:+.2f}″)", fontsize=12); a.set_ylabel("mJy/beam"); a.set_xlabel(r"velocity [km s$^{-1}$]")
    stats["peaks"].append(dict(offset=[round(float(dx), 2), round(float(dy), 2)], peak_cmp=round(1e3 * float(r_cmp[:, j, i].max()), 2),
                               peak_cln=round(1e3 * float(r_cln[:, j, i].max()), 2)))
ax[0, 0].legend(frameon=False)
fig.suptitle(f"{TOP}: one-beam spectra at the eight brightest CLEAN peaks", fontsize=14)
save(fig, "47_spectra_peaks.png", "One-beam spectra at the eight brightest local maxima of CLEAN's moment 0: compact, CLEAN components.")

# ======================================================================== 48 radial profile
rr = np.hypot((xx - C0) * CELL, (yy - C0) * CELL)
edges = np.arange(0, 1.28 * HALF + 0.01, 0.2); mids = 0.5 * (edges[1:] + edges[:-1])
fig, ax = plt.subplots(1, 3, figsize=(20, 6.2), constrained_layout=True)
prof = {}
for key, col, ls in (("cmp", BLUE, "-"), ("cln", GREEN, "-")):
    f = M0[key] / BEAM_PX                                                 # Jy km/s per pixel
    ring = np.array([f[(rr >= a0) & (rr < a1)].sum() for a0, a1 in zip(edges[:-1], edges[1:])])
    prof[key] = ring
    ax[0].plot(mids, ring, ls, color=col, lw=2 if ls == "-" else 1.5, label=LAB[key])
    ax[1].plot(edges[1:], np.cumsum(ring), ls, color=col, lw=2 if ls == "-" else 1.5, label=LAB[key])
ax[2].plot(mids, prof["cmp"] / np.maximum(prof["cln"], 1e-9), "-o", color=BLUE, lw=2, ms=5)
ax[2].axhline(1, color="k", lw=1, ls="--"); ax[2].set_ylim(0, 3)
for a, t, yl in ((ax[0], "flux per 0.2″ annulus", "Jy km/s"), (ax[1], "cumulative flux", "Jy km/s"),
                 (ax[2], "compact / CLEAN per annulus", "ratio")):
    a.set_xlabel("radius from the phase centre [″]"); a.set_ylabel(yl); a.set_title(t, fontsize=12); a.grid(alpha=0.25)
ax[0].legend(frameon=False, fontsize=10)
fig.suptitle(f"{TOP}: radial distribution of moment 0", fontsize=14)
save(fig, "48_radial.png", "Moment-0 flux per annulus, cumulative flux and the compact/CLEAN ratio per annulus.")

# ======================================================================== 49 summary
er = stats["ext_residual_mJy_beam"]
metrics = [(f"line flux {IN}", "Jy km/s", {"compact": flux["cmp"], "CLEAN": flux["cln"], "CLEAN residual": flux["res"]}),
           ("moment-0 peak (ext. clean beam)", UNIT, {"compact": stats["mom0"]["peak_cmp"], "CLEAN": stats["mom0"]["peak_cln"]}),
           ("ext. residual rms, line channels,\nwhole field (↓ better)", "mJy/beam", {"compact": er["cmp"]["rms_line"], "CLEAN": er["cln"]["rms_line"]}),
           ("ext. residual rms, line channels,\nin the footprint (↓ better)", "mJy/beam", {"compact": er["cmp"]["rms_line_in_footprint"], "CLEAN": er["cln"]["rms_line_in_footprint"]}),
           ("ext. residual rms, line channels,\nin the 3D mask (↓ better)", "mJy/beam", {"compact": er["cmp"]["rms_line_in_mask"], "CLEAN": er["cln"]["rms_line_in_mask"]}),
           ("ext. residual mean in the 3D mask\n(→ 0 better)", "mJy/beam", {"compact": er["cmp"]["mean_line_in_mask"], "CLEAN": er["cln"]["mean_line_in_mask"]}),
           ("agreement compact vs CLEAN", "", {"moment-0 r": stats["mom0"]["pearson"], "voxel r": stats["voxel_pearson"],
                                               "mom-0 slope": stats["mom0_slope_through_0"]}),
           ("moment-1 difference", r"km s$^{-1}$", {"rms": stats["mom1"]["rms_diff_kms"], "median": stats["mom1"]["median_diff_kms"]})]
COLS = {"compact": BLUE, "CLEAN": GREEN, "CLEAN residual": "0.4"}
fig, ax = plt.subplots(2, 4, figsize=(22, 8.5), constrained_layout=True)
for a, (t, u, vals) in zip(ax.ravel(), metrics):
    keys = list(vals); x = np.array([vals[k] for k in keys])
    lo, hi = min(x.min(), 0), max(x.max(), 0)
    pad = 0.25 * (hi - lo if hi > lo else 1)
    for yv_, k in enumerate(keys):
        a.hlines(yv_, lo - pad if lo < 0 else 0, vals[k], color=COLS.get(k, "0.3"), lw=1, alpha=0.35)
        a.plot(vals[k], yv_, "o", ms=11, color=COLS.get(k, "0.3"), mec="white", mew=2, zorder=3)
        a.annotate(f"{vals[k]:.3g}", (vals[k], yv_), xytext=(0, 11), textcoords="offset points", ha="center", fontsize=12)
    a.axvline(0, color=MUTED, lw=0.8)
    a.set_xlim(lo - pad if lo < 0 else 0, hi + pad); a.set_ylim(len(keys) - 0.4, -0.6)
    a.set_yticks(range(len(keys))); a.set_yticklabels(keys); a.set_title(t, fontsize=12); a.set_xlabel(u)
    a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4)); a.grid(axis="x", alpha=0.25)
    for s_ in ("top", "right"):
        a.spines[s_].set_visible(False)
fig.suptitle(f"{TOP}: the numbers (zero-based axes)", fontsize=14)
save(fig, "49_summary.png", "The numbers: flux, moment-0 peak, extended-data residual rms (whole field, footprint, 3D mask) and mean, "
     "compact-vs-CLEAN correlations and moment-0 slope, moment-1 difference.")

json.dump(stats, open(os.path.join(OUT_DIR, "stats.json"), "w"), indent=1)
with open(os.path.join(OUT_DIR, "INDEX.md"), "w") as fh:
    how = ("No mask is applied: both full models are convolved with the extended clean beam; CLEAN's own mask "
           "(mask.fits) only draws the grey outline and bounds the 'in the mask' statistics."
           if NOMASK else f"Mask: `{os.path.relpath(MASK, REPO)}`. Both models are multiplied by the 3D mask and "
           "convolved with the extended clean beam.")
    fh.write(f"# Compact deconvolved vs extended MS-CLEAN{'' if NOMASK else ', inside ' + MASK_DESC}\n\n"
             f"Script: `scripts/compact_vs_msclean_dendro/make_figures.py`. Compact: `{os.path.relpath(COMPACT, REPO)}` "
             f"(pdtau k = 5, raw, Jy/pixel). CLEAN: `{os.path.relpath(CDIR, REPO)}/`. {how} Extended clean beam {BL}. "
             f"Numbers in `stats.json`.\n\n" + "\n".join(INDEX) + "\n")
print("flux in mask:", {k: round(v, 1) for k, v in flux.items()}, "| mom0 r", round(stats["mom0"]["pearson"], 3),
      "| mom1 rms diff", round(stats["mom1"]["rms_diff_kms"], 1), "| ext resid rms line", {k: round(er[k]["rms_line"], 4) for k in ("cmp", "cln")},
      "in mask", {k: round(er[k]["rms_line_in_mask"], 4) for k in ("cmp", "cln")})
