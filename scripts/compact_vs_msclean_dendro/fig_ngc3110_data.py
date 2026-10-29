#!/usr/bin/env python
"""
NGC 3110 data overview (paper, Data section), 2 x 2 square panels: left column the MS-CLEAN line moment 0 of the
compact (top) and extended (bottom) configuration, colorbars above / below them; right column the two uv coverages overlaid
(top) and the two integrated spectra (bottom). One colour per configuration throughout: two shades of blue, compact
darker.

  moment 0   CLEAN components inside the extended astrodendro mask * that configuration's clean beam, no residual
             (compact/msclean, extended/msclean_finished; as fig_msclean_pairs_grid.py), Jy arcsec^-2 km/s
  uv         (u, v) and (-u, -v) of the target field in the CO spectral window, at the line frequency, in klambda;
             read from the MSs of results/NGC3110/config.json and cached in results/NGC3110/uv_coverage.npz
  spectra    sum over the field of the same masked components (Jy) per channel; line channels shaded

    [COLUMN=1] [OUT=...] python scripts/compact_vs_msclean_dendro/fig_ngc3110_data.py      (CASA env: casatools for the uv)
    -> results/gallery/ngc3110_data_overview.{png,pdf}            (large, for viewing)
       results/gallery/ngc3110_data_overview_column.{png,pdf}     (COLUMN=1: drawn at A&A column width, 88 mm, at its
                                                                   final size: fonts 6-9 pt; the paper's version)
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

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
from restore import convolve_with_beam  # noqa: E402

T = "NGC3110"
R = os.path.join(REPO, "results", T)
COLUMN = os.environ.get("COLUMN", "0") == "1"
FS, LW = (0.56, 0.6) if COLUMN else (1.0, 1.0)                       # font and line scale (COLUMN: final size)
BOTTOM = os.environ.get("BOTTOM", "extended")   # "joint": bottom row = MS-CLEAN of compact + extended (natural, no taper,
#                                                  scripts/experiments/joint_natural.py)
OUT = os.environ.get("OUT", os.path.join(REPO, "results", "gallery", "ngc3110_data_overview"
                                         + ("_joint" if BOTTOM == "joint" else "") + ("20" if os.environ.get("GRID") == "20kms" else "")
                                         + ("_column" if COLUMN else "") + ".png"))
CFGS = ("compact", "extended")
COL = {"compact": "#0b3c6e", "extended": "#3f8fd8", "joint": "#3f8fd8"}  # same hue, compact darker
C_KMS = 299792.458

plt.rcParams.update({"font.family": "serif", "xtick.direction": "in", "ytick.direction": "in", "font.size": 13 * FS,
                     "xtick.major.size": 3.5 * FS, "ytick.major.size": 3.5 * FS, "axes.linewidth": 0.8 * max(LW, 0.75)})
rd = lambda p: np.nan_to_num(fits.getdata(p).astype(np.float64)).squeeze()
omega = lambda b: np.pi * b[0] * b[1] / (4 * np.log(2))
c = json.load(open(os.path.join(R, "config.json")))
L = json.load(open(os.path.join(R, "line_channels.json")))["line"]
cell, dv, half = c["cell_arcsec"], c["width_kms"], float(c["half_arcsec"])
vel = c["start_kms"] + dv * np.arange(c["nchan"])                    # channel centres, LSRK radio


def uv_coverage():
    """{cfg: (u, v) in klambda} of the target field in the line spw, cached"""
    cache = os.path.join(R, "uv_coverage.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: (z[k + "_u"], z[k + "_v"]) for k in CFGS}
    from casatools import table, msmetadata
    tb, md, out = table(), msmetadata(), {}
    lam_m = C_KMS * 1e3 / (c["f_line_ghz"] * 1e9)
    for k in CFGS:
        us, vs = [], []
        for ms, spw, field in zip(c["configs"][k]["ms"], c["configs"][k]["spw"], c["configs"][k]["field"]):
            spw_id = int(spw.split(":")[0])
            md.open(ms); fid = int(md.fieldsforname(field)[0]); md.close()
            tb.open(ms + "/DATA_DESCRIPTION"); dd = int(np.nonzero(tb.getcol("SPECTRAL_WINDOW_ID") == spw_id)[0][0]); tb.close()
            tb.open(ms); q = tb.query(f"FIELD_ID=={fid} && DATA_DESC_ID=={dd} && ANTENNA1!=ANTENNA2 && !FLAG_ROW", columns="UVW")
            uvw = q.getcol("UVW"); q.close(); tb.close()
            us.append(uvw[0] / lam_m / 1e3); vs.append(uvw[1] / lam_m / 1e3)
        out[k] = (np.concatenate(us).astype(np.float32), np.concatenate(vs).astype(np.float32))
    np.savez_compressed(cache, **{f"{k}_{a}": out[k][i] for k in CFGS for i, a in enumerate("uv")})
    return out


M = np.load(os.path.join(R, "extended/msclean_finished/mask_final.npy")).astype(bool)
FOOT0 = M[L].any(0)                                          # the extended emission mask's footprint (0.02", 16")
G20 = os.environ.get("GRID") == "20kms"                      # joint version on the 64 x 20 km/s grid of the ladder
if G20:
    _g = json.load(open(os.path.join(R, "consistency_20kms_0.08/grid.json")))
    L, dv = _g["line"], _g["width_kms"]; vel = _g["start_kms"] + dv * np.arange(_g["nchan"])
ROWS = ("compact", BOTTOM)
JOINT_DIR = os.environ.get("JOINT_DIR", "joint_robust0")      # joint: no uv taper (scripts/experiments/joint_natural.py)
LIN100 = os.environ.get("STRETCH", "linear100" if BOTTOM == "joint" else "sqrt") == "linear100"   # maps: min-max linear
APER = float(os.environ.get("APER", 0.7))                     # restored spectra: mask footprint widened by APER arcsec
img, spec, beam, CELLS, MASKS = {}, {}, {}, {}, {}
if BOTTOM == "joint":
    # restored images (CLEAN components * clean beam + residual) of the compact data (51" field: compact/big_finished,
    # data/ngc3110) and of the joint visibilities, both 0.04" pixels, the same 32 x 25 km/s channels; maps unmasked,
    # spectra summed over one common aperture (the extended emission mask's footprint widened by APER)
    from scipy.ndimage import binary_dilation
    SRC = ({"compact": os.path.join(R, "consistency_20kms_0.08/clean_compact_finished"),
            "joint": os.path.join(R, "ladder_joint_20kms/b10/clean_finished")} if G20 else
           {"compact": os.path.join(REPO, "data/ngc3110/compact/big_finished"), "joint": os.path.join(R, JOINT_DIR)})
    for k in ROWS:
        h = fits.getheader(os.path.join(SRC[k], "image.fits")); ck = abs(h["CDELT2"]) * 3600
        B = beam[k] = (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"])
        I = rd(os.path.join(SRC[k], "image.fits")); n0 = I.shape[-1]
        f = int(round(ck / cell)); nm = M.shape[-1] // f; o = (n0 - nm) // 2
        A = np.zeros((n0, n0), bool); A[o:o + nm, o:o + nm] = FOOT0.reshape(nm, f, nm, f).any((1, 3))
        A = binary_dilation(A, iterations=max(1, int(round(APER / ck))))
        no = int(round(16.0 / ck)); oo = (n0 - no) // 2                 # the maps: the central 16"
        CELLS[k], MASKS[k] = ck, np.broadcast_to(A[oo:oo + no, oo:oo + no], (I.shape[0], no, no))
        img[k] = (I[L].sum(0) * dv / omega(B))[oo:oo + no, oo:oo + no]   # Jy arcsec^-2 km/s
        spec[k] = (I * A).sum((1, 2)) / (omega(B) / ck ** 2)            # Jy
    _a = (MASKS["compact"][0].sum() * CELLS["compact"] ** 2)
    print(f"aperture {_a:.0f} arcsec^2 (16\" crop); restored line flux", {k: round(float(spec[k][L].sum() * dv), 1) for k in ROWS})
else:
    for k, D in (("compact", "compact/msclean"), ("extended", "extended/msclean_finished")):
        if k not in ROWS:
            continue
        D = os.path.join(R, D); h = fits.getheader(os.path.join(D, "image.fits"))
        B = beam[k] = (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"])
        x = rd(os.path.join(D, "model.fits")) * M                      # Jy/pixel components inside the mask
        CELLS[k], MASKS[k] = cell, M
        img[k] = convolve_with_beam((x[L].sum(0) * dv)[None], cell, *B)[0] / omega(B)
        spec[k] = x.sum((1, 2))
uv = uv_coverage()

# ======================================================================== layout (inches)
# panel, colorbar thickness / pad, gaps; MT / MB hold the colorbars and their labels (and the uv / spectrum x labels)
if COLUMN:                                                            # 88 mm wide in total
    S, CBW, CBP, G, RG = 1.40, 0.055, 0.03, 0.05, 0.05
    ML, MR, MB, MT = 0.04, 0.50, 0.42, 0.45
else:
    S, CBW, CBP, G, RG = 3.6, 0.13, 0.08, 0.15, 0.15
    ML, MR, MB, MT = 0.15, 0.95, 0.95, 0.95
x_img = ML; x_r = x_img + S + G
FW, FH = x_r + S + MR, MB + 2 * S + RG + MT
fig = plt.figure(figsize=(FW, FH))
inch = lambda x, y, w, h: fig.add_axes([x / FW, y / FH, w / FW, h / FH])
y_top, y_bot = MB + S + RG, MB
for k, y0 in (("compact", y_top), (BOTTOM, y_bot)):
    c0 = img[k].shape[0] // 2; hp = int(round(half / CELLS[k])); sl = np.s_[c0 - hp:c0 + hp, c0 - hp:c0 + hp]
    a = inch(x_img, y0, S, S)
    im = a.imshow(img[k][sl], origin="lower", extent=[half, -half, -half, half], cmap="inferno",
                  norm=(matplotlib.colors.Normalize(img[k][sl].min(), img[k][sl].max()) if LIN100 else  # 100% linear
                        matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=np.percentile(img[k][MASKS[k][L].any(0)], 99.9))),
                  interpolation="nearest")
    a.set_xticks([]); a.set_yticks([])
    a.text(0.04, 0.96, k.capitalize(), transform=a.transAxes, color="w", ha="left", va="top",
           fontsize=13 * FS, fontweight="bold")                  # the size of the axis labels
    if k == "joint":                                                   # what "joint" means, under the name
        a.text(0.04, 0.96 - 0.085, "compact + extended\nvisibilities", transform=a.transAxes, color="w", ha="left",
               va="top", fontsize=10.5 * FS, linespacing=1.05)
    B = beam[k]; pad = 0.06 * half; bx, by = half - pad - B[0] / 2, -half + pad + B[0] / 2
    a.add_patch(Ellipse((bx, by), width=B[0], height=B[1], angle=90 - B[2], facecolor="w", edgecolor="w"))
    if BOTTOM == "joint":                                              # the beam's size, right of it (RA increases left)
        a.text(bx - max(B[0], 0.25) / 2 - 0.08 * half, by, f"{B[0]:.2f}″ × {B[1]:.2f}″", color="w", ha="left",
               va="center", fontsize=10.5 * FS)
    xs, ys, cap = -half + 0.08 * half, half - 0.1 * half, 0.03 * half    # scale bar, top right (RA increases left)
    a.plot([xs + 2, xs], [ys, ys], color="w", lw=2.0 * LW, solid_capstyle="butt")
    for xe in (xs, xs + 2):                                            # end caps
        a.plot([xe, xe], [ys - cap, ys + cap], color="w", lw=1.4 * LW, solid_capstyle="butt")
    a.annotate("2″", (xs + 1, ys - cap), xytext=(0, -2.5), textcoords="offset points", color="w", ha="center",
               va="top", fontsize=12 * FS, fontweight="bold")
    top = k == "compact"                                                # compact: colorbar above, extended: below
    cax = inch(x_img, y0 + S + CBP if top else y0 - CBP - CBW, S, CBW)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.locator = matplotlib.ticker.MaxNLocator(4); cb.update_ticks()
    cax.xaxis.set_ticks_position("top" if top else "bottom"); cax.xaxis.set_label_position("top" if top else "bottom")
    cax.tick_params(labelsize=11 * FS, direction="out", length=2.5 * FS, pad=2.5 * FS)
    cb.set_label(r"Jy arcsec$^{-2}$ km s$^{-1}$", fontsize=12 * FS, labelpad=5 * FS)

# uv coverage: extended first, compact (darker) on top
a = inch(x_r, y_top, S, S)
for k in ("extended", "compact"):
    u, v = uv[k]
    a.scatter(np.r_[u, -u], np.r_[v, -v], s=(1.0 if k == "extended" else 0.6) * LW ** 2, color=COL[k], alpha=0.9, rasterized=True,
              linewidths=0)
    a.plot([], [], "s", ms=7 * FS, color=COL[k],
           label=(rf"{k} ({c['bmax_m' if k == 'extended' else 'bmax_compact_m']:.0f} m)" if COLUMN else
                  rf"{k} ($B_{{\max}}$ = {c['bmax_m' if k == 'extended' else 'bmax_compact_m']:.0f} m)"))
lim = 1.08 * np.hypot(*uv["extended"]).max()
a.set_xlim(lim, -lim); a.set_ylim(-lim, lim); a.set_aspect("equal")
a.set_xlabel(r"$u$ (k$\lambda$)", fontsize=13 * FS, labelpad=5 * FS); a.set_ylabel(r"$v$ (k$\lambda$)", fontsize=13 * FS, labelpad=7 * FS)
a.tick_params(labelsize=11 * FS, top=True, right=True, labeltop=True, labelbottom=False, labelright=True, labelleft=False,
              pad=3 * FS)
a.xaxis.set_label_position("top"); a.yaxis.set_label_position("right")
h, l = a.get_legend_handles_labels()
LEG = dict(fontsize=11 * FS - (1 if COLUMN else 0), frameon=True, framealpha=0.95, edgecolor="0.6", fancybox=False)
a.legend(h[::-1], l[::-1], loc="upper left", handletextpad=0.4, borderaxespad=0.5, **LEG).get_frame().set_linewidth(0.6 * LW)

# integrated spectra
a = inch(x_r, y_bot, S, S)
lo, hi = vel[min(L)] - dv / 2, vel[max(L)] + dv / 2
a.axvspan(lo, hi, color="0.93", zorder=0, lw=0)
for k in ROWS:
    fl = spec[k][L].sum() * dv                                          # line flux, Jy km/s (COLUMN: units in the caption)
    a.step(vel, spec[k], where="mid", color=COL[k], lw=2.0 * LW,
           label=f"{k} ({fl:.0f})" if COLUMN else f"{k}  ({fl:.0f} Jy km s$^{{-1}}$)")
a.axhline(0, color="0.5", lw=0.8 * LW)
a.set_xlim(vel[0] - dv / 2, vel[-1] + dv / 2)
ymax = max(spec[k].max() for k in ROWS); a.set_ylim(-0.09 * ymax, 1.35 * ymax)     # headroom for the legend
a.set_xlabel(r"$v_\mathrm{LSRK}$ (km s$^{-1}$)", fontsize=13 * FS, labelpad=5 * FS)
a.set_ylabel(r"$S_\nu$ (Jy)", fontsize=13 * FS, labelpad=7 * FS)
a.tick_params(labelsize=11 * FS, top=True, right=True, labelright=True, labelleft=False, pad=3 * FS)
a.yaxis.set_label_position("right")
a.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(400 if (COLUMN and G20) else 300 if COLUMN else 200))
a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5 if COLUMN else 0.2))
a.legend(loc="upper left", handlelength=1.5, borderaxespad=0.5, **LEG).get_frame().set_linewidth(0.6 * LW)
if not COLUMN:                                                        # COLUMN: the caption says what is shaded
    a.text(0.5 * (lo + hi), 0.012, "line channels", transform=a.get_xaxis_transform(), ha="center", va="bottom",
           fontsize=10 * FS, color="0.45")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=400 if COLUMN else 150)
fig.savefig(OUT.replace(".png", ".pdf"), dpi=600 if COLUMN else 300)
print("saved", OUT, "and .pdf", {k: len(uv[k][0]) for k in CFGS}, "line flux", {k: round(float(spec[k][L].sum() * dv), 1) for k in ROWS},
      "beams", beam)
