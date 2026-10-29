#!/usr/bin/env python
"""
One-figure summary of the NGC 3110 compact-vs-extended consistency test at the compact beam.

  row 1 (compact array)            uv coverage (zoomed) | dirty image | MS-CLEAN image | (compact - extended) / extended
  row 2 (extended array, tapered)  uv coverage coloured by the tapered weight | dirty image | MS-CLEAN image | spectra

Images are line-only moment 0 (each cube minus its line-free mean, line channels, x 25 km/s). Dirty images:
data/ngc3110/{compact/dirty_big, extended/dirty_big_taper} (82"); MS-CLEAN: compact/big_finished and
extended/big_taper_finished (51", restored; the tapered run brought to the compact run's beam). The difference and the
spectra are inside the tapered extended run's astrodendro mask (mask_final.npy); RMSE over every masked voxel (cube).
uv: raw_visibilities.npz (baselines of the target field in the CO spw); the taper's uv-plane weight from the image-plane
uvtaper of extended/msclean_taper/taper.json. Style: scripts/compact_vs_msclean_dendro/fig_ngc3110_data.py (scale bar
with end caps, bold labels, beam without text).

    python scripts/experiments/fig_consistency_overview.py   -> results/NGC3110/figures/consistency/fig5_overview.{png,pdf}
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
for _p in ("simple", "scripts/compact_vs_msclean_dendro"):
    sys.path.insert(0, os.path.join(REPO, _p))
import beams  # noqa: E402

D_ = os.path.join(REPO, "data/ngc3110"); RES = os.path.join(REPO, "results/NGC3110")
EXT_RUN = os.environ.get("EXT_RUN", "big_taper_finished")      # the tapered extended MS-CLEAN shown (and compared)
TAG = os.environ.get("TAG", "")
OUT = os.path.join(RES, "figures", "consistency"); os.makedirs(OUT, exist_ok=True)
GC = json.load(open(os.path.join(RES, "config.json")))
SET = os.environ.get("SET")                     # a results/NGC3110/consistency_* folder (consistency_target.py); else the
if SET:                                         # original 0.04" / 32 x 25 km/s runs in data/ngc3110
    G_ = json.load(open(os.path.join(SET, "grid.json")))
    L, F, DV, CELL, NCH = G_["line"], G_["line_free"], G_["width_kms"], G_["cell"], G_["nchan"]
    vel = G_["start_kms"] + DV * np.arange(NCH)
    P_CC = f"{SET}/clean_compact_finished/image.fits"; P_EXT = f"{SET}/clean_extended_taper_cmask/image.fits"
    P_DC, P_DT = f"{SET}/dirty_compact", f"{SET}/dirty_extended_taper"
    MASK_PATH = f"{SET}/clean_extended_taper_cmask/dendro_mask_min{os.environ.get('MASKSIG', '16')}.npy"
    RUNJ = f"{SET}/clean_extended_taper_cmask/run.json"
    OUT = os.path.join(SET, "figures"); os.makedirs(OUT, exist_ok=True)
    if os.environ.get("BOTTOM") == "joint":     # bottom row: compact + extended jointly, tapered to the compact beam
        P_EXT = f"{SET}/clean_joint_taper_finished/image.fits"; P_DT = f"{SET}/dirty_joint_taper"
        MASK_PATH = f"{SET}/clean_joint_taper_finished/dendro_mask_min{os.environ.get('MASKSIG', '16')}.npy"
else:
    L = json.load(open(os.path.join(RES, "line_channels.json")))["line"]
    F = [c for c in range(32) if c not in L and 1 < c < 30]
    DV, CELL, NCH = 25.0, 0.04, 32
    vel = GC["vsys_kms"] + DV * (np.arange(32) - 15.5)
    P_CC = P_CC; P_EXT = P_EXT
    P_DC, P_DT = f"{D_}/compact/dirty_big", f"{D_}/extended/dirty_big_taper"
    MASK_PATH = f"{D_}/" + os.environ.get("MASK", "extended/big_taper_finished/mask_final.npy")
    RUNJ = f"{D_}/extended/{EXT_RUN}/run.json"
COMPACT, EXTENDED, MUTED = "#0b3c6e", "#e07b00", "#8a8a8a"
TICK, LAB = 13, 15
plt.rcParams.update({"font.family": "serif", "font.size": 13, "xtick.direction": "in", "ytick.direction": "in"})
rd = lambda p: np.nan_to_num(fits.getdata(p)).astype(np.float64).squeeze()
bm = lambda p: (lambda h: (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"]))(fits.getheader(p))
CONTSUB = os.environ.get("CONTSUB", "1") == "1"                  # CONTSUB=0: keep the continuum (line channels as imaged)
line = (lambda cube: cube[L] - cube[F].mean(0)) if CONTSUB else (lambda cube: cube[L])
n = fits.getheader(P_CC)["NAXIS1"]
N = n if SET else 2048; o = (N - n) // 2         # the original dirty images are 82" (2048 px), the rest 51"


def embed(x):
    out = np.zeros(x.shape[:-2] + (N, N), dtype=x.dtype); out[..., o:o + n, o:o + n] = x
    return out


# ======================================================================== data
BC = bm(P_CC); BT = bm(P_EXT)
bpx = beams.omega(BC) / CELL ** 2
Ic = embed(line(rd(P_CC)))                       # Jy/beam (compact clean beam)
It = line(rd(P_EXT))
It = embed(beams.to_beam(It, BT, BC, CELL) if not np.allclose(BT, BC, rtol=0.005) else It)
Dc = embed(line(rd(f"{P_DC}/dirty.fits"))) if SET else line(rd(f"{P_DC}/dirty.fits"))
Dt = embed(line(rd(f"{P_DT}/dirty.fits"))) if SET else line(rd(f"{P_DT}/dirty.fits"))
BDc = bm(f"{P_DC}/psf.fits") if "BMAJ" in fits.getheader(f"{P_DC}/psf.fits") else BC
BDt = bm(f"{P_DT}/psf.fits") if "BMAJ" in fits.getheader(f"{P_DT}/psf.fits") else BT
M = embed(np.load(MASK_PATH).astype(bool)[L])                   # the comparison region (astrodendro)
FOOT = M.any(0)
flux_c, flux_t = float((Ic * M).sum() / bpx * DV), float((It * M).sum() / bpx * DV)
total = 100 * (flux_c - flux_t) / flux_t
rmse = 100 * np.linalg.norm((Ic - It)[M]) / np.linalg.norm(It[M])
spec_c = np.zeros(NCH); spec_t = np.zeros(NCH)
spec_c[L] = (Ic * M).sum((1, 2)) / bpx; spec_t[L] = (It * M).sum((1, 2)) / bpx
if SET and not CONTSUB:                          # every channel (line-free too: the continuum), over the mask's footprint
    _fp = M.any(0)
    _all = lambda p: (embed(rd(p)) * _fp).sum((1, 2)) / bpx
    spec_c = _all(P_CC)
    _e = rd(P_EXT); spec_t = (embed(beams.to_beam(_e, BT, BC, CELL) if not np.allclose(BT, BC, rtol=0.005) else _e) * _fp).sum((1, 2)) / bpx
    del _e
del Ic, It
m0 = lambda c: c.sum(0) * DV
print(f"in the mask: compact {flux_c:.1f}, {os.environ.get('BOTTOM', 'extended')} (tapered) {flux_t:.1f} Jy km/s -> total {total:+.1f}%, RMSE (cube) {rmse:.1f}%")

# uv: the target field's baselines; the taper's uv-plane weight
z = np.load(f"{D_}/raw_visibilities.npz"); cols = list(z["columns"])
UV = {k: (z[k][cols.index("u_kl")], z[k][cols.index("v_kl")], z[k][cols.index("weight")]) for k in ("compact", "extended")}
JOINT = os.environ.get("BOTTOM") == "joint"
NAME2 = "joint" if JOINT else "extended"        # the bottom row's data
tap = json.load(open(f"{SET}/joint_taper.json" if JOINT else f"{D_}/extended/msclean_taper/taper.json"))["uvtaper"]
ta, tb_, tpa = (float(tap[0].rstrip("arcsec")), float(tap[1].rstrip("arcsec")), float(tap[2].rstrip("deg")))
asr = np.pi / 180 / 3600
ue, ve, we = (tuple(np.r_[UV["compact"][i], UV["extended"][i]] for i in range(3)) if JOINT else UV["extended"])


def taper_weight(u, v):
    """uv-plane weight of an image-plane Gaussian taper (FWHM ta x tb_ at PA tpa, north through east); u, v in klambda"""
    p = np.deg2rad(tpa)
    up, vp = u * np.sin(p) + v * np.cos(p), u * np.cos(p) - v * np.sin(p)
    return np.exp(-np.pi ** 2 / (4 * np.log(2)) * ((ta * asr * up * 1e3) ** 2 + (tb_ * asr * vp * 1e3) ** 2))


tw = taper_weight(ue, ve)
eff = we * tw / (we * tw).max()

# ======================================================================== layout (inches)
S, G, CBH, CBP = 3.7, 0.5, 0.15, 0.1
ML, MR, MB, MT = 0.95, 1.05, 1.0, 1.75
FW, FH = ML + 4 * S + 3 * G + MR, MB + CBP + CBH + 0.25 + 2 * S + 0.45 + MT
fig = plt.figure(figsize=(FW, FH))
inch = lambda x, y, w, h: fig.add_axes([x / FW, y / FH, w / FW, h / FH])
PX = lambda c: ML + c * (S + G) + (0.25 if c > 0 else 0)
Y2 = MB + CBP + CBH + 0.25                         # bottom row (its colorbars below)
Y1 = Y2 + S + 0.45                                 # top row (its colorbars above)
H = 8.0; hp = int(H / CELL); sl = np.s_[N // 2 - hp:N // 2 + hp, N // 2 - hp:N // 2 + hp]; EXT = [H, -H, -H, H]


def label(a, text, light=False, box=False):
    a.text(0.04, 0.96, text, transform=a.transAxes, ha="left", va="top", fontsize=LAB, fontweight="bold",
           color="k" if light else "w", zorder=10,
           bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85) if box else None)


def image(x, y, img, beam, text, cbar_above, unit, cmap="inferno", lo=None, hi=None, light=False, outline=False, bar=True,
          outline_alpha=1.0):
    a = inch(x, y, S, S)
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("0.9")
    im = a.imshow(img[sl], origin="lower", extent=EXT, cmap=cm, norm=Normalize(lo, hi), interpolation="nearest")
    if outline:
        a.contour(FOOT[sl].astype(float), levels=[0.5], colors="k" if light else "w", linewidths=0.6, extent=EXT,
                  origin="lower", alpha=outline_alpha)
    a.set_xticks([]); a.set_yticks([])
    ink = "k" if light else "w"
    if beam is not None:                           # beam bottom left, no text
        pad = 0.06 * H
        a.add_patch(Ellipse((H - pad - beam[0] / 2, -H + pad + beam[0] / 2), width=beam[0], height=beam[1],
                            angle=90 - beam[2], facecolor="0.2" if light else "w", edgecolor=ink))
    if bar:                                        # scale bar, top right, with end caps; bold label below it
        xs, ys, cap = -H + 0.08 * H, H - 0.1 * H, 0.03 * H
        a.plot([xs + 2, xs], [ys, ys], color=ink, lw=2.0, solid_capstyle="butt")
        for xe in (xs, xs + 2):
            a.plot([xe, xe], [ys - cap, ys + cap], color=ink, lw=1.4, solid_capstyle="butt")
        a.text(xs + 1, ys - 2.2 * cap, "2″", color=ink, ha="center", va="top", fontsize=12, fontweight="bold")
    label(a, text, light)
    cax = inch(x, y + S + CBP if cbar_above else y - CBP - CBH, S, CBH)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    if cbar_above:
        cax.xaxis.set_ticks_position("top"); cax.xaxis.set_label_position("top")
    cax.tick_params(labelsize=TICK, direction="out", length=4, pad=3)
    cb.locator = matplotlib.ticker.MaxNLocator(5); cb.update_ticks()
    cb.set_label(unit, fontsize=LAB - 1, labelpad=5)
    return a


def uv_panel(x, y, u, v, text, c=None, taper=False, xlabel=True):
    a = inch(x, y, S, S)
    R = 200.0
    sel = (np.abs(u) < R) & (np.abs(v) < R)
    rng = np.random.default_rng(1)
    pick = np.nonzero(sel)[0]; pick = rng.choice(pick, min(pick.size, 40000), replace=False)
    if c is None:
        a.scatter(np.r_[u[pick], -u[pick]], np.r_[v[pick], -v[pick]], s=0.6, color=COMPACT, lw=0, alpha=0.6, rasterized=True)
        sc = None
    else:
        cc = np.r_[c[pick], c[pick]]; o_ = np.argsort(cc)
        sc = a.scatter(np.r_[u[pick], -u[pick]][o_], np.r_[v[pick], -v[pick]][o_], s=1.2, c=cc[o_], cmap="viridis",
                       vmin=0, vmax=1, lw=0, rasterized=True)
    if taper:                                      # the taper's half-weight contour
        gu, gv = np.meshgrid(np.linspace(-R, R, 400), np.linspace(-R, R, 400))
        a.contour(gu, gv, taper_weight(gu, gv), levels=[0.5], colors="w", linewidths=1.2)
    a.set_xlim(R, -R); a.set_ylim(-R, R); a.set_aspect("equal")
    a.set_facecolor("0.12" if taper else "white")
    a.tick_params(which="both", top=True, right=True, labelsize=TICK, color="0.6" if taper else "k")
    a.set_ylabel(r"$v$ (k$\lambda$)", fontsize=LAB, labelpad=3)
    if xlabel:
        a.set_xlabel(r"$u$ (k$\lambda$)", fontsize=LAB, labelpad=3)
    else:
        a.tick_params(labelbottom=False)
    label(a, text, light=not taper, box=not taper)
    return a, sc


# row 1: compact
uv_panel(ML, Y1, *UV["compact"][:2], "Compact visibilities", xlabel=False)
# dirty images: each its own scale (its own min and max in the shown box)
mDc = m0(Dc)
image(PX(1), Y1, mDc, BDc, "Compact dirty", True, r"Jy (dirty beam)$^{-1}$ km s$^{-1}$", lo=mDc[sl].min(), hi=mDc[sl].max())
Icm = m0(embed(line(rd(P_CC))))
# MS-CLEAN images: one scale, the common min and max of both (in the shown box)
_tt = line(rd(P_EXT))
_tt = embed(beams.to_beam(_tt, BT, BC, CELL) if not np.allclose(BT, BC, rtol=0.005) else _tt)
_Itm = m0(_tt); del _tt
clo, chi = min(Icm[sl].min(), _Itm[sl].min()), max(Icm[sl].max(), _Itm[sl].max())
ac = image(PX(2), Y1, Icm, BC, "Compact MS-CLEAN", True, r"Jy beam$^{-1}$ km s$^{-1}$", lo=clo, hi=chi,
           outline=True, outline_alpha=0.4)
PARAM = ("multiscale (0, 1.2″, 3.6″)\nnatural, 51″ field\nnsigma 3, auto-multithresh\n→ nsigma 1.5, dendrogram mask")
ac.text(0.96, 0.04, PARAM, transform=ac.transAxes, ha="right", va="bottom", fontsize=10.5, color="w", linespacing=1.15)
# difference, % of the tapered extended CLEAN, inside the mask
Ia, Ib = (embed(line(rd(P_CC))) * M).sum(0), None
_t = line(rd(P_EXT))
_t = embed(beams.to_beam(_t, BT, BC, CELL) if not np.allclose(BT, BC, rtol=0.005) else _t)
Ib = (_t * M).sum(0); Itm = m0(_t); del _t
pct = np.where(FOOT & (Ib > 0.05 * Ib[FOOT].max()), 100 * (Ia - Ib) / np.where(Ib > 0, Ib, 1), np.nan)
ar = image(PX(3), Y1, pct, None, f"Difference / {NAME2}", True, f"(compact − {NAME2}) / {NAME2} (%)", cmap="RdBu_r",
           lo=-50, hi=50, light=True, outline=True, bar=False)
ar.text(0.04, 0.86, f"total {total:+.1f}%\nRMSE (cube) {rmse:.1f}%", transform=ar.transAxes, ha="left", va="top",
        fontsize=13)

# row 2: extended, tapered to the compact beam
_, sc = uv_panel(ML, Y2, ue, ve, f"{NAME2.capitalize()} visibilities,\ntapered", c=eff, taper=True)
cax = inch(ML, Y2 - CBP - CBH - 0.62, S, CBH)
cb = fig.colorbar(sc, cax=cax, orientation="horizontal"); cax.tick_params(labelsize=TICK, direction="out", length=4)
cb.set_label("weight × taper (normalised)", fontsize=LAB - 1, labelpad=5)
mDt = m0(Dt)
image(PX(1), Y2, mDt, BDt, f"{NAME2.capitalize()} dirty\ntapered", False, r"Jy (dirty beam)$^{-1}$ km s$^{-1}$", lo=mDt[sl].min(), hi=mDt[sl].max())
ae = image(PX(2), Y2, Itm, BC, f"{NAME2.capitalize()} MS-CLEAN\ntapered", False, r"Jy beam$^{-1}$ km s$^{-1}$", lo=clo, hi=chi,
           outline=True, outline_alpha=0.4)
ae.text(0.96, 0.04, f"uvtaper {ta:.2f}″ × {tb_:.2f}″\n" + (PARAM if ((EXT_RUN == "big_taper_finished" and not SET) or JOINT) else
              "multiscale (0, 1.2″, 3.6″)\nnatural, 51″ field\n" + ("" if SET else "nsigma 3, auto-multithresh\n→ ")
              + "in the compact mask, "
              + (lambda j: j.get("stopping", {}).get("threshold") or (f"{j['threshold_mJy']:.2f} mJy" if "threshold_mJy" in j else "nsigma 1.5"))(json.load(open(RUNJ)))), transform=ae.transAxes, ha="right", va="bottom",
        fontsize=10.5, color="w", linespacing=1.15)
# spectra, 2 in 1
a = inch(PX(3), Y2, S, S)
a.axvspan(vel[L[0]] - DV / 2, vel[L[-1]] + DV / 2, color=MUTED, alpha=0.1, lw=0)
sf_c, sf_t = (float(spec_c[L].sum() * DV), float(spec_t[L].sum() * DV)) if (SET and not CONTSUB) else (flux_c, flux_t)
a.step(vel, spec_c, where="mid", color=COMPACT, lw=2, label=f"compact: {sf_c:.0f}")
a.step(vel, spec_t, where="mid", color=EXTENDED, lw=2, label=f"{NAME2}, tapered: {sf_t:.0f}")
a.step(vel, spec_c - spec_t, where="mid", color=COMPACT, lw=1.3, ls="--", label=f"difference: {sf_c - sf_t:+.0f}")
a.axhline(0, color=MUTED, lw=0.8)
a.set_xlim((vel[0] - DV / 2, vel[-1] + DV / 2) if SET else (vel[L[0]] - 60, vel[L[-1]] + 60))
a.set_ylim(top=1.55 * max(spec_c.max(), spec_t.max()))
a.yaxis.tick_right(); a.yaxis.set_label_position("right")
a.tick_params(which="both", top=True, bottom=True, left=True, right=True, labelleft=False, labelright=True, labelsize=TICK)
a.minorticks_on(); a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
a.set_xlabel(r"$v_\mathrm{LSRK}$ (km s$^{-1}$)", fontsize=LAB, labelpad=3)
a.set_ylabel(r"$S_\nu$ in the mask (Jy)", fontsize=LAB, labelpad=6)
label(a, "MS-CLEAN spectra" + ("\nall channels, mask footprint" if (SET and not CONTSUB) else ""), light=True)
a.legend(frameon=False, fontsize=11.5, loc="upper right", bbox_to_anchor=(1.0, 0.86), title="Jy km s$^{-1}$", title_fontsize=11.5)
fig.text(0.5, 1 - 0.2 / FH, f"NGC 3110: compact vs {NAME2}, both at the compact beam"
         + (f"   ({CELL:g}″ pixels, {NCH} × {DV:g} km s$^{{-1}}$)" if SET else ""), ha="center", va="top", fontsize=22,
         fontweight="bold")
fig.savefig(os.path.join(OUT, f"fig5_overview{TAG}.png"), dpi=110)
fig.savefig(os.path.join(OUT, f"fig5_overview{TAG}.pdf"), dpi=300)
print("saved", os.path.join(OUT, f"fig5_overview{TAG}.png"))
