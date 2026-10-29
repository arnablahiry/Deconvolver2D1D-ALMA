#!/usr/bin/env python
"""
53-55: compact pdtau (k = 5) deconvolved WITHOUT a target resolution vs WITH one (gpu_pd.Op target_fwhm_px), both
convolved afterwards with the EXTENDED configuration's clean beam (header of data/ngc3110/extended/ms_clean/image.fits),
so the two are compared at one resolution. Writes, in figures/experiments/11_target_resolution/:

  53_mom0_extbeam_<tag>.png       moment 0 of both restored cubes and their difference (central 10" and 3")
  54_residuals_<tag>.png          residual d - N y on the compact dirty cube (0.04" grid, compact Jy/beam):
                                  moment 0 of each, their difference, rms per channel
  55_spectra_extbeam_<tag>.png    integrated spectra (model, Jy) and one-beam spectra at the brightest peaks
                                  of the restored moment 0 (Jy/beam of the extended beam)

and the numbers to data/experiments/11_target_resolution/compare_extbeam_<tag>.json.


MASK (optional): a (32, 800, 800) 0/1 cube on the 0.02" grid (e.g. fig61's dendrogram mask of the extended CLEAN
image). Every model cube is multiplied by it BEFORE the convolution with the extended beam and before the residuals,
moment-0 comparisons are taken over its footprint (pixels masked in any line channel) instead of the whole box, and
the outputs get the suffix _<mask file name>.

    MODEL=... TARGET_MODEL=... TAG=t0.1134 [MASK=...] python scripts/superres_figures/fig53_target_vs_none_extbeam.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import maximum_filter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from common import G, REPO, EXP_DATA, EXP_FIG, problem, to_native, vel  # noqa: E402
from restore import convolve_with_beam  # noqa: E402

TAG = os.environ.get("TAG", "t0.1134")
MODEL = os.environ.get("MODEL", os.path.join(EXP_DATA, "04_pd_tau_reweighting/compact/pdtau_k5.npy"))
TARGET_MODEL = os.environ.get("TARGET_MODEL", os.path.join(EXP_DATA, f"11_target_resolution/pdtau_k5_{TAG}.npy"))
TARGET_RES = float(json.load(open(TARGET_MODEL.replace(".npy", ".json")))["target_res"])
FIG_DIR = os.path.join(EXP_FIG, "11_target_resolution")
MASK = os.environ.get("MASK")
OTAG = TAG + (f"_{os.path.basename(MASK).replace('.npy', '')}" if MASK else "")
OUT_JSON = os.path.join(EXP_DATA, "11_target_resolution", f"compare_extbeam_{OTAG}.json")
CELL, DV = 0.02, 25.0
BLUE, ORANGE, MUTED = "#2a6fdb", "#e07b00", "#8a8a8a"

h = fits.getheader(os.path.join(REPO, "data/ngc3110/extended/ms_clean/image.fits"))
BE = (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"])
BEAM_PX = np.pi * BE[0] * BE[1] / (4 * np.log(2)) / CELL ** 2           # extended beam area, 0.02" pixels

P = problem("ngc3110", "compact")                                       # no target: d and N of the original problem
L, NC = P.line, P.noise
v = vel("ngc3110")
names = ("no target", f"target {TARGET_RES:g}″")
models = [np.load(f).astype(np.float64) for f in (MODEL, TARGET_MODEL)]  # Jy/pixel at 0.02"
M = np.load(MASK).astype(bool) if MASK else None
if MASK:
    models = [m * M for m in models]
    print(f"masked by {MASK}: {M.mean():.2%} of voxels")
print(f"extended clean beam {BE[0]:.3f}\" x {BE[1]:.3f}\" @ {BE[2]:.1f} deg; line channels {L[0]}-{L[-1]}, "
      f"{len(NC)} line-free channels")

restored = [convolve_with_beam(m, CELL, *BE) for m in models]           # Jy/beam (extended)
resid = [P.op.residual_sky(G.to_t(to_native(P, m))).cpu().numpy().astype(np.float64) for m in models]  # compact Jy/beam, 0.04"
mom0 = [r[L].sum(0) * DV for r in restored]                             # Jy/beam km/s
rmom0 = [r[L].sum(0) * DV for r in resid]

# ---------------------------------------------------------------- numbers
cy = P.crop                                                             # central 10" of the 0.04" grid
c2 = np.zeros((800, 800), bool); c2[150:650, 150:650] = True           # the same 10" on the 0.02" grid
if MASK:
    c2 &= M[L].any(0)                                                   # ... inside the mask's footprint
stats = dict(mask=MASK, extended_beam_arcsec_deg=BE, target_res_arcsec=TARGET_RES, model=MODEL, target_model=TARGET_MODEL,
             line_channels=L, line_free_channels=NC)
for n, m, r, rc, m0 in zip(names, models, restored, resid, mom0):
    stats[n] = dict(
        flux_line_Jy_kms=float(m[L].sum() * DV), flux_line_free_Jy=float(m[NC].sum()),
        resid_rms_line_mJy_beam=1e3 * float(rc[L].std()), resid_rms_line_free_mJy_beam=1e3 * float(rc[NC].std()),
        resid_rms_line_central10_mJy_beam=1e3 * float(rc[L][(slice(None),) + cy].std()),
        resid_rms_per_channel_mJy_beam=[round(1e3 * float(rc[c].std()), 4) for c in range(32)],
        resid_mean_per_channel_mJy_beam=[round(1e3 * float(rc[c].mean()), 4) for c in range(32)],
        resid_mean_line_mJy_beam=1e3 * float(rc[L].mean()),
        resid_mom0_rms_Jy_beam_kms=float(rmom0[names.index(n)][cy].std()),
        restored_rms_line_free_mJy_beam=1e3 * float(r[NC].std()),
        restored_peak_mJy_beam=1e3 * float(r.max()), mom0_peak_Jy_beam_kms=float(m0.max()))
d = mom0[1] - mom0[0]
stats["mom0_difference"] = dict(rms_central10_Jy_beam_kms=float(d[c2].std()),
                                max_abs_Jy_beam_kms=float(np.abs(d).max()),
                                rms_over_peak=float(d[c2].std() / mom0[0].max()),
                                pearson_central10=float(np.corrcoef(mom0[0][c2].ravel(), mom0[1][c2].ravel())[0, 1]))
rd = resid[1] - resid[0]
stats["resid_difference_rms_line_mJy_beam"] = 1e3 * float(rd[L].std())
json.dump(stats, open(OUT_JSON, "w"), indent=1)
for n in names:
    s = stats[n]
    print(f"{n:16s} flux {s['flux_line_Jy_kms']:7.2f} Jy km/s (line-free {s['flux_line_free_Jy']:.3f} Jy) | "
          f"resid rms line {s['resid_rms_line_mJy_beam']:.4f}, line-free {s['resid_rms_line_free_mJy_beam']:.4f}, "
          f"line central 10\" {s['resid_rms_line_central10_mJy_beam']:.4f}, line mean {s['resid_mean_line_mJy_beam']:+.4f} mJy/beam | restored: line-free rms "
          f"{s['restored_rms_line_free_mJy_beam']:.4f} mJy/beam, peak {s['restored_peak_mJy_beam']:.2f} mJy/beam, "
          f"mom0 peak {s['mom0_peak_Jy_beam_kms']:.4f} Jy/beam km/s")
print("mom0 difference:", {k: round(v, 5) for k, v in stats["mom0_difference"].items()},
      f"| residual difference rms (line) {stats['resid_difference_rms_line_mJy_beam']:.4f} mJy/beam")
print("saved", OUT_JSON)

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.family": "serif", "font.size": 12, "xtick.direction": "in", "ytick.direction": "in"})
os.makedirs(FIG_DIR, exist_ok=True)
beam_lab = f"{BE[0]:.3f}″×{BE[1]:.3f}″"
MT = " (models × the dendrogram mask)" if MASK else ""


def panel(a, img, cell, half_arcsec, cmap, lo, hi, title, label):
    n = img.shape[0]; c = n // 2; hp = int(round(half_arcsec / cell))
    sl = np.s_[c - hp:c + hp, c - hp:c + hp]; ext = [hp * cell, -hp * cell, -hp * cell, hp * cell]
    im = a.imshow(img[sl], origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    a.set_title(title, fontsize=12); a.set_xlabel(r'$\Delta$RA ["]'); a.set_ylabel(r'$\Delta$Dec ["]')
    plt.colorbar(im, ax=a, shrink=0.85, label=label)


# 51: moment 0 at the extended beam
fig, ax = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
vm = max(np.percentile(m, 99.95) for m in mom0); dm = np.percentile(np.abs(d[c2]), 99.9)
for row, half in enumerate((5.0, 1.5)):
    for k, (img, title, cm, lo, hi) in enumerate(((mom0[0], f"no target ∗ extended beam", "inferno", 0, vm),
                                                  (mom0[1], f"target {TARGET_RES:g}″ ∗ extended beam", "inferno", 0, vm),
                                                  (d, "target − no target", "RdBu_r", -dm, dm))):
        panel(ax[row, k], img, CELL, half, cm, lo, hi, title if row == 0 else f"central {2 * half:g}″",
              r"Jy beam$^{-1}$ km s$^{-1}$")
fig.suptitle(f"NGC 3110 compact, pdtau k = 5{MT}: moment 0, both models convolved with the extended clean beam "
             f"({beam_lab}, PA {BE[2]:.1f}°)", fontsize=14)
fig.savefig(os.path.join(FIG_DIR, f"53_mom0_extbeam_{OTAG}.png"), dpi=70); plt.close(fig)

# 52: residuals on the compact dirty cube
fig = plt.figure(figsize=(17, 10), constrained_layout=True)
gs = fig.add_gridspec(2, 3)
rm = max(np.percentile(np.abs(r[cy]), 99.9) for r in rmom0); rdm = np.percentile(np.abs((rmom0[1] - rmom0[0])[cy]), 99.9)
for k, (img, title, lim) in enumerate(((rmom0[0], "residual, no target", rm), (rmom0[1], f"residual, target {TARGET_RES:g}″", rm),
                                       (rmom0[1] - rmom0[0], "target − no target", rdm))):
    panel(fig.add_subplot(gs[0, k]), img, P.cell, 5.0, "RdBu_r", -lim, lim, title, r"compact Jy beam$^{-1}$ km s$^{-1}$")
a = fig.add_subplot(gs[1, :])
a.axvspan(v[L[0]] - DV / 2, v[L[-1]] + DV / 2, color=MUTED, alpha=0.12, lw=0)
a.text(v[L[0]] - DV / 2 + 5, 0.92, "line channels", transform=a.get_xaxis_transform(), color="0.35", fontsize=11)
for n, col, s in zip(names, (BLUE, ORANGE), (stats[names[0]], stats[names[1]])):
    a.plot(v, s["resid_rms_per_channel_mJy_beam"], "-o", color=col, lw=2, ms=6, label=f"{n}: rms")
    a.plot(v, s["resid_mean_per_channel_mJy_beam"], "--s", color=col, lw=2, ms=6, label=f"{n}: mean")
a.axhline(0, color=MUTED, lw=0.8)
a.set_xlabel(r"velocity [km s$^{-1}$]"); a.set_ylabel("residual per channel [mJy/beam]")
a.set_title("residual rms (solid) and mean (dashed) per channel, whole 16″ field (compact beam)", fontsize=12); a.legend(frameon=False)
a.grid(alpha=0.25)
fig.suptitle(f"NGC 3110 compact, pdtau k = 5{MT}: residual d − N y on the compact dirty cube (0.04″ grid), moment 0 over the "
             "line channels", fontsize=14)
fig.savefig(os.path.join(FIG_DIR, f"54_residuals_{OTAG}.png"), dpi=70); plt.close(fig)

# 53: spectra. Positions: the brightest local maxima of the no-target restored moment 0, >= 0.6" apart, central 10"
loc = (mom0[0] == maximum_filter(mom0[0], size=31)) & (mom0[0] > 0.1 * mom0[0].max())
loc[:150] = loc[650:] = False; loc[:, :150] = loc[:, 650:] = False
pk = sorted(zip(*np.nonzero(loc)), key=lambda p: -mom0[0][p])[:5]
fig, ax = plt.subplots(2, 3, figsize=(17, 9.5), constrained_layout=True)
a = ax[0, 0]
for n, m, col in zip(names, models, (BLUE, ORANGE)):
    a.step(v, m.sum((1, 2)), where="mid", color=col, lw=2, label=n)
a.axhline(0, color=MUTED, lw=0.8); a.set_title("integrated spectrum (model, whole field)", fontsize=12)
a.set_ylabel("flux density [Jy]"); a.legend(frameon=False)
stats["peaks"] = []
for a, (j, i) in zip(ax.ravel()[1:], pk):
    dx, dy = -(i - 400) * CELL, (j - 400) * CELL
    for n, r, col in zip(names, restored, (BLUE, ORANGE)):
        a.step(v, 1e3 * r[:, j, i], where="mid", color=col, lw=2, label=n)
    a.axhline(0, color=MUTED, lw=0.8)
    a.set_title(f"one extended beam at ({dx:+.2f}″, {dy:+.2f}″)", fontsize=12); a.set_ylabel("mJy/beam (extended)")
    stats["peaks"].append(dict(offset_arcsec=[round(float(dx), 3), round(float(dy), 3)],
                               peak_mJy_beam=[round(1e3 * float(r[:, j, i].max()), 3) for r in restored]))
for a in ax[1]:
    a.set_xlabel(r"velocity [km s$^{-1}$]")
for a in ax.ravel():
    a.grid(alpha=0.25)
fig.suptitle(f"NGC 3110 compact, pdtau k = 5{MT}: spectra, both models convolved with the extended clean beam ({beam_lab})",
             fontsize=14)
fig.savefig(os.path.join(FIG_DIR, f"55_spectra_extbeam_{OTAG}.png"), dpi=70); plt.close(fig)
json.dump(stats, open(OUT_JSON, "w"), indent=1)
for p in stats["peaks"]:
    print("peak", p)
print("saved figures 53-55 to", FIG_DIR)
