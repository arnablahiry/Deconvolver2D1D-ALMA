#!/usr/bin/env python
"""
56-59: compact pdtau (k = 5) without and with a target resolution (gpu_pd.Op target_fwhm_px) vs the EXTENDED
configuration's multiscale CLEAN (data/ngc3110/extended/<CLEAN>/, scripts/msclean_cube.py), all at the extended
clean beam. The compact models are convolved with that beam (no residual added); CLEAN appears both as its
components convolved with the same beam (like for like) and as its image (components * beam + residual).

The residuals are all taken on the EXTENDED dirty cube with one operator (gpu_pd.Op on the extended PSF, same
natural weighting as tclean): d_ext - N_ext x for the two compact models and the CLEAN components. For the compact
models this asks how well a model built from the compact data alone predicts the long-baseline data; tclean's own
residual is printed and plotted as a check on the operator. Writes, in figures/experiments/11_target_resolution/:

  56_vs_clean_mom0_<tag>.png       moment 0: the four maps, central 10" and 3"
  57_vs_clean_diff_<tag>.png       moment 0 differences against CLEAN's components, central 10" and 3"
  58_vs_clean_residuals_<tag>.png  residual on the extended dirty cube: moment 0 maps, rms and mean per channel
  59_vs_clean_spectra_<tag>.png    integrated model spectra and one-beam spectra at the brightest peaks

and the numbers to data/experiments/11_target_resolution/vs_clean_<tag>.json.


MASK (optional): a (32, 800, 800) 0/1 cube on the 0.02" grid (e.g. fig61's dendrogram mask of the extended CLEAN
image). Every model cube is multiplied by it BEFORE the convolution with the extended beam and before the residuals,
moment-0 comparisons are taken over its footprint (pixels masked in any line channel) instead of the whole box, and
the outputs get the suffix _<mask file name>.

    CLEAN=ms_clean MODEL=... TARGET_MODEL=... TAG=t0.1134 [MASK=...] python scripts/superres_figures/fig56_compact_vs_extended_clean.py
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
from common import G, DATA, EXP_DATA, EXP_FIG, line_channels, noise_channels, read, vel  # noqa: E402
from restore import convolve_with_beam  # noqa: E402

TAG = os.environ.get("TAG", "t0.1134")
CLEAN = os.environ.get("CLEAN", "ms_clean")
MODEL = os.environ.get("MODEL", os.path.join(EXP_DATA, "04_pd_tau_reweighting/compact/pdtau_k5.npy"))
TARGET_MODEL = os.environ.get("TARGET_MODEL", os.path.join(EXP_DATA, f"11_target_resolution/pdtau_k5_{TAG}.npy"))
TARGET_RES = float(json.load(open(TARGET_MODEL.replace(".npy", ".json")))["target_res"])
CDIR = os.path.join(DATA, "ngc3110", "extended", CLEAN)
FIG_DIR = os.path.join(EXP_FIG, "11_target_resolution")
MASK = os.environ.get("MASK")
SUFFIX = (TAG if CLEAN == "ms_clean" else f"{TAG}_{CLEAN}") + (f"_{os.path.basename(MASK).replace('.npy', '')}" if MASK else "")
OUT_JSON = os.path.join(EXP_DATA, "11_target_resolution", f"vs_clean_{SUFFIX}.json")
CELL, DV = 0.02, 25.0
C10, C3 = np.s_[150:650, 150:650], np.s_[325:475, 325:475]
R10, R3 = np.zeros((800, 800), bool), np.zeros((800, 800), bool)          # regions for the moment-0 comparisons
R10[C10] = True; R3[C3] = True
BLUE, ORANGE, GREEN, MUTED = "#2a6fdb", "#e07b00", "#1a9e77", "#8a8a8a"

rd = lambda p: np.nan_to_num(fits.getdata(p).astype(np.float64)).squeeze()
h = fits.getheader(os.path.join(CDIR, "image.fits"))
BE = (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"])
BEAM_PX = np.pi * BE[0] * BE[1] / (4 * np.log(2)) / CELL ** 2
L = line_channels("ngc3110"); NC = [c for c in range(32) if c not in L]           # outside the line window (11 channels)
v = vel("ngc3110")

names = ("compact, no target", f"compact, target {TARGET_RES:g}″", "extended CLEAN")
cols = (BLUE, ORANGE, GREEN)
models = [np.load(MODEL).astype(np.float64), np.load(TARGET_MODEL).astype(np.float64), rd(os.path.join(CDIR, "model.fits"))]
c_image, c_resid, c_mask = rd(os.path.join(CDIR, "image.fits")), rd(os.path.join(CDIR, "residual.fits")), rd(os.path.join(CDIR, "mask.fits")) > 0
M = np.load(MASK).astype(bool) if MASK else None
IMG_MASK = M if MASK else c_mask                                          # where the CLEAN image's flux is summed
if MASK:
    models = [m * M for m in models]
    c_image = c_image * M
    R10 &= M[L].any(0); R3 &= M[L].any(0)
    print(f"masked by {MASK}: {M.mean():.2%} of voxels; CLEAN components and image masked too")
print(f"{CLEAN}: extended clean beam {BE[0]:.3f}\" x {BE[1]:.3f}\" @ {BE[2]:.1f} deg ({BEAM_PX:.0f} px); "
      f"line channels {L[0]}-{L[-1]}, extended line-free channels {NC}")

restored = [convolve_with_beam(m, CELL, *BE) for m in models]          # Jy/beam (extended), no residual
op = G.Op(read("ngc3110", "extended", "dirty_beam_cube.fits"), read("ngc3110", "extended", "dirty_cube.fits"))
d_ext = op.d.cpu().numpy().astype(np.float64)
resid = [op.residual_sky(G.to_t(m)).cpu().numpy().astype(np.float64) for m in models]   # d_ext - N_ext x
del op
sigma = float(d_ext[NC].std())                                         # extended line-free noise, Jy/beam
mom0 = [r[L].sum(0) * DV for r in restored] + [c_image[L].sum(0) * DV]  # Jy/beam km/s
rmom0 = [r[L].sum(0) * DV for r in resid] + [c_resid[L].sum(0) * DV]
mlab = names + ("extended CLEAN image",)

# ---------------------------------------------------------------- numbers
stats = dict(mask=MASK, clean=CLEAN, model=MODEL, target_model=TARGET_MODEL, target_res_arcsec=TARGET_RES,
             extended_beam_arcsec_deg=BE, line_channels=L, extended_line_free_channels=NC,
             extended_noise_sigma_mJy_beam=1e3 * sigma,
             tclean_vs_operator_residual_rms_mJy_beam=1e3 * float((resid[2] - c_resid)[L].std()))
ref = mom0[2][R10]
for n, m, r, rr, m0 in zip(names, models, restored, resid, mom0):
    s = dict(flux_line_Jy_kms=float(m[L].sum() * DV), flux_line_central10_Jy_kms=float(m[L][(slice(None),) + C10].sum() * DV),
             flux_line_free_Jy=float(m[NC].sum()),
             resid_rms_line_mJy_beam=1e3 * float(rr[L].std()),
             resid_rms_line_central10_mJy_beam=1e3 * float(rr[L][(slice(None),) + C10].std()),
             resid_rms_line_in_clean_mask_mJy_beam=1e3 * float(rr[L][c_mask[L]].std()),
             resid_rms_line_free_mJy_beam=1e3 * float(rr[NC].std()),
             resid_mean_line_mJy_beam=1e3 * float(rr[L].mean()),
             resid_rms_per_channel_mJy_beam=[round(1e3 * float(rr[c].std()), 4) for c in range(32)],
             resid_mean_per_channel_mJy_beam=[round(1e3 * float(rr[c].mean()), 4) for c in range(32)],
             restored_peak_mJy_beam=1e3 * float(r.max()), mom0_peak_Jy_beam_kms=float(m0.max()),
             mom0_vs_clean_components=dict(
                 rel_rms_diff_central10=float(np.linalg.norm(m0[R10] - ref) / np.linalg.norm(ref)),
                 pearson_central10=float(np.corrcoef(m0[R10], ref)[0, 1]),
                 pearson_central3=float(np.corrcoef(m0[R3], mom0[2][R3])[0, 1])),
             mom0_vs_clean_image=dict(
                 rel_rms_diff_central10=float(np.linalg.norm(m0[R10] - mom0[3][R10]) / np.linalg.norm(mom0[3][R10])),
                 pearson_central10=float(np.corrcoef(m0[R10], mom0[3][R10])[0, 1])))
    stats[n] = s
stats["extended CLEAN"]["tclean_residual_rms_line_mJy_beam"] = 1e3 * float(c_resid[L].std())
stats["extended CLEAN"]["tclean_residual_rms_per_channel_mJy_beam"] = [round(1e3 * float(c_resid[c].std()), 4) for c in range(32)]
stats["extended CLEAN"]["image_flux_line_in_mask_Jy_kms"] = float((c_image * IMG_MASK)[L].sum() / BEAM_PX * DV)
stats["extended CLEAN"]["residual_flux_line_Jy_kms"] = float(c_resid[L].sum() / BEAM_PX * DV)

print(f"extended noise sigma {1e3 * sigma:.4f} mJy/beam; tclean residual vs this operator's residual of the CLEAN "
      f"components: rms difference {stats['tclean_vs_operator_residual_rms_mJy_beam']:.4f} mJy/beam (line channels)")
for n in names:
    s = stats[n]; c = s["mom0_vs_clean_components"]
    print(f"{n:22s} flux {s['flux_line_Jy_kms']:6.1f} Jy km/s (central 10\" {s['flux_line_central10_Jy_kms']:6.1f}) | "
          f"ext. residual rms line {s['resid_rms_line_mJy_beam']:.4f} (central 10\" {s['resid_rms_line_central10_mJy_beam']:.4f}, "
          f"in CLEAN mask {s['resid_rms_line_in_clean_mask_mJy_beam']:.4f}), line-free {s['resid_rms_line_free_mJy_beam']:.4f}, "
          f"mean {s['resid_mean_line_mJy_beam']:+.4f} mJy/beam | mom0 peak {s['mom0_peak_Jy_beam_kms']:.3f} | vs CLEAN comps: "
          f"rel rms {c['rel_rms_diff_central10']:.3f}, r10 {c['pearson_central10']:.3f}, r3 {c['pearson_central3']:.3f}")
c = stats["extended CLEAN"]
print(f"CLEAN image: flux in mask {c['image_flux_line_in_mask_Jy_kms']:.1f} Jy km/s; residual flux {c['residual_flux_line_Jy_kms']:.1f} "
      f"Jy km/s; tclean residual rms line {c['tclean_residual_rms_line_mJy_beam']:.4f} mJy/beam")

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.family": "serif", "font.size": 12, "xtick.direction": "in", "ytick.direction": "in"})
os.makedirs(FIG_DIR, exist_ok=True)
beam_lab = f"{BE[0]:.3f}″×{BE[1]:.3f}″, PA {BE[2]:.1f}°"
UNIT = r"Jy beam$^{-1}$ km s$^{-1}$"
title_pre = f"NGC 3110: compact pdtau k = 5 vs extended multiscale CLEAN ({CLEAN})" + (", all models × the dendrogram mask" if MASK else "")


def panel(a, img, half_arcsec, cmap, lo, hi, title, label=UNIT, contour=None):
    c = img.shape[0] // 2; hp = int(round(half_arcsec / CELL))
    sl = np.s_[c - hp:c + hp, c - hp:c + hp]; ext = [hp * CELL, -hp * CELL, -hp * CELL, hp * CELL]
    im = a.imshow(img[sl], origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    if contour is not None:
        a.contour(contour[sl], levels=[0.5], colors="k", linewidths=0.6, extent=ext, origin="lower")
    a.set_title(title, fontsize=11); a.set_xlabel(r'$\Delta$RA ["]'); a.set_ylabel(r'$\Delta$Dec ["]')
    plt.colorbar(im, ax=a, shrink=0.85, label=label)


# 54: maps
fig, ax = plt.subplots(2, 4, figsize=(22, 10), constrained_layout=True)
vm = np.percentile(mom0[2][C10], 99.95)
titles = (f"{names[0]} ∗ ext. beam", f"{names[1]} ∗ ext. beam", "extended CLEAN: components ∗ beam",
          "extended CLEAN: image (+ residual)")
for row, half in enumerate((5.0, 1.5)):
    for k in range(4):
        panel(ax[row, k], mom0[k], half, "inferno", 0, vm, titles[k] if row == 0 else f"central {2 * half:g}″")
fig.suptitle(f"{title_pre}: moment 0 at the extended clean beam ({beam_lab})", fontsize=14)
fig.savefig(os.path.join(FIG_DIR, f"56_vs_clean_mom0_{SUFFIX}.png"), dpi=60); plt.close(fig)

# 55: differences against CLEAN's components (and CLEAN's residual for scale)
diffs = [(mom0[0] - mom0[2], f"{names[0]} − CLEAN comps"), (mom0[1] - mom0[2], f"{names[1]} − CLEAN comps"),
         (mom0[1] - mom0[0], "target − no target"), (rmom0[3], "CLEAN residual (tclean)")]
dm = 0.5 * vm
mask_any = IMG_MASK[L].any(0)
fig, ax = plt.subplots(2, 4, figsize=(22, 10), constrained_layout=True)
for row, half in enumerate((5.0, 1.5)):
    for k, (img, t) in enumerate(diffs):
        panel(ax[row, k], img, half, "RdBu_r", -dm, dm, t if row == 0 else f"central {2 * half:g}″",
              contour=mask_any.astype(float) if k < 2 else None)
fig.suptitle(f"{title_pre}: moment-0 differences (black: union of the {'dendrogram' if MASK else 'CLEAN'} mask over the line channels), "
             f"±{dm:.2f} Jy/beam km/s", fontsize=13)
fig.savefig(os.path.join(FIG_DIR, f"57_vs_clean_diff_{SUFFIX}.png"), dpi=60); plt.close(fig)

# 56: residuals on the extended dirty cube
fig = plt.figure(figsize=(22, 11), constrained_layout=True)
gs = fig.add_gridspec(2, 4)
rm = np.percentile(np.abs(np.stack([r[C10] for r in rmom0])), 99.9)
rt = [f"{names[0]}: d − N x", f"{names[1]}: d − N y", "CLEAN comps: d − N x (this operator)", "CLEAN: tclean residual"]
for k in range(4):
    panel(fig.add_subplot(gs[0, k]), rmom0[k], 5.0, "RdBu_r", -rm, rm, rt[k], label=f"extended {UNIT}")
a = fig.add_subplot(gs[1, :])
a.axvspan(v[L[0]] - DV / 2, v[L[-1]] + DV / 2, color=MUTED, alpha=0.12, lw=0)
a.text(v[L[0]] - DV / 2 + 5, 0.93, "line channels", transform=a.get_xaxis_transform(), color="0.35", fontsize=11)
for n, col in zip(names, cols):
    s = stats[n]
    a.plot(v, s["resid_rms_per_channel_mJy_beam"], "-o", color=col, lw=2, ms=6, label=f"{n}: rms")
    a.plot(v, s["resid_mean_per_channel_mJy_beam"], "--s", color=col, lw=2, ms=5, label=f"{n}: mean")
a.plot(v, stats["extended CLEAN"]["tclean_residual_rms_per_channel_mJy_beam"], ":", color=GREEN, lw=2,
       label="CLEAN: tclean residual rms")
a.axhline(0, color=MUTED, lw=0.8); a.axhline(1e3 * sigma, color="0.3", lw=1, ls="-.")
a.text(v[-1], 1e3 * sigma, f" σ = {1e3 * sigma:.3f}", va="bottom", ha="right", color="0.3", fontsize=11)
a.set_xlabel(r"velocity [km s$^{-1}$]"); a.set_ylabel("residual per channel [mJy/beam, extended]")
a.set_title("residual rms (solid) and mean (dashed) per channel on the extended dirty cube, whole 16″ field", fontsize=12)
a.legend(frameon=False, ncol=4, fontsize=10); a.grid(alpha=0.25)
fig.suptitle(f"{title_pre}: residuals on the EXTENDED dirty cube, one operator for all three models; "
             "moment 0 over the line channels", fontsize=14)
fig.savefig(os.path.join(FIG_DIR, f"58_vs_clean_residuals_{SUFFIX}.png"), dpi=60); plt.close(fig)

# 57: spectra. Positions: brightest local maxima of the CLEAN components' moment 0, >= 0.6" apart, central 10"
loc = (mom0[2] == maximum_filter(mom0[2], size=31)) & (mom0[2] > 0.1 * mom0[2].max())
keep = np.zeros_like(loc); keep[C10] = True; loc &= keep
pk = sorted(zip(*np.nonzero(loc)), key=lambda p: -mom0[2][p])[:5]
fig, ax = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
a = ax[0, 0]
for n, m, col in zip(names, models, cols):
    a.step(v, m.sum((1, 2)), where="mid", color=col, lw=2, label=f"{n} (model)")
a.step(v, (c_image * IMG_MASK).sum((1, 2)) / BEAM_PX, where="mid", color="0.3", lw=1.5, ls="--",
       label="CLEAN image in the " + ("dendrogram" if MASK else "its") + " mask")
a.axhline(0, color=MUTED, lw=0.8); a.set_title("integrated spectrum, " + ("inside the mask" if MASK else "whole field"), fontsize=12)
a.set_ylabel("flux density [Jy]"); a.legend(frameon=False, fontsize=10)
stats["peaks"] = []
for a, (j, i) in zip(ax.ravel()[1:], pk):
    dx, dy = -(i - 400) * CELL, (j - 400) * CELL
    for n, r, col in zip(names, restored, cols):
        a.step(v, 1e3 * r[:, j, i], where="mid", color=col, lw=2, label=n)
    a.step(v, 1e3 * c_image[:, j, i], where="mid", color="0.3", lw=1.5, ls="--", label="CLEAN image")
    a.axhline(0, color=MUTED, lw=0.8)
    a.set_title(f"one extended beam at ({dx:+.2f}″, {dy:+.2f}″)", fontsize=12); a.set_ylabel("mJy/beam (extended)")
    stats["peaks"].append(dict(offset_arcsec=[round(float(dx), 3), round(float(dy), 3)],
                               peak_mJy_beam={n: round(1e3 * float(r[:, j, i].max()), 3)
                                              for n, r in zip(names + ("CLEAN image",), restored + [c_image])}))
ax[0, 1].legend(frameon=False, fontsize=10)
for a in ax[1]:
    a.set_xlabel(r"velocity [km s$^{-1}$]")
for a in ax.ravel():
    a.grid(alpha=0.25)
fig.suptitle(f"{title_pre}: spectra at the extended clean beam ({beam_lab})", fontsize=14)
fig.savefig(os.path.join(FIG_DIR, f"59_vs_clean_spectra_{SUFFIX}.png"), dpi=60); plt.close(fig)

json.dump(stats, open(OUT_JSON, "w"), indent=1)
for p in stats["peaks"]:
    print("peak", p)
print("saved", OUT_JSON, "and figures 56-59 to", FIG_DIR)
