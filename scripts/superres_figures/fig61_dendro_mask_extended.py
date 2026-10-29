#!/usr/bin/env python
"""
61: a tight 3D emission mask of the EXTENDED configuration from astrodendro, on its multiscale CLEAN image
(data/ngc3110/extended/<CLEAN>/image.fits: components * clean beam + residual, so the sidelobes of what was cleaned
are gone and what was not cleaned is still there). Recipe of the GUI's dendro_mask (gui/deconvgui/worker.py): the
voxels of every trunk structure above MIN_VALUE sigma (split at MIN_DELTA sigma, at least MIN_BEAMS clean beams of
voxels) whose peak reaches PEAK sigma. No binning (the beam is ~10 px FWHM here) and no dilation. Two changes:

  sigma PER CHANNEL: the robust rms of the tclean residual in that channel. In the line channels the residual holds
    the sidelobes of the emission CLEAN left behind, 1.3-4.5x the line-free noise; with one line-free sigma the
    NEGATED image has about as many > 10 sigma structures as the image itself (all sidelobes), so no threshold
    separates emission from sidelobes. SIGMA=global restores the GUI's single value.
  PEAK=auto (default): the same dendrogram on the negated image (which has no emission, only noise and sidelobes)
    sets the bar: PEAK = the brightest negated structure's peak + 1 sigma (at least 5), so the mask keeps nothing that
    the negated image would also produce.

Saves the mask to data/ngc3110/extended/ms_clean_masks/dendro_<CLEAN>.npy (uint8, (32, 800, 800)) with a JSON, and
the figure figures/experiments/11_target_resolution/61_dendro_mask_extended_<CLEAN>.png: moment 0 of the CLEAN image
and of the compact no-target model at the extended beam, both outlined by the dendrogram mask (cyan) and CLEAN's own
mask (white, dashed); the number of masked channels per pixel; the masked area per channel; and channel maps.
Prints the compact model's and CLEAN components' flux inside / outside the mask.

    CLEAN=ms_clean MIN_VALUE=3 MIN_DELTA=1 PEAK=auto MIN_BEAMS=2 SIGMA=channel python scripts/superres_figures/fig61_dendro_mask_extended.py
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
from astrodendro import Dendrogram

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
from restore import convolve_with_beam  # noqa: E402

CLEAN = os.environ.get("CLEAN", "ms_clean")
P = dict(min_value=float(os.environ.get("MIN_VALUE", 3)), min_delta=float(os.environ.get("MIN_DELTA", 1)),
         peak=os.environ.get("PEAK", "auto"), min_beams=float(os.environ.get("MIN_BEAMS", 2)),
         sigma=os.environ.get("SIGMA", "channel"))
COMPACT = os.environ.get("COMPACT", os.path.join(REPO, "data/experiments/04_pd_tau_reweighting/compact/pdtau_k5.npy"))
EXT = os.path.join(REPO, "data/ngc3110/extended")
CDIR = os.path.join(EXT, CLEAN)
OUT_MASK = os.path.join(EXT, "ms_clean_masks", f"dendro_{CLEAN}.npy")
OUT = os.environ.get("OUT", os.path.join(REPO, f"figures/experiments/11_target_resolution/61_dendro_mask_extended_{CLEAN}.png"))
CELL, DV = 0.02, 25.0
vel = 5073.9 + 25.0 * (np.arange(32) - 15.5)
L = [i for i, v in enumerate(vel) if 4750 <= v <= 5280]
FREE = [i for i in range(32) if i not in L]

rd = lambda p: np.nan_to_num(fits.getdata(p).astype(np.float64)).squeeze()
h = fits.getheader(os.path.join(CDIR, "image.fits"))
BE = (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"])
BEAM_PX = np.pi * BE[0] * BE[1] / (4 * np.log(2)) / CELL ** 2
img, res, cmask = rd(os.path.join(CDIR, "image.fits")), rd(os.path.join(CDIR, "residual.fits")), rd(os.path.join(CDIR, "mask.fits")) > 0
cmodel = rd(os.path.join(CDIR, "model.fits"))

# ---------------------------------------------------------------- the mask
t0 = time.time()
rms = 1.4826 * np.median(np.abs(res - np.median(res, axis=(1, 2), keepdims=True)), axis=(1, 2))
quiet = rms <= 1.5 * rms.min()
sigma = float(np.median(rms[quiet]))                                         # line-free noise
sig_c = rms if P["sigma"] == "channel" else np.full(32, sigma)               # the sigma each channel is measured in
snr = img / sig_c[:, None, None]
kw = dict(min_value=P["min_value"], min_delta=P["min_delta"], min_npix=max(1, int(round(P["min_beams"] * BEAM_PX))))
neg_peaks = sorted((s.get_peak()[1] for s in Dendrogram.compute(-snr, **kw).trunk), reverse=True)
P["peak"] = max(5.0, np.ceil(neg_peaks[0]) + 1) if P["peak"] == "auto" and neg_peaks else (5.0 if P["peak"] == "auto" else float(P["peak"]))
dg = Dendrogram.compute(snr, **kw)
keep = [s for s in dg.trunk if s.get_peak()[1] >= P["peak"]]
neg_kept = sum(p >= P["peak"] for p in neg_peaks)
mask = np.zeros(img.shape, bool)
for s in keep:
    mask[s.indices(subtree=True)] = True
os.makedirs(os.path.dirname(OUT_MASK), exist_ok=True)
np.save(OUT_MASK, mask.astype(np.uint8))

# ---------------------------------------------------------------- numbers
compact = np.load(COMPACT).astype(np.float64)
inside = lambda cube, m: float((cube * m)[L].sum() * DV)
ap = lambda m2: float(m2.sum() * CELL ** 2)                                  # arcsec^2
info = dict(clean=CLEAN, params=P, sigma_mJy_beam=1e3 * sigma, quiet_channels=[int(c) for c in np.nonzero(quiet)[0]],
            sigma_per_channel_mJy_beam=[round(1e3 * float(v), 4) for v in sig_c],
            negated_image=dict(trunks=len(neg_peaks), peaks=[round(float(v), 2) for v in neg_peaks[:20]], kept=int(neg_kept)),
            structures=len(keep), trunk=len(dg.trunk), voxel_fraction=float(mask.mean()),
            line_free_voxels_masked=int(mask[FREE].sum()),
            area_any_channel_arcsec2=ap(mask.any(0)), clean_mask_area_any_channel_arcsec2=ap(cmask[L].any(0)),
            voxels_vs_clean_mask=dict(dendro=int(mask.sum()), clean=int(cmask.sum()), both=int((mask & cmask).sum())),
            area_per_channel_arcsec2=[round(ap(mask[c]), 3) for c in range(32)],
            clean_mask_area_per_channel_arcsec2=[round(ap(cmask[c]), 3) for c in range(32)],
            compact_model_flux_Jy_kms=dict(total=float(compact[L].sum() * DV), in_dendro=inside(compact, mask),
                                           in_clean_mask=inside(compact, cmask)),
            clean_components_flux_Jy_kms=dict(total=float(cmodel[L].sum() * DV), in_dendro=inside(cmodel, mask),
                                              in_clean_mask=inside(cmodel, cmask)),
            clean_image_flux_in_dendro_Jy_kms=float((img * mask)[L].sum() / BEAM_PX * DV),
            seconds=round(time.time() - t0))
json.dump(info, open(OUT_MASK.replace(".npy", ".json"), "w"), indent=1)
cf, kf = info["compact_model_flux_Jy_kms"], info["clean_components_flux_Jy_kms"]
print(f"negated image: {len(neg_peaks)} trunk structures, peaks {np.round(neg_peaks[:8], 1).tolist()} -> PEAK {P['peak']:g} sigma "
      f"({neg_kept} negated structures would pass)")
print(f"{CLEAN}: line-free sigma {1e3 * sigma:.4f} mJy/beam ({quiet.sum()} quiet channels), sigma {P['sigma']}; "
      f"{len(keep)} of {len(dg.trunk)} trunk "
      f"structures kept; {mask.mean():.2%} of voxels, {info['line_free_voxels_masked']} in line-free channels")
print(f"area (any channel): dendro {info['area_any_channel_arcsec2']:.1f} arcsec^2, CLEAN mask "
      f"{info['clean_mask_area_any_channel_arcsec2']:.1f}; voxels dendro {mask.sum()}, CLEAN {cmask.sum()}, both "
      f"{(mask & cmask).sum()}")
print(f"compact model: {cf['total']:.0f} Jy km/s, {cf['in_dendro']:.0f} ({cf['in_dendro'] / cf['total']:.0%}) in the dendro mask, "
      f"{cf['in_clean_mask']:.0f} ({cf['in_clean_mask'] / cf['total']:.0%}) in CLEAN's mask")
print(f"CLEAN components: {kf['total']:.0f} Jy km/s, {kf['in_dendro']:.0f} in the dendro mask, {kf['in_clean_mask']:.0f} in "
      f"CLEAN's mask; CLEAN image in the dendro mask {info['clean_image_flux_in_dendro_Jy_kms']:.0f} Jy km/s")
print("saved", OUT_MASK)

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.family": "serif", "font.size": 12, "xtick.direction": "in", "ytick.direction": "in"})
HALF = 6.0
hp = int(HALF / CELL); c0 = 400; sl = np.s_[c0 - hp:c0 + hp, c0 - hp:c0 + hp]
ext = [HALF, -HALF, -HALF, HALF]
m_img = img[L].sum(0) * DV
m_cmp = convolve_with_beam(compact[L].sum(0, keepdims=True) * DV, CELL, *BE)[0]          # Jy/beam km/s
dmask2, cmask2 = mask[L].any(0), cmask[L].any(0)
UNIT = r"Jy beam$^{-1}$ km s$^{-1}$"


def outlines(a, d2, c2):
    a.contour(d2[sl].astype(float), levels=[0.5], colors="cyan", linewidths=1.0, extent=ext, origin="lower")
    a.contour(c2[sl].astype(float), levels=[0.5], colors="white", linewidths=0.9, linestyles="--", extent=ext, origin="lower")


def axes_lab(a):
    a.set_xlabel(r'$\Delta$RA ["]'); a.set_ylabel(r'$\Delta$Dec ["]')


fig = plt.figure(figsize=(24, 17), constrained_layout=True)
gs = fig.add_gridspec(3, 4)
vm = np.percentile(m_img[sl], 99.9)
for k, (m0, t) in enumerate(((m_img, f"extended CLEAN image ({CLEAN})"), (m_cmp, "compact pdtau k = 5 (no target) ∗ ext. beam"))):
    a = fig.add_subplot(gs[0, k])
    im = a.imshow(m0[sl], origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=vm, interpolation="nearest")
    outlines(a, dmask2, cmask2); a.set_title(t + "\ncyan: dendrogram mask, white dashed: CLEAN's mask", fontsize=12)
    axes_lab(a); plt.colorbar(im, ax=a, shrink=0.85, label=UNIT)
a = fig.add_subplot(gs[0, 2])
nch = mask.sum(0).astype(float); nch[nch == 0] = np.nan
im = a.imshow(nch[sl], origin="lower", extent=ext, cmap="viridis", vmin=1, vmax=np.nanmax(nch), interpolation="nearest")
a.contour(cmask2[sl].astype(float), levels=[0.5], colors="0.3", linewidths=0.9, linestyles="--", extent=ext, origin="lower")
a.set_title("dendrogram mask: channels per pixel\n(grey dashed: CLEAN's mask)", fontsize=12); axes_lab(a)
plt.colorbar(im, ax=a, shrink=0.85, label="channels")
a = fig.add_subplot(gs[0, 3])
a.axvspan(vel[L[0]] - DV / 2, vel[L[-1]] + DV / 2, color="0.5", alpha=0.12, lw=0)
a.plot(vel, info["area_per_channel_arcsec2"], "-o", color="#1a9e77", lw=2, ms=6, label="dendrogram mask")
a.plot(vel, info["clean_mask_area_per_channel_arcsec2"], "--s", color="0.35", lw=2, ms=5, label="CLEAN's mask")
a.set_xlabel(r"velocity [km s$^{-1}$]"); a.set_ylabel(r"masked area [arcsec$^2$]"); a.legend(frameon=False)
a.set_title("masked area per channel (grey band: line channels)", fontsize=12); a.grid(alpha=0.25)
chans = L[1::3][:8] if len(L[1::3]) >= 8 else L[::max(1, len(L) // 8)][:8]
for k, c in enumerate(chans):
    a = fig.add_subplot(gs[1 + k // 4, k % 4])
    cm_ = snr[c]
    im = a.imshow(cm_[sl], origin="lower", extent=ext, cmap="inferno", vmin=-2, vmax=np.percentile(cm_[sl], 99.8),
                  interpolation="nearest")
    outlines(a, mask[c], cmask[c]); axes_lab(a)
    a.set_title(f"channel {c}, {vel[c]:.0f} km/s", fontsize=12)
    plt.colorbar(im, ax=a, shrink=0.85, label="σ")
fig.suptitle(f"NGC 3110 extended: astrodendro mask on the multiscale CLEAN image ({CLEAN}; σ {P['sigma']}; ≥{P['min_value']:g}σ, "
             f"split {P['min_delta']:g}σ, peak ≥{P['peak']:g}σ, ≥{P['min_beams']:g} beams; negated image: {neg_kept} of "
             f"{len(neg_peaks)} structures pass): "
             f"{len(keep)} structures, {info['area_any_channel_arcsec2']:.0f} arcsec² in any channel "
             f"(CLEAN's mask {info['clean_mask_area_any_channel_arcsec2']:.0f})", fontsize=14)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=60)
print("saved", OUT)
