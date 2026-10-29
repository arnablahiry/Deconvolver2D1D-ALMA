#!/usr/bin/env python
"""
Does CLEAN give the same NGC 3110 from the compact and the extended data at one beam? The large-field CLEANs of
msclean_bigfield.py compared at the compact clean beam, on the compact 0.04" grid:

  C   compact          data/ngc3110/compact/big_finished          (restored with its clean beam)
  T   extended tapered data/ngc3110/extended/big_taper_finished    (uv taper to the compact beam, restored with it)
  E   extended         data/ngc3110/extended/big_finished          (0.02" grid: model 2x2-summed to 0.04" and * the
                                                                    compact beam; residual brought to the compact beam
                                                                    (beams.to_beam) and 2x2-averaged)
For each pair (C-T, C-E), restored images and models alone (components * the compact beam), over the line voxels of
several regions: the astrodendro mask of each run's own image (its mask_final.npy, E's at its own beam excluded), their
union and intersection, and where the mean of the two restored cubes > 5 sigma:
  flux 100 (sum C - sum X) / sum X,  RMSE 100 ||C - X|| / ||X||,  the RMSE noise alone would give (line-free rms of the
  difference), and the best scale s with the RMSE after it. Also per channel, and each run's convergence (residual rms
  in the line channels / line-free noise; image = model * beam + residual with tclean's circular convolution).

    python scripts/experiments/compare_clean_bigfield.py
    -> results/NGC3110/figures/clean_bigfield/{compare.json, clean_bigfield.png}
"""
import json
import os
import sys

import numpy as np
from astropy.io import fits
from numpy.fft import rfft2, irfft2, ifftshift

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in ("simple", "scripts/compact_vs_msclean_dendro"):
    sys.path.insert(0, os.path.join(REPO, _p))
from restore import convolve_with_beam  # noqa: E402
import beams  # noqa: E402

D_ = os.path.join(REPO, "data/ngc3110"); RES = os.path.join(REPO, "results/NGC3110")
OUT = os.path.join(RES, "figures", "clean_bigfield"); os.makedirs(OUT, exist_ok=True)
L = json.load(open(os.path.join(RES, "line_channels.json")))["line"]
F = [c for c in range(32) if c not in L and 1 < c < 30]
DV = 25.0
rd = lambda p: np.nan_to_num(fits.getdata(p)).astype(np.float64).squeeze()
bm = lambda p: (lambda h: (h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"]))(fits.getheader(p))
mad = lambda a: 1.4826 * np.median(np.abs(a - np.median(a)))
RUNS = dict(C=f"{D_}/compact/big_finished", T=f"{D_}/extended/big_taper_finished", E=f"{D_}/extended/big_finished")


def circ(x, b, cell):
    """x (nchan, n, n) * beam b, circularly (as tclean restores)"""
    n = x.shape[-1]; d = np.zeros((1, n, n)); d[0, n // 2, n // 2] = 1
    k = convolve_with_beam(d, cell, *b)[0]
    return irfft2(rfft2(x) * rfft2(ifftshift(k)), s=x.shape[1:])


def run_info(d):
    h = fits.getheader(d + "/image.fits"); cell = abs(h["CDELT1"]) * 3600; B = bm(d + "/image.fits")
    img, mod, res = rd(d + "/image.fits"), rd(d + "/model.fits"), rd(d + "/residual.fits")
    noise = np.median([mad(res[c]) for c in F])
    chk = dict(cell=cell, beam=B, components_Jy_kms=float(mod[L].sum() * DV), resid_rms_over_noise=float(np.std(res[L]) / noise),
               resid_extremes_sigma=[float(res[L].max() / noise), float(res[L].min() / noise)],
               restore_mismatch=float(np.linalg.norm((img - circ(mod, B, cell) - res)[L]) / np.linalg.norm(res[L])))
    return chk, cell, B, img[L], mod[L], res[L]


info, cubes = {}, {}
for k, d in RUNS.items():
    if not os.path.exists(d + "/image.fits"):
        print("missing", d); continue
    chk, cell, B, img, mod, res = run_info(d)
    info[k] = chk
    print(f"{k} {os.path.relpath(d, REPO)}: {cell}\" grid, beam {B[0]:.3f}x{B[1]:.3f}\"; components {chk['components_Jy_kms']:.1f} Jy km/s; "
          f"residual {chk['resid_rms_over_noise']:.2f}x noise, extremes {chk['resid_extremes_sigma'][0]:+.0f}/{chk['resid_extremes_sigma'][1]:+.0f} sigma; "
          f"restore mismatch {chk['restore_mismatch']:.3f}", flush=True)
    cubes[k] = (cell, B, img, mod, res)

cC, BC, imgC, modC, resC = cubes["C"]
bpx = beams.omega(BC) / cC ** 2
at_compact = {"C": (imgC, circ(modC, BC, cC))}
if "T" in cubes:
    _, BT, imgT, modT, _ = cubes["T"]
    if not np.allclose(BT, BC, rtol=0.005):                                 # restored with another beam: bring it to C's
        print(f"T restored with {BT[0]:.3f}x{BT[1]:.3f}\" vs C's {BC[0]:.3f}x{BC[1]:.3f}\": T image brought to C's beam", flush=True)
        imgT = beams.to_beam(imgT, BT, BC, cC)
    at_compact["T"] = (imgT, circ(modT, BC, cC))
if "E" in cubes:
    cE, BE, imgE, modE, resE = cubes["E"]
    f = int(round(cC / cE))
    bin_sum = lambda x: x.reshape(x.shape[0], x.shape[1] // f, f, x.shape[2] // f, f).sum((2, 4))
    mE = circ(bin_sum(modE), BC, cC)                                       # components * compact beam, 0.04" grid
    rE = bin_sum(beams.to_beam(resE, BE, BC, cE)) / f ** 2                 # residual at the compact beam, averaged
    at_compact["E"] = (mE + rE, mE)
masks = {"C mask": np.load(RUNS["C"] + "/mask_final.npy").astype(bool)[L]}
if "T" in cubes:
    masks["T mask"] = np.load(RUNS["T"] + "/mask_final.npy").astype(bool)[L]
    masks["C or T"] = masks["C mask"] | masks["T mask"]; masks["C and T"] = masks["C mask"] & masks["T mask"]

out = dict(runs=info, pairs={})
for X in [k for k in ("T", "E") if k in at_compact]:
    for kind, j in (("restored", 0), ("models", 1)):
        A, Bx = at_compact["C"][j], at_compact[X][j]
        # noise of the difference (and of the mean) from an emission-free annulus near the field edge
        n = A.shape[-1]; yy, xx = np.indices((n, n)); r = np.hypot(yy - n // 2, xx - n // 2) * cC
        ann = (r > 0.8 * n * cC / 2) & (r < 0.95 * n * cC / 2)
        sigd = np.median([mad(dd[ann]) for dd in (at_compact["C"][0] - at_compact[X][0])])
        sig = np.median([mad(dd[ann]) for dd in 0.5 * (at_compact["C"][0] + at_compact[X][0])])
        regs = dict(masks); regs["mean > 5 sigma"] = 0.5 * (at_compact["C"][0] + at_compact[X][0]) > 5 * sig
        for rn, Rg in regs.items():
            a, b = A[Rg], Bx[Rg]
            s = float((a * b).sum() / (b * b).sum())
            m = dict(area_arcsec2=float(Rg.any(0).sum() * cC ** 2), flux_C=float(a.sum() / bpx * DV), flux_X=float(b.sum() / bpx * DV),
                     flux_pct=float(100 * (a.sum() - b.sum()) / b.sum()), rmse_pct=float(100 * np.linalg.norm(a - b) / np.linalg.norm(b)),
                     noise_rmse_pct=float(100 * sigd * np.sqrt(a.size) / np.linalg.norm(b)), scale=s,
                     rmse_scaled_pct=float(100 * np.linalg.norm(a - s * b) / np.linalg.norm(s * b)))
            out["pairs"][f"C vs {X} | {kind} | {rn}"] = m
            print(f"C vs {X} {kind:8s} {rn:15s} ({m['area_arcsec2']:5.0f} arcsec2): C {m['flux_C']:6.1f} {X} {m['flux_X']:6.1f} Jy km/s "
                  f"-> flux {m['flux_pct']:+6.2f}% RMSE {m['rmse_pct']:5.2f}% (noise {m['noise_rmse_pct']:4.2f}%), "
                  f"scale {s:.3f} -> {m['rmse_scaled_pct']:5.2f}%", flush=True)
        if kind == "restored":
            Rg = regs["C mask"]
            out["pairs"][f"C vs {X} | per channel (C mask)"] = [
                dict(channel=c, flux_pct=float(100 * (A[i][Rg[i]].sum() - Bx[i][Rg[i]].sum()) / Bx[i][Rg[i]].sum()),
                     rmse_pct=float(100 * np.linalg.norm((A[i] - Bx[i])[Rg[i]]) / np.linalg.norm(Bx[i][Rg[i]])))
                for i, c in enumerate(L) if Rg[i].sum() > 100]
json.dump(out, open(os.path.join(OUT, "compare.json"), "w"), indent=1)
np.savez_compressed(os.path.join(OUT, "moment0.npz"), cell=cC, beam=BC,
                    **{f"{k}_restored": v[0].sum(0) * DV / beams.omega(BC) for k, v in at_compact.items()},
                    **{f"{k}_model": v[1].sum(0) * DV / beams.omega(BC) for k, v in at_compact.items()},
                    **{f"mask_{k.split()[0]}": v.any(0) for k, v in masks.items() if k.endswith("mask")})
print("saved", OUT)
