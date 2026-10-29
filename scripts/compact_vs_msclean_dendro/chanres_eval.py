#!/usr/bin/env python
"""
Why 2D-1D: NGC 3110 deconvolved on 20, 10 and 5 km/s channels (results/NGC3110/g<DV>_c0.16: 0.16" pixels, 51" field,
gallery20_casa.py + gallery20_deconv.py with CHANKMS, GCELL=0.16), with the 2D-1D dictionary (L1D = 5) and with the 2D
starlet alone in every channel (L1D = 0), both against the joint compact + extended image on the same channels at the
compact clean beam, lambda/B_max and 0.40" (ladder_joint_<DV>kms/b00, b05, b07; 20 km/s: ladder_joint_20kms).

Region: the footprint of the 20 km/s ladder mask at that beam (masks.npz, the same sky area at every channel width),
over the line channels of each grid. Metrics as the ladder: cube RMSE and flux (%), moment-0 RMSE (%), moment-1 and -2
RMSE (km/s; voxels above 3 sigma of the joint image, the same voxels in both cubes); faint emission: the RMSE over the
voxels where the joint image is 3-10 sigma, and the fraction of their joint flux the model recovers; and the S/N per
channel (the joint cube's peak over its line-free noise).

    python scripts/compact_vs_msclean_dendro/chanres_eval.py
    -> results/NGC3110/figures/resolution_ladder_joint20/chanres_eval.json
"""
import json
import os
import sys

import numpy as np
from astropy.io import fits
from scipy.ndimage import convolve1d

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
from restore import convolve_with_beam  # noqa: E402

R = os.path.join(REPO, "results", "NGC3110"); OUT = os.path.join(R, "figures", "resolution_ladder_joint20")
LAD = json.load(open(os.path.join(OUT, "resolution_ladder.json"))); MK = np.load(os.path.join(OUT, "masks.npz"))
CELL = 0.16; IB = (0, 5, 7, 10)                       # compact beam, lambda/B_max, 0.40", extended beam
rd = lambda p: np.nan_to_num(fits.getdata(p)).astype(np.float64).squeeze()


def to016(x):                                      # 0.04" (1280 px) -> 0.16" (320 px), pixel centres aligned, averaged
    k = np.array([0.5, 1, 1, 1, 0.5]) / 4
    return convolve1d(convolve1d(x, k, axis=-1, mode="constant"), k, axis=-2, mode="constant")[..., ::4, ::4]


def moments(cube, foot, thr, v):
    w = np.where(cube >= thr[:, None, None], cube, 0.0); s = w.sum(0); ok = foot & (s > 0)
    m1 = np.where(ok, (w * v[:, None, None]).sum(0) / np.where(ok, s, 1), np.nan)
    m2 = np.where(ok, np.sqrt(np.clip((w * (v[:, None, None] - np.nan_to_num(m1)) ** 2).sum(0) / np.where(ok, s, 1), 0, None)), np.nan)
    return m1, m2, s


out = {}
for dv in (20, 15, 10, 5, 2.6):
    gd = os.path.join(R, f"g{dv:g}_c0.16"); G = json.load(open(os.path.join(gd, "grid.json")))
    L, F = G["line"], G["line_free"]; v = (G["start_kms"] + dv * np.arange(G["nchan"]))[L]
    mods = {n: os.path.join(gd, f"model_0.16_support{s}.npy") for n, s in (("2D-1D", ""), ("2D per channel", "_L0"))}
    if not all(os.path.exists(p) for p in mods.values()):
        print(f"{dv} km/s: models missing"); continue
    X = {n: np.load(p).astype(np.float64)[L] for n, p in mods.items()}
    out[f"{dv:g}"] = {}
    for ib in IB:
        jd = os.path.join(R, ("ladder_joint_20kms" if dv == 20 else f"ladder_joint_{dv:g}kms") + f"/b{ib:02d}/clean_finished")
        if not os.path.exists(os.path.join(jd, "image.fits")):
            print(f"{dv} km/s b{ib:02d}: joint reference missing"); continue
        b = tuple(LAD["rows"][ib][k] for k in ("bmaj", "bmin", "bpa"))
        Ia, Ra = rd(os.path.join(jd, "image.fits")), rd(os.path.join(jd, "residual.fits"))
        I, res = to016(Ia[L]), to016(Ra)
        foot = MK[f"b{ib:02d}"].any(0)[::2, ::2]; M = np.broadcast_to(foot, I.shape)
        sig = 1.4826 * np.median(np.abs(res - np.median(res, axis=(1, 2), keepdims=True)), axis=(1, 2))
        thr = 3 * sig[L]; snr = float(np.max(I) / np.median(sig[F]))
        j1, j2, j0 = moments(I, foot, thr, v)
        row = dict(beam=b, snr_peak_per_channel=snr)
        for n, x in X.items():
            C = convolve_with_beam(x, CELL, *b); c, i = C[M], I[M]; c1, c2, c0 = moments(C, foot, thr, v)
            ok = np.isfinite(c1) & np.isfinite(j1)
            snr_v = I / sig[L][:, None, None]; fm = M & (snr_v >= 3) & (snr_v < 10)          # the faint voxels
            row[n + " faint"] = dict(rmse=float(100 * np.linalg.norm((C - I)[fm]) / np.linalg.norm(I[fm])),
                                     recovered=float(C[fm].sum() / I[fm].sum()), voxels=int(fm.sum()))
            row[n] = dict(rmse=float(100 * np.linalg.norm(c - i) / np.linalg.norm(i)), flux=float(100 * (c.sum() - i.sum()) / i.sum()),
                          mom0=float(100 * np.linalg.norm((c0 - j0)[ok]) / np.linalg.norm(j0[ok])),
                          mom1=float(np.sqrt(np.mean((c1 - j1)[ok] ** 2))), mom2=float(np.sqrt(np.mean((c2 - j2)[ok] ** 2))))
        out[f"{dv:g}"][f"b{ib:02d}"] = row
        a_, b_ = row["2D-1D"], row["2D per channel"]
        print(f"{dv:4g} km/s b{ib:02d} ({np.sqrt(b[0] * b[1]):.2f}\", peak S/N per channel {snr:.0f}): 2D-1D vs 2D per channel  "
              f"RMSE {a_['rmse']:.1f}/{b_['rmse']:.1f}%  mom0 {a_['mom0']:.1f}/{b_['mom0']:.1f}%  mom1 {a_['mom1']:.1f}/{b_['mom1']:.1f}  "
              f"mom2 {a_['mom2']:.1f}/{b_['mom2']:.1f} km/s  flux {a_['flux']:+.1f}/{b_['flux']:+.1f}%  | faint (3-10 sigma): RMSE "
              f"{row['2D-1D faint']['rmse']:.1f}/{row['2D per channel faint']['rmse']:.1f}%, recovered "
              f"{row['2D-1D faint']['recovered']:.2f}/{row['2D per channel faint']['recovered']:.2f}", flush=True)
json.dump(out, open(os.path.join(OUT, "chanres_eval.json"), "w"), indent=1)
