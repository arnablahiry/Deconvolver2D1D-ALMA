#!/usr/bin/env python
"""
Do the compact and extended configurations of target T have the same flux scale? The visibility test of
raw_visibility_compare.py (NGC 3110), generalised to every target, on the pre-averaged 20 km/s MSs
(/scratch/alahiry/regrid/<T>_{64,68}x20kms): per row, the line-integrated visibility (the line channels of the 20 km/s
grid minus the mean of its line-free channels, x 20 km/s; polarisations averaged; rows with a flagged line channel
dropped), referred to ONE common model -- the compact MS-CLEAN of the same grid (its line channels minus its line-free
mean), Fourier transformed and interpolated at every (u, v), with the offset between the MS phase centre and the
image centre applied as a phase.

  r = sum w V conj(V_model) / sum w |V_model|^2 per array, over the uv range where both arrays have data and the model
  is reliable (from the extended array's 5th percentile to where the compact clean beam's transfer function falls to
  10%, as in raw_visibility_compare.py), and in bins out to the longest compact baseline; the test is the ratio
  |r_extended| / |r_compact|, which does not depend on the model being right. The axis and phase-centre conventions
  are those that best fit the compact data (checked on the compact data only).

    python scripts/experiments/vis_fluxscale.py T   -> results/<T>/figures/vis_fluxscale.json
"""
import json
import os
import sys

import numpy as np
from astropy.io import fits
from numpy.fft import fft2, ifftshift
from scipy.ndimage import map_coordinates

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "simple"))
from casa_logs import redirect_casa_logs  # noqa: E402
T = sys.argv[1]
redirect_casa_logs(f"vis_fluxscale_{T}.log")
import casatools  # noqa: E402

R = os.path.join(REPO, "results", T); GC = json.load(open(os.path.join(R, "config.json")))
S3110 = T == "NGC3110"
RG = json.load(open(f"/scratch/alahiry/regrid/{T}_{'64' if S3110 else '68'}x20kms/ms.json"))
G = json.load(open(os.path.join(R, "consistency_20kms_0.08/grid.json" if S3110 else "g20/grid.json")))
L, F, DV = G["line"], G["line_free"], G["width_kms"]; OFF = (RG["nchan"] - G["nchan"]) // 2      # MS channel = image + OFF
MODEL = os.path.join(R, "consistency_20kms_0.08" if S3110 else "g20", "clean_compact_finished/model.fits")
AS = np.pi / 180 / 3600
log = lambda m: print(f"{T}: {m}", flush=True)

# ---------------------------------------------------------------- the common model on a fine uv grid
h = fits.getheader(MODEL); cell = abs(h["CDELT1"]) * 3600 * AS
M = np.nan_to_num(fits.getdata(MODEL)).squeeze().astype(np.float64)
Ml = (M[L] - M[F].mean(0)).sum(0) * DV; n0 = Ml.shape[0]; N = 2 * n0
pad = np.zeros((N, N)); pad[(N - n0) // 2:(N - n0) // 2 + n0, (N - n0) // 2:(N - n0) // 2 + n0] = Ml
FM = fft2(ifftshift(pad))


def predict(u, v, su, sv):
    kx = (su * u * 1e3 * N * cell) % N; ky = (sv * v * 1e3 * N * cell) % N
    return map_coordinates(FM.real, [ky, kx], order=1, mode="grid-wrap") + 1j * map_coordinates(FM.imag, [ky, kx], order=1, mode="grid-wrap")


def parse_pc(s):                                   # "J2000 hh:mm:ss.s -dd.mm.ss.s" -> radians
    _, ra, dec = s.split(); h_, m_, s_ = map(float, ra.split(":"))
    sg = -1 if dec.strip().startswith("-") else 1; d_, dm, ds = (abs(float(x)) for x in dec.split(".", 2)[:2] + [".".join(dec.split(".")[2:])])
    return np.radians(15 * (h_ + m_ / 60 + s_ / 3600)), sg * np.radians(d_ + dm / 60 + ds / 3600)


ra0, de0 = parse_pc(GC["phasecenter"])
tb = casatools.table()


def load(cfg):
    us, vs, Vs, Ws = [], [], [], []
    for ms in RG["ms"][cfg]:
        tb.open(ms + "/FIELD"); pd = tb.getcol("PHASE_DIR").squeeze(); tb.close()
        tb.open(ms + "/SPECTRAL_WINDOW"); fr = tb.getcol("CHAN_FREQ")[:, 0]; tb.close()
        dra = (pd[0] - ra0 + np.pi) % (2 * np.pi) - np.pi                    # PHASE_DIR's RA can be in (-pi, pi]
        dl, dm = dra * np.cos(de0), pd[1] - de0                              # field centre - image centre (rad)
        lam = 299792458.0 / fr[[c + OFF for c in L]].mean()
        tb.open(ms); n = tb.nrows(); step = 200000
        for s0 in range(0, n, step):
            k = min(step, n - s0)
            d = tb.getcol("DATA", s0, k); fl = tb.getcol("FLAG", s0, k); uvw = tb.getcol("UVW", s0, k); w = tb.getcol("WEIGHT", s0, k)
            d = np.where(fl, 0, d); ng = (~fl).sum(0); dp = d.sum(0) / np.maximum(ng, 1)       # (nchan, nrow), pols averaged
            Li, Fi = [c + OFF for c in L], [c + OFF for c in F]
            good = (ng[Li] > 0).all(0) & ((ng[Fi] > 0).sum(0) > 3)
            cont = np.where(ng[Fi] > 0, dp[Fi], 0).sum(0) / np.maximum((ng[Fi] > 0).sum(0), 1)
            V = (dp[Li] - cont).sum(0) * DV
            us.append(uvw[0, good] / lam / 1e3); vs.append(uvw[1, good] / lam / 1e3)
            Vs.append(V[good]); Ws.append(w.mean(0)[good] / len(L))
            if s0 == 0:
                off = (dl, dm)
        tb.close()
    return dict(u=np.concatenate(us), v=np.concatenate(vs), V=np.concatenate(Vs), w=np.concatenate(Ws), off=off)


D = {c: load(c) for c in ("compact", "extended")}
for c, d in D.items():
    d["q"] = np.hypot(d["u"], d["v"]); log(f"{c}: {d['u'].size} visibilities, phase-centre offset "
                                           f"{d['off'][0] / AS:+.2f}\", {d['off'][1] / AS:+.2f}\"")


def model_at(d, su, sv, sp):
    ph = np.exp(sp * 2j * np.pi * (d["u"] * 1e3 * d["off"][0] + d["v"] * 1e3 * d["off"][1]))
    return predict(d["u"], d["v"], su, sv) * ph


best = None
for su in (1, -1):
    for sv in (1, -1):
        for sp in (1, -1):
            d = D["compact"]; s = d["q"] < np.percentile(d["q"], 70); vm = model_at(d, su, sv, sp)[s]
            r = (d["w"][s] * d["V"][s] * np.conj(vm)).sum() / (d["w"][s] * np.abs(vm) ** 2).sum()
            res = (d["w"][s] * np.abs(d["V"][s] - r * vm) ** 2).sum()
            if best is None or res < best[0]:
                best = (res, su, sv, sp)
_, SU, SV, SP = best
for d in D.values():
    d["m"] = model_at(d, SU, SV, SP)
# the band where the common model is reliable: up to where the compact clean beam's transfer function falls to 10%
with fits.open(MODEL.replace("model.fits", "image.fits")) as _h:          # one beam, or the median of per-channel beams
    _b = (_h[0].header["BMAJ"] * 3600, _h[0].header["BMIN"] * 3600) if "BMAJ" in _h[0].header else \
        (float(np.median(_h["BEAMS"].data["BMAJ"])), float(np.median(_h["BEAMS"].data["BMIN"])))
CB = np.sqrt(_b[0] * _b[1]) * AS
QB = 2 * np.sqrt(np.log(2) * np.log(10)) / (np.pi * CB) / 1e3
lo, hi = float(np.percentile(D["extended"]["q"], 5)), float(min(D["compact"]["q"].max(), QB))


def fit(d, a, b):
    s = (d["q"] >= a) & (d["q"] < b)
    if s.sum() < 50:
        return None
    w, x, vm = d["w"][s], d["V"][s], d["m"][s]
    r = (w * x * np.conj(vm)).sum() / (w * np.abs(vm) ** 2).sum()
    k = (w * np.abs(x - r * vm) ** 2).sum() / (2 * s.sum() - 2)            # the weights' absolute scale from the scatter
    return dict(n=int(s.sum()), ratio=float(np.abs(r)), err=float(np.sqrt(k / (w * np.abs(vm) ** 2).sum())),
                phase_deg=float(np.degrees(np.angle(r))))


out = dict(target=T, model=os.path.relpath(MODEL, REPO), convention=[SU, SV, SP], overlap_klambda=[lo, hi], compact_beam_10pct_klambda=QB)
ov = {c: fit(D[c], lo, hi) for c in D}
rat = ov["extended"]["ratio"] / ov["compact"]["ratio"]
rerr = rat * np.hypot(ov["extended"]["err"] / ov["extended"]["ratio"], ov["compact"]["err"] / ov["compact"]["ratio"])
out.update(overlap=ov, extended_over_compact=rat, extended_over_compact_err=rerr)
bins = np.geomspace(max(lo, 5), float(D["compact"]["q"].max()), 9)
out["bins"] = [dict(q=[float(a), float(b)], **{c: fit(D[c], a, b) for c in D}) for a, b in zip(bins[:-1], bins[1:])]
os.makedirs(os.path.join(R, "figures"), exist_ok=True)
json.dump(out, open(os.path.join(R, "figures", "vis_fluxscale.json"), "w"), indent=1)
log(f"overlap {lo:.0f}-{hi:.0f} klambda: compact |r| {ov['compact']['ratio']:.3f}, extended |r| {ov['extended']['ratio']:.3f}, "
    f"phases {ov['compact']['phase_deg']:+.1f} / {ov['extended']['phase_deg']:+.1f} deg -> extended / compact = {rat:.3f} +- {rerr:.3f}")
