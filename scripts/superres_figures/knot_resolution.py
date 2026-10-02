"""
Per-knot effective resolution of the compact deconvolved model, tested against the independently
observed EXTENDED dirty cube (exactly linear: dirty_ext = sky * PSF_ext + noise, no CLEAN involved),
and validated on the simulation with known truth. Moment 0 of the line channels.

If, around a knot, the model is model = sky * K_theta (K a circular Gaussian of FWHM theta), then
        dirty_ext * K_theta  =  model * PSF_ext  (+ smoothed noise).
For each knot (peaks of the extended CLEAN image, S/N > 8, >= 0.3" apart) we fit theta^2 and a flux
scale a in a Gaussian window (FWHM 0.5"):
    theta^2 >= 0:  dirty_ext * K_theta   ~  a * model * PSF_ext
    theta^2 <  0:  dirty_ext             ~  a * model * PSF_ext * K_|theta|   (model knot over-sharpened)
Smoothing the noisy reference lowers chi^2 by itself, so the expected noise term sum w (n * K)^2
(from noise realisations) is subtracted for every theta. The model cube is convolved channel by
channel with the per-channel PSF (2x field, linear convolution). One global compact-extended offset
is fitted first. Errors: the fit repeated on dirty_ext + n_i (n_i: random-sign sums of the extended
dirty's line-free channels, scaled to len(L) channels), with the doubled noise term.

    python scripts/superres_figures/knot_resolution.py  -> data/experiments/09_knot_resolution/knot_resolution.json
"""
import json
import os
import sys

import numpy as np
from astropy.io import fits
from scipy.ndimage import maximum_filter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.environ.get("MODEL", os.path.join(REPO, "data/experiments/04_pd_tau_reweighting/compact/pdtau_k5.npy"))
SIM_MODEL = os.environ.get("SIM_MODEL", os.path.join(REPO, "data/experiments/01_simulations/compact/pdtau_k5.npy"))
SIMI = os.path.join(REPO, "data/experiments/01_simulations/inputs")
OUTD = os.path.join(REPO, "data/experiments/09_knot_resolution")
D = os.path.join(REPO, "data/ngc3110")
CELL, N = 0.02, 800
WIN_FWHM, NKNOT, NNOISE, MIN_SEP, HALF = 0.5, 25, 24, 0.3, 40
vel = 5073.9 + 25.0 * (np.arange(32) - 15.5)
L = [i for i, v in enumerate(vel) if 4750 <= v <= 5280]
F = [i for i in range(32) if i not in L]
rd = lambda p: np.nan_to_num(fits.getdata(p).astype(np.float64)).squeeze()
rng = np.random.default_rng(1)
kx, ky = np.meshgrid(np.fft.fftfreq(N, CELL), np.fft.fftfreq(N, CELL))
k2 = kx ** 2 + ky ** 2
yy, xx = (np.indices((N, N)) - N // 2) * CELL
S2 = np.concatenate([-np.linspace(0.3, 0.0, 31) ** 2, np.linspace(0.01, 0.5, 50) ** 2])   # theta^2 [arcsec^2]
gk = lambda s: np.exp(-np.pi ** 2 * k2 * abs(s) / (4 * np.log(2)))                         # FT of K_sqrt|s|
conv = lambda img, Fk: np.fft.ifft2(np.fft.fft2(img) * Fk).real
box = np.s_[N // 2 - 250:N // 2 + 250, N // 2 - 250:N // 2 + 250]


def through_psf(cube, psf):
    """sum over line channels of cube[c] (*) psf[c], linear convolution with the 2x-field PSF"""
    n2 = psf.shape[-1]
    out = np.zeros((n2, n2))
    for c in L:
        pad = np.zeros((n2, n2)); pad[n2 // 4:n2 // 4 + N, n2 // 4:n2 // 4 + N] = cube[c]
        out += np.fft.ifft2(np.fft.fft2(pad) * np.fft.fft2(np.fft.ifftshift(psf[c]))).real
    return out[n2 // 4:n2 // 4 + N, n2 // 4:n2 // 4 + N] * 25.0


def global_shift(a, ref):
    w = np.zeros_like(a); w[box] = 1
    cc = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(ref * w) * np.conj(np.fft.fft2(a * w))).real)
    j, i = np.unravel_index(np.argmax(cc), cc.shape)
    p = lambda m, z, q: 0.5 * (m - q) / (m - 2 * z + q)
    return ((i - N // 2 + p(cc[j, i - 1], cc[j, i], cc[j, i + 1])) * CELL,
            (j - N // 2 + p(cc[j - 1, i], cc[j, i], cc[j + 1, i])) * CELL)


def knots(det, sig):
    mx = maximum_filter(det, size=int(MIN_SEP / CELL) | 1)
    pk = (det == mx) & (det > 8 * sig) & (np.abs(xx) < 4.4) & (np.abs(yy) < 4.4)
    j, i = np.nonzero(pk)
    o = np.argsort(det[j, i])[::-1][:NKNOT]
    return list(zip(j[o], i[o]))


C = 100                                                              # cutout half-size [px]
ckx, cky = np.meshgrid(np.fft.fftfreq(2 * C + 1, CELL), np.fft.fftfreq(2 * C + 1, CELL))
cgk = lambda s: np.exp(-np.pi ** 2 * (ckx ** 2 + cky ** 2) * abs(s) / (4 * np.log(2)))
cyy, cxx = (np.indices((2 * C + 1, 2 * C + 1)) - C) * CELL
W = np.exp(-4 * np.log(2) * (cxx ** 2 + cyy ** 2) / WIN_FWHM ** 2) * (np.maximum(abs(cxx), abs(cyy)) <= HALF * CELL)
smooth = lambda img, s: np.fft.ifft2(np.fft.fft2(img) * cgk(s)).real if s != 0 else img


def fit_knot(A, ref, noises, extra=None, nfac=1.0, nt=None):
    """theta^2 for one knot from local cutouts (all (2C+1)^2, centred on the knot)"""
    if nt is None:
        nt = np.array([np.mean([np.sum(W * (smooth(n, s) if s > 0 else n) ** 2) for n in noises]) for s in S2])
    R0 = ref if extra is None else ref + extra
    chi = np.empty(len(S2))
    for n_, s in enumerate(S2):
        R, m = (smooth(R0, s), A) if s > 0 else (R0, smooth(A, s))
        a = np.sum(W * m * R) / np.sum(W * m * m)
        chi[n_] = np.sum(W * (R - a * m) ** 2) - nfac * nt[n_]
    return S2[int(np.argmin(chi))], nt


def analyse(model_cube, ref, psf, noises, kn, label, boot=True):
    A0 = through_psf(model_cube, psf)
    dx, dy = global_shift(A0, ref)
    A = np.fft.ifft2(np.fft.fft2(A0) * np.exp(-2j * np.pi * (kx * dx + ky * dy))).real
    sig = float(np.std(noises[-1][box])) or 1.0
    out = []
    th = lambda v: float(np.sign(v) * np.sqrt(abs(v)))
    for (j, i) in kn:
        cut = lambda img: img[j - C:j + C + 1, i - C:i + C + 1]
        nz = [cut(n) for n in noises]
        s, nt = fit_knot(cut(A), cut(ref), nz[:NNOISE // 2])
        bs = [fit_knot(cut(A), cut(ref), None, extra=e, nfac=2.0, nt=nt)[0] for e in nz[NNOISE // 2:-1]] if boot else [s]
        lo, hi = np.percentile(bs, [16, 84])
        out.append(dict(x=float(-xx[j, i]), y=float(yy[j, i]), snr=float(ref[j, i] / sig), theta=th(s),
                        theta_16=th(lo), theta_84=th(hi), at_edge=bool(s in (S2[0], S2[-1]))))
        r = out[-1]
        print(f"  ({r['x']:+.2f},{r['y']:+.2f})  S/N {r['snr']:6.1f}  theta {r['theta']:+.3f}\"  [{r['theta_16']:+.3f}, {r['theta_84']:+.3f}]"
              f"{'  EDGE' if r['at_edge'] else ''}", flush=True)
    print(f"[{label}] global offset dx {dx * 1000:+.1f} mas, dy {dy * 1000:+.1f} mas; {len(out)} knots", flush=True)
    return dict(offset_mas=[dx * 1000, dy * 1000], knots=out)


def noise_maps(cube, n):
    return [np.tensordot(rng.choice([-1, 1], len(F)), cube[F], 1) * 25.0 * np.sqrt(len(L) / len(F)) for _ in range(n)]


if __name__ == "__main__":
    os.makedirs(OUTD, exist_ok=True)
    psf = rd(os.path.join(D, "extended/cube/dirty_beam_cube.fits"))
    dirty = rd(os.path.join(D, "extended/cube/dirty_cube.fits"))
    img = rd(os.path.join(D, "extended/ms_clean/image.fits"))
    noises = noise_maps(dirty, NNOISE + 1)
    img_m0 = img[L].sum(0) * 25.0
    res = {}
    print("real: compact model vs extended dirty", flush=True)
    res["real"] = analyse(np.load(MODEL).astype(np.float64), dirty[L].sum(0) * 25.0, psf, noises,
                          knots(img_m0, float(np.std(noise_maps(img, 1)[0][box]))), "real")
    # ---- simulation: the same sky observed by the extended array (real PSF, real-spectrum noise)
    up = lambda x: np.kron(x, np.ones((1, 2, 2))) / 4
    ms = up(np.load(SIM_MODEL).astype(np.float64))
    tr = up(np.load(os.path.join(SIMI, "compact_sim_truth_0.04arcsec.npy")).astype(np.float64))
    sim_ed = np.load(os.path.join(SIMI, "extended_sim_dirty.npy")).astype(np.float64)
    sim_noises = noise_maps(sim_ed, NNOISE + 1)
    sim_m0 = sim_ed[L].sum(0) * 25.0
    kn_sim = knots(conv(sim_m0, gk(0.1 ** 2)), float(np.std(conv(sim_noises[0], gk(0.1 ** 2))[box])))
    print("sim: compact sim model vs extended sim dirty", flush=True)
    res["sim"] = analyse(ms, sim_m0, psf, sim_noises, kn_sim, "sim")
    print("sim, exact: compact sim model vs truth itself (noise-free reference: truth * PSF_ext)", flush=True)
    zero = [np.zeros((N, N))] * (NNOISE + 1)
    res["sim_exact"] = analyse(ms, through_psf(tr, psf), psf, zero, kn_sim, "sim exact", boot=False)
    res.update(window_fwhm=WIN_FWHM, model=MODEL, sim_model=SIM_MODEL)
    json.dump(res, open(os.path.join(OUTD, "knot_resolution.json"), "w"), indent=1)
    print("saved", os.path.join(OUTD, "knot_resolution.json"))
