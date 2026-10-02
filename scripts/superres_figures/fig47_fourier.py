"""
47: the compact dirty map and the compact deconvolved model in Fourier space (moment 0, line channels),
radially averaged against spatial frequency k (cycles/arcsec; top axis: baseline in kilo-lambda).

(a) amplitude |F(k)| [Jy km/s] of: compact dirty, compact deconvolved (raw), extended CLEAN image, and
    the noise of the two dirty cubes (line-free channels, scaled to the number of line channels).
    Jy/beam maps are divided by their clean-beam area in pixels, so F(0) is a flux.
(b) the "beam" of each map, as a transfer function T(k) = F_map / F_sky:
    - expected: compact dirty beam (FT of the PSF / clean-beam area), compact and extended clean beams
      (Gaussian MTFs), the lambda/B_max beam;
    - measured, against the independently observed EXTENDED dirty cube (F_ed = P_ed F_sky + noise, P_ed
      the FT of its dirty beam):
          T = sum Re(F_map conj F_ed) P_ed / sum (|F_ed|^2 - N_ed)    per ring,
      so noise uncorrelated between the maps drops out of the numerator and its power is removed from
      the denominator. Applied to the compact dirty map, this must give back its dirty beam (a check);
      applied to the deconvolved model, it gives the deconvolution's effective beam.
      Shown where the extended array samples the sky (k > 0.25 cycles/arcsec, i.e. inside its 2.4" largest
      recoverable scale) with ring S/N > 5; error bars: spread over 8 azimuthal sectors. The deconvolved model's
      'beam' curve is a Gaussian MTF fitted to its measured T.

    MODEL=... OUT=... python scripts/superres_figures/fig47_fourier.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.environ.get("MODEL", os.path.join(REPO, "data/experiments/04_pd_tau_reweighting/compact/pdtau_k5.npy"))
OUT = os.environ.get("OUT", os.path.join(REPO, "figures/experiments/04_pd_tau_reweighting/5_sigma/47_fourier_compact.png"))
LABEL = os.environ.get("LABEL", r"compact model: primal-dual ($\tau\lambda$ reweighting), k = 5, 20,000 iterations")
D = os.path.join(REPO, "data/ngc3110")
CELL, N = 0.02, 800
LBMAX_C, LBMAX_E, CELL_SOLVE = 0.60, 0.1134, 0.04          # compact / extended lambda/B_max ["], compact solve grid
KLAM = 206.265                                             # cycles/arcsec -> kilo-lambda
vel = 5073.9 + 25.0 * (np.arange(32) - 15.5)
L = [i for i, v in enumerate(vel) if 4750 <= v <= 5280]
F = [i for i in range(32) if i not in L]
FS = 17
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": FS, "axes.labelsize": FS + 1,
                     "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1, "xtick.direction": "in", "ytick.direction": "in",
                     "ytick.right": True, "axes.linewidth": 1.3, "legend.fontsize": FS - 3})
rd = lambda p: np.nan_to_num(fits.getdata(p).astype(np.float64)).squeeze()


def beam(path):
    h = fits.getheader(path)
    return h["BMAJ"] * 3600, h["BMIN"] * 3600, h["BPA"]


omega_px = lambda b: np.pi * b[0] * b[1] / (4 * np.log(2)) / CELL ** 2

# ---- maps (moment 0, Jy km/s per pixel or per beam), one taper for all: cosine from r = 6.5" to 7.9"
y, x = (np.indices((N, N)) - N // 2) * CELL
r = np.hypot(x, y)
taper = np.clip((7.9 - r) / 1.4, 0, 1); taper = 0.5 - 0.5 * np.cos(np.pi * taper)
ft = lambda m: np.fft.fft2(np.fft.ifftshift(m * taper))
BC, BE = beam(os.path.join(D, "compact/ms_clean/image.fits")), beam(os.path.join(D, "extended/ms_clean/image.fits"))
cd, ed = rd(os.path.join(D, "compact/cube/dirty_cube.fits")), rd(os.path.join(D, "extended/cube/dirty_cube.fits"))
noise_m0 = lambda c: c[F].sum(0) * 25.0 * np.sqrt(len(L) / len(F))     # line-free channels, scaled to len(L) channels
F_cd, F_ed = ft(cd[L].sum(0) * 25.0), ft(ed[L].sum(0) * 25.0)
N_cd, N_ed = np.abs(ft(noise_m0(cd))) ** 2, np.abs(ft(noise_m0(ed))) ** 2
F_m = ft(np.load(MODEL).astype(np.float64)[L].sum(0) * 25.0)
F_ei = ft(rd(os.path.join(D, "extended/ms_clean/image.fits"))[L].sum(0) * 25.0)


def psf_ft(cfg):
    """FT of the line-channel mean dirty beam (2x field), on the 800-px map's frequency grid (units: px)"""
    p = rd(os.path.join(D, cfg, "cube/dirty_beam_cube.fits"))[L].mean(0)
    return np.fft.fft2(np.fft.ifftshift(p)).real[::2, ::2]


P_c, P_e = psf_ft("compact"), psf_ft("extended")

# ---- rings
k = np.hypot(*np.meshgrid(np.fft.fftfreq(N, CELL), np.fft.fftfreq(N, CELL)))     # cycles/arcsec
dk = 1.0 / (N * CELL)
edges = np.unique(np.round(np.geomspace(dk, 25.0, 48) / dk)) * dk + dk / 2
edges = np.concatenate([[dk / 2], edges])
idx = np.digitize(k.ravel(), edges) - 1
nb = len(edges) - 1
kc = np.array([k.ravel()[idx == i].mean() if np.any(idx == i) else np.nan for i in range(nb)])
ring = lambda a: np.array([a.ravel()[idx == i].mean() if np.any(idx == i) else np.nan for i in range(nb)])
amp = lambda Fm: np.sqrt(ring(np.abs(Fm) ** 2))

A_cd, A_m, A_ei = amp(F_cd) / omega_px(BC), amp(F_m), amp(F_ei) / omega_px(BE)
A_ncd, A_ned = np.sqrt(ring(N_cd)) / omega_px(BC), np.sqrt(ring(N_ed)) / omega_px(BE)
den = ring(np.abs(F_ed) ** 2 - N_ed)
snr = den / ring(N_ed)
ok = (snr > 5) & (kc > 0.25) & (kc < 1 / LBMAX_E)     # inside the extended array's uv coverage
# T per ring = ratio of ring sums; its error from the spread over 8 azimuthal sectors (angle mod pi, since
# F(-k) = conj F(k)): the sky's own anisotropic structure, not the thermal noise, dominates the scatter
NS = 8
kx, ky = np.meshgrid(np.fft.fftfreq(N, CELL), np.fft.fftfreq(N, CELL))
sec = (np.floor(np.mod(np.arctan2(ky, kx), np.pi) / (np.pi / NS)).astype(int) % NS).ravel()
valid = (idx >= 0) & (idx < nb)
rsum = lambda a: np.bincount(idx[valid], a.ravel()[valid], minlength=nb)[:nb]
ssum = lambda a: np.bincount((idx * NS + sec)[valid], a.ravel()[valid], minlength=nb * NS)[:nb * NS].reshape(nb, NS)
den_a = np.abs(F_ed) ** 2 - N_ed


def T(Fm):
    num = (Fm * np.conj(F_ed)).real * P_e
    t, ts = rsum(num) / rsum(den_a), ssum(num) / ssum(den_a)
    return t, np.nanstd(ts, axis=1) / np.sqrt(NS - 1)


(T_cd, eT_cd), (T_m, eT_m) = T(F_cd), T(F_m)
T_cd, eT_cd = T_cd / omega_px(BC), eT_cd / omega_px(BC)
gauss = lambda b, kk: np.exp(-(np.pi * kk) ** 2 * b[0] * b[1] / (4 * np.log(2)))     # geometric-mean Gaussian MTF
kf = np.geomspace(dk, 25, 400)
s0 = LBMAX_E / np.sqrt(BE[0] * BE[1]); b1 = (BE[0] * s0, BE[1] * s0)

# effective beam of the deconvolved model: Gaussian MTF exp(-(pi theta k)^2 / (4 ln 2)) fitted to the measured T
good = ok & np.isfinite(T_m)
th = np.linspace(0.05, 1.5, 2901)
chi = [np.sum(((T_m[good] - np.exp(-(np.pi * t * kc[good]) ** 2 / (4 * np.log(2)))) / eT_m[good]) ** 2) for t in th]
theta_m = th[int(np.argmin(chi))]
fwhm = lambda kh: 2 * np.log(2) / (np.pi * kh)                 # Gaussian MTF = 1/2 at kh
k_half_m = 2 * np.log(2) / (np.pi * theta_m)
P_c_ring = ring(P_c) / omega_px(BC)
k_half_c = kc[(kc > kc[np.nanargmax(P_c_ring)]) & (P_c_ring < 0.5)][0]

# ---- figure: two square panels, shared x
Pw, g, XLb, YLb, TM, XT = 7.6, 1.45, 0.95, 1.25, 1.45, 0.95
W, H = YLb + 2 * Pw + g + 0.3, XLb + Pw + XT + TM
fig = plt.figure(figsize=(W, H))
axs = [fig.add_axes([(YLb + i * (Pw + g)) / W, XLb / H, Pw / W, Pw / H]) for i in range(2)]
marks = [(1 / LBMAX_C, "compact $B_{\\rm max}/\\lambda$"), (1 / LBMAX_E, "extended $B_{\\rm max}/\\lambda$"),
         (1 / (2 * CELL_SOLVE), "0.04$^{\\prime\\prime}$ grid Nyquist")]
for a in axs:
    for kv, lab in marks:
        a.axvline(kv, color="0.55", ls=":", lw=1.4, zorder=0)
        a.text(kv * 0.93, 0.03, lab, rotation=90, transform=a.get_xaxis_transform(), ha="right", va="bottom",
               fontsize=FS - 4, color="0.35", bbox=dict(fc="w", ec="none", pad=1, alpha=0.85))
    a.set_xscale("log"); a.set_xlim(dk, 25)
    a.set_xlabel(r"spatial frequency $k$ [cycles arcsec$^{-1}$]")
    top = a.secondary_xaxis("top", functions=(lambda v: v * KLAM, lambda v: v / KLAM))
    top.set_xlabel(r"baseline [k$\lambda$]", labelpad=8); top.tick_params(direction="in", which="both")
    a.tick_params(which="both", top=False)

a = axs[0]
a.loglog(kc, A_cd, color="tab:blue", lw=2.6, label="compact dirty")
a.loglog(kc, A_ncd, color="tab:blue", lw=1.6, ls="--", label="compact dirty: noise")
a.loglog(kc, A_m, color="tab:red", lw=2.6, label="compact deconvolved (raw)")
a.loglog(kc, A_ei, color="k", lw=2.0, label="extended CLEAN image")
a.loglog(kc, A_ned, color="k", lw=1.4, ls="--", label="extended dirty: noise")
a.set_ylabel(r"amplitude $|\tilde{I}(k)|$ [Jy km s$^{-1}$]")
a.set_ylim(np.nanmin(A_ncd[kc < 20]) / 3, np.nanmax(A_m) * 2)
a.text(0.97, 0.97, "(a) signal amplitude", transform=a.transAxes, ha="right", va="top", fontsize=FS + 1, fontweight="bold")
a.legend(loc="lower left", bbox_to_anchor=(0.0, 0.09), frameon=True)

a = axs[1]
a.semilogx(kc, P_c_ring, color="tab:blue", lw=2.6, label="compact dirty beam (FT of the PSF)")
a.semilogx(kf, gauss(BC, kf), color="tab:blue", lw=1.6, ls="--", label="compact clean beam")
a.errorbar(kc[ok], T_cd[ok], eT_cd[ok], fmt="o", color="tab:blue", ms=6, mfc="none", mew=1.6, capsize=2,
           label="compact dirty, measured")
a.errorbar(kc[ok], T_m[ok], eT_m[ok], fmt="o", color="tab:red", ms=6, capsize=2, label="compact deconvolved (raw), measured")
a.semilogx(kf, np.exp(-(np.pi * theta_m * kf) ** 2 / (4 * np.log(2))), color="tab:red", lw=2.6,
           label=f"deconvolved beam (Gaussian fit, {theta_m:.2f}$^{{\\prime\\prime}}$)")
a.semilogx(kf, gauss(BE, kf), color="k", lw=1.6, ls="--", label="extended clean beam")
a.semilogx(kf, gauss(b1, kf), color="0.45", lw=1.6, ls="-.", label=r"$\lambda/B_{\rm max}$ beam")
a.axhline(0.5, color="0.7", lw=1, zorder=0)
a.set_ylim(-0.15, 2.7)
a.set_ylabel(r"transfer function $T(k)$ (the beam in Fourier space)")
a.text(0.03, 0.97, "(b) beam", transform=a.transAxes, ha="left", va="top", fontsize=FS + 1, fontweight="bold")
a.legend(loc="upper right", frameon=True)
fig.text(0.5, 1 - 0.15 / H, "NGC 3110 CO(2-1), moment 0 (4750-5280 km/s): compact dirty map and its deconvolution in Fourier space\n"
         f"{LABEL}\n"
         f"measured $T$: cross-spectrum with the independent extended dirty cube, at "
         f"$k$ = {kc[ok].min():.2f}-{kc[ok].max():.1f} cycles/$^{{\\prime\\prime}}$ (where its S/N > 5)",
         ha="center", va="top", fontsize=FS - 1)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=60)
print("saved", OUT)
print(f"half-power k: compact dirty beam {k_half_c:.2f} c/\" (FWHM-eq {fwhm(k_half_c):.3f}\"), deconvolved fit {k_half_m:.2f} c/\" "
      f"(FWHM {theta_m:.3f}\", chi2/dof {min(chi) / (good.sum() - 1):.2f}); measured range k = {kc[ok].min():.2f}-{kc[ok].max():.2f}")
for kk, t1, e1, t2, e2, pc in zip(kc[ok], T_cd[ok], eT_cd[ok], T_m[ok], eT_m[ok], P_c_ring[ok]):
    print(f"  k {kk:5.2f}  T_dirty {t1:5.2f}+-{e1:.2f} (PSF {pc:5.2f})  T_model {t2:5.2f}+-{e2:.2f}")
