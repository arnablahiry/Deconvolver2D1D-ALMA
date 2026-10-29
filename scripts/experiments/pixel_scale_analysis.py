#!/usr/bin/env python
"""
Pixel-scale experiment, the analysis: does the 2D-1D deconvolution of the NGC 3110 compact cube stop at the pixel
(features of 1 px on both grids: 0.16" and 0.08") or at a fixed angular size (1 px at 0.16" = 2 px at 0.08")?
Models from pixel_scale_deconv.py (Jy/pixel, (64, n, n)), raw, no beam.

  peak sizes          every local maximum of a raw line-channel map inside the 25 sigma astrodendro mask of the tapered
                      extended MS-CLEAN (same footprint on both grids), brighter than FRAC x its channel's max, not
                      touching the run's support edge: a 2D elliptical Gaussian + constant fitted to a +-0.48" box,
                      the Gaussian averaged over each pixel (so a one-pixel spike fits below 1 px instead of failing);
                      FWHM = geometric mean of the axes. (The half-max diameter d = 2 sqrt(area above half the peak
                      / pi) is kept in the JSON: it measures the bright top of a clump, the fit the whole clump.)
  power spectrum      azimuthally averaged |FFT|^2 of the raw line-channel maps. A model in Jy/pixel has the
                      continuous Fourier transform of its surface brightness as its DFT, so the two grids are on
                      the same scale; with the compact array's B_max/lambda and both Nyquist frequencies
  spectra agree to    the spatial frequency up to which the two models' power spectra agree to 25% (beyond it each grid
                      fills its own extra frequencies), and the share of the power above the compact B_max/lambda

    TAG=_support python scripts/experiments/pixel_scale_analysis.py
    -> results/NGC3110/pixel_scale/figures/pixel_scale<TAG>.png/.pdf and pixel_scale<TAG>.json
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, label, binary_dilation
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(REPO, "results/NGC3110"); PS = os.path.join(R, "pixel_scale")
TAG = os.environ.get("TAG", "_support"); FRAC = float(os.environ.get("FRAC", 0.1)); MSIG = int(os.environ.get("MASKSIG", 25))
OUT = os.path.join(PS, "figures"); os.makedirs(OUT, exist_ok=True)
grid = json.load(open(os.path.join(R, "consistency_20kms_0.16/grid.json"))); L, DV = grid["line"], grid["width_kms"]
GC = json.load(open(os.path.join(R, "config.json"))); LBMAX_C = GC["lbmax_compact_arcsec"]       # lambda / B_max, compact
CELLS = (0.16, 0.08)
sys.path.insert(0, os.path.join(REPO, "simple"))
from astropy.io import fits  # noqa: E402
X = {c: np.load(os.path.join(PS, f"model_{c:g}{TAG}.npy")).astype(np.float64) for c in CELLS}
SUP = {c: np.load(os.path.join(R, f"consistency_20kms_{c:g}/clean_compact_finished/mask_final.npy")).astype(bool) for c in CELLS}
# the comparison mask: 25 sigma on the 0.16" grid; on 0.08" each pixel takes the 0.16" pixel it falls in (0.16" pixel i
# covers 0.08" pixels 2i - 1 .. 2i + 1, both grids centred on pixel N/2)
M16 = np.load(os.path.join(R, f"consistency_20kms_0.16/clean_extended_taper_cmask/dendro_mask_min{MSIG}.npy")).astype(bool)
idx = np.clip((np.arange(640) + 1) // 2, 0, 319)
MASK = {0.16: M16, 0.08: M16[:, idx][:, :, idx]}


def halfmax_peaks(x, cell, mask, sup):
    """half-max diameter (arcsec) of every local maximum of the line channels; see the docstring"""
    w = max(3, int(round(0.4 / cell)) | 1); hw = int(round(0.8 / cell))
    out = []
    for c in L:
        m = x[c]; mx = m[mask[c]].max() if mask[c].any() else 0
        if mx <= 0:
            continue
        edge = sup[c] & binary_dilation(~sup[c])                    # the support's outermost pixels
        loc = (m == maximum_filter(m, size=w)) & mask[c] & (m > FRAC * mx)
        for j, i in zip(*np.nonzero(loc)):
            sl = np.s_[max(j - hw, 0):j + hw + 1, max(i - hw, 0):i + hw + 1]
            cut = m[sl]; bg = np.percentile(cut, 20); thr = (m[j, i] + bg) / 2
            lab, _ = label(cut > thr); reg = lab == lab[j - sl[0].start, i - sl[1].start]
            if (reg & edge[sl]).any() or reg[0].any() or reg[-1].any() or reg[:, 0].any() or reg[:, -1].any():
                continue
            out.append(dict(ch=int(c), j=int(j), i=int(i), peak=float(m[j, i]), area_px=int(reg.sum()),
                            d_px=float(2 * np.sqrt(reg.sum() / np.pi)), d_arcsec=float(2 * np.sqrt(reg.sum() / np.pi) * cell)))
    return out


def power(x, cell):
    """azimuthally averaged |FFT|^2 over the line channels against spatial frequency (cycles/arcsec)"""
    n = x.shape[-1]; P = np.mean([np.abs(np.fft.fftshift(np.fft.fft2(x[c]))) ** 2 for c in L], 0)
    f = np.fft.fftshift(np.fft.fftfreq(n, d=cell)); r = np.hypot(*np.meshgrid(f, f))
    edges = np.arange(0, 1 / (2 * cell) + 1e-9, 1 / (n * cell) * 2)   # bins of 2 frequency cells
    k = 0.5 * (edges[1:] + edges[:-1]); idx_ = np.digitize(r.ravel(), edges) - 1
    ok = (idx_ >= 0) & (idx_ < len(k))
    pk = np.bincount(idx_[ok], P.ravel()[ok], len(k)) / np.maximum(np.bincount(idx_[ok], None, len(k)), 1)
    return k, pk, float(P[r > 1 / LBMAX_C].sum() / P.sum())


def gauss_fwhm(x, cell, pk, os_=5):
    """2D elliptical Gaussian + constant fitted to a +-0.48" box around each peak, the Gaussian averaged over each
    pixel (os_ x os_ sub-samples), so a single-pixel spike fits with a FWHM below a pixel instead of failing.
    Adds fwhm_px / fwhm_arcsec (geometric mean of the two axes) and fwhm_maj/min_px to each peak."""
    from scipy.optimize import least_squares
    hb = int(round(0.48 / cell)); g = np.arange(-hb, hb + 1)
    sub = (np.arange(os_) + 0.5) / os_ - 0.5
    YY, XX = np.meshgrid(g, g, indexing="ij")
    sy = YY[:, :, None, None] + sub[None, None, :, None]; sx = XX[:, :, None, None] + sub[None, None, None, :]

    def model(q):
        A, x0, y0, sa, sb, th, Bg = q
        u = (sx - x0) * np.cos(th) + (sy - y0) * np.sin(th); v = -(sx - x0) * np.sin(th) + (sy - y0) * np.cos(th)
        return Bg + A * np.exp(-0.5 * ((u / sa) ** 2 + (v / sb) ** 2)).mean((2, 3))
    for p in pk:
        m = x[p["ch"]]; j, i = p["j"], p["i"]
        cut = m[j - hb:j + hb + 1, i - hb:i + hb + 1]
        if cut.shape != YY.shape:
            p["fwhm_px"] = np.nan; continue
        bg = float(np.percentile(cut, 20)); A0 = max(float(cut[hb, hb]) - bg, 1e-12)
        q0 = [A0 * 1.5, 0, 0, 1.0, 1.0, 0.0, bg]
        lo = [0, -1.5, -1.5, 0.05, 0.05, -np.pi / 2, float(cut.min()) - abs(A0)]
        hi = [np.inf, 1.5, 1.5, 2.0 * hb, 2.0 * hb, np.pi / 2, float(cut.max())]
        r = least_squares(lambda q: (model(q) - cut).ravel(), q0, bounds=(lo, hi), x_scale="jac")
        sa, sb = r.x[3], r.x[4]; f = 2 * np.sqrt(2 * np.log(2))
        p.update(fit=[float(v) for v in r.x], fit_hb=hb,                         # A, x0, y0, sa, sb, theta, background (px)
                 fwhm_px=float(f * np.sqrt(sa * sb)), fwhm_arcsec=float(f * np.sqrt(sa * sb) * cell),
                 fwhm_maj_px=float(f * max(sa, sb)), fwhm_min_px=float(f * min(sa, sb)))
    return pk


res = {}
for c in CELLS:
    pk = gauss_fwhm(X[c], c, halfmax_peaks(X[c], c, MASK[c], SUP[c])); d = np.array([p["d_arcsec"] for p in pk])
    fw = np.array([p["fwhm_arcsec"] for p in pk if np.isfinite(p.get("fwhm_px", np.nan))])
    k, p, hf = power(X[c], c)
    # the complete compact MS-CLEAN (finished run; restored image = components * clean beam + residual) on the same
    # grid, whole field (no mask: cutting the smooth image at a mask edge adds edge power at high k), Jy/pixel
    _f = os.path.join(R, f"consistency_20kms_{c:g}/clean_compact_finished/image.fits"); _hc = fits.getheader(_f)
    bpx = np.pi * _hc["BMAJ"] * _hc["BMIN"] * 3600 ** 2 / (4 * np.log(2)) / c ** 2
    pc = power(np.nan_to_num(fits.getdata(_f)).astype(np.float64).squeeze() / bpx, c)[1]
    res[c] = dict(n_peaks=len(pk), d_median_arcsec=float(np.median(d)), d_median_px=float(np.median(d) / c),
                  d_p10_arcsec=float(np.percentile(d, 10)), d_min_arcsec=float(d.min()),
                  frac_single_pixel=float(np.mean([p["area_px"] == 1 for p in pk])),
                  fwhm_median_arcsec=float(np.median(fw)), fwhm_median_px=float(np.median(fw) / c),
                  fwhm_p10_arcsec=float(np.percentile(fw, 10)), fwhm_p10_px=float(np.percentile(fw, 10) / c),
                  frac_subpixel=float(np.mean(fw / c < 1.0)), n_fit=int(fw.size),
                  power_k=k.tolist(), power=p.tolist(), power_conv=pc.tolist(), power_frac_above_bmax=hf,
                  flux_line_Jy_kms=float(X[c][L].sum() * DV), flux_in_mask=float((X[c] * MASK[c])[L].sum() * DV), peaks=pk)
    print(f"{c}\": {len(pk)} peaks, half-max diameter median {np.median(d):.3f}\" = {np.median(d) / c:.2f} px "
          f"(10th pct {np.percentile(d, 10):.3f}\"), single-pixel peaks {res[c]['frac_single_pixel']:.0%}; "
          f"power above B_max/lambda {hf:.2e}")
    print(f"      Gaussian FWHM median {np.median(fw):.3f}\" = {np.median(fw) / c:.2f} px, 10th pct {np.percentile(fw, 10):.3f}\" "
          f"= {np.percentile(fw, 10) / c:.2f} px, below 1 px: {np.mean(fw / c < 1):.0%} ({fw.size} fits)")
cmp = {}
lam_px = {c: LBMAX_C / c for c in CELLS}
k16, p16, p08 = np.array(res[0.16]["power_k"]), np.array(res[0.16]["power"]), np.array(res[0.08]["power"])[:len(res[0.16]["power"])]
bad = np.nonzero((np.abs(p08 / p16 - 1) > 0.25) & (k16 > 0.2))[0]                # the same bins up to the 0.16" Nyquist
cmp["spectra_agree_to_cyc_arcsec"] = float(k16[bad[0]]) if bad.size else float(k16[-1])
cmp["spectra_agree_to_scale_arcsec"] = 1 / cmp["spectra_agree_to_cyc_arcsec"]
print(f"power spectra agree (25%) up to {cmp['spectra_agree_to_cyc_arcsec']:.2f} cycles/arcsec = {cmp['spectra_agree_to_scale_arcsec']:.3f}\" "
      f"(B_max/lambda = {1 / LBMAX_C:.2f})")
json.dump(dict(tag=TAG, mask_sigma=MSIG, frac=FRAC, lbmax_compact_arcsec=LBMAX_C, comparison=cmp,
               **{f"cell_{c:g}": r for c, r in res.items()}), open(os.path.join(PS, f"pixel_scale{TAG}.json"), "w"), indent=1)

# ======================================================================== figure
FS = 15
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": FS, "axes.labelsize": FS + 1,
                     "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True})
COL = {0.16: "#2a6fdb", 0.08: "#d1495b"}
fig = plt.figure(figsize=(21, 14))
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.62], hspace=0.3, wspace=0.22, left=0.04, right=0.97, top=0.94, bottom=0.03)
m0 = {c: X[c][L].sum(0) * DV / c ** 2 for c in CELLS}                      # Jy arcsec^-2 km/s
jc, ic = np.unravel_index(np.argmax(m0[0.16]), m0[0.16].shape)            # the bar's peak, 0.16" pixels
off = ((ic - 160) * 0.16, (jc - 160) * 0.16)                              # its offset, arcsec (+x = pixel index)
HALF = 2.5


def zoom(a, img, cell, title, vmax, ch_label=None):
    n = img.shape[0]; c0 = n // 2
    ci, cj = c0 + off[0] / cell, c0 + off[1] / cell; hp = HALF / cell
    sl = np.s_[int(round(cj - hp)):int(round(cj + hp)), int(round(ci - hp)):int(round(ci + hp))]
    a.imshow(img[sl], origin="lower", cmap="inferno", vmin=0, vmax=vmax, interpolation="nearest",
             extent=[HALF, -HALF, -HALF, HALF])
    a.set_xticks([]); a.set_yticks([])
    a.text(0.03, 0.97, title, transform=a.transAxes, ha="left", va="top", color="w", fontsize=FS + 1, fontweight="bold",
           bbox=dict(facecolor="k", alpha=0.75, edgecolor="0.4", boxstyle="round,pad=0.35"))
    x0, y0, sb = HALF - 1.3, -HALF + 0.55, 1.0                              # 1" scale bar bottom left, end caps, label below
    a.plot([x0 + sb, x0], [y0, y0], color="w", lw=2.2)
    for xe in (x0, x0 + sb):
        a.plot([xe, xe], [y0 - 0.08, y0 + 0.08], color="w", lw=2.2)
    a.text(x0 + sb / 2, y0 - 0.16, "1″", color="w", ha="center", va="top", fontsize=FS + 1, fontweight="bold")
    a.text(0.97, 0.03, f"pixel {cell:g}″", transform=a.transAxes, ha="right", va="bottom", color="w", fontsize=FS - 1)


# each map on its own scale (99.7th percentile inside the mask): finer pixels reach higher surface brightness
pct = lambda img, c: np.percentile(img[MASK[c][L].any(0)], 99.7)
ch = L[int(np.argmax([X[0.16][c, jc, ic] for c in L]))]                  # the line channel brightest at the bar's peak
v = grid["start_kms"] + DV * ch
for k_, c in enumerate(CELLS):
    zoom(fig.add_subplot(gs[0, k_]), m0[c], c, f"({'ab'[k_]}) {c:g}″ pixels, moment 0", pct(m0[c], c))
    chm = X[c][ch] / c ** 2
    zoom(fig.add_subplot(gs[0, 2 + k_]), chm, c, f"({'cd'[k_]}) {c:g}″ pixels, {v:.0f} km/s",
         np.percentile(chm[MASK[c][ch]], 99.5))

# (e) cut along declination through the bar's peak (the bar runs N-S), surface brightness of the channel
a = fig.add_subplot(gs[1, 0])
for c in CELLS:
    n = X[c].shape[-1]; i0 = int(round(n // 2 + off[0] / c)); j0 = n // 2 + off[1] / c
    jj = np.arange(int(j0 - HALF / c), int(j0 + HALF / c)); y = (jj - j0) * c
    a.step(y, X[c][ch][jj, i0] / c ** 2, where="mid", color=COL[c], lw=1.8, label=f"{c:g}″ pixels")
a.set_xlabel(r"$\Delta$Dec from the bar's peak [″]"); a.set_ylabel(r"Jy arcsec$^{-2}$")
a.legend(frameon=False, fontsize=FS - 1, loc="center right"); a.set_xlim(-HALF, HALF); a.set_ylim(top=a.get_ylim()[1] * 1.25)
a.text(0.03, 0.97, f"(e) cut N–S, {v:.0f} km/s", transform=a.transAxes, ha="left", va="top", fontsize=FS + 1, fontweight="bold")

# (f) Gaussian FWHM of the peaks, arcsec and pixels
a = fig.add_subplot(gs[1, 1])
bins = np.arange(0, 1.62, 0.06)
for c in CELLS:
    d = np.array([p["fwhm_arcsec"] for p in res[c]["peaks"] if np.isfinite(p.get("fwhm_px", np.nan))])
    a.hist(d, bins=bins, histtype="step", lw=2, color=COL[c], density=True,
           label=f"{c:g}″: median {np.median(d):.2f}″ = {np.median(d) / c:.1f} px (N = {len(d)})")
    a.axvline(c, color=COL[c], ls=":", lw=1.4)                              # FWHM of one pixel
a.axvline(LBMAX_C, color="0.4", ls="--", lw=1.4)
a.text(LBMAX_C + 0.02, a.get_ylim()[1] * 0.6, "λ/B$_{\\rm max}$", color="0.4", fontsize=FS - 1)
a.set_xlabel("Gaussian FWHM of model peaks [″]"); a.set_ylabel("density")
a.set_ylim(top=a.get_ylim()[1] * 1.45)
a.legend(frameon=False, fontsize=FS - 3, loc="upper right", bbox_to_anchor=(1.0, 0.9))
a.text(0.03, 0.97, "(f) peak sizes", transform=a.transAxes, ha="left", va="top", fontsize=FS + 1, fontweight="bold")

# (g) power spectra
a = fig.add_subplot(gs[1, 2:])
for c in CELLS:
    k, p = np.array(res[c]["power_k"]), np.array(res[c]["power"])
    a.loglog(k, p, color=COL[c], lw=2, label=f"{c:g}″ raw model")
    a.loglog(k, res[c]["power_conv"], color=COL[c], lw=1.4, ls="--", label=f"{c:g}″ MS-CLEAN (restored)")
    a.axvline(1 / (2 * c), color=COL[c], ls=":", lw=1.4)
a.axvline(1 / LBMAX_C, color="0.4", ls="--", lw=1.4)
yl = a.get_ylim()
for xv, t, cl in ((1 / LBMAX_C, "B$_{\\rm max}$/λ", "0.4"), (1 / 0.32, "Nyquist 0.16″", COL[0.16]), (1 / 0.16, "Nyquist 0.08″", COL[0.08])):
    a.text(xv * 0.96, yl[0] * (yl[1] / yl[0]) ** 0.55, t, color=cl, ha="right", va="top", fontsize=FS - 1, rotation=90)
a.axvline(cmp["spectra_agree_to_cyc_arcsec"], color="0.2", lw=1, alpha=0.5); a.set_xlim(right=8)
a.set_xlabel("spatial frequency [cycles / arcsec]"); a.set_ylabel(r"power $|\tilde{x}|^2$ [Jy$^2$]")
a.legend(frameon=False, fontsize=FS - 1, loc="lower left")
a.text(0.0, 1.02, "(g) power spectrum of the raw models", transform=a.transAxes, ha="left", va="bottom", fontsize=FS + 1,
       fontweight="bold")

# (h) the numbers
a = fig.add_subplot(gs[2, :]); a.axis("off")
r16, r08 = res[0.16], res[0.08]
rows = [("", "0.16″ grid", "0.08″ grid"),
        ("peaks fitted", f"{r16['n_fit']}", f"{r08['n_fit']}"),
        ("Gaussian FWHM, median", f"{r16['fwhm_median_arcsec']:.3f}″  ({r16['fwhm_median_px']:.2f} px)",
         f"{r08['fwhm_median_arcsec']:.3f}″  ({r08['fwhm_median_px']:.2f} px)"),
        ("Gaussian FWHM, 10th pct", f"{r16['fwhm_p10_arcsec']:.3f}″  ({r16['fwhm_p10_px']:.2f} px)",
         f"{r08['fwhm_p10_arcsec']:.3f}″  ({r08['fwhm_p10_px']:.2f} px)"),
        ("FWHM below 1 px", f"{r16['frac_subpixel']:.0%}", f"{r08['frac_subpixel']:.0%}"),
        (r"power above B$_{\rm max}$/λ", f"{r16['power_frac_above_bmax']:.1e}", f"{r08['power_frac_above_bmax']:.1e}"),
        ("λ/B$_{\\rm max}$ (compact)", f"{LBMAX_C:.3f}″  ({lam_px[0.16]:.1f} px)", f"{LBMAX_C:.3f}″  ({lam_px[0.08]:.1f} px)"),
        ("line flux [Jy km/s]", f"{r16['flux_line_Jy_kms']:.0f}", f"{r08['flux_line_Jy_kms']:.0f}")]
tb = a.table(cellText=[r[1:] for r in rows[1:]], rowLabels=[r[0] for r in rows[1:]], colLabels=list(rows[0][1:]),
             loc="center", cellLoc="center", colWidths=[0.22, 0.22])
tb.auto_set_font_size(False); tb.set_fontsize(FS); tb.scale(1, 1.75)
a.text(0.5, -0.06, f"the two models' power spectra agree (to 25%) up to {cmp['spectra_agree_to_cyc_arcsec']:.2f} cycles/″, "
       f"i.e. on scales ≥ {cmp['spectra_agree_to_scale_arcsec']:.2f}″  (B$_{{\\rm max}}$/λ of the compact array = "
       f"{1 / LBMAX_C:.2f} cycles/″)", transform=a.transAxes, fontsize=FS + 1, ha="center", va="top")
a.text(0.0, 1.0, "(h) finest raw features", transform=a.transAxes, ha="left", va="top", fontsize=FS + 1, fontweight="bold")
fig.suptitle(f"NGC 3110 compact, 2D-1D deconvolution on two pixel grids (raw models{', support mask' if TAG else ', no mask'})",
             fontsize=FS + 6, fontweight="bold")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"pixel_scale{TAG}.{ext}"), dpi=80 if ext == "png" else None)
print("saved", os.path.join(OUT, f"pixel_scale{TAG}.png"))

# ======================================================================== Gaussian-fit examples (for the figures below)
# four peaks found at the same place (same channel, within 0.12") on both grids, both fits centred on their peak,
# from narrow to wide (10th, 40th, 60th, 90th percentile of the 0.16" FWHM)
from matplotlib.patches import Ellipse as _Ell  # noqa: E402
off_ = lambda p, c: ((p["i"] - X[c].shape[-1] // 2) * c, (p["j"] - X[c].shape[-1] // 2) * c)   # arcsec, (x, y)
ok_ = lambda p: np.isfinite(p.get("fwhm_px", np.nan))
pairs = []
for p16 in filter(ok_, res[0.16]["peaks"]):
    o16 = off_(p16, 0.16)
    near = [q for q in filter(ok_, res[0.08]["peaks"]) if q["ch"] == p16["ch"] and np.hypot(*np.subtract(off_(q, 0.08), o16)) < 0.12]
    if near:
        pairs.append((p16, min(near, key=lambda q: np.hypot(*np.subtract(off_(q, 0.08), o16)))))
# examples only where both fits are centred on their peak (within 0.1"), so the ellipse describes that peak
centred = lambda p, c: np.hypot(p["fit"][1], p["fit"][2]) * c < 0.1
pairs = [t for t in pairs if centred(t[0], 0.16) and centred(t[1], 0.08)]
pairs.sort(key=lambda t: t[0]["fwhm_arcsec"])
EX = [pairs[int(q * (len(pairs) - 1))] for q in (0.1, 0.4, 0.6, 0.9)]
print(f"{len(pairs)} peaks found on both grids with centred fits; examples:",
      [(round(a_["fwhm_arcsec"], 2), round(b_["fwhm_arcsec"], 2)) for a_, b_ in EX])
fg = 2 * np.sqrt(2 * np.log(2))


def fit_example(a, ac, p, c, HB, fs, label=None, scalebar=False):
    """cutout (+-HB arcsec) of a peak with its fit box (dashed) and fitted FWHM ellipse on axes a; its cut along the
    fit's major axis (model pixels and the fitted Gaussian + background) added to axes ac"""
    A, x0, y0, sa, sb, th, Bg = p["fit"]; hb = p["fit_hb"]; h = int(round(HB / c))
    cut = X[c][p["ch"], p["j"] - h:p["j"] + h + 1, p["i"] - h:p["i"] + h + 1] / c ** 2
    e = [-(h + 0.5) * c, (h + 0.5) * c, -(h + 0.5) * c, (h + 0.5) * c]
    a.imshow(cut, origin="lower", cmap="inferno", extent=e, interpolation="nearest", vmin=0)
    a.add_patch(plt.Rectangle((-(hb + 0.5) * c, -(hb + 0.5) * c), (2 * hb + 1) * c, (2 * hb + 1) * c, fill=False,
                              ec="0.75", lw=0.9, ls="--"))
    a.add_patch(_Ell((x0 * c, y0 * c), fg * sa * c, fg * sb * c, angle=np.degrees(th), fill=False, ec="c", lw=1.6))
    a.set_xticks([]); a.set_yticks([])
    a.text(0.05, 0.95, label or f"{c:g}″", transform=a.transAxes, ha="left", va="top", color="w", fontsize=fs,
           fontweight="bold", path_effects=OUTL)
    a.text(0.05, 0.04, f"{p['fwhm_arcsec']:.2f}″ = {p['fwhm_px']:.1f} px", transform=a.transAxes, ha="left",
           va="bottom", color="w", fontsize=fs - 1, path_effects=OUTL)
    if scalebar:                                                         # 0.5" scale bar, top right
        xs, ys = (h + 0.5) * c - 0.6, (h + 0.5) * c - 0.12
        a.plot([xs, xs + 0.5], [ys, ys], color="w", lw=1.6, solid_capstyle="butt")
        for xe in (xs, xs + 0.5):
            a.plot([xe, xe], [ys - 0.04, ys + 0.04], color="w", lw=1.2)
        a.text(xs + 0.25, ys - 0.06, "0.5″", color="w", ha="center", va="top", fontsize=fs - 1, fontweight="bold",
               path_effects=OUTL)
    ang = th if sa >= sb else th + np.pi / 2; sm = max(sa, sb)
    t = np.linspace(-HB / c, HB / c, 400); xx, yy = x0 + t * np.cos(ang), y0 + t * np.sin(ang)
    jj, ii = np.round(yy).astype(int) + p["j"], np.round(xx).astype(int) + p["i"]
    ac.plot(t * c, X[c][p["ch"], jj, ii] / c ** 2, color=COL[c], lw=1.2, alpha=0.75, drawstyle="steps-mid",
            label=f"{c:g}″ model")
    ac.plot(t * c, (Bg + A * np.exp(-0.5 * (t / sm) ** 2)) / c ** 2, color=COL[c], lw=1.5, ls="--",
            label=f"{c:g}″ Gaussian fit")


# ======================================================================== paper versions: tight, no title; the second
# with the numbers as a booktabs-style table below (rules only, bold column headings)
for WITH_TABLE, WITH_FITS in ((False, False), (True, False), (True, True), (False, True)):
    # top: moment 0 and one channel, each on both grids (colorbars above); bottom: N-S cut, peak sizes, power spectra
    plt.close("all")
    FS = 12
    plt.rcParams.update({"font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS - 1.5, "ytick.labelsize": FS - 1.5,
                         "legend.fontsize": FS - 3.5, "legend.frameon": True, "legend.framealpha": 0.95,
                         "legend.edgecolor": "0.6", "legend.fancybox": False, "axes.linewidth": 1.0})
    FW, ML, MR, G = 12.0, 0.66, 0.12, 0.08
    GP = 0.3                                                                 # extra gap between the two pairs
    S = (FW - ML - MR - 3 * G - GP) / 4                                      # map side, inches
    XM = lambda k_: ML + k_ * (S + G) + (GP if k_ >= 2 else 0)               # left edge of map k_
    TBH = 2.35 if WITH_TABLE else 0.0                                      # the table's height, inches
    # the fit examples (optional) between the bottom row and the table: a row of cuts (HC high) under a row of
    # 0.16" | 0.08" cutout pairs
    HC, GXF = 1.55, 0.45
    SC = ((12.0 - 0.66 - 0.12 - 3 * GXF) / 4 - 0.05) / 2                    # cutout side, inches
    FBH = HC + 0.12 + SC + 0.32 if WITH_FITS else 0.0
    Y_F = 0.12 + (TBH + 0.62 if WITH_TABLE else 0.45)                       # bottom of the fits block
    MB = Y_F + FBH + 0.6 if WITH_FITS else 0.5 + (TBH + 0.25 if WITH_TABLE else 0)
    H2, GR, CBP, CBH, MT = 2.55, 0.42, 0.05, 0.1, 0.55
    FH = MB + H2 + GR + S + CBP + CBH + MT
    fig = plt.figure(figsize=(FW, FH))
    inch = lambda x, y, w, h: fig.add_axes([x / FW, y / FH, w / FW, h / FH])
    y_map = MB + H2 + GR


    import matplotlib.patheffects as pe
    OUTL = [pe.withStroke(linewidth=2.5, foreground="k")]


    def pzoom(k_, img, cell, label, vmax):
        a = inch(XM(k_), y_map, S, S)
        n = img.shape[0]; c0 = n // 2
        ci, cj = c0 + off[0] / cell, c0 + off[1] / cell; hp = HALF / cell
        sl = np.s_[int(round(cj - hp)):int(round(cj + hp)), int(round(ci - hp)):int(round(ci + hp))]
        im = a.imshow(img[sl], origin="lower", cmap="inferno", vmin=0, vmax=vmax, interpolation="nearest",
                      extent=[HALF, -HALF, -HALF, HALF])
        a.set_xticks([]); a.set_yticks([])
        a.text(0.04, 0.96, label, transform=a.transAxes, ha="left", va="top", color="w", fontsize=FS, fontweight="bold",
               path_effects=OUTL)
        x0, y0, sb, cap = HALF - 1.25, -HALF + 0.5, 1.0, 0.07                # 1" scale bar, bottom left, end caps
        a.plot([x0, x0 - sb], [y0, y0], color="w", lw=1.8, solid_capstyle="butt")
        for xe in (x0, x0 - sb):
            a.plot([xe, xe], [y0 - cap, y0 + cap], color="w", lw=1.4, solid_capstyle="butt")
        a.text(x0 - sb / 2, y0 - 0.14, "1″", color="w", ha="center", va="top", fontsize=FS, fontweight="bold",
               path_effects=OUTL)
        return im


    def pcbar(k_, im, unit):
        """one colorbar above the pair of maps k_, k_ + 1"""
        cax = inch(XM(k_), y_map + S + CBP, 2 * S + G, CBH)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cax.xaxis.set_ticks_position("top"); cax.xaxis.set_label_position("top")
        cax.tick_params(labelsize=FS - 2, length=3, pad=2); cb.set_label(unit, fontsize=FS - 1, labelpad=3)


    # one linear scale per pair (the brighter grid's 99.7th / 99.5th percentile inside the mask)
    vm0 = max(pct(m0[c], c) for c in CELLS)
    vch = max(np.percentile((X[c][ch] / c ** 2)[MASK[c][ch]], 99.5) for c in CELLS)
    for k_, c in enumerate(CELLS):
        im0 = pzoom(k_, m0[c], c, f"({'ab'[k_]}) {c:g}″ pixels", vm0)
        im1 = pzoom(2 + k_, X[c][ch] / c ** 2, c, f"({'cd'[k_]}) {c:g}″, {v:.0f} km s$^{{-1}}$", vch)
    pcbar(0, im0, r"moment 0 [Jy arcsec$^{-2}$ km s$^{-1}$]")
    pcbar(2, im1, r"channel [Jy arcsec$^{-2}$]")

    GX = 0.78                                                                # gap between the bottom panels (y labels)
    w1 = 4.3; w3 = FW - ML - MR - w1 - GX                                    # peak sizes | power spectrum
    style = lambda a: (a.tick_params(which="both", direction="in", top=True, right=True), a.minorticks_on())
    plab = lambda a, t: a.text(0.04, 0.95, t, transform=a.transAxes, ha="left", va="top", fontsize=FS, fontweight="bold")

    a = inch(ML, MB, w1, H2); style(a)                                       # (e) peak sizes
    for c in CELLS:
        d = np.array([p["fwhm_arcsec"] for p in res[c]["peaks"] if np.isfinite(p.get("fwhm_px", np.nan))])
        a.hist(d, bins=np.arange(0, 1.62, 0.06), histtype="step", lw=1.6, color=COL[c], density=True,
               label=f"{c:g}″: median {np.median(d):.2f}″ ({np.median(d) / c:.1f} px)")
        a.axvline(c, color=COL[c], ls=":", lw=1.2)
    a.axvline(LBMAX_C, color="0.35", ls="--", lw=1.2)
    a.set_ylim(top=a.get_ylim()[1] * 2.0)
    a.text(LBMAX_C + 0.03, a.get_ylim()[1] * 0.08, r"$\lambda/B_{\rm max}$", color="0.35", fontsize=FS - 1.5)
    a.plot([], [], ":", color="0.35", lw=1.2, label="1 pixel (each grid)")
    a.set_xlim(0, 1.6); a.set_xlabel("Gaussian FWHM of peaks [″]"); a.set_ylabel("density")
    a.legend(loc="upper right", bbox_to_anchor=(1.0, 0.88)); plab(a, "(e) peak sizes")

    a = inch(ML + w1 + GX, MB, w3, H2); style(a)                             # (f) power spectra
    for c in CELLS:
        a.loglog(res[c]["power_k"], res[c]["power"], color=COL[c], lw=1.6, label=f"{c:g}″, deconvolved")
        a.loglog(res[c]["power_k"], res[c]["power_conv"], color=COL[c], lw=1.2, ls="--", label=f"{c:g}″, MS-CLEAN (restored)")
        a.axvline(1 / (2 * c), color=COL[c], ls=":", lw=1.2)
    a.axvline(1 / LBMAX_C, color="0.35", ls="--", lw=1.2)
    a.axvspan(a.get_xlim()[0], cmp["spectra_agree_to_cyc_arcsec"], color="0.5", alpha=0.08, lw=0)
    pmin = min(min(res[c]["power"]) for c in CELLS)
    a.set_xlim(right=7.5); a.set_ylim(pmin * 1e-3, max(max(res[c]["power"]) for c in CELLS) * 40); yl = a.get_ylim()
    for xv, t, cl in ((1 / LBMAX_C, r"$B_{\rm max}/\lambda$", "0.35"), (1 / 0.32, "Nyquist 0.16″", COL[0.16]),
                      (1 / 0.16, "Nyquist 0.08″", COL[0.08])):
        a.text(xv * 0.93, yl[1] * 0.4, t, color=cl, ha="right", va="top", fontsize=FS - 2.5, rotation=90)
    a.text(0.04, 0.06, f"agree to 25% on scales ≥ {cmp['spectra_agree_to_scale_arcsec']:.2f}″", transform=a.transAxes,
           fontsize=FS - 2, color="0.25")
    a.set_xlabel(r"spatial frequency [arcsec$^{-1}$]"); a.set_ylabel(r"power [Jy$^2$]")
    a.legend(loc="lower left", bbox_to_anchor=(0.0, 0.12)); plab(a, "(f) power spectrum")
    if WITH_FITS:
        WC = (FW - ML - MR - 3 * GXF) / 4                                    # one example's column width
        for k_, pair in enumerate(EX):
            x_ = ML + k_ * (WC + GXF)
            ac = inch(x_, Y_F, WC, HC); style(ac)
            for r_, (p, c) in enumerate(zip(pair, CELLS)):
                a = inch(x_ + r_ * (SC + 0.05), Y_F + HC + 0.12, SC, SC)
                fit_example(a, ac, p, c, 0.8, FS - 2, scalebar=(k_ == 0 and r_ == 0))
                if r_ == 0:
                    a.text(0.0, 1.04, f"({'ghij'[k_]}) {grid['start_kms'] + DV * p['ch']:.0f} km s$^{{-1}}$",
                           transform=a.transAxes, ha="left", va="bottom", fontsize=FS, fontweight="bold")
            ac.set_xlim(-0.8, 0.8); ac.set_xlabel("offset along the fit's major axis [″]", fontsize=FS - 1)
            if k_ == 0:
                ac.set_ylabel(r"Jy arcsec$^{-2}$"); ac.legend(loc="upper left", fontsize=FS - 4.5, handlelength=1.6)
    if WITH_TABLE:
        r16, r08 = res[0.16], res[0.08]
        two = lambda f: (f(r16, 0.16), f(r08, 0.08))
        ROWS = [("Peaks fitted (all line channels)", *two(lambda r, c: f"{r['n_fit']}")),
                ("Gaussian FWHM, median", *two(lambda r, c: f"{r['fwhm_median_arcsec']:.2f}″  ({r['fwhm_median_px']:.1f} px)")),
                ("Gaussian FWHM, 10th percentile", *two(lambda r, c: f"{r['fwhm_p10_arcsec']:.2f}″  ({r['fwhm_p10_px']:.1f} px)")),
                ("Peaks with FWHM below 1 px", *two(lambda r, c: f"{100 * r['frac_subpixel']:.0f}%")),
                (r"Power at $k > B_{\rm max}/\lambda$", *two(lambda r, c: f"{100 * r['power_frac_above_bmax']:.1f}%")),
                (r"$\lambda/B_{\rm max}$ (compact)", *two(lambda r, c: f"{LBMAX_C:.2f}″  ({LBMAX_C / c:.1f} px)")),
                ("Line flux [Jy km s$^{-1}$]", *two(lambda r, c: f"{r['flux_line_Jy_kms']:.0f}")),
                ("Power spectra agree (25%) on scales", f"≥ {cmp['spectra_agree_to_scale_arcsec']:.2f}″", None)]
        TW = 8.4; a = inch((FW - TW) / 2, 0.12, TW, TBH); a.axis("off"); a.set_xlim(0, 1); a.set_ylim(0, 1)
        n_ = len(ROWS) + 1; dy = 1 / (n_ + 0.6); X0, X1, X2 = 0.0, 0.62, 0.86
        y = 1 - 0.3 * dy
        a.axhline(1 - 0.02, color="k", lw=1.2)                              # top rule
        for x_, t, ha in ((X0, "Quantity", "left"), (X1, "0.16″ grid", "center"), (X2, "0.08″ grid", "center")):
            a.text(x_, y - 0.5 * dy, t, ha=ha, va="center", fontsize=FS, fontweight="bold")
        y -= dy; a.axhline(y - 0.1 * dy, color="k", lw=0.7)                 # rule under the headings
        y -= 0.2 * dy
        for lab, c1, c2 in ROWS:
            a.text(X0, y - 0.5 * dy, lab, ha="left", va="center", fontsize=FS - 0.5)
            if c2 is None:                                                   # one value for both grids
                a.text((X1 + X2) / 2, y - 0.5 * dy, c1, ha="center", va="center", fontsize=FS - 0.5)
            else:
                a.text(X1, y - 0.5 * dy, c1, ha="center", va="center", fontsize=FS - 0.5)
                a.text(X2, y - 0.5 * dy, c2, ha="center", va="center", fontsize=FS - 0.5)
            y -= dy
        a.axhline(y + 0.05 * dy, color="k", lw=1.2)                         # bottom rule
    name = f"pixel_scale{TAG}_paper{'_table' if WITH_TABLE else ''}{'_fits' if WITH_FITS else ''}"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200 if ext == "png" else None)
    plt.close(fig); print("saved", os.path.join(OUT, f"{name}.png"))

# ======================================================================== the Gaussian fits, illustrated
# four peaks found at the same place (same channel, within 0.12") on both grids, from narrow to wide (10th, 40th,
# 60th, 90th percentile of the 0.16" FWHM): the cutout with the fit box and the fitted FWHM ellipse (top two rows), and
# a cut through the peak along the fit's major axis, data and fitted Gaussian + background (bottom row)
plt.close("all")
FS = 12; HB = 0.8                                                        # cutout half-size, arcsec
NC = len(EX); S2, G2, ML2, MB2, H3, GR2, MT2 = 2.55, 0.5, 0.62, 0.5, 2.0, 0.55, 0.3
FW2 = ML2 + NC * S2 + (NC - 1) * G2 + 0.12; FH2 = MB2 + H3 + GR2 + 2 * S2 + 0.1 + MT2
fig = plt.figure(figsize=(FW2, FH2))
inch2 = lambda x, y, w, h: fig.add_axes([x / FW2, y / FH2, w / FW2, h / FH2])
fg = 2 * np.sqrt(2 * np.log(2))
for k_, pair in enumerate(EX):
    x_ = ML2 + k_ * (S2 + G2)
    ac = inch2(x_, MB2, S2, H3); ac.tick_params(which="both", direction="in", top=True, right=True); ac.minorticks_on()
    for r_, (p, c) in enumerate(zip(pair, CELLS)):
        A, x0, y0, sa, sb, th, Bg = p["fit"]; hb = p["fit_hb"]; h = int(round(HB / c))
        cut = X[c][p["ch"], p["j"] - h:p["j"] + h + 1, p["i"] - h:p["i"] + h + 1] / c ** 2
        a = inch2(x_, MB2 + H3 + GR2 + (1 - r_) * (S2 + 0.1), S2, S2)
        e = [-(h + 0.5) * c, (h + 0.5) * c, -(h + 0.5) * c, (h + 0.5) * c]
        a.imshow(cut, origin="lower", cmap="inferno", extent=e, interpolation="nearest", vmin=0)
        a.add_patch(plt.Rectangle((-(hb + 0.5) * c, -(hb + 0.5) * c), (2 * hb + 1) * c, (2 * hb + 1) * c, fill=False,
                                  ec="0.75", lw=1.0, ls="--"))
        a.add_patch(_Ell((x0 * c, y0 * c), fg * sa * c, fg * sb * c, angle=np.degrees(th), fill=False, ec="c", lw=1.8))
        a.set_xticks([]); a.set_yticks([])
        a.text(0.04, 0.96, f"{c:g}″ pixels", transform=a.transAxes, ha="left", va="top", color="w", fontsize=FS - 1,
               fontweight="bold", path_effects=OUTL)
        a.text(0.04, 0.04, f"FWHM {p['fwhm_arcsec']:.2f}″ = {p['fwhm_px']:.1f} px", transform=a.transAxes, ha="left",
               va="bottom", color="w", fontsize=FS - 1.5, path_effects=OUTL)
        if r_ == 0:
            a.set_title(f"({'abcd'[k_]})  {grid['start_kms'] + DV * p['ch']:.0f} km s$^{{-1}}$", fontsize=FS, fontweight="bold")
        if k_ == 0:                                                       # 0.5" scale bar
            xs, ys = (h + 0.5) * c - 0.62, (h + 0.5) * c - 0.12
            a.plot([xs, xs + 0.5], [ys, ys], color="w", lw=1.8, solid_capstyle="butt")
            for xe in (xs, xs + 0.5):
                a.plot([xe, xe], [ys - 0.04, ys + 0.04], color="w", lw=1.4)
            a.text(xs + 0.25, ys - 0.06, "0.5″", color="w", ha="center", va="top", fontsize=FS - 1, fontweight="bold",
                   path_effects=OUTL)
        # cut along the fit's major axis through the fitted centre: data (nearest pixel) and the fitted profile
        ang = th if sa >= sb else th + np.pi / 2; sm = max(sa, sb)
        t = np.linspace(-HB / c, HB / c, 400); xx, yy = x0 + t * np.cos(ang), y0 + t * np.sin(ang)
        jj, ii = np.round(yy).astype(int) + p["j"], np.round(xx).astype(int) + p["i"]
        ac.plot(t * c, X[c][p["ch"], jj, ii] / c ** 2, color=COL[c], lw=1.3, alpha=0.75, drawstyle="steps-mid",
                label=f"{c:g}″ data")
        ac.plot(t * c, (Bg + A * np.exp(-0.5 * (t / sm) ** 2)) / c ** 2, color=COL[c], lw=1.6, ls="--",
                label=f"{c:g}″ Gaussian fit")
    ac.set_xlim(-HB, HB); ac.set_xlabel("offset along the major axis [″]")
    if k_ == 0:
        ac.set_ylabel(r"Jy arcsec$^{-2}$"); ac.legend(loc="upper left", fontsize=FS - 3.5)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"pixel_scale{TAG}_fits.{ext}"), dpi=200 if ext == "png" else None)
print("saved", os.path.join(OUT, f"pixel_scale{TAG}_fits.png"))
