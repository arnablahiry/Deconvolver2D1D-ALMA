"""
Experiment 21, analysis (refine_preserve.py analyze): does 2D1D resolve the data more finely while keeping its
flux and spectra? Every section states its criterion; the report says which pass and which do not.

  1  real data: flux and spectra of 2D1D vs a converged CLEAN on the same inputs, field and inside 3D masks
  2  simulations with a known 0.2" sky: flux, spectra, fake flux, resolution (error vs smoothing scale, Fourier
     ring correlation, transfer function per spatial frequency)
  3  injection into the real data: flux recovery of sources, and pairs closer than the beam: resolved or not
  4  null: fake flux from noise alone
  5  the independent extended configuration (0.2"): how well each compact-data model predicts it, per spatial frequency
  6  jackknife: the two execution blocks alone: flux and the scales on which structure repeats
"""
import json
import os
import sys

import numpy as np

import refine_preserve as R

OUT, FIG, CELL, DV, BEAM, BPX = R.OUT, R.FIG, R.CELL, R.DV, R.BEAM, R.BPX
LINE, FREE, NZ, NY, NX = R.LINE, R.FREE, R.NZ, R.NY, R.NX
KL = 206.265            # cycles/arcsec -> kilo-lambda
THETAS = [0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0]


def _uv_range():
    """the compact array's shortest and longest projected baselines (kλ), from a GUI dataset's uv.json (metres)"""
    import glob
    lam = 299792458.0 / (float(R.META["restfreq_ghz"] if "restfreq_ghz" in R.META else R.META["imaging"]["restfreq_ghz"]) * 1e9
                         / (1 + float(np.mean(R.VEL)) / 299792.458))
    for f in sorted(glob.glob("/scratch/alahiry/deconvgui/datasets/*NGC3110-natural*/uv.json")):
        r = np.concatenate([np.hypot(np.asarray(s["u"]), np.asarray(s["v"])) for s in json.load(open(f))["sets"]])
        return float(r[r > 0].min() / lam / 1e3), float(r.max() / lam / 1e3)
    return 9.5, 290.0


UV_MIN, UV_MAX = _uv_range()


def _uv_lines(a, y):
    for x, t in ((UV_MIN, " shortest baseline"), (UV_MAX, " longest baseline")):
        a.axvline(x, color="gray", ls=":", lw=0.9); a.text(x, y, t, fontsize=8, rotation=90, va="top")


def p(*a):
    print(*a, flush=True)


def load(path):
    return np.load(path).astype(np.float64) if os.path.exists(path) else None


def models(name):
    """the 2D1D and CLEAN results of cube `name`, all in Jy/pixel: models, and beam-convolved maps / the beam area.
    CLEAN± (standard: |peaks|, negative components allowed) and CLEAN+ (positive components only, as 2D1D), both the
    exact-PSF GPU multiscale CLEAN with no mask, to 2 sigma."""
    out = {}
    m = load(os.path.join(OUT, "res", name, "model.npy"))
    if m is not None:
        out["2D1D"] = m
        # the noise-driven positive floor, measured per pixel in the line-free channels (no emission there), removed
        out["2D1D − floor"] = m - m[FREE].mean(axis=0, keepdims=True)
    beam = json.load(open(os.path.join(OUT, "ds", name, "meta.json")))["beam"]
    bpx = 1.1331 * beam[0] * beam[1] / CELL ** 2
    for tag, folder in (("CLEAN±", "gclean"), ("CLEAN+", "gclean_pos")):
        c = os.path.join(OUT, folder, name)
        cm = load(os.path.join(c, "model.npy"))
        if cm is None:
            continue
        rs = load(os.path.join(c, "restored.npy"))
        out[f"{tag} model"] = cm
        out[f"{tag} ∗ beam"] = rs / bpx
        out[f"{tag} image"] = (rs + load(os.path.join(c, "residual.npy"))) / bpx
    return out


def gauss_smooth(cube, fwhm):
    """per-channel Gaussian smoothing, flux kept (FFT)"""
    if fwhm <= 0:
        return cube
    s = fwhm / 2.3548 / CELL
    ky = np.fft.fftfreq(cube.shape[-2])[:, None]
    kx = np.fft.rfftfreq(cube.shape[-1])[None, :]
    g = np.exp(-2 * np.pi ** 2 * s ** 2 * (kx ** 2 + ky ** 2))
    return np.fft.irfft2(np.fft.rfft2(cube) * g, s=cube.shape[-2:])


def ring_stats(a, b, nbin=40):
    """over the line channels: per ring of spatial frequency, the Fourier ring correlation of a and b and the transfer
    Re<A B*>/<|B|^2> (b = the reference). Returns k (kλ), frc, transfer"""
    A, B = np.fft.rfft2(a[LINE]), np.fft.rfft2(b[LINE])
    ky = np.fft.fftfreq(a.shape[-2], d=CELL)[:, None]
    kx = np.fft.rfftfreq(a.shape[-1], d=CELL)[None, :]
    k = np.hypot(kx, ky) * KL
    edges = np.geomspace(5, 1000, nbin + 1)
    idx = np.digitize(k, edges) - 1
    w = np.ones_like(k)
    w[:, 1:-1 if a.shape[-1] % 2 == 0 else None] = 2         # rfft: the other half-plane
    xs, frc, tr = [], [], []
    for i in range(nbin):
        sel = idx == i
        if not sel.any():
            continue
        ab = (A[:, sel] * np.conj(B[:, sel]) * w[sel]).sum()
        aa, bb = (np.abs(A[:, sel]) ** 2 * w[sel]).sum(), (np.abs(B[:, sel]) ** 2 * w[sel]).sum()
        xs.append(np.sqrt(edges[i] * edges[i + 1])); frc.append(float(ab.real / np.sqrt(aa * bb + 1e-300))); tr.append(float(ab.real / (bb + 1e-300)))
    return np.array(xs), np.array(frc), np.array(tr)


def crossing(x, y, level):
    """first x where y falls below level (log-interpolated); None if never"""
    for i in range(1, len(y)):
        if y[i - 1] >= level > y[i]:
            f = (y[i - 1] - level) / (y[i - 1] - y[i])
            return float(np.exp(np.log(x[i - 1]) + f * (np.log(x[i]) - np.log(x[i - 1]))))
    return None


def spectrum_stats(s, ref):
    """spectral fidelity over the line channels: total ratio, correlation, centroid and width shifts (km/s), max |Δ|/peak"""
    v = R.VEL[LINE]
    s, ref = np.asarray(s)[LINE], np.asarray(ref)[LINE]
    mom = lambda y: ((y * v).sum() / y.sum(), np.sqrt((y * (v - (y * v).sum() / y.sum()) ** 2).sum() / y.sum()))
    (c1, w1), (c0, w0) = mom(np.clip(s, 0, None)), mom(np.clip(ref, 0, None))
    return dict(ratio=float(s.sum() / ref.sum()), corr=float(np.corrcoef(s, ref)[0, 1]), d_centroid=float(c1 - c0),
                d_width=float(w1 - w0), max_dev=float(np.abs(s - ref).max() / ref.max()))


def psf_conv(cube, psf):
    """cube (Jy/pixel) * a dirty beam (peak 1, 2x the image, centred on its peak): Jy/beam on the image grid"""
    from scipy.signal import fftconvolve
    out = np.zeros_like(cube)
    for c in LINE:
        py, px = np.unravel_index(np.argmax(psf[c]), psf.shape[1:])
        k = psf[c, py - NY: py + NY, px - NX: px + NX]
        full = fftconvolve(cube[c], k, mode="full")
        out[c] = full[NY: 2 * NY, NX: 2 * NX]
    return out


# ================================================================ sections
def sec_real(rep, res, plt):
    from deconvgui.worker import dendro_mask
    from restore import convolve_with_beam
    M = models("real")
    if "CLEAN± model" not in M or "CLEAN+ model" not in M:
        rep.append("## 1. Real data\nCLEAN on the real cube not finished.\n"); return
    dirty = load(os.path.join(OUT, "ds", "real", "dirty.npy"))
    sig = R.noise_sigma(dirty)
    resid = {"2D1D": load(os.path.join(OUT, "res", "real", "residual.npy")),
             "CLEAN±": load(os.path.join(OUT, "gclean", "real", "residual.npy")), "CLEAN+": load(os.path.join(OUT, "gclean_pos", "real", "residual.npy"))}
    M["2D1D ∗ beam"] = convolve_with_beam(M["2D1D"], CELL, *BEAM) / BPX
    M["2D1D − floor ∗ beam"] = convolve_with_beam(M["2D1D − floor"], CELL, *BEAM) / BPX
    # 3D masks: astrodendro on the converged CLEAN± image (worker.dendro_mask), several thresholds
    tmp = os.path.join(OUT, "masks")
    os.makedirs(tmp, exist_ok=True)
    for f in ("restored.npy", "residual.npy"):
        if os.path.lexists(os.path.join(tmp, f)):
            os.remove(os.path.join(tmp, f))
        os.symlink(os.path.join(OUT, "gclean", "real", f), os.path.join(tmp, f))
    masks = {"field": np.ones((NZ, NY, NX), bool)}
    for lv in (3, 10, 40):
        dendro_mask(tmp, BPX, dict(min_value=lv, min_delta=1, peak=lv + 2, min_beams=2))
        masks[f"astrodendro {lv}σ"] = np.load(os.path.join(tmp, "dendro_mask.npy")) > 0
        os.replace(os.path.join(tmp, "dendro_mask.npy"), os.path.join(tmp, f"dendro_{lv}.npy"))
    keys = ["2D1D", "2D1D ∗ beam", "2D1D − floor ∗ beam", "CLEAN± model", "CLEAN± image", "CLEAN+ model", "CLEAN+ image"]
    rows, spec = [], {}
    for mn, mk in masks.items():
        sp = {k: (M[k] * mk).sum(axis=(1, 2)) for k in keys}
        spec[mn] = sp
        rows.append((mn, float(mk.mean()), {k: float(sp[k].sum() * DV) for k in keys},
                     spectrum_stats(sp["2D1D ∗ beam"], sp["CLEAN± image"]), spectrum_stats(sp["2D1D ∗ beam"], sp["CLEAN+ image"]),
                     spectrum_stats(sp["2D1D − floor ∗ beam"], sp["CLEAN± image"]), spectrum_stats(sp["2D1D − floor ∗ beam"], sp["CLEAN+ image"])))
    res["real"] = dict(noise=sig, residual_rms={k: dict(line=float(v[LINE].std() / sig), free=float(v[FREE].std() / sig)) for k, v in resid.items()},
                       regions={r[0]: dict(fraction=r[1], flux=r[2], vs_clean_pm=r[3], vs_clean_pos=r[4], floor_vs_clean_pm=r[5], floor_vs_clean_pos=r[6]) for r in rows})
    rr = res["real"]["residual_rms"]
    rep.append("## 1. Real data: 2D1D vs a converged CLEAN on the same inputs\n"
               "CLEAN: multiscale (Cornwell 2008; Gaussian components of FWHM " + ", ".join(str(v) for v in R.CLEAN_SCALES_GPU) + " px, scale bias 0.6, gain 0.1) "
               "with the exact full PSF, no mask, on the same dirty cube and PSF as 2D1D, to 2σ "
               f"({2 * sig * 1e3:.3f} mJy/beam). CLEAN±: |peaks| (as CASA; negative components allowed); CLEAN+: positive components only (as 2D1D). "
               "Residual rms / noise, line | line-free channels: " + "; ".join(f"{k} {v['line']:.2f} | {v['free']:.2f}" for k, v in rr.items()) +
               " (≈ 1: everything above the noise is in the model).\n\n"
               "Flux (Jy km/s) over the field and inside astrodendro masks (made on CLEAN±'s image, model ∗ beam + residual). "
               "Image = model ∗ clean beam + residual.\n\n"
               "| region | cube % | 2D1D | 2D1D ∗ beam | 2D1D − floor ∗ beam | CLEAN± model | CLEAN± image | CLEAN+ model | CLEAN+ image | 2D1D∗beam / CLEAN± image | / CLEAN+ image | (2D1D − floor)∗beam / CLEAN± image | / CLEAN+ image | spectral corr. | Δ centroid | Δ width (km/s) |\n"
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for mn, fr, fl, s1, s2, s3, s4 in rows:
        rep.append(f"| {mn} | {100 * fr:.1f} | " + " | ".join(f"{fl[k]:.1f}" for k in keys) +
                   f" | {s1['ratio']:.3f} | {s2['ratio']:.3f} | **{s3['ratio']:.3f}** | **{s4['ratio']:.3f}** | {s1['corr']:.4f} | {s1['d_centroid']:+.1f} | {s1['d_width']:+.1f} |")
    rep.append("\nCriterion: inside the emission (the masks) 2D1D ∗ beam matches CLEAN's image within the flux calibration accuracy "
               "(~5–10 %) with the same line shape; over the whole field the difference is the unmeasured large scales.\n")
    fig, ax = plt.subplots(2, len(masks), figsize=(5 * len(masks), 7), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    v = R.VEL
    for j, (mn, sp) in enumerate(spec.items()):
        for k, col, ls in [("CLEAN± image", "C0", "-"), ("CLEAN+ image", "C2", "--"), ("2D1D ∗ beam", "C3", ":"), ("2D1D − floor ∗ beam", "C3", "-")]:
            ax[0, j].plot(v, sp[k], color=col, ls=ls, label=f"{k} ({sp[k].sum() * DV:.0f})")
            if "2D1D" not in k:
                ax[1, j].plot(v[LINE], sp["2D1D − floor ∗ beam"][LINE] / sp[k][LINE], color=col, ls=ls)
        ax[1, j].axhline(1, color="k", lw=0.6); ax[1, j].set_ylim(0.6, 1.6)
        ax[0, j].set_title(mn); ax[0, j].legend(fontsize=8); ax[1, j].set_xlabel("velocity (km/s)")
    ax[0, 0].set_ylabel("flux density (Jy)"); ax[1, 0].set_ylabel("(2D1D − floor)∗beam / CLEAN")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "21a_real_spectra.png"), dpi=110); plt.close(fig)


def sec_sims(rep, res, plt):
    rep.append("## 2. Simulations: a known sky with 0.2″ structure, observed with the compact PSF and noise (the real noise power spectrum)\n"
               "Truth: the multiscale CLEAN model of the extended configuration (0.22″ × 0.17″ beam) smoothed to that beam "
               "(`sim_clumpy`), and the same + a 4″ smooth component with 30 % of the flux (`sim_diffuse`). Each method's result is compared "
               "with the truth.\n")
    res["sims"] = {}
    for name in R.SIMS:
        t = load(os.path.join(OUT, "ds", name, "truth.npy"))
        M = models(name)
        if t is None or "2D1D" not in M or "CLEAN+ model" not in M or "CLEAN± model" not in M:
            rep.append(f"`{name}`: not finished.\n"); continue
        dirty = load(os.path.join(OUT, "ds", name, "dirty.npy"))
        M = {"dirty / beam area": dirty / BPX, **M}
        tot = t.sum() * DV
        tb = gauss_smooth(t, np.sqrt(BEAM[0] * BEAM[1]))
        inside = tb > 0.01 * tb.max()                        # where the truth, at the clean beam, has any emission
        S = {}
        for k, m in M.items():
            st = spectrum_stats(m.sum(axis=(1, 2)), t.sum(axis=(1, 2)))
            err = [float(np.linalg.norm(gauss_smooth(m[LINE] - t[LINE], th)) / np.linalg.norm(gauss_smooth(t[LINE], th))) for th in THETAS]
            kk, frc, tr = ring_stats(m, t)
            S[k] = dict(flux=float(m.sum() * DV), flux_ratio=float(m.sum() / t.sum()), outside=float((m * ~inside).sum() * DV),
                        spectrum=st, err=err, theta_err30=_res_scale(THETAS, err, 0.3),
                        frc_k=kk.tolist(), frc=frc.tolist(), transfer=tr.tolist(), frc_res=_frc_res(kk, frc, 0.5))
        res["sims"][name] = dict(truth_flux=tot, methods=S)
        rep.append(f"### `{name}` (truth {tot:.1f} Jy km/s)\n"
                   "| method | flux / truth | flux outside the truth (Jy km/s) | spectrum corr. | Δ centroid | Δ width (km/s) | max spectral error / peak | error at 0.2″ | at 0.5″ | at 1.0″ | finest scale with error < 30 % | FRC = 0.5 at |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k, s in S.items():
            e = dict(zip(THETAS, s["err"]))
            rep.append(f"| {k} | **{s['flux_ratio']:.3f}** | {s['outside']:.1f} | {s['spectrum']['corr']:.4f} | {s['spectrum']['d_centroid']:+.1f} | "
                       f"{s['spectrum']['d_width']:+.1f} | {s['spectrum']['max_dev']:.3f} | {e[0.2]:.2f} | {e[0.5]:.2f} | {e[1.0]:.2f} | "
                       f"{_fmt_scale(s['theta_err30'])} | {_fmt_scale(s['frc_res'])} |")
        rep.append("")
        _fig_sim(plt, name, t, M, S)
    rep.append("Errors: ‖G_θ ∗ (x − truth)‖ / ‖G_θ ∗ truth‖ over the line channels, both smoothed to θ (FWHM). "
               "FRC: Fourier ring correlation with the truth; its 0.5 crossing, as a scale (1/k). "
               "Criteria: flux / truth ≈ 1 with little flux outside the truth (preservation), and a finer scale than CLEAN at the same error (refinement).\n")


def _res_scale(thetas, err, level):
    """smallest smoothing scale at which the error is below `level`"""
    for th, e in zip(thetas, err):
        if e < level:
            return th
    return None


def _frc_res(k, frc, level):
    kc = None
    for i in range(1, len(k)):
        if frc[i - 1] >= level > frc[i]:
            f = (frc[i - 1] - level) / (frc[i - 1] - frc[i])
            kc = float(np.exp(np.log(k[i - 1]) + f * (np.log(k[i]) - np.log(k[i - 1]))))
            break
    return None if kc is None else KL / kc            # arcsec: the scale of that spatial frequency


def _fmt_scale(v):
    return "–" if v is None else f"{v:.2f}″"


def _fig_sim(plt, name, t, M, S):
    fig = plt.figure(figsize=(26, 11))
    gs = fig.add_gridspec(2, 6)
    show = [("truth", t), ("dirty / beam area", M["dirty / beam area"]), ("CLEAN+ ∗ beam", M["CLEAN+ ∗ beam"]),
            ("CLEAN+ model (0.1″)", gauss_smooth(M["CLEAN+ model"], 0.1)), ("2D1D (0.1″)", gauss_smooth(M["2D1D"], 0.1)), ("truth (0.1″)", gauss_smooth(t, 0.1))]
    vmax = np.percentile(t[LINE].sum(0), 99.7)
    for j, (title, c) in enumerate(show):
        a = fig.add_subplot(gs[0, j]); im = c[LINE].sum(0) * DV
        a.imshow(np.sqrt(np.clip(im, 0, None)), origin="lower", cmap="inferno", vmin=0, vmax=np.sqrt(vmax * DV))
        a.set_title(title); a.set_xticks([]); a.set_yticks([])
    a = fig.add_subplot(gs[1, 0:2])
    for k, s in S.items():
        if "image" not in k:
            a.plot(THETAS, s["err"], "o-", label=k)
    a.axvline(np.sqrt(BEAM[0] * BEAM[1]), color="k", ls=":", lw=0.8); a.text(np.sqrt(BEAM[0] * BEAM[1]), 0.02, " clean beam", fontsize=8)
    a.axhline(0.3, color="gray", lw=0.6); a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("smoothing FWHM θ (″)"); a.set_ylabel("error vs truth at θ"); a.legend(fontsize=8)
    a = fig.add_subplot(gs[1, 2:4])
    for k, s in S.items():
        if "image" not in k:
            a.plot(s["frc_k"], s["transfer"], label=k)
    a.axhline(1, color="k", lw=0.6); _uv_lines(a, 1.55)
    a.set_xscale("log"); a.set_ylim(-0.2, 1.6); a.set_xlabel("spatial frequency (kλ)"); a.set_ylabel("transfer Re⟨X T*⟩/⟨|T|²⟩"); a.legend(fontsize=8)
    a = fig.add_subplot(gs[1, 4:6])
    v = R.VEL
    a.plot(v, t.sum(axis=(1, 2)), "k-", lw=2.5, label=f"truth ({t.sum() * DV:.0f})")
    for k in ("CLEAN± image", "CLEAN+ image", "CLEAN+ ∗ beam", "2D1D", "2D1D − floor"):
        a.plot(v, M[k].sum(axis=(1, 2)), label=f"{k} ({S[k]['flux']:.0f})")
    a.set_xlabel("velocity (km/s)"); a.set_ylabel("flux density (Jy)"); a.legend(fontsize=8)
    fig.suptitle(f"{name}: moment 0 (sqrt), error vs smoothing scale, transfer per spatial frequency, spectra (Jy km/s)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"21b_{name}.png"), dpi=90); plt.close(fig)


def sec_inj(rep, res, plt):
    rep.append("## 3. Injection into the real data: model(data + sources) − model(data)\n")
    base = models("real")
    keys = ["2D1D", "CLEAN± model", "CLEAN+ model", "CLEAN+ ∗ beam"]
    res["inj"] = {}
    for name in R.INJ:
        M = models(name)
        if any(k not in M or k not in base for k in keys):
            rep.append(f"`{name}`: not finished.\n"); continue
        truth = json.load(open(os.path.join(OUT, "ds", name, "meta.json")))["injected"]
        D = {k: M[k] - base[k] for k in keys}
        yy, xx = np.mgrid[0:NY, 0:NX]
        rows = []
        for s_ in [t for t in truth if t["kind"] == "source"]:
            x0, y0 = R.pix(s_["dx"], s_["dy"])
            ap = np.hypot(xx - x0, yy - y0) <= (np.hypot(s_["fwhm"], np.sqrt(BEAM[0] * BEAM[1])) + 0.3) / CELL
            rows.append((s_["name"], s_["fwhm"], s_["flux"], *[float((D[k][:, ap]).sum() * DV / s_["flux"]) for k in keys]))
        pairs = {}
        for pname, cx, cy, sep, v0 in R.PAIRS:
            flux = sum(t["flux"] for t in truth if t.get("pair") == pname)
            x0, y0 = R.pix(cx, cy)
            ap = np.hypot(xx - x0, yy - y0) <= (sep / 2 + np.sqrt(BEAM[0] * BEAM[1]) + 0.3) / CELL
            chans = np.abs(R.VEL - v0) <= 2.5 * R.LINE_SIG
            out = {}
            for k in keys:
                img = gauss_smooth(D[k][chans], 0.1 if "beam" not in k else 0).sum(0)
                prof = img[:, int(round(x0)) - 3: int(round(x0)) + 4].sum(1)             # north-south through the pair
                half = int((sep / 2 + 0.5) / CELL)
                seg = prof[int(round(y0)) - half: int(round(y0)) + half + 1]
                fr = float(D[k][:, ap].sum() * DV / flux)
                pk = _two_peaks(np.arange(-half, half + 1) * CELL, seg, sep)
                if fr < 0.3:                      # nothing recovered: no peaks to speak of
                    pk = dict(resolved=False, dip=None, sep_found=None)
                out[k] = dict(flux_ratio=fr, **pk)
            pairs[pname] = dict(sep=sep, flux=flux, **out)
        res["inj"][name] = dict(sources=rows, pairs=pairs)
        rep.append(f"### `{name}` (peak S/N {R.INJ[name]:.0f} per clean beam): recovered / injected flux\n| source | FWHM″ | injected (Jy km/s) | " +
                   " | ".join(keys) + " |\n|---|---|---|" + "---|" * len(keys))
        for n, fw, fl, *vals in rows:
            rep.append(f"| {n} | {fw} | {fl:.3f} | " + " | ".join(f"{'**' if i == 0 else ''}{v:.2f}{'**' if i == 0 else ''}" for i, v in enumerate(vals)) + " |")
        rep.append(f"\nPoint pairs, separated north–south (the beam's minor axis is {BEAM[1]:.2f}″), models smoothed to 0.1″ (∗ beam as is): "
                   "recovered flux / injected; resolved = a dip below 0.8 of the fainter peak (only if ≥ 30 % of the flux came back); recovered separation\n\n"
                   "| pair | separation | " + " | ".join(f"{k}: flux, resolved, sep." for k in keys) + " |\n|---|---|" + "---|" * len(keys))
        for pn, pr in pairs.items():
            cells = []
            for k in keys:
                a = pr[k]
                cells.append(f"{a['flux_ratio']:.2f}, {'**yes**' if a['resolved'] else 'no'}, {_fmt_scale(a['sep_found'])}")
            rep.append(f"| {pn} | {pr['sep']}″ | " + " | ".join(cells) + " |")
        rep.append("")
        _fig_pairs(plt, name, D, truth)


def _two_peaks(ys, seg, sep):
    """the two highest local maxima of a profile and the dip between them"""
    from scipy.signal import find_peaks
    pk, _ = find_peaks(seg)
    if len(pk) < 2:
        return dict(resolved=False, dip=1.0, sep_found=None)
    top = sorted(pk[np.argsort(seg[pk])[-2:]])
    lo = seg[top[0]: top[1] + 1].min()
    dip = float(lo / min(seg[top[0]], seg[top[1]])) if min(seg[top[0]], seg[top[1]]) > 0 else 1.0
    return dict(resolved=bool(dip < 0.8), dip=dip, sep_found=float(ys[top[1]] - ys[top[0]]))


def _fig_pairs(plt, name, D, truth):
    fig, ax = plt.subplots(len(R.PAIRS), 4, figsize=(13, 3.2 * len(R.PAIRS)))
    sky = load(os.path.join(OUT, "ds", name, "injected_sky.npy"))
    for i, (pname, cx, cy, sep, v0) in enumerate(R.PAIRS):
        x0, y0 = [int(round(c)) for c in R.pix(cx, cy)]
        h = int(1.6 / CELL)
        chans = np.abs(R.VEL - v0) <= 2.5 * R.LINE_SIG
        for j, (title, cube, sm) in enumerate([("injected (0.1″)", sky, 0.1), ("2D1D (0.1″)", D["2D1D"], 0.1),
                                               ("CLEAN+ model (0.1″)", D["CLEAN+ model"], 0.1), ("CLEAN+ ∗ beam", D["CLEAN+ ∗ beam"], 0)]):
            img = np.pad(gauss_smooth(cube[chans], sm).sum(0), h)[y0: y0 + 2 * h, x0: x0 + 2 * h]   # padded: pairs near the edge
            ax[i, j].imshow(img, origin="lower", cmap="inferno", extent=[-1.6, 1.6, -1.6, 1.6])
            ax[i, j].set_title(f"{pname}: {title}", fontsize=9); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"21c_pairs_{name}.png"), dpi=90); plt.close(fig)


def sec_null(rep, res):
    M, real = models("null"), models("real")
    if "2D1D" not in M or "CLEAN+ model" not in M:
        rep.append("## 4. Null\nnot finished.\n"); return
    rows = {k: dict(flux=float(M[k].sum() * DV), of_real=float(M[k].sum() / real[k].sum()) if k in real else None, peak_of_real=float(M[k].max() / real[k].max()) if k in real else None)
            for k in ("2D1D", "2D1D − floor", "CLEAN± model", "CLEAN+ model")}
    res["null"] = rows
    rep.append("## 4. Null test: noise only (the measured noise power spectrum), no sky\n| method | fake flux (Jy km/s) | of its real-data total | peak / real peak |\n|---|---|---|---|")
    for k, r in rows.items():
        rep.append(f"| {k} | {r['flux']:.2f} | {100 * r['of_real']:.2f} % | {r['peak_of_real']:.3f} |")
    rep.append("\nCriterion: a few % at most; anything here is created from noise.\n")


def sec_ext(rep, res, plt):
    from deconvgui.worker import _read_fits_cube
    M = models("real")
    if "CLEAN+ model" not in M:
        rep.append("## 5. Extended configuration\nnot finished.\n"); return
    ed = R.image_contsub(_read_fits_cube(os.path.join(R.EXT, "dirty_cube.fits")).astype(np.float64))
    ep = _read_fits_cube(os.path.join(R.EXT, "dirty_beam_cube.fits")).astype(np.float64)
    from restore import convolve_with_beam
    cand = {k: M[k] for k in ("2D1D", "2D1D − floor", "CLEAN± model", "CLEAN+ model", "CLEAN+ ∗ beam")}
    rows, curves = {}, {}
    for k, m in cand.items():
        pm = psf_conv(m, ep)
        d, x = ed[LINE].ravel(), pm[LINE].ravel()
        a = float(d @ x / (x @ x))
        rows[k] = dict(scale=a, unexplained=float(((d - a * x) ** 2).sum() / (d ** 2).sum()), unexplained_unscaled=float(((d - x) ** 2).sum() / (d ** 2).sum()))
        kk, frc, tr = ring_stats(pm, ed)
        curves[k] = (kk, frc)
        rows[k]["frc_k"], rows[k]["frc"] = kk.tolist(), frc.tolist()
        rows[k]["frc_at"] = {f"{x:.0f}": float(np.interp(np.log(x), np.log(kk), frc)) for x in (50, 100, 175, 250, UV_MAX, 400)}
    res["ext"] = rows
    rep.append("## 5. The independent extended configuration (0.22″ × 0.17″): what each compact-data model predicts\n"
               "Each model ∗ the extended dirty beam vs the extended dirty cube (continuum removed), line channels. The compact data "
               "never saw these baselines: agreement on scales finer than the compact beam is detail the method got right.\n\n"
               "| model | best flux scale a | unexplained (scaled) | unexplained (as is) | correlation at 50 | 100 | 175 | 250 | " + f"{UV_MAX:.0f}" + " | 400 kλ |\n|---|---|---|---|---|---|---|---|---|---|")
    for k, r in rows.items():
        rep.append(f"| {k} | {r['scale']:.3f} | **{r['unexplained']:.3f}** | {r['unexplained_unscaled']:.3f} | " + " | ".join(f"{v:.2f}" for v in r["frc_at"].values()) + " |")
    rep.append("\nPer spatial frequency (correlation with the extended data, line channels): see 21d. Criterion: 2D1D correlates "
               "with the extended data beyond the compact array's resolution, where CLEAN ∗ beam cannot.\n")
    fig, a = plt.subplots(figsize=(8, 5))
    for k, (kk, frc) in curves.items():
        a.plot(kk, frc, label=k)
    a.axvline(KL / np.sqrt(BEAM[0] * BEAM[1]), color="C7", ls="--", lw=0.8); a.text(KL / np.sqrt(BEAM[0] * BEAM[1]), 0.05, " 1 / compact beam", fontsize=8, rotation=90)
    _uv_lines(a, 1.0)
    a.set_xscale("log"); a.set_xlabel("spatial frequency (kλ)"); a.set_ylabel("correlation with the extended data"); a.axhline(0, color="k", lw=0.5); a.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "21d_extended_frc.png"), dpi=110); plt.close(fig)


def sec_jack(rep, res, plt):
    A, B, J = models("jack_eb1"), models("jack_eb2"), models("real")
    if "2D1D" not in A or "2D1D" not in B or "CLEAN+ model" not in A:
        rep.append("## 6. Jackknife\nnot finished.\n"); return
    rows = {}
    fig, a = plt.subplots(figsize=(8, 5))
    for k in ("2D1D", "2D1D − floor", "CLEAN± model", "CLEAN+ model", "CLEAN+ ∗ beam"):
        if k not in A or k not in B:
            continue
        kk, frc, _ = ring_stats(A[k], B[k])
        rows[k] = dict(flux_eb1=float(A[k].sum() * DV), flux_eb2=float(B[k].sum() * DV), flux_joint=float(J[k].sum() * DV) if k in J else None,
                       frc_res=_frc_res(kk, frc, 0.5), frc_k=kk.tolist(), frc=frc.tolist(),
                       frc_at={f"{x:.0f}": float(np.interp(np.log(x), np.log(kk), frc)) for x in (50, 100, 175, 250, UV_MAX, 400)})
        a.plot(kk, frc, label=k)
    res["jack"] = rows
    a.axhline(0.5, color="gray", lw=0.6); _uv_lines(a, 1.0); a.set_xscale("log"); a.set_xlabel("spatial frequency (kλ)"); a.set_ylabel("FRC, EB1 vs EB2"); a.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "21e_jackknife_frc.png"), dpi=110); plt.close(fig)
    rep.append("## 6. Jackknife: each execution block deconvolved alone\n| method | EB1 | EB2 | both (Jy km/s) | EB1–EB2 correlation at 50 | 100 | 175 | 250 | " +
               f"{UV_MAX:.0f}" + " | 400 kλ |\n|---|---|---|---|---|---|---|---|---|---|")
    for k, r in rows.items():
        rep.append(f"| {k} | {r['flux_eb1']:.1f} | {r['flux_eb2']:.1f} | {r['flux_joint'] or float('nan'):.1f} | " + " | ".join(f"{v:.2f}" for v in r["frc_at"].values()) + " |")
    rep.append("\nCriterion: the two independent halves agree on flux and repeat the fine structure: what repeats is in the data, not noise.\n")


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    rep, res = [f"# Experiment 21: does 2D1D refine the data while preserving flux and spectra?\n\n"
                f"Continuum-subtracted NGC 3110 compact cube `{R.DS}`: {NX} px × {CELL}″, {NZ} × {DV} km/s, clean beam "
                f"{BEAM[0]:.2f}″ × {BEAM[1]:.2f}″; baselines {UV_MIN:.1f}–{UV_MAX:.0f} kλ (λ/B_max = {KL / UV_MAX:.2f}″). "
                f"2D1D: the GUI's r002 recipe. CLEAN: exact-PSF multiscale on the same inputs, no mask, converged to 2σ (± and positive-only).\n"], {}
    for f in (lambda: sec_real(rep, res, plt), lambda: sec_sims(rep, res, plt), lambda: sec_inj(rep, res, plt),
              lambda: sec_null(rep, res), lambda: sec_ext(rep, res, plt), lambda: sec_jack(rep, res, plt)):
        f()
        open(os.path.join(OUT, "report.md"), "w").write("\n".join(rep) + "\n")
        R.jdump(res, os.path.join(OUT, "results.json"))
    p("\n".join(rep))
