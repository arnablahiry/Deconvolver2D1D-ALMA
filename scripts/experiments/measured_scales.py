#!/usr/bin/env python
"""
Experiment 20: 2D1D vs MS-CLEAN on MEASURED scales only. The total flux (zero spacing) and everything below the
shortest baseline (9.5 kλ) is not measured; there the two methods extrapolate differently (experiments 18, 19).
Here every quantity is one the visibilities constrain:

  1. what the interferometer sees: each model convolved with the dirty beam, N(x), against the dirty cube and the
     other model (image domain, exact): spectra over the field and in apertures, per-channel data/model scale
  2. channel by channel against the raw visibilities: per cube channel and per uv band (9.5-15, 15-30, 30-60, 60-120,
     120-350 kλ), the weighted vector-averaged visibility of the data and of each model at the same uv points
  3. band-pass maps: line moment 0 of each model with everything outside 9.5-220 kλ removed (below: not measured;
     above: noise-dominated), compared pixel by pixel

    python scripts/experiments/measured_scales.py extract orig|contsub      per-channel visibilities of that set's MSs
    python scripts/experiments/measured_scales.py analyze orig|contsub      -> data/experiments/20_measured_scales/,
                                                                             figures/experiments/20_measured_scales/
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "gui")); sys.path.insert(0, os.path.join(REPO, "simple"))
GUI_DS = sorted(glob.glob("/scratch/alahiry/deconvgui/datasets/*NGC3110-natural*"))[0]
OUT = os.path.join(REPO, "data/experiments/20_measured_scales")
FIG = os.path.join(REPO, "figures/experiments/20_measured_scales")
E18, E19 = os.path.join(REPO, "data/experiments/18_flux_audit"), os.path.join(REPO, "data/experiments/19_contsub")
SETS = {   # dataset folder (dirty, psf, meta) and the models compared
    "orig": dict(ds=GUI_DS, models={
        "2D1D (r002)": f"{GUI_DS}/results/r002-gpu_pd/model.npy",
        "MS-CLEAN (c001)": f"{GUI_DS}/msclean/c001/model.npy",
        "MS-CLEAN (beam-matched)": f"{E18}/msclean_beam/model.npy"}),
    "contsub": dict(ds="/scratch/alahiry/contsub/ds_contsub", models={
        "2D1D recipe": f"{E19}/res/recipe/model.npy",
        "2D1D coarse kept": f"{E19}/res/keepcoarse/model.npy",
        "2D1D coarse free": f"{E19}/res/nocoarse/model.npy",
        "2D1D field-int κ=1": f"{E19}/res/fi1/model.npy",
        "2D1D field-int κ=5": f"{E19}/res/fi5/model.npy",
        "2D1D field-int κ=25": f"{E19}/res/fi25/model.npy",
        "2D1D mask 3σ": f"{E19}/res/mask3/model.npy",
        "2D1D mask 2σ": f"{E19}/res/mask2/model.npy",
        "2D1D mask 3σ + κ=1": f"{E19}/res/mask3_fi1/model.npy",
        "2D1D sidelobe mask": f"{E19}/res/mask_sl/model.npy",
        "2D1D sidelobe mask + κ=1": f"{E19}/res/mask_sl_fi1/model.npy",
        "2D1D wavelet-denoised residual": f"{E19}/res/mrs_resid/model.npy",
        "2D1D fixed wavelet support": f"{E19}/res/mrs_fixed/model.npy",
        "MS-CLEAN auto-mask": f"{E19}/clean_auto/model.npy",
        "MS-CLEAN no mask": f"{E19}/clean_nomask/model.npy"}),
}
BANDS = [(9.5, 15), (15, 30), (30, 60), (60, 120), (120, 350)]
PASS = (9.5, 220.0)                                   # kλ band kept in the band-pass maps
COLS = ["#d08c00", "#e35d00", "#a03000", "#2ca25f", "#1b7d45", "#0b4d2a", "#c51b7d", "#e7298a", "#7a0177",
        "#8c510a", "#bf812d", "#01665e", "#35978f", "#2a78d6", "#6a3d9a"]
p = lambda *a: print(*a, flush=True)


def setup(name):
    ds = SETS[name]["ds"]
    meta = json.load(open(os.path.join(ds, "meta.json")))
    nz, n = int(meta["nchan"]), int(meta["imsize"])
    vel = float(meta["start_kms"]) + float(meta["width_kms"]) * np.arange(nz)
    return ds, meta, nz, n, vel, float(meta["cell_arcsec"]), abs(float(meta["width_kms"]))


# ============================================================== extract
def extract(a):
    """per cube channel (its LSRK radio-velocity bin): flag-weighted mean Stokes-I visibility of each baseline"""
    import casatools
    from casa_logs import redirect_casa_logs
    redirect_casa_logs("measured_scales_extract.log")
    ds, meta, nz, n, vel, cell, dv = setup(a.set)
    im = meta["imaging"]; rest = float(im["restfreq_ghz"]) * 1e9
    U, V, W, VIS = [], [], [], []
    for ms, spw in zip(im["ms"], im["spw_list"]):
        t0 = time.time()
        md = casatools.msmetadata(); md.open(ms)
        fld = int(md.fieldsforname(im["field"])[0])
        dd = int([d for d in md.datadescids() if md.spwfordatadesc(d) == int(spw)][0]); md.close()
        t = casatools.ms(); t.open(ms)
        f = np.array(t.cvelfreqs(spwids=[int(spw)], fieldids=[fld], mode="channel", outframe="LSRK")); t.close()
        vch = 299792.458 * (1 - f / rest)
        tb = casatools.table(); tb.open(ms)
        q = tb.query(f"FIELD_ID == {fld} && DATA_DESC_ID == {dd} && ANTENNA1 != ANTENNA2")
        uvw, w = q.getcol("UVW"), q.getcol("WEIGHT").sum(0)
        out = np.zeros((nz, q.nrows()), np.complex64)
        wch = np.zeros((nz, q.nrows()), np.float32)   # per-channel weight: WEIGHT x unflagged fraction (flagged rows can
                                                      # carry large weights: EB2's fully flagged rows hold 64 % of its weight)
        for c in range(nz):
            sel = np.where((vch >= vel[c] - dv / 2) & (vch < vel[c] + dv / 2))[0]
            d = q.getcolslice("DATA", [0, int(sel.min())], [1, int(sel.max())], [1, 1])
            fl = q.getcolslice("FLAG", [0, int(sel.min())], [1, int(sel.max())], [1, 1])
            keep = np.isin(np.arange(sel.min(), sel.max() + 1), sel)
            ok = (~fl[:, keep]).astype(np.float64)
            out[c] = ((d[:, keep] * ok).sum((0, 1)) / np.maximum(ok.sum((0, 1)), 1)).astype(np.complex64)
            wch[c] = w * ok.mean((0, 1))
        q.close(); tb.close()
        U.append(uvw[0]); V.append(uvw[1]); W.append(wch); VIS.append(out)
        p(f"  {os.path.basename(ms)}: {out.shape[1]} baselines x {nz} channels, {time.time() - t0:.0f} s")
    fch = rest * (1 - vel / 299792.458)                                    # each cube channel's frequency -> λ
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"vis_{a.set}.npz"), u_m=np.concatenate(U), v_m=np.concatenate(V), w=np.concatenate(W, axis=1),
             vis=np.concatenate(VIS, axis=1), lam=299792458.0 / fch)


# ============================================================== helpers
def vis_fft(img, u, v, cell, pad=16):
    """model visibilities at (u, v) [wavelengths]: zero-padded FFT (pad x the image) + bilinear interpolation,
    same convention as flux_audit.model_vis(sign=+1): V = sum I exp(-2πi u δ (x - x0)) exp(+2πi v δ (y - y0))"""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ny, nx = img.shape; N = pad * nx; d = np.deg2rad(cell / 3600)
    P = torch.zeros((N, N), dtype=torch.complex64, device=dev)
    P[:ny, :nx] = torch.tensor(img, dtype=torch.float32, device=dev)
    P = torch.roll(P, shifts=(-(ny // 2), -(nx // 2)), dims=(0, 1))       # reference pixel -> index 0
    F = torch.fft.fft2(P)
    fx = torch.tensor(u * d * N, device=dev, dtype=torch.float64) % N
    fy = torch.tensor(-v * d * N, device=dev, dtype=torch.float64) % N
    x0, y0 = torch.floor(fx).long(), torch.floor(fy).long()
    tx, ty = (fx - x0).to(torch.float32), (fy - y0).to(torch.float32)
    x1, y1 = (x0 + 1) % N, (y0 + 1) % N
    val = (F[y0, x0] * (1 - tx) * (1 - ty) + F[y0, x1] * tx * (1 - ty) + F[y1, x0] * (1 - tx) * ty + F[y1, x1] * tx * ty)
    return val.cpu().numpy()


def bandpass(img, cell, lo, hi, ramp=0.15):
    """keep spatial frequencies lo..hi kλ (cosine ramps of ramp x the edge)"""
    ny, nx = img.shape; d = np.deg2rad(cell / 3600)
    ku, kv = np.fft.fftfreq(nx, d) / 1e3, np.fft.fftfreq(ny, d) / 1e3
    q = np.hypot(*np.meshgrid(ku, kv))
    def edge(x, e, up):
        t = np.clip((x - e * (1 - ramp)) / (2 * e * ramp), 0, 1)
        r = 0.5 - 0.5 * np.cos(np.pi * t)
        return r if up else 1 - r
    T = edge(q, lo, True) * edge(q, hi, False)
    return np.real(np.fft.ifft2(np.fft.fft2(img) * T))


# ============================================================== analyze
def analyze(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from uv_to_image import ShiftInvariantOperator
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ds, meta, nz, n, vel, cell, dv = setup(a.set)
    os.makedirs(FIG, exist_ok=True)
    dirty = np.load(os.path.join(ds, "dirty.npy")).astype(np.float64)
    psf = np.load(os.path.join(ds, "psf.npy")).astype(np.float64)
    b = meta["beam"]; omega = np.pi * b[0] * b[1] / (4 * np.log(2)) / cell ** 2
    models = {k: np.load(f).astype(np.float64) for k, f in SETS[a.set]["models"].items() if os.path.exists(f)}
    keys = list(models)
    rep = [f"# Measured scales only: {a.set} ({meta.get('name')})\n", f"Models: {', '.join(keys)}. Clean beam {b[0]:.2f}″ × {b[1]:.2f}″ ({omega:.0f} px).\n"]
    R = {}

    # ---- 1. what the interferometer sees: N(x) vs the dirty cube
    op = ShiftInvariantOperator(psf, dirty)
    NX = {k: op.apply(x) for k, x in models.items()}
    yy, xx = np.mgrid[0:n, 0:n]; rad = np.hypot(xx - n // 2, yy - n // 2) * cell
    aps = {"r ≤ 2″": rad <= 2, "r ≤ 4″": rad <= 4, "whole 16″ field": np.ones_like(rad, bool)}
    free = json.load(open(os.path.join(E18 if a.set == "orig" else E19, "line_free.json")))["channels"]
    rep.append("## 1. What the interferometer sees: model ∗ dirty beam vs the dirty cube\n"
               "Each column: Σ over the aperture of the map / clean-beam area, summed over channels × Δv (Jy km/s, dirty-beam units: "
               "NOT a flux, the measured quantity). Equal to the dirty cube's = the model reproduces the data there.\n"
               "| | " + " | ".join(aps) + " | per-channel data/model (median, range) | residual rms / noise |\n|---|" + "---|" * (len(aps) + 2))
    noise = float(np.mean([(dirty[c] - dirty[free].mean(0)).std() for c in free]))   # line-free, continuum removed
    row = lambda cube: [float(cube[:, m].sum() / omega * dv) for m in aps.values()]
    rep.append("| dirty cube (data) | " + " | ".join(f"{v_:.1f}" for v_ in row(dirty)) + " | | |")
    R["image"] = {"dirty": row(dirty)}
    for k in keys:
        sc = [float((dirty[c] * NX[k][c]).sum() / max((NX[k][c] ** 2).sum(), 1e-30)) for c in range(nz)]
        line = [c for c in range(nz) if (NX[k][c] ** 2).sum() > 0.05 * max((NX[k][i] ** 2).sum() for i in range(nz))]
        scl = [sc[c] for c in line]
        rr = float(np.mean([(dirty[c] - NX[k][c]).std() for c in line]) / noise)
        R["image"][k] = dict(apertures=row(NX[k]), scale_median=float(np.median(scl)), scale_min=float(min(scl)), scale_max=float(max(scl)), resid=rr)
        rep.append(f"| {k} ∗ dirty beam | " + " | ".join(f"{v_:.1f}" for v_ in row(NX[k])) +
                   f" | {np.median(scl):.3f} ({min(scl):.3f}–{max(scl):.3f}) | {rr:.2f} |")
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            d12 = np.sqrt(((NX[k1] - NX[k2]) ** 2).sum() / (dirty ** 2).sum())
            rep.append(f"\n{k1} vs {k2}: ‖N x₁ − N x₂‖ / ‖dirty‖ = {d12:.4f} (the two models' measured parts differ by this fraction of the data)")
    rep.append("")

    # ---- 2. per channel, per uv band, against the raw visibilities
    vf = os.path.join(OUT, f"vis_{a.set}.npz")
    if os.path.exists(vf):
        Z = np.load(vf); um, vm_, w, vis, lam = Z["u_m"], Z["v_m"], Z["w"], Z["vis"], Z["lam"]
        # check the FFT interpolation against the exact DFT once
        import flux_audit as A
        x0 = models[keys[0]][nz // 2]; uu, vv = um[::200] / lam[nz // 2], vm_[::200] / lam[nz // 2]   # (w: per channel)
        ex = A.model_vis(x0, uu, vv, +1); ap_ = vis_fft(x0, uu, vv, cell)
        err = float(np.sqrt(np.sum(np.abs(ex - ap_) ** 2) / np.sum(np.abs(ex) ** 2)))
        rep.append(f"## 2. Channel by channel against the raw visibilities\n(FFT interpolation vs exact DFT: rms error {err:.2e})\n")
        spec = {k: np.zeros((len(BANDS), nz)) for k in ["data"] + keys}
        for c in range(nz):
            u, v = um / lam[c], vm_ / lam[c]; qk = np.hypot(u, v) / 1e3
            mv = {k: vis_fft(models[k][c], u, v, cell) for k in keys}
            wc = w[c]
            for j, (lo, hi) in enumerate(BANDS):
                s = (qk >= lo) & (qk < hi) & (wc > 0)
                spec["data"][j, c] = abs(np.sum(wc[s] * vis[c, s])) / np.sum(wc[s])
                for k in keys:
                    spec[k][j, c] = abs(np.sum(wc[s] * mv[k][s])) / np.sum(wc[s])
        R["uv_spectra"] = {k: v_.tolist() for k, v_ in spec.items()}
        rep.append("Line-channel sum of the weighted vector-averaged visibility per band (Jy km/s), data and models at the SAME uv points:\n"
                   "| band (kλ) | data | " + " | ".join(keys) + " |\n|---|---|" + "---|" * len(keys))
        lc = [c for c in range(nz) if spec["data"][0, c] > 0.15 * spec["data"][0].max()]
        for j, (lo, hi) in enumerate(BANDS):
            dsum = spec["data"][j, lc].sum() * dv
            rep.append(f"| {lo:g}–{hi:g} | {dsum:.1f} | " + " | ".join(f"{spec[k][j, lc].sum() * dv:.1f} ({spec[k][j, lc].sum() / max(spec['data'][j, lc].sum(), 1e-30):.3f})" for k in keys) + " |")
        rep.append("(in brackets: model / data). Equal numbers = the model has the measured flux on those scales, channel by channel.\n")
        fig, ax = plt.subplots(1, len(BANDS), figsize=(4.4 * len(BANDS), 4.2), sharex=True)
        for j, (lo, hi) in enumerate(BANDS):
            ax[j].plot(vel, spec["data"][j], "ko", ms=3.5, label="data")
            for i, k in enumerate(keys):
                ax[j].step(vel, spec[k][j], where="mid", color=COLS[i % len(COLS)], label=k)
            ax[j].set_title(f"{lo:g}–{hi:g} kλ"); ax[j].set_xlabel("v (km/s)")
        ax[0].set_ylabel("|vector-averaged visibility| (Jy)"); ax[0].legend(fontsize=7)
        fig.suptitle(f"measured scales only ({a.set}): spectra per uv band, data vs models at the same uv points")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"20a_uv_spectra_{a.set}.png"), dpi=120); plt.close(fig)

    # ---- 3. band-pass maps (9.5-220 kλ), line moment 0
    lc = [c for c in range(nz) if dirty[c].std() > 1.5 * noise] or list(range(nz))
    bp = {k: bandpass(x[lc].sum(0) * dv, cell, *PASS) for k, x in models.items()}
    rep.append(f"## 3. Band-pass ({PASS[0]}–{PASS[1]} kλ) line moment-0 maps\n| pair | correlation | ‖difference‖ / ‖first‖ |\n|---|---|---|")
    R["bandpass"] = {}
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            cc = float(np.corrcoef(bp[k1].ravel(), bp[k2].ravel())[0, 1]); dd = float(np.linalg.norm(bp[k1] - bp[k2]) / np.linalg.norm(bp[k1]))
            R["bandpass"][f"{k1} | {k2}"] = dict(corr=cc, rel_diff=dd)
            rep.append(f"| {k1} vs {k2} | {cc:.4f} | {dd:.3f} |")
    fig, ax = plt.subplots(1, len(keys) + 1, figsize=(4.5 * (len(keys) + 1), 4.3))
    vmax = max(np.percentile(np.abs(x), 99.8) for x in bp.values()); ext = [n * cell / 2, -n * cell / 2, -n * cell / 2, n * cell / 2]
    for i, k in enumerate(keys):
        ax[i].imshow(bp[k], origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=ext); ax[i].set_title(k, fontsize=9)
    ax[-1].imshow(bp[keys[0]] - bp[keys[-1]], origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=ext)
    ax[-1].set_title(f"{keys[0]} − {keys[-1]}", fontsize=9)
    fig.suptitle(f"band-pass {PASS[0]}–{PASS[1]} kλ line moment 0 (Jy km/s per pixel): only measured scales")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"20b_bandpass_maps_{a.set}.png"), dpi=110); plt.close(fig)

    # spectra figure: model * dirty beam vs dirty (field and r <= 4")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for j, (nm, m) in enumerate([("r ≤ 4″", aps["r ≤ 4″"]), ("whole 16″ field", aps["whole 16″ field"])]):
        ax[j].plot(vel, dirty[:, m].sum(1) / omega, "ko", ms=3.5, label="dirty cube (data)")
        for i, k in enumerate(keys):
            ax[j].step(vel, NX[k][:, m].sum(1) / omega, where="mid", color=COLS[i % len(COLS)], label=f"{k} ∗ dirty beam")
        ax[j].set_title(f"what the interferometer sees, {nm}"); ax[j].set_xlabel("v (km/s)")
    ax[0].set_ylabel("Σ / clean-beam area (dirty-beam units)"); ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"20c_seen_spectra_{a.set}.png"), dpi=120); plt.close(fig)
    json.dump(R, open(os.path.join(OUT, f"results_{a.set}.json"), "w"), indent=1)
    open(os.path.join(OUT, f"report_{a.set}.md"), "w").write("\n".join(rep) + "\n")
    p("\n".join(rep))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("extract", "analyze")); ap.add_argument("set", choices=list(SETS))
    a = ap.parse_args()
    {"extract": extract, "analyze": analyze}[a.cmd](a)
