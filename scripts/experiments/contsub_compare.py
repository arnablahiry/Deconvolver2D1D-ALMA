#!/usr/bin/env python
"""
Experiment 19: MS-CLEAN and 2D1D on the SAME line-only data, both converged, compared like for like.

The audit (flux_audit.py, experiment 18) found the two models agree on every measured scale but differ in total
flux by ~30 %, from three things: continuum (never subtracted: the .ms.split.cal DATA column has it), MS-CLEAN
stopping early (CASA stop code 4, residual ~9 x noise), and 2D1D's flux on scales below the shortest baseline.
Here every one of those is controlled:

  contsub     CASA uvcontsub (order 1) of each MS, fitted on the channels outside 4700-5330 km/s (radio LSRK) ->
              /scratch/alahiry/contsub/<ms>.contsub.ms (target field, CO spw only)
  image       the dirty cube + PSF of the continuum-subtracted visibilities, with the GUI dataset's own CASA call and
              grid -> a GUI-style dataset folder (DS2) + data/experiments/19_contsub/ds/real (for flux_audit.py)
  then, with flux_audit.py (DS=DS2, OUT=data/experiments/19_contsub):
     run real --tag recipe                        the GUI's r002 recipe
     run real --tag keepcoarse --extra '{"reweight_keep_coarse": true}'      coarse plane never reweighted
     run real --tag nocoarse   --extra '{"coarse_lambda_scale": 0}'          coarse plane not penalised at all
     msclean --tag clean_auto   relaxed auto-mask, nsigma 2, scales 0, 1, 3, 6 beams
     msclean --tag clean_nomask no mask, nsigma 3, same scales
     uv                                           the continuum-subtracted visibilities, for the uv test
  analyze     spectra, curves of growth, uv test, residuals, line-free flux and a data-only total flux estimate
              -> data/experiments/19_contsub/report.md, figures/experiments/19_contsub/
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "gui")); sys.path.insert(0, os.path.join(REPO, "simple"))
SRC = sorted(glob.glob("/scratch/alahiry/deconvgui/datasets/*NGC3110-natural*"))[0]   # the GUI dataset (continuum in)
RES = "r002-gpu_pd"
SCR = "/scratch/alahiry/contsub"
DS2 = os.path.join(SCR, "ds_contsub")                                                   # GUI-style folder, line only
OUT = os.path.join(REPO, "data/experiments/19_contsub")
FIG = os.path.join(REPO, "figures/experiments/19_contsub")
OLD = os.path.join(REPO, "data/experiments/18_flux_audit")                             # the continuum-in runs
LINE_V = (4700.0, 5330.0)                                                               # excluded from the continuum fit
p = lambda *a: print(*a, flush=True)


def lsrk_velocities(ms, spw, field, rest_ghz):
    import casatools
    msmd = casatools.msmetadata(); msmd.open(ms); fld = int(msmd.fieldsforname(field)[0]); msmd.close()
    t = casatools.ms(); t.open(ms)
    f = np.array(t.cvelfreqs(spwids=[int(spw)], fieldids=[fld], mode="channel", outframe="LSRK")); t.close()
    return 299792.458 * (1 - f / (rest_ghz * 1e9))


def ranges(idx):
    idx = sorted(int(i) for i in idx)
    out, s = [], idx[0]
    for a, b in zip(idx, idx[1:] + [None]):
        if b != a + 1:
            out.append(f"{s}~{a}"); s = b
    return ";".join(out)


def contsub(a):
    from casatasks import uvcontsub
    sys.path.insert(0, os.path.join(REPO, "simple"))
    from casa_logs import redirect_casa_logs
    redirect_casa_logs("contsub_compare.log")
    meta = json.load(open(os.path.join(SRC, "meta.json"))); im = meta["imaging"]
    os.makedirs(SCR, exist_ok=True)
    for ms, spw in zip(im["ms"], im["spw_list"]):
        v = lsrk_velocities(ms, spw, im["field"], float(im["restfreq_ghz"]))
        free = np.where((v < LINE_V[0]) | (v > LINE_V[1]))[0]
        out = os.path.join(SCR, os.path.basename(ms).replace(".ms.split.cal", ".contsub.ms"))
        if os.path.exists(out):
            shutil.rmtree(out)
        t0 = time.time()
        uvcontsub(vis=ms, outputvis=out, field=im["field"], spw=str(spw), fitspec=f"{spw}:{ranges(free)}", fitorder=1, datacolumn="data")
        p(f"{os.path.basename(ms)}: continuum fitted on {len(free)} of {len(v)} channels "
          f"({v.min():.0f}-{v.max():.0f} km/s, line {LINE_V} excluded), order 1 -> {out} in {time.time() - t0:.0f} s")


def image(a):
    import casatools
    import grid_with_casa as g
    from deconvgui import worker
    worker._setup_paths(REPO)
    meta = json.load(open(os.path.join(SRC, "meta.json"))); im = dict(meta["imaging"])
    ms = [os.path.join(SCR, os.path.basename(m).replace(".ms.split.cal", ".contsub.ms")) for m in im["ms"]]
    spws = []
    for m in ms:   # the output keeps only the CO spw, renumbered
        md = casatools.msmetadata(); md.open(m); spws.append(str(int(md.spwsforfield(im["field"])[0]))); md.close()
    im.update(ms=ms, spw_list=spws, spw=spws[0])
    os.makedirs(os.path.join(DS2, "results", RES), exist_ok=True)
    t0 = time.time()
    out = g.tclean_dirty_cube(ms_list=ms, spw_list=spws, field=im["field"], phasecenter=im["phasecenter"], out_dir=os.path.join(DS2, "casa"),
                              imsize=int(im["imsize"]), cell_arcsec=float(im["cell_arcsec"]), start_kms=float(im["start_kms"]),
                              width_kms=float(im["width_kms"]), nchan=int(im["nchan"]), restfreq_ghz=float(im["restfreq_ghz"]),
                              weighting=im.get("weighting", "natural"), log_name="contsub_compare_image.log")
    np.save(os.path.join(DS2, "dirty.npy"), worker._read_fits_cube(out["dirty"]))
    np.save(os.path.join(DS2, "psf.npy"), worker._read_fits_cube(out["psf"]))
    m2 = dict(meta, name="NGC3110 compact, continuum subtracted", ms=ms, imaging=im, beam=[float(b) for b in out["beam"]])
    json.dump(m2, open(os.path.join(DS2, "meta.json"), "w"), indent=1)
    shutil.copy(os.path.join(SRC, "results", RES, "params.json"), os.path.join(DS2, "results", RES, "params.json"))
    # flux_audit's layout: OUT/ds/real -> this cube
    d = os.path.join(OUT, "ds", "real"); os.makedirs(d, exist_ok=True)
    for f in ("dirty.npy", "psf.npy"):
        if not os.path.exists(os.path.join(d, f)):
            os.symlink(os.path.join(DS2, f), os.path.join(d, f))
    json.dump(dict(m2, audit="real"), open(os.path.join(d, "meta.json"), "w"), indent=1)
    p(f"continuum-subtracted cube imaged in {time.time() - t0:.0f} s; beam {out['beam']}; spws {spws}")


def mask(a):
    """emission masks from the dirty cube: |dirty| > n sigma (sigma: line-free, continuum removed), grown by one
    beam FWHM in space and +-1 channel, islands smaller than one beam removed -> OUT/mask_<n>sig.npy"""
    from scipy import ndimage
    d = np.load(os.path.join(DS2, "dirty.npy")).astype(np.float64)
    meta = json.load(open(os.path.join(DS2, "meta.json"))); cell = float(meta["cell_arcsec"]); b = meta["beam"]
    free = json.load(open(os.path.join(OUT, "line_free.json")))["channels"]
    sig = float(np.mean([(d[c] - d[free].mean(0)).std() for c in free]))
    beam_px = np.pi * b[0] * b[1] / (4 * np.log(2)) / cell ** 2
    r = int(round(b[0] / cell)); yy, xx = np.mgrid[-r:r + 1, -r:r + 1]; disk = xx ** 2 + yy ** 2 <= r * r
    for n in (3, 2):
        m = d > n * sig                                         # emission is positive
        for c in range(m.shape[0]):
            lab, k = ndimage.label(m[c])
            if k:
                sizes = ndimage.sum(m[c], lab, range(1, k + 1))
                m[c] = np.isin(lab, 1 + np.where(sizes >= beam_px)[0])
            m[c] = ndimage.binary_dilation(m[c], structure=disk)
        m = m | np.roll(m, 1, 0) | np.roll(m, -1, 0)            # +-1 channel
        m[0] |= m[1]; m[-1] |= m[-2]
        np.save(os.path.join(OUT, f"mask_{n}sig.npy"), m)
        p(f"mask {n} sigma: {100 * m.mean():.1f}% of the cube; per channel {[round(100 * float(x.mean()), 1) for x in m]}")


# ============================================================== analyze
def analyze(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.environ.update(DS=DS2, OUT=OUT, FIG=FIG)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import flux_audit as A
    os.makedirs(FIG, exist_ok=True)
    free = A.FREE_FROM_FILE(); line = [c for c in range(A.NZ) if c not in free]
    DV, VEL = A.DV, A.VEL
    L = lambda f: np.load(f).astype(np.float64) if os.path.exists(f) else None
    dirty = np.load(os.path.join(DS2, "dirty.npy")).astype(np.float64)
    noise = float(np.mean([(dirty[c] - dirty[free].mean(0)).std() for c in free]))
    M = {   # name: (model, residual, colour, style)
        "2D1D recipe": (L(f"{OUT}/res/recipe/model.npy"), L(f"{OUT}/res/recipe/residual.npy"), "#d08c00", "-"),
        "2D1D coarse kept": (L(f"{OUT}/res/keepcoarse/model.npy"), L(f"{OUT}/res/keepcoarse/residual.npy"), "#e35d00", "-"),
        "2D1D coarse free": (L(f"{OUT}/res/nocoarse/model.npy"), L(f"{OUT}/res/nocoarse/residual.npy"), "#a03000", ":"),
        "2D1D field-int κ=1": (L(f"{OUT}/res/fi1/model.npy"), L(f"{OUT}/res/fi1/residual.npy"), "#2ca25f", "-"),
        "2D1D field-int κ=5": (L(f"{OUT}/res/fi5/model.npy"), L(f"{OUT}/res/fi5/residual.npy"), "#1b7d45", "-"),
        "2D1D field-int κ=25": (L(f"{OUT}/res/fi25/model.npy"), L(f"{OUT}/res/fi25/residual.npy"), "#0b4d2a", "-"),
        "2D1D mask 3σ": (L(f"{OUT}/res/mask3/model.npy"), L(f"{OUT}/res/mask3/residual.npy"), "#c51b7d", "-"),
        "2D1D mask 2σ": (L(f"{OUT}/res/mask2/model.npy"), L(f"{OUT}/res/mask2/residual.npy"), "#e7298a", "-"),
        "2D1D mask 3σ + κ=1": (L(f"{OUT}/res/mask3_fi1/model.npy"), L(f"{OUT}/res/mask3_fi1/residual.npy"), "#7a0177", "-"),
        "2D1D sidelobe mask": (L(f"{OUT}/res/mask_sl/model.npy"), L(f"{OUT}/res/mask_sl/residual.npy"), "#8c510a", "-"),
        "2D1D sidelobe mask + κ=1": (L(f"{OUT}/res/mask_sl_fi1/model.npy"), L(f"{OUT}/res/mask_sl_fi1/residual.npy"), "#bf812d", "-"),
        "2D1D wavelet-denoised residual": (L(f"{OUT}/res/mrs_resid/model.npy"), L(f"{OUT}/res/mrs_resid/residual.npy"), "#01665e", "-"),
        "2D1D fixed wavelet support": (L(f"{OUT}/res/mrs_fixed/model.npy"), L(f"{OUT}/res/mrs_fixed/residual.npy"), "#35978f", "-"),
        "MS-CLEAN auto-mask": (L(f"{OUT}/clean_auto/model.npy"), L(f"{OUT}/clean_auto/residual.npy"), "#2a78d6", "-"),
        "MS-CLEAN no mask": (L(f"{OUT}/clean_nomask/model.npy"), L(f"{OUT}/clean_nomask/residual.npy"), "#6a3d9a", "-"),
        "2D1D, continuum in (r002)": (L(f"{SRC}/results/{RES}/model.npy"), None, "#d08c00", "--"),
        "MS-CLEAN, continuum in (c001)": (L(f"{SRC}/msclean/c001/model.npy"), None, "#2a78d6", "--"),
    }
    M = {k: v for k, v in M.items() if v[0] is not None}
    summ = lambda d: json.load(open(d)) if os.path.exists(d) else {}
    rep = ["# MS-CLEAN vs 2D1D on the same continuum-subtracted data (NGC 3110 compact)\n",
           f"Line-free channels {free}; noise {noise * 1e3:.3f} mJy/beam (continuum subtracted in the visibilities).\n"]
    for tag in ("clean_auto", "clean_nomask"):
        s = summ(f"{OUT}/{tag}/summary.json")
        if s:
            rep.append(f"- {tag}: stop code {s.get('stopcode')} (2 / 7 = threshold reached; 4 = stalled), {s.get('iterations')} iterations, "
                       f"{s.get('major_cycles')} major cycles, {s.get('seconds', 0) / 60:.0f} min, scales {s['params'].get('scales')}, nsigma {s['params'].get('nsigma')}")
    rep.append("\n## Flux and residuals (16″ field, Jy km/s)\n| model | total | line ch. | line-free ch. | residual rms / noise (line) | (line-free) |\n|---|---|---|---|---|---|")
    for k, (x, r, _, _) in M.items():
        rl = f"{np.mean([r[c].std() for c in line]) / noise:.2f}" if r is not None else "–"
        rf = f"{np.mean([r[c].std() for c in free]) / noise:.2f}" if r is not None else "–"
        rep.append(f"| {k} | {x.sum() * DV:.1f} | {x[line].sum() * DV:.1f} | {x[free].sum() * DV:.1f} | {rl} | {rf} |")

    # ---- uv test on the continuum-subtracted visibilities
    U = np.load(f"{OUT}/uv_line.npz"); u, v, vd, w = U["u"], U["v"], U["vis"], U["w"]; q = np.hypot(u, v) / 1e3
    bins = np.array([9.5, 12, 15, 20, 25, 30, 40, 50, 65, 80, 100, 130, 170, 220, 350])
    UV = {}
    for k, (x, _, _, _) in M.items():
        if "continuum in" in k:
            continue
        vm = A.model_vis(x[line].sum(0) * DV, u, v, +1)
        UV[k] = [float(np.real(np.sum(w[s] * vd[s] * np.conj(vm[s]))) / np.sum(w[s] * np.abs(vm[s]) ** 2))
                 for s in [(q >= lo) & (q < hi) for lo, hi in zip(bins[:-1], bins[1:])]]
    rep.append(f"\n## Data / model per uv annulus (continuum-subtracted visibilities; 1 = the model predicts the data, < 1 = too much flux)\n"
               "| annulus (kλ) | " + " | ".join(UV) + " |\n|---|" + "---|" * len(UV))
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        rep.append(f"| {lo:g}–{hi:g} | " + " | ".join(f"{UV[k][i]:.3f}" for k in UV) + " |")

    # ---- the total flux from the visibilities alone: vector-averaged amplitude per annulus, extrapolated to 0
    from scipy.optimize import curve_fit
    cen = (bins[:-1] + bins[1:]) / 2
    amp = np.array([abs(np.sum(w[s] * vd[s])) / np.sum(w[s]) for s in [(q >= lo) & (q < hi) for lo, hi in zip(bins[:-1], bins[1:])]])
    fit = {}
    sel = cen < 45
    for name, f, p0 in (("Gaussian", lambda x, S, s: S * np.exp(-0.5 * (x / s) ** 2), (500, 15)),
                        ("exponential disk", lambda x, S, s: S / (1 + (x / s) ** 2) ** 1.5, (500, 15))):
        try:
            pp, _ = curve_fit(f, cen[sel], amp[sel], p0=p0, maxfev=20000)
            fit[name] = float(pp[0])
        except Exception as e:
            fit[name] = None
    rep.append(f"\n## The total flux from the visibilities alone\nVector-averaged amplitude per annulus (Jy km/s): "
               + ", ".join(f"{c:.1f} kλ: {x_:.0f}" for c, x_ in zip(cen, amp) if c < 60)
               + f".\nExtrapolated to zero spacing (fit to < 45 kλ): " + ", ".join(f"{k} {v_:.0f}" if v_ else f"{k} failed" for k, v_ in fit.items())
               + " Jy km/s. This is a model-independent bracket; total-power data would settle it.\n")

    # ---- figures: spectra, growth, uv scale
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.4))
    cx, cy = A.NX // 2, A.NY // 2
    yy, xx = np.mgrid[0:A.NY, 0:A.NX]; rad = np.hypot(xx - cx, yy - cy) * A.CELL; redges = np.arange(0.5, 8.01, 0.5)
    for k, (x, _, c, ls) in M.items():
        ax[0].step(VEL, x.sum((1, 2)), where="mid", color=c, ls=ls, label=f"{k} ({x.sum() * DV:.0f})")
        m0 = x.sum(0) * DV
        ax[1].plot(redges, [m0[rad <= r_].sum() for r_ in redges], color=c, ls=ls, label=k)
    for c in free:
        ax[0].axvspan(VEL[c] - DV / 2, VEL[c] + DV / 2, color="0.93", zorder=0)
    ax[0].set_xlabel("v (km/s)"); ax[0].set_ylabel("flux density (Jy)"); ax[0].legend(fontsize=8); ax[0].set_title("spectra, 16″ field (Jy km/s)")
    ax[1].set_xlabel("radius (″)"); ax[1].set_ylabel("cumulative flux (Jy km/s)"); ax[1].legend(fontsize=8)
    for k, vals in UV.items():
        ax[2].plot(cen, vals, "o-", color=M[k][2], ls=M[k][3], label=k)
    ax[2].axhline(1, color="k", lw=0.8); ax[2].set_xscale("log"); ax[2].set_xlabel("uv distance (kλ)"); ax[2].set_ylabel("data / model")
    ax[2].legend(fontsize=8); ax[2].set_title("< 1: model brighter than the measured visibilities")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "19_contsub_compare.png"), dpi=125); plt.close(fig)
    json.dump(dict(uv=UV, bins=bins.tolist(), amp=amp.tolist(), V0_fit=fit,
                   flux={k: float(x.sum() * DV) for k, (x, _, _, _) in M.items()}), open(os.path.join(OUT, "results.json"), "w"), indent=1)
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(rep) + "\n")
    p("\n".join(rep))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("contsub", "image", "mask", "analyze"))
    a = ap.parse_args()
    {"contsub": contsub, "image": image, "mask": mask, "analyze": analyze}[a.cmd](a)
