#!/usr/bin/env python
"""
Experiment 18: is the 2D1D flux real? An audit of the GUI's pdtau result on NGC 3110 compact, which recovers
~40 % more flux than multiscale CLEAN (697 vs 487 Jy km/s over the 16" field), for fake flux and garbage.

Every 2D1D solve here is the GUI's own code (gui/deconvgui/worker.deconvolve_job) with the result's own
parameters (pdtau, k = 5, 2x2 rebin, reweighting from 6000 every 2000, 20 000 iterations), on a modified cube;
the line-free channels are fixed to those the GUI detected on the real cube, so lambda is calibrated alike.

  prepare          the input cubes:
                     real          the dataset's dirty cube (reproducibility: must give the GUI's result back)
                     null_synth    pure Gaussian noise with the measured 2D noise power spectrum: no sky at all
                     null_linefree the real line-free channels, continuum removed, tiled to 32: real noise, no sky
                     inj_bright    real cube + dirty-beam images of 8 known Gaussian sources (0.12-3" FWHM, on and
                     inj_faint     off the galaxy, Gaussian lines), peak S/N 30 / 6 per clean beam
                     jack_eb1/2    each execution block imaged alone (CASA, same grid): structure must repeat
  run NAME --device cuda:N
  msclean          multiscale CLEAN with scales matched to the beam (0, 1, 3 x FWHM px), the fair reference
  uv               line-integrated (and line-free) visibilities of the measurement sets, for the uv test
  analyze          -> data/experiments/18_flux_audit/report.md, results.json, figures/experiments/18_flux_audit/

    CUDA_VISIBLE_DEVICES=0 python scripts/experiments/flux_audit.py prepare
    python scripts/experiments/flux_audit.py run null_synth --device cuda:1     (one per GPU, in parallel)
    python scripts/experiments/flux_audit.py analyze

DS (a GUI dataset folder), RES (its 2D1D result) and CLEAN (its MS-CLEAN run) choose what is audited.
"""
import argparse
import glob
import json
import os
import sys
import threading
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "gui"))
sys.path.insert(0, os.path.join(REPO, "simple"))
DS = os.environ.get("DS") or sorted(glob.glob("/scratch/alahiry/deconvgui/datasets/*NGC3110-natural*"))[0]
RES = os.environ.get("RES", "r002-gpu_pd")
CLEAN = os.environ.get("CLEAN", "c001")
OUT = os.environ.get("OUT") or os.path.join(REPO, "data/experiments/18_flux_audit")
FIG = os.environ.get("FIG") or os.path.join(REPO, "figures/experiments/18_flux_audit")
EXT = os.path.join(REPO, "data/ngc3110/extended/cube")          # the extended configuration, same grid
RUNS = ["real", "null_synth", "null_linefree", "inj_bright", "inj_faint", "jack_eb1", "jack_eb2"]

META = json.load(open(os.path.join(DS, "meta.json")))
PARAMS = json.load(open(os.path.join(DS, "results", RES, "params.json")))["params"]
CELL, DV = float(META["cell_arcsec"]), abs(float(META["width_kms"]))
NZ, NY, NX = int(META["nchan"]), int(META["imsize"]), int(META["imsize"])
VEL = float(META["start_kms"]) + float(META["width_kms"]) * np.arange(NZ)
BEAM = META["beam"]                                               # clean beam of the dirty cube
OMEGA = np.pi * BEAM[0] * BEAM[1] / (4 * np.log(2)) / CELL ** 2   # its area, px
# injected sources: (name, dx", dy" (east, north), FWHM", line centre km/s); off = outside the galaxy
SOURCES = [("off_pt", -6.0, 5.5, 0.12, 4900.0), ("off_0.5", 6.0, 5.5, 0.5, 5000.0), ("off_1.5", -6.0, -5.5, 1.5, 5100.0),
           ("off_2", 6.2, -6.2, 2.0, 4950.0), ("off_pt_b", 0.0, 6.8, 0.12, 5050.0), ("off_1.5_b", 6.8, 1.5, 1.5, 4850.0),
           ("on_pt", 1.5, 2.0, 0.12, 5000.0), ("on_pt_b", -2.0, -1.5, 0.12, 5100.0)]
APERTURE = lambda fwhm: np.hypot(fwhm, np.sqrt(BEAM[0] * BEAM[1])) + 0.3   # radius ("): ~94 % of a source, never touching another
LINE_SIG = 40.0                                                   # km/s, Gaussian line width of the injected sources
SNR = {"inj_bright": 30.0, "inj_faint": 6.0}


def p(*a):
    print(*a, flush=True)


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1, default=float)


def pix(dx, dy):
    """sky offset (east, north) in arcsec -> (x, y) pixel; x grows to the west (RA axis), y to the north"""
    return NX // 2 - dx / CELL, NY // 2 + dy / CELL


# ============================================================== prepare
def line_free(dirty, psf):
    """the GUI's automatic line-free channels on the GUI's (rebinned) grid"""
    import torch
    import gpu_pd as G
    from deconvgui import worker
    b = int(PARAMS.get("rebin") or 1)
    d_s, psf_s = G.rebin(dirty, psf, b) if b > 1 else (dirty, psf)
    J, _ = worker._auto_scales(*d_s.shape[1:], d_s.shape[0], PARAMS.get("num_scales_2d"), PARAMS.get("num_levels_1d"))
    return worker._auto_line_free(G, G.Op(psf_s, d_s).d, J)


def source_cube(snr):
    """sky cube (Jy/pixel per channel) of the injected sources and their true fluxes (Jy km/s)"""
    d = np.load(os.path.join(DS, "dirty.npy"), mmap_mode="r")
    free = FREE_FROM_FILE()
    lf = np.asarray(d[free], dtype=np.float64)
    sigma = float((lf - lf.mean(0)).std() * np.sqrt(len(free) / (len(free) - 1)))   # noise per channel, Jy/beam
    sky, truth = np.zeros((NZ, NY, NX)), []
    yy, xx = np.mgrid[0:NY, 0:NX]
    for name, dx, dy, fwhm, v0 in SOURCES:
        x0, y0 = pix(dx, dy)
        s = max(fwhm, 0.5 * CELL) / CELL / 2.3548
        img = np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / s ** 2)
        img /= img.sum()                                                    # unit total flux
        g = np.exp(-0.5 * ((VEL - v0) / LINE_SIG) ** 2)
        # peak brightness in the clean beam (Jy/beam) = snr x noise per channel: S_ch = that x the dilution
        dil = np.sqrt((BEAM[0] ** 2 + fwhm ** 2) * (BEAM[1] ** 2 + fwhm ** 2)) / (BEAM[0] * BEAM[1])
        s_peak = snr * sigma * dil                                          # Jy, in the peak channel
        spec = s_peak * g / g.max()
        sky += spec[:, None, None] * img[None]
        truth.append(dict(name=name, dx=dx, dy=dy, fwhm=fwhm, v0=v0, flux=float(spec.sum() * DV), peak_channel_jy=float(s_peak)))
    return sky, truth


def FREE_FROM_FILE():
    return json.load(open(os.path.join(OUT, "line_free.json")))["channels"]


def prepare(a):
    import torch
    import gpu_pd as G
    from uv_to_image import ShiftInvariantOperator
    dirty = np.load(os.path.join(DS, "dirty.npy")).astype(np.float64)
    psf = np.load(os.path.join(DS, "psf.npy")).astype(np.float64)
    free = line_free(dirty, psf)
    jdump(dict(channels=free, velocities=[float(VEL[c]) for c in free]), os.path.join(OUT, "line_free.json"))
    p(f"line-free channels (the GUI's detection): {free}")

    def write(name, cube, extra=None):
        d = os.path.join(OUT, "ds", name)
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "dirty.npy"), cube.astype(np.float32))
        if not os.path.exists(os.path.join(d, "psf.npy")):
            os.symlink(os.path.join(DS, "psf.npy"), os.path.join(d, "psf.npy"))
        jdump(dict(META, audit=name, **(extra or {})), os.path.join(d, "meta.json"))
        p(f"  {name}: rms {cube.std():.3e}, line-free rms {cube[free].std():.3e}")

    write("real", dirty)
    d_t = torch.tensor(dirty, device=G.DEV, dtype=G.DT)
    ps = G.measured_ps(G.noise_planes(d_t, free))
    write("null_synth", G.noise_with_ps(ps, tuple(dirty.shape), seed=4242).cpu().numpy())
    lf = dirty[free] - dirty[free].mean(0, keepdims=True)                 # continuum removed: noise only
    lf *= np.sqrt(len(free) / (len(free) - 1))                            # undo the variance the mean took away
    write("null_linefree", lf[[i % len(free) for i in range(NZ)]])
    op = ShiftInvariantOperator(psf, dirty)
    for name, snr in SNR.items():
        sky, truth = source_cube(snr)
        write(name, dirty + op.apply(sky), dict(injected=truth))
        np.save(os.path.join(OUT, "ds", name, "injected_sky.npy"), sky.astype(np.float32))
        p("   " + ", ".join(f"{t['name']} {t['flux']:.3f}" for t in truth) + "  (Jy km/s)")
    if not a.skip_jack:
        jackknife()


def jackknife():
    """each execution block alone, imaged with the dataset's own CASA call (same grid)"""
    import grid_with_casa as g
    from deconvgui import worker
    worker._setup_paths(REPO)
    im = META["imaging"]
    for i, (ms, spw) in enumerate(zip(im["ms"], im["spw_list"])):
        name = f"jack_eb{i + 1}"
        d = os.path.join(OUT, "ds", name)
        os.makedirs(d, exist_ok=True)
        t0 = time.time()
        out = g.tclean_dirty_cube(ms_list=[ms], spw_list=[str(spw)], field=im["field"], phasecenter=im["phasecenter"],
                                  out_dir=os.path.join(d, "casa"), imsize=int(im["imsize"]), cell_arcsec=float(im["cell_arcsec"]),
                                  start_kms=float(im["start_kms"]), width_kms=float(im["width_kms"]), nchan=int(im["nchan"]),
                                  restfreq_ghz=float(im["restfreq_ghz"]), weighting=im.get("weighting", "natural"),
                                  log_name=f"flux_audit_{name}.log")
        np.save(os.path.join(d, "dirty.npy"), worker._read_fits_cube(out["dirty"]))
        np.save(os.path.join(d, "psf.npy"), worker._read_fits_cube(out["psf"]))
        jdump(dict(META, audit=name, ms=[ms], beam=[float(v) for v in out["beam"]]), os.path.join(d, "meta.json"))
        p(f"  {name}: {os.path.basename(ms)} imaged in {time.time() - t0:.0f} s, beam {out['beam']}")


# ================================================================== run
class _Q:
    """the job queue of the GUI's worker, as a log on stdout (frames dropped)"""
    def __init__(self, name):
        self.name, self.last = name, 0

    def put(self, item):
        kind, payload = item
        if kind == "log":
            p(f"[{self.name}] {payload}")
        elif kind == "progress" and time.time() - self.last > 60:
            self.last = time.time()
            p(f"[{self.name}] {payload.get('message')}")


def run(a):
    """the GUI's solver with the result's parameters (+ --extra JSON settings) on cube a.name -> res/<a.tag or a.name>"""
    from deconvgui import worker
    worker._setup_paths(REPO)
    d, r = os.path.join(OUT, "ds", a.name), os.path.join(OUT, "res", a.tag or a.name)
    os.makedirs(r, exist_ok=True)
    meta = json.load(open(os.path.join(d, "meta.json")))
    params = dict(PARAMS, noise_channels=",".join(str(c) for c in FREE_FROM_FILE()), **json.loads(a.extra or "{}"))
    params.pop("frame_every", None)
    t0 = time.time()
    out = worker.deconvolve_job(d, r, params, meta, _Q(a.tag or a.name), threading.Event(), device=a.device)
    jdump(dict(out, seconds=time.time() - t0, params=params), os.path.join(r, "summary.json"))
    p(f"[{a.name}] done in {time.time() - t0:.0f} s: flux {out.get('flux')}")


def msclean(a):
    """multiscale CLEAN; default scales matched to the beam: 0, 1, 3 x the beam FWHM (major axis) in pixels.
    --params JSON overrides (scales, nsigma, mask, auto-multithresh values); --tag names the output folder."""
    from deconvgui import worker
    worker._setup_paths(REPO)
    b = round(BEAM[0] / CELL)
    rdir = os.path.join(OUT, a.tag or "msclean_beam")
    os.makedirs(rdir, exist_ok=True)
    params = dict(dict(scales=[0, b, 3 * b], nsigma=3.0, niter=1000000, mask="auto-multithresh"), **json.loads(a.params or "{}"))
    t0 = time.time()
    out = worker.msclean_job(DS, rdir, params, META, _Q("msclean"), threading.Event())
    jdump(dict(out, params=params, seconds=time.time() - t0), os.path.join(rdir, "summary.json"))


# =================================================================== uv
def uv(a):
    """vector-averaged visibilities per baseline over the line channels and over the line-free channels
    (the cube's LSRK radio velocities -> each MS's own channels), target field, the imaged spw"""
    import casatools
    from casa_logs import redirect_casa_logs
    redirect_casa_logs("flux_audit_uv.log")
    im, free = META["imaging"], FREE_FROM_FILE()
    line = [c for c in range(NZ) if c not in free]
    groups = dict(line=line, linefree=free)
    res = {}
    for ms, spw in zip(im["ms"], im["spw_list"]):
        msmd = casatools.msmetadata(); msmd.open(ms)
        fld = int(msmd.fieldsforname(im["field"])[0])
        dd = int([d for d in msmd.datadescids() if msmd.spwfordatadesc(d) == int(spw)][0])
        msmd.close()
        tool = casatools.ms(); tool.open(ms)
        lsrk = np.array(tool.cvelfreqs(spwids=[int(spw)], fieldids=[fld], mode="channel", outframe="LSRK"))
        tool.close()
        vch = 299792.458 * (1 - lsrk / (float(im["restfreq_ghz"]) * 1e9))         # radio velocity of each MS channel
        tb = casatools.table(); tb.open(ms)
        q = tb.query(f"FIELD_ID == {fld} && DATA_DESC_ID == {dd} && ANTENNA1 != ANTENNA2")
        uvw = q.getcol("UVW")
        for gname, chans in groups.items():
            edges = [(VEL[c] - DV / 2, VEL[c] + DV / 2) for c in chans]
            sel = np.where(np.any([(vch >= lo) & (vch < hi) for lo, hi in edges], axis=0))[0]
            vsum, wsum = 0, 0
            for c0 in range(0, len(sel), 64):                                      # blocks of channels, low memory
                cs = sel[c0:c0 + 64]
                dat = q.getcolslice("DATA", [0, int(cs.min())], [1, int(cs.max())], [1, 1])   # (npol, nch, nrow)
                flg = q.getcolslice("FLAG", [0, int(cs.min())], [1, int(cs.max())], [1, 1])
                keep = np.isin(np.arange(cs.min(), cs.max() + 1), cs)
                dat, flg = dat[:, keep], flg[:, keep]
                w = (~flg).astype(np.float64)
                vsum = vsum + (dat * w).sum(axis=(0, 1)); wsum = wsum + w.sum(axis=(0, 1))
            ok = wsum > 0
            v = np.where(ok, vsum / np.maximum(wsum, 1), 0)                         # Stokes I mean over channels (Jy)
            wt = q.getcol("WEIGHT").sum(0)                                          # natural weights (both pols)
            # x the velocity span -> Jy km/s (the moment the model images are compared in)
            span = len(chans) * DV
            res.setdefault(gname, []).append(dict(u=uvw[0][ok], v=uvw[1][ok], vis=v[ok] * span, w=wt[ok],
                                                  lam=299792458.0 / lsrk.mean()))
            p(f"  {os.path.basename(ms)} {gname}: {ok.sum()} baselines, {len(sel)} channels")
        q.close(); tb.close()
    for gname, parts in res.items():
        np.savez(os.path.join(OUT, f"uv_{gname}.npz"),
                 u=np.concatenate([x["u"] / x["lam"] for x in parts]), v=np.concatenate([x["v"] / x["lam"] for x in parts]),
                 vis=np.concatenate([x["vis"] for x in parts]), w=np.concatenate([x["w"] for x in parts]))


def model_vis(img, u, v, sign=-1):
    """exact DFT of an image (Jy per pixel, row 0 south, x to the west) at (u, v) in wavelengths"""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rad = np.deg2rad(CELL / 3600)
    l = torch.tensor(-(np.arange(NX) - NX // 2) * rad, device=dev, dtype=torch.float64)    # east positive
    m = torch.tensor((np.arange(NY) - NY // 2) * rad, device=dev, dtype=torch.float64)
    I = torch.tensor(np.asarray(img, dtype=np.float64), device=dev).to(torch.complex128)
    out = np.empty(len(u), complex)
    for s in range(0, len(u), 4096):
        uu = torch.tensor(u[s:s + 4096], device=dev, dtype=torch.float64)[:, None]
        vv = torch.tensor(v[s:s + 4096], device=dev, dtype=torch.float64)[:, None]
        Ex = torch.exp(sign * 2j * np.pi * uu * l[None])                   # (P, nx)
        Ey = torch.exp(sign * 2j * np.pi * vv * m[None])                   # (P, ny)
        out[s:s + 4096] = ((Ey @ I) * Ex).sum(1).cpu().numpy()
    return out


# ============================================================== analyze
def analyze(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from uv_to_image import ShiftInvariantOperator
    os.makedirs(FIG, exist_ok=True)
    free = FREE_FROM_FILE(); line = [c for c in range(NZ) if c not in free]
    R = {}
    dirty = np.load(os.path.join(DS, "dirty.npy")).astype(np.float64)
    noise = float(np.mean([(dirty[c] - dirty[free].mean(0)).std() for c in free]))
    load = lambda path: np.load(path).astype(np.float64) if os.path.exists(path) else None
    gui = load(os.path.join(DS, "results", RES, "model.npy"))
    runs = {n: load(os.path.join(OUT, "res", n, "model.npy")) for n in RUNS}
    clean = {"c001": load(os.path.join(DS, "msclean", CLEAN, "model.npy")), "beam": load(os.path.join(OUT, "msclean_beam", "model.npy"))}
    resid = {n: load(os.path.join(OUT, "res", n, "residual.npy")) for n in RUNS}
    cres = {"c001": load(os.path.join(DS, "msclean", CLEAN, "residual.npy")), "beam": load(os.path.join(OUT, "msclean_beam", "residual.npy"))}
    F = lambda x, ch=None: float(x[ch].sum() * DV) if ch is not None else float(x.sum() * DV)
    rep = [f"# Flux audit: {META['name']} — 2D1D {RES} vs MS-CLEAN\n",
           f"Dataset `{DS}`; {NX} px × {CELL}″, {NZ} × {DV} km/s; clean beam {BEAM[0]:.2f}″ × {BEAM[1]:.2f}″ ({OMEGA:.0f} px).  ",
           f"Line-free channels (the GUI's detection): {free}. Noise (line-free, continuum removed): {noise * 1e3:.3f} mJy/beam.\n"]

    # ---- 0. the GUI result is reproduced by this runner
    if runs["real"] is not None and gui is not None:
        dif = np.abs(runs["real"] - gui).max() / np.abs(gui).max()
        R["reproduce"] = dict(max_rel_diff=float(dif), flux_gui=F(gui), flux_rerun=F(runs["real"]))
        rep.append(f"## 0. Reproducibility\nThe audit runner on the unmodified cube vs the GUI's {RES}: max |Δ| / max = {dif:.2e}, "
                   f"flux {F(runs['real']):.1f} vs {F(gui):.1f} Jy km/s. All other 2D1D runs use this exact recipe.\n")

    # ---- 1. per-channel budget: model flux, residual / noise
    models = {"2D1D": gui, "MS-CLEAN (c001, scales 0-60 px)": clean["c001"], "MS-CLEAN (beam-matched)": clean["beam"]}
    resids = {"2D1D": load(os.path.join(DS, "results", RES, "residual.npy")), "MS-CLEAN (c001, scales 0-60 px)": cres["c001"],
              "MS-CLEAN (beam-matched)": cres["beam"]}
    rows = []
    for k, x in models.items():
        if x is None:
            continue
        rr = resids[k]
        rows.append(dict(model=k, total=F(x), line=F(x, line), linefree=F(x, free),
                         resid_line=float(np.mean([rr[c].std() for c in line]) / noise) if rr is not None else None,
                         resid_free=float(np.mean([rr[c].std() for c in free]) / noise) if rr is not None else None,
                         neg=float(np.clip(x, None, 0).sum() * DV)))
    R["budget"] = rows
    rep.append("## 1. Flux and residual budget (16″ field)\n| model | total | line ch. | line-free ch. | negative | residual rms / noise, line | line-free |\n|---|---|---|---|---|---|---|")
    for r in rows:
        rep.append(f"| {r['model']} | {r['total']:.1f} | {r['line']:.1f} | {r['linefree']:.1f} | {r['neg']:.1f} | "
                   f"{r['resid_line']:.2f} | {r['resid_free']:.2f} |" if r["resid_line"] is not None else f"| {r['model']} | {r['total']:.1f} | | | | | |")
    rep.append("(Jy km/s). Residual rms below 1 in line-free channels = the model absorbs noise; above 1 in line channels = emission left in the residual.\n")

    # ---- 2. null tests: flux from no sky at all
    rep.append("## 2. Null tests: the recipe on cubes without any sky\n| cube | model flux (Jy km/s) | of the real 2D1D total | in line ch. | line-free ch. | peak / real peak |\n|---|---|---|---|---|---|")
    R["null"] = {}
    for n in ("null_synth", "null_linefree"):
        x = runs[n]
        if x is None:
            continue
        R["null"][n] = dict(flux=F(x), frac=F(x) / F(gui), line=F(x, line), linefree=F(x, free), peak_ratio=float(x.max() / gui.max()))
        rep.append(f"| {n} | {F(x):.2f} | {100 * F(x) / F(gui):.2f} % | {F(x, line):.2f} | {F(x, free):.2f} | {x.max() / gui.max():.3f} |")
    rep.append("Any flux here is fake by construction: it bounds what noise alone turns into model flux.\n")

    # ---- 3. injection recovery
    R["injection"] = {}
    yy, xx = np.mgrid[0:NY, 0:NX]
    if runs["real"] is not None:
        rep.append("## 3. Injection – recovery: model(data + sources) − model(data), apertures of radius √(FWHM² + beam²) + 0.3″\n"
                   "| set | source | FWHM″ | where | injected | recovered | ratio |\n|---|---|---|---|---|---|---|")
        for n in ("inj_bright", "inj_faint"):
            x = runs[n]
            if x is None:
                continue
            truth = json.load(open(os.path.join(OUT, "ds", n, "meta.json")))["injected"]
            dm = (x - runs["real"]).sum(0) * DV
            inside = np.zeros((NY, NX), bool); out_rows = []
            for t in truth:
                x0, y0 = pix(t["dx"], t["dy"])
                rap = APERTURE(t["fwhm"]) / CELL
                ap = (xx - x0) ** 2 + (yy - y0) ** 2 <= rap ** 2
                inside |= ap
                rec = float(dm[ap].sum())
                out_rows.append(dict(t, recovered=rec, ratio=rec / t["flux"]))
                rep.append(f"| {n} | {t['name']} | {t['fwhm']} | {'galaxy' if t['name'].startswith('on') else 'off'} | "
                           f"{t['flux']:.3f} | {rec:.3f} | **{rec / t['flux']:.2f}** |")
            elsewhere = float(dm[~inside].sum())
            R["injection"][n] = dict(sources=out_rows, change_elsewhere=elsewhere, injected_total=sum(t["flux"] for t in truth))
            rep.append(f"| {n} | change outside the apertures | | | | {elsewhere:+.3f} | |")
        rep.append("Ratio > 1 would be flux created by the recipe (overshoot); < 1 flux lost (L1 shrinkage, resolution).\n")

    # ---- 4. jackknife: the two execution blocks
    if runs["jack_eb1"] is not None and runs["jack_eb2"] is not None:
        e1, e2 = runs["jack_eb1"], runs["jack_eb2"]
        sm = lambda img: _smooth(img, 0.3 / CELL)
        m1, m2, mj = sm(e1[line].sum(0)), sm(e2[line].sum(0)), sm(gui[line].sum(0))
        corr = float(np.corrcoef(m1.ravel(), m2.ravel())[0, 1])
        sig = (m1 + m2) / 2; dif = (m1 - m2) / 2
        R["jackknife"] = dict(flux_eb1=F(e1), flux_eb2=F(e2), flux_joint=F(gui), corr_0p3=corr,
                              diff_over_signal_energy=float((dif ** 2).sum() / (sig ** 2).sum()),
                              linefree_eb1=F(e1, free), linefree_eb2=F(e2, free))
        rep.append(f"## 4. Jackknife: each execution block deconvolved alone\nFlux EB1 {F(e1):.1f}, EB2 {F(e2):.1f}, joint {F(gui):.1f} Jy km/s "
                   f"(line-free: {F(e1, free):.1f}, {F(e2, free):.1f}). Line moment 0 at 0.3″: correlation EB1–EB2 {corr:.3f}; "
                   f"half-difference / half-sum energy {R['jackknife']['diff_over_signal_energy']:.3f}. Structure or flux the two halves do not "
                   f"share is noise-driven.\n")

    # ---- 5. uv test against the raw visibilities
    R["uv"] = {}
    for gname, chans in (("line", line), ("linefree", free)):
        f = os.path.join(OUT, f"uv_{gname}.npz")
        if not os.path.exists(f):
            continue
        U = np.load(f); uu, vv, vd, w = U["u"], U["v"], U["vis"], U["w"]
        r = np.hypot(uu, vv) / 1e3                                             # kλ
        cand = {k: (x[chans].sum(0) * DV if x is not None else None) for k, x in models.items()}
        # the sign/orientation convention: the one under which the MS-CLEAN model (made by tclean from these very
        # visibilities on this grid) predicts them best
        ref = cand["MS-CLEAN (c001, scales 0-60 px)"] if cand["MS-CLEAN (c001, scales 0-60 px)"] is not None else cand["2D1D"]
        if gname == "line":
            fits = {s: float(np.sum(w * np.abs(vd - model_vis(ref, uu, vv, s)) ** 2)) for s in (-1, 1)}
            SIGN = min(fits, key=fits.get)
            R["uv"]["sign"] = SIGN
        bins = np.array([0, 15, 20, 25, 30, 40, 50, 65, 80, 100, 130, 170, 220, 300])
        out = dict(bins=bins.tolist())
        for k, img in cand.items():
            if img is None:
                continue
            vm = model_vis(img, uu, vv, R["uv"].get("sign", -1))
            a_bin, chi = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                s = (r >= lo) & (r < hi)
                if s.sum() < 20:
                    a_bin.append(None); chi.append(None); continue
                a_bin.append(float(np.real(np.sum(w[s] * vd[s] * np.conj(vm[s]))) / max(np.sum(w[s] * np.abs(vm[s]) ** 2), 1e-30)))
                chi.append(float(np.sum(w[s] * np.abs(vd[s] - vm[s]) ** 2) / np.sum(w[s] * np.abs(vd[s]) ** 2)))
            out[k] = dict(scale=a_bin, resid_frac=chi, V0=float(img.sum()))
        out["data_vector_avg"] = [float(np.abs(np.sum(w[(r >= lo) & (r < hi)] * vd[(r >= lo) & (r < hi)])) / max(np.sum(w[(r >= lo) & (r < hi)]), 1e-30))
                                  for lo, hi in zip(bins[:-1], bins[1:])]
        out["shortest_klambda"] = float(r.min())
        R["uv"][gname] = out
    if R["uv"].get("line"):
        U = R["uv"]["line"]; b = U["bins"]
        rep.append(f"## 5. Models vs the raw visibilities (line channels, shortest baseline {U['shortest_klambda']:.1f} kλ)\n"
                   "Per uv annulus, the least-squares scale a = Re Σ w V_data V_model* / Σ w |V_model|²: a = 1 means the model predicts the "
                   "measured visibilities; a < 1 the model has MORE flux on those scales than the data (overshoot), a > 1 less.\n")
        rep.append("| annulus (kλ) | " + " | ".join(k for k in models if k in U) + " |\n|---|" + "---|" * sum(k in U for k in models))
        for i, (lo, hi) in enumerate(zip(b[:-1], b[1:])):
            vals = [U[k]["scale"][i] for k in models if k in U]
            if all(v is None for v in vals):
                continue
            rep.append(f"| {lo}–{hi} | " + " | ".join("–" if v is None else f"{v:.3f}" for v in vals) + " |")
        rep.append("\nTotal flux V(0) (Jy km/s, not measured by any baseline): " + ", ".join(f"{k} {U[k]['V0']:.1f}" for k in models if k in U) + "\n")
    if R["uv"].get("linefree"):
        U = R["uv"]["linefree"]
        rep.append(f"Line-free channels: data vector-average amplitude per annulus {[round(x, 3) for x in U['data_vector_avg']]} Jy km/s "
                   f"(continuum, if any); 2D1D scale per annulus {[None if x is None else round(x, 2) for x in U['2D1D']['scale']]}.\n")

    # ---- 6. cross-validation: the independent extended-configuration data
    if os.path.exists(os.path.join(EXT, "dirty_cube.fits")):
        from deconvgui.worker import _read_fits_cube
        ed = _read_fits_cube(os.path.join(EXT, "dirty_cube.fits")).astype(np.float64)
        ep = _read_fits_cube(os.path.join(EXT, "dirty_beam_cube.fits")).astype(np.float64)
        op = ShiftInvariantOperator(ep, ed)
        e_free = free
        en = float(np.mean([(ed[c] - ed[e_free].mean(0)).std() for c in e_free]))
        R["xval"] = {}
        rep.append("## 6. Cross-validation on the extended configuration (independent visibilities, same grid)\n"
                   "Each model convolved with the EXTENDED dirty beam vs the extended dirty cube, line channels: least-squares flux "
                   "scale a (1 = the model's flux is what the extended array sees; < 1 = too much) and the fraction of the extended "
                   "signal energy left unexplained.\n| model | a | unexplained | residual rms / ext. noise |\n|---|---|---|---|")
        for k, x in models.items():
            if x is None:
                continue
            nx_ = op.apply(x)
            dl, ml = ed[line], nx_[line]
            aa = float((dl * ml).sum() / (ml * ml).sum())
            un = float(((dl - aa * ml) ** 2).sum() / (dl ** 2).sum())
            rr = float(np.mean([(ed[c] - nx_[c]).std() for c in line]) / en)
            R["xval"][k] = dict(scale=aa, unexplained=un, resid_over_noise=rr)
            rep.append(f"| {k} | {aa:.3f} | {un:.3f} | {rr:.2f} |")
        rep.append("")

    # ---- 7. where the extra flux sits: curve of growth, and the 2D1D − CLEAN difference
    cx, cy = NX // 2, NY // 2
    rad = np.hypot(xx - cx, yy - cy) * CELL
    redges = np.arange(0.5, 8.01, 0.5)
    R["growth"] = {"r": redges.tolist()}
    for k, x in models.items():
        if x is not None:
            m0 = x.sum(0) * DV
            R["growth"][k] = [float(m0[rad <= q].sum()) for q in redges]

    # ---- figures
    _figures(plt, models, runs, R, free, line)
    jdump(R, os.path.join(OUT, "results.json"))
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(rep) + "\n")
    p("\n".join(rep))


def _smooth(img, sigma_pix):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, sigma_pix)


def _figures(plt, models, runs, R, free, line):
    cols = {"2D1D": "#d08c00", "MS-CLEAN (c001, scales 0-60 px)": "#2a78d6", "MS-CLEAN (beam-matched)": "#6a3d9a"}
    # spectra
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for k, x in models.items():
        if x is not None:
            ax[0].step(VEL, x.sum((1, 2)), where="mid", color=cols[k], label=f"{k} ({x.sum() * DV:.0f} Jy km/s)")
    for n, ls in (("null_synth", ":"), ("null_linefree", "--")):
        if runs.get(n) is not None:
            ax[0].step(VEL, runs[n].sum((1, 2)), where="mid", color="k", ls=ls, label=f"{n} (no sky): {runs[n].sum() * DV:.1f}")
    for c in free:
        ax[0].axvspan(VEL[c] - DV / 2, VEL[c] + DV / 2, color="0.92", zorder=0)
    ax[0].set_xlabel("v (km/s)"); ax[0].set_ylabel("flux density (Jy)"); ax[0].legend(fontsize=8); ax[0].set_title("16″ field; grey: line-free channels")
    if "growth" in R:
        for k in models:
            if k in R["growth"]:
                ax[1].plot(R["growth"]["r"], R["growth"][k], color=cols[k], label=k)
        ax[1].set_xlabel("radius from the phase centre (″)"); ax[1].set_ylabel("cumulative flux (Jy km/s)"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "18a_spectra_growth.png"), dpi=130); plt.close(fig)
    # uv scale per annulus
    if R["uv"].get("line"):
        U = R["uv"]["line"]; b = np.array(U["bins"]); mid = (b[:-1] + b[1:]) / 2
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        for k in models:
            if k in U:
                ax[0].plot(mid, [np.nan if v is None else v for v in U[k]["scale"]], "o-", color=cols[k], label=k)
                ax[1].plot(mid, [np.nan if v is None else v for v in U[k]["resid_frac"]], "o-", color=cols[k], label=k)
        ax[0].axhline(1, color="k", lw=0.8); ax[0].set_xscale("log"); ax[0].set_xlabel("uv distance (kλ)")
        ax[0].set_ylabel("data / model (least-squares scale)"); ax[0].set_title("< 1: model has more flux than the data on that scale")
        ax[1].set_xscale("log"); ax[1].set_xlabel("uv distance (kλ)"); ax[1].set_ylabel("Σw|V_d − V_m|² / Σw|V_d|²")
        for a_ in ax:
            a_.axvline(U["shortest_klambda"], color="0.6", ls=":"); a_.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "18b_uv_test.png"), dpi=130); plt.close(fig)
    # null-test and jackknife maps
    show = [(n, runs[n]) for n in ("null_synth", "null_linefree", "jack_eb1", "jack_eb2") if runs.get(n) is not None]
    if show:
        fig, ax = plt.subplots(1, len(show), figsize=(4.6 * len(show), 4.4))
        ax = np.atleast_1d(ax)
        for a_, (n, x) in zip(ax, show):
            m0 = x.sum(0) * DV
            a_.imshow(m0, origin="lower", cmap="inferno", extent=[NX * CELL / 2, -NX * CELL / 2, -NY * CELL / 2, NY * CELL / 2])
            a_.set_title(f"{n}: {m0.sum():.1f} Jy km/s")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "18c_null_jackknife_maps.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prepare"); s.add_argument("--skip-jack", action="store_true")
    s = sub.add_parser("jackknife")
    s = sub.add_parser("run"); s.add_argument("name"); s.add_argument("--device", default="cuda:0")
    s.add_argument("--extra", help="JSON solver settings added to the result's parameters"); s.add_argument("--tag", help="output name")
    s = sub.add_parser("msclean"); s.add_argument("--params", help="JSON MS-CLEAN settings"); s.add_argument("--tag", help="output folder")
    sub.add_parser("uv"); sub.add_parser("analyze")
    a = ap.parse_args()
    {"prepare": prepare, "jackknife": lambda a: jackknife(), "run": run, "msclean": msclean, "uv": uv, "analyze": analyze}[a.cmd](a)
