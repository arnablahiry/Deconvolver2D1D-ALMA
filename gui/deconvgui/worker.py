"""
Job bodies. Each runs in a FRESH worker process (multiprocessing "spawn"),
which imports the repository's `simple/` modules at that moment -- so a
change to the solver (or a `git pull`) is used by the next job with no
restart, and a CASA crash can never take the server down.

Messages to the server go through `q` as (kind, payload) tuples:
    ("log", str)                       a line for the job log
    ("progress", dict)                 fraction 0..1 (or None = indeterminate), message
    ("stat", dict)                     per-iteration diagnostics for the live plot
    ("frame", (iteration, float32 2D)) a moment-0 snapshot of the current iterate
    ("result", dict)                   final payload (job-specific)
    ("error", str)                     traceback text
"""

import inspect
import json
import os
import sys
import time
import traceback

import numpy as np


class Cancelled(Exception):
    pass


def _setup_paths(repo_root):
    for sub in ("simple", "scripts"):
        p = os.path.join(repo_root, sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    # CASA writes timestamped logs into the CWD: keep them out of the repo root.
    logs = os.path.join(repo_root, "casa_logs")
    os.makedirs(logs, exist_ok=True)
    os.chdir(repo_root)


def run_job(kind, args, q, cancel, repo_root):
    try:
        _setup_paths(repo_root)
        fn = {"inspect": inspect_ms_job, "image": image_job, "deconvolve": deconvolve_job,
              "devices": devices_job, "msclean": msclean_job}[kind]
        result = fn(q=q, cancel=cancel, **args)
        q.put(("result", result))
    except Cancelled:
        q.put(("cancelled", "cancelled by user"))
    except BaseException:
        q.put(("error", traceback.format_exc()))


def _kwargs_for(fn, kw):
    """Only pass the keyword arguments `fn` actually accepts, so the GUI keeps
    working when the solver's signature changes in the repo."""
    sig = inspect.signature(fn)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kw)
    return {k: v for k, v in kw.items() if k in sig.parameters}


# ======================================================================
# Inspect an MS
# ======================================================================
def inspect_ms_job(ms_paths, q, cancel):
    """Inspect one or several MSs (e.g. the execution blocks of one array
    configuration, imaged together). The first MS supplies fields and spws;
    `per_ms` keeps each one's own spws so the form can match them up."""
    per_ms = []
    for i, ms_path in enumerate(ms_paths):
        q.put(("progress", dict(fraction=i / len(ms_paths),
                                message=f"reading MS metadata (casatools): {os.path.basename(ms_path.rstrip('/'))}")))
        per_ms.append(_inspect_one(ms_path))
    info = dict(per_ms[0], ms_list=list(ms_paths), per_ms=per_ms)
    info["nant"] = max(m["nant"] for m in per_ms)
    bl = [m for m in per_ms if m.get("bl_max")]
    if bl:
        info["bl_min"] = min(m["bl_min"] for m in bl)
        info["bl_max"] = max(m["bl_max"] for m in bl)
    # Only targets present in every MS can be imaged jointly (by name: field
    # ids differ between execution blocks with different calibrators).
    common = set.intersection(*({t["name"] for t in m["target_fields"]} for m in per_ms))
    info["target_fields"] = [t for t in per_ms[0]["target_fields"] if t["name"] in common]
    return info


def _inspect_one(ms_path):
    from casatools import msmetadata, measures, quanta  # noqa: F401

    info = None
    try:
        import inspect_ms  # scripts/inspect_ms.py
        info = inspect_ms.describe(ms_path)
    except ImportError:
        pass

    md = msmetadata()
    md.open(ms_path)
    try:
        fields = list(md.fieldnames())
        try:
            target_ids = list(md.fieldsforintent("OBSERVE_TARGET#ON_SOURCE"))
        except Exception:
            target_ids = list(range(len(fields)))
        targets = []
        for fid in target_ids:
            pc = md.phasecenter(int(fid))
            ra = pc["m0"]["value"]
            dec = pc["m1"]["value"]
            targets.append(dict(id=int(fid), name=fields[fid], phasecenter=_fmt_j2000(ra, dec)))
        if info is None:
            spws = []
            for s in range(md.nspw()):
                f = md.chanfreqs(s) / 1e9
                if f.size < 2:
                    continue
                spws.append(dict(spw=s, nchan=int(f.size), f_lo=float(f.min()), f_hi=float(f.max()),
                                 chanwidth_khz=float(np.mean(np.abs(md.chanwidths(s))) / 1e3)))
            info = dict(path=ms_path, fields=fields, spws=spws, nant=int(md.nantennas()))
    finally:
        md.close()
    info["target_fields"] = targets
    info["mosaic"] = len(targets) > 1 and len({t["name"] for t in targets}) == 1
    try:
        info["uv"] = _uv_sample(ms_path, [t["id"] for t in targets])
    except Exception as e:   # the plot is a nicety: never fail the inspection over it
        info["uv_error"] = repr(e)
    return info


DENDRO_DEFAULTS = dict(min_value=3.0, min_delta=1.0, peak=5.0, min_beams=2.0)


def dendro_mask(run_dir, beam_px, params=None):
    """A 3D signal mask from astrodendro, on MS-CLEAN run run_dir's image (model * clean beam + residual, which has
    the sidelobes of what was cleaned removed), in units of its noise: the voxels of every trunk structure above
    min_value sigma (structures split at min_delta sigma, at least min_beams clean beams of voxels) whose peak
    reaches `peak` sigma. sigma: the median robust rms of the line-free channels (those within 1.5 x the quietest;
    an early-stopped CLEAN leaves emission in the residual of the line channels), one value for all channels.
    Computed on a cube binned so the beam's FWHM is ~8 px (the mask is beam-smooth anyway), then brought back
    to the full grid. Writes dendro_mask.npy (uint8) and dendro_mask.json, returns the latter."""
    from astrodendro import Dendrogram

    p = dict(DENDRO_DEFAULTS, **{k: float(v) for k, v in (params or {}).items() if k in DENDRO_DEFAULTS and v not in (None, "")})
    t0 = time.time()
    res = np.load(os.path.join(run_dir, "residual.npy")).astype(np.float64)
    img = np.load(os.path.join(run_dir, "restored.npy")).astype(np.float64) + res
    nz, ny, nx = img.shape
    f = max(1, int(np.sqrt(beam_px / 1.1331) / 8))
    rebin = lambda a: a[:, :ny // f * f, :nx // f * f].reshape(nz, ny // f, f, nx // f, f).mean(axis=(2, 4))
    ib, rb = rebin(img), rebin(res)
    rms = 1.4826 * np.median(np.abs(rb - np.median(rb, axis=(1, 2), keepdims=True)), axis=(1, 2))
    quiet = rms <= 1.5 * rms.min()
    sigma = float(np.median(rms[quiet]))
    d = Dendrogram.compute(ib / sigma, min_value=p["min_value"], min_delta=p["min_delta"],
                           min_npix=max(1, int(round(p["min_beams"] * beam_px / f ** 2))))
    keep = [st for st in d.trunk if st.get_peak()[1] >= p["peak"]]
    mb = np.zeros(ib.shape, bool)
    for st in keep:
        mb[st.indices(subtree=True)] = True
    m = np.zeros((nz, ny, nx), np.uint8)
    m[:, :ny // f * f, :nx // f * f] = np.repeat(np.repeat(mb, f, axis=1), f, axis=2)
    np.save(os.path.join(run_dir, "dendro_mask.npy"), m)
    info = dict(params=p, sigma=sigma, sigma_unit="Jy/beam (binned)", bin=f, line_free_channels=int(quiet.sum()),
                structures=len(keep), trunk=len(d.trunk), fraction=float(m.mean()),
                channel_fraction=[float(v) for v in m.mean(axis=(1, 2))], seconds=time.time() - t0, created=time.time())
    with open(os.path.join(run_dir, "dendro_mask.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    return info


def uv_coverage(ms_list, field, out_path):
    """The uv plot of a dataset (run by the server in a separate process, see /api/datasets/{id}/uv):
    _uv_sample of each MS for the imaged field (by name; the target fields if no name), written to
    out_path as JSON."""
    import json
    from casatools import msmetadata
    sets = []
    for ms in ms_list:
        md = msmetadata()
        md.open(ms)
        try:
            ids = md.fieldsforname(field) if field else md.fieldsforintent("OBSERVE_TARGET#ON_SOURCE")
            ids = [int(i) for i in ids]
        finally:
            md.close()
        sets.append(dict(ms=ms, **_uv_sample(ms, ids)))
    with open(out_path + ".part", "w") as f:
        json.dump(dict(sets=sets), f)
    os.replace(out_path + ".part", out_path)


def _uv_sample(ms_path, field_ids, n=12000):
    """About n baselines' (u, v), in metres, for the uv-coverage plot: target fields, one spectral
    window (the rows repeat per spw), no autocorrelations, evenly thinned in time."""
    from casatools import table
    tb = table()
    tb.open(ms_path)
    try:
        sel = f"FIELD_ID IN [{','.join(str(int(f)) for f in field_ids) or '0'}] && ANTENNA1 != ANTENNA2"
        # the spw of the first target row: the DATA_DESCRIPTION table may list spws with no data
        first = tb.query(sel, columns="DATA_DESC_ID")   # row references only: cheap
        try:
            if not first.nrows():
                return dict(u=[], v=[], rows=0)
            dd = int(first.getcell("DATA_DESC_ID", 0))
        finally:
            first.close()
        q = tb.query(f"{sel} && DATA_DESC_ID == {dd}", columns="UVW")
        try:
            nrow = q.nrows()
            uvw = q.getcol("UVW", 0, -1, max(1, nrow // n)) if nrow else np.zeros((3, 0))
        finally:
            q.close()
    finally:
        tb.close()
    return dict(u=np.round(uvw[0], 1).tolist(), v=np.round(uvw[1], 1).tolist(), rows=int(nrow))


def _fmt_j2000(ra_rad, dec_rad):
    ra_h = (np.degrees(ra_rad) % 360.0) / 15.0
    h = int(ra_h); m = int((ra_h - h) * 60); s = ((ra_h - h) * 60 - m) * 60
    sign = "-" if dec_rad < 0 else "+"
    d_abs = abs(np.degrees(dec_rad))
    d = int(d_abs); dm = int((d_abs - d) * 60); ds = ((d_abs - d) * 60 - dm) * 60
    return f"J2000 {h:02d}:{m:02d}:{s:07.4f} {sign}{d:02d}.{dm:02d}.{ds:06.3f}"


# ======================================================================
# MS -> dirty cube + PSF (CASA tclean, niter=0)
# ======================================================================
def image_job(ds_dir, params, q, cancel):
    import grid_with_casa as g

    p = params
    q.put(("progress", dict(fraction=None, message="CASA: gridding PSF and dirty cube (tclean niter=0)")))
    ms_list = p["ms"] if isinstance(p["ms"], list) else [p["ms"]]
    spw_list = [str(s) for s in p.get("spw_list") or [p["spw"]] * len(ms_list)]
    for ms, spw in zip(ms_list, spw_list):
        q.put(("log", f"[image] {ms} spw={spw}"))
    q.put(("log", f"[image] tclean_dirty_cube on {len(ms_list)} MS, field={p['field']}"))
    t0 = time.time()
    out_dir = os.path.join(ds_dir, "casa")
    call = dict(ms_list=ms_list, spw_list=spw_list, field=p["field"], phasecenter=p["phasecenter"],
                out_dir=out_dir, imsize=int(p["imsize"]), cell_arcsec=float(p["cell_arcsec"]),
                start_kms=float(p["start_kms"]), width_kms=float(p["width_kms"]), nchan=int(p["nchan"]),
                restfreq_ghz=float(p["restfreq_ghz"]), weighting=p.get("weighting", "natural"),
                log_name=f"gui_tclean_{os.path.basename(ds_dir)}.log")
    out = g.tclean_dirty_cube(**_kwargs_for(g.tclean_dirty_cube, call))
    q.put(("log", f"[image] tclean done in {time.time() - t0:.0f} s"))

    dirty = _read_fits_cube(out["dirty"])
    psf = _read_fits_cube(out["psf"])
    np.save(os.path.join(ds_dir, "dirty.npy"), dirty)
    np.save(os.path.join(ds_dir, "psf.npy"), psf)
    bmaj, bmin, bpa = (float(v) for v in out["beam"])
    q.put(("log", f"[image] dirty {dirty.shape}, psf {psf.shape}, beam {bmaj:.3f}\" x {bmin:.3f}\" @ {bpa:.1f} deg"))
    return dict(shape=list(dirty.shape), psf_shape=list(psf.shape), beam=[bmaj, bmin, bpa])


# ======================================================================
# MS-CLEAN: the method in use, to compare the 2D-1D model against
# ======================================================================
MSCLEAN_AUTOMASK = dict(sidelobethreshold=3.0, noisethreshold=5.0, lownoisethreshold=1.5,
                        negativethreshold=7.0, minbeamfrac=0.3, growiterations=75)


def msclean_job(ds_dir, run_dir, params, meta, q, cancel):
    """Multiscale CLEAN (tclean deconvolver="multiscale"), the recipe of scripts/msclean_cube.py, on the
    dataset's own visibilities and grid: same MSs, field, spws, phase centre, channels, image size, cell
    and weighting as its dirty cube, so every pixel and channel lines up (cropped datasets: the crops
    are applied afterwards, the cleaning itself always sees the full field). Saves model.npy (Jy/pixel),
    residual.npy, restored.npy = model * the common clean beam (Jy/beam; no residual added) and, when
    tclean made one, mask.npy (uint8, the clean mask per channel)."""
    import shutil
    import casatools
    from casa_logs import redirect_casa_logs
    from restore import convolve_with_beam

    redirect_casa_logs(f"gui_msclean_{os.path.basename(ds_dir)}_{os.path.basename(run_dir)}.log")
    from casatasks import exportfits, tclean

    im, p = meta["imaging"], params
    ms_list = im["ms"] if isinstance(im["ms"], list) else [im["ms"]]
    spw_list = [str(s) for s in im.get("spw_list") or [im["spw"]] * len(ms_list)]
    scales = [int(s) for s in p.get("scales") or [0, 10, 30, 60]]
    base = os.path.join(run_dir, "_tclean")
    common = dict(vis=list(ms_list), field=im["field"], spw=spw_list, datacolumn="data", imagename=base,
                  imsize=int(im["imsize"]), cell=f"{float(im['cell_arcsec'])}arcsec", phasecenter=im["phasecenter"],
                  specmode="cube", start=f"{float(im['start_kms'])}km/s", width=f"{float(im['width_kms'])}km/s",
                  nchan=int(im["nchan"]), restfreq=f"{float(im['restfreq_ghz'])}GHz", outframe="LSRK", veltype="radio",
                  gridder="standard", weighting=im.get("weighting", "natural"), deconvolver="multiscale",
                  scales=scales, restoringbeam="common", pbcor=False)
    if p.get("smallscalebias") not in (None, ""):
        common["smallscalebias"] = float(p["smallscalebias"])
    mask = p.get("mask") or "auto-multithresh"
    if mask == "auto-multithresh":
        common.update(usemask="auto-multithresh",
                      **{k: type(v)(p[k]) if p.get(k) not in (None, "") else v for k, v in MSCLEAN_AUTOMASK.items()})
    else:
        common.update(usemask="user", mask="")
    q.put(("log", f"[msclean] {len(ms_list)} MS, field {im['field']}, spw {spw_list}, {im['imsize']} px x {im['cell_arcsec']}\", "
                  f"{im['nchan']} x {im['width_kms']} km/s from {im['start_kms']}, {im.get('weighting', 'natural')}; "
                  f"scales {scales} px, nsigma {p.get('nsigma', 3)}, niter {p.get('niter', 1000000)}, mask {mask}"))
    q.put(("progress", dict(fraction=None, message="CASA tclean, multiscale: minor and major cycles (minutes to hours)")))
    t0 = time.time()
    rec = tclean(niter=int(p.get("niter") or 1000000), nsigma=float(p.get("nsigma") or 3.0), **common)
    q.put(("log", f"[msclean] stopcode {rec.get('stopcode')}, {rec.get('iterdone')} iterations, "
                  f"{rec.get('nmajordone')} major cycles, {time.time() - t0:.0f} s"))

    q.put(("progress", dict(fraction=None, message="exporting model and residual, convolving with the clean beam")))
    cubes = {}
    for ext in ("model", "residual"):
        f = os.path.join(run_dir, f"{ext}.fits")
        exportfits(imagename=f"{base}.{ext}", fitsimage=f, velocity=True, overwrite=True)
        cubes[ext] = _read_fits_cube(f).astype(np.float64)
        os.remove(f)
    if os.path.isdir(f"{base}.mask"):   # the clean mask (auto-multithresh: one per channel), for like-for-like comparisons
        f = os.path.join(run_dir, "mask.fits")
        exportfits(imagename=f"{base}.mask", fitsimage=f, velocity=True, overwrite=True)
        cubes["mask"] = _read_fits_cube(f) > 0.5
        os.remove(f)
    ia = casatools.image()
    ia.open(f"{base}.image")
    b = ia.restoringbeam()
    ia.close()
    beam = [float(b["major"]["value"]), float(b["minor"]["value"]), float(b["positionangle"]["value"])]
    if b["major"]["unit"] != "arcsec" or b["minor"]["unit"] != "arcsec":
        qa = casatools.quanta()
        beam[:2] = [qa.convert(b["major"], "arcsec")["value"], qa.convert(b["minor"], "arcsec")["value"]]
    cubes["restored"] = convolve_with_beam(cubes["model"], float(im["cell_arcsec"]), *beam)   # model * clean beam
    for c in meta.get("crops") or []:   # the dataset's crops, in the order they were made
        x0, x1, y0, y1 = c["box"]
        c0, c1 = c.get("chans") or (0, None)
        for k in cubes:
            cubes[k] = cubes[k][c0:None if c1 is None else c1 + 1, y0:y1 + 1, x0:x1 + 1]
    for k, a in cubes.items():
        np.save(os.path.join(run_dir, f"{k}.npy"), np.ascontiguousarray(a, dtype=np.uint8 if k == "mask" else np.float32))
    for path in [x for x in os.listdir(run_dir) if x.startswith("_tclean")]:
        shutil.rmtree(os.path.join(run_dir, path), ignore_errors=True)   # CASA images: large, all kept as .npy
    rms = cubes["residual"].std(axis=(1, 2))
    dirty_shape = list(np.load(os.path.join(ds_dir, "dirty.npy"), mmap_mode="r").shape)
    if list(cubes["model"].shape) != dirty_shape:
        q.put(("log", f"[warn] MS-CLEAN cube {list(cubes['model'].shape)} does not match the dirty cube {dirty_shape}"))
    q.put(("log", f"[msclean] common beam {beam[0]:.3f}\" x {beam[1]:.3f}\" @ {beam[2]:.1f} deg; residual rms "
                  f"{rms.min() * 1e3:.3f}-{rms.max() * 1e3:.3f} mJy/beam; model flux {cubes['model'].sum():.3f} Jy"))
    return dict(beam=beam, stopcode=rec.get("stopcode"), iterations=rec.get("iterdone"), major_cycles=rec.get("nmajordone"),
                flux=float(cubes["model"].sum()), residual_rms=[float(rms.min()), float(rms.max())], seconds=time.time() - t0)


def _read_fits_cube(path):
    from astropy.io import fits
    with fits.open(path) as h:
        data = np.asarray(h[0].data, dtype=np.float32)
    data = np.squeeze(data)
    if data.ndim == 2:
        data = data[None]
    return np.nan_to_num(data, nan=0.0)


# ======================================================================
# Deconvolution
# ======================================================================
MAX_SNAPSHOTS = 1000      # each is a full moment-0 map kept in memory (800 x 800 px: 2.6 MB)


def _frame_marks(n_iter, every=None):
    """Iterations to keep as snapshots: every `every` iterations if given (at most MAX_SNAPSHOTS: `every` is
    raised if needed), else ~140 log-spaced ones (as the scripts' evolution GIFs: the model changes fastest
    early on); the first and last iteration always. The live view shows every iteration regardless."""
    if every:
        every = max(int(every), -(-n_iter // MAX_SNAPSHOTS))
        marks = set(range(0, n_iter + 1, max(1, every)))
    else:
        marks = set(np.unique(np.geomspace(1, max(n_iter, 1), 140).astype(int)).tolist())
    return marks | {1, n_iter}


def _surface_brightness(meta):
    """(factor, unit): moment 0 in Jy/pixel km/s -> Jy km/s arcsec^-2 (per pixel when the cell is unknown)"""
    cell = meta.get("cell_arcsec")
    v = "km/s" if meta.get("width_kms") else "ch"
    if cell:
        return 1.0 / float(cell) ** 2, f"Jy {v} arcsec\u207b\u00b2"
    return 1.0, f"Jy/pixel {v}"


class _WatchedOperator:
    """
    Wraps the fast operator. Both solvers call `operator.gradient(...)`
    exactly once per iteration on the current iterate, so this is where the
    GUI hooks in -- progress, live moment-0 frames, per-iteration stats and
    cancellation -- without any change to the solver code.
    """

    def __init__(self, op, lipschitz, n_iter, frame_every, dv, q, cancel, sb=1.0):
        self._op = op
        self.lipschitz = lipschitz
        self.n_iter = n_iter
        self.marks = _frame_marks(n_iter, frame_every)
        self.dv = dv * sb
        self.q = q
        self.cancel = cancel
        self.it = 0
        self.frames, self.frame_iters = [], []
        self.t0 = time.time()

    def __getattr__(self, name):
        return getattr(self._op, name)

    def gradient(self, x):
        if self.cancel.is_set():
            raise Cancelled()
        g = self._op.gradient(x)
        it = self.it
        self.it += 1
        m0 = (np.asarray(x).sum(axis=0) * self.dv).astype(np.float32)
        el = time.time() - self.t0
        eta = el / max(self.it, 1) * max(self.n_iter - self.it, 0)
        # live display, every iteration; the progress travels with the frame so the two always agree
        self.q.put(("frame", (it, m0, dict(fraction=min(self.it / self.n_iter, 1.0),
                                           message=f"iteration {it}/{self.n_iter}  ({el:.0f} s, ~{eta:.0f} s left)"))))
        if it in self.marks:
            self.frames.append(m0)
            self.frame_iters.append(it)
        self.q.put(("stat", dict(iter=it, residual_rms=float(g.std()), flux=float(np.asarray(x).sum()))))
        return g


def deconvolve_job(ds_dir, res_dir, params, meta, q, cancel, device="cpu"):
    from uv_to_image import ShiftInvariantOperator

    p = dict(params)
    backend = p.pop("backend", "pd")
    p.pop("devices", None)
    if backend in ("gpu_pd", "gpu_fista"):
        return torch_job(ds_dir, res_dir, p, meta, q, cancel, device, algo="pd" if backend == "gpu_pd" else "fista")
    if device != "cpu":
        q.put(("log", f"[warn] the {backend} solver is numpy-only; running on the CPU, not {device}"))
    frame_every = p.pop("frame_every", None)
    lrs_arcsec = p.pop("lrs_arcsec", None)
    noise_spec = p.pop("noise_channels", None)

    q.put(("progress", dict(fraction=None, message="loading cubes")))
    dirty = np.load(os.path.join(ds_dir, "dirty.npy")).astype(np.float64)
    psf = np.load(os.path.join(ds_dir, "psf.npy")).astype(np.float64)
    q.put(("log", f"[setup] dirty {dirty.shape}, psf {psf.shape}, backend={backend}"))
    if psf.shape[1] < 2 * dirty.shape[1] or psf.shape[2] < 2 * dirty.shape[2]:
        q.put(("log", "[warn] the PSF is smaller than twice the image: the normal operator may be "
                      "indefinite (see uv_to_image.ShiftInvariantOperator)"))

    fast = ShiftInvariantOperator(psf, dirty)
    if backend == "pd":
        import deconvolve_pd as mod
        q.put(("progress", dict(fraction=None, message="power iteration for the step size")))
        lip = mod.field_lipschitz(fast, dirty.shape)
    elif backend == "fista":
        import deconvolve as mod
        lip = fast.lipschitz
    else:
        raise ValueError(f"unknown backend {backend}")

    n_iter = int(p.get("n_iter") or getattr(mod, "N_ITER", 100))
    p["n_iter"] = n_iter
    dv = abs(float(meta.get("width_kms") or 1.0))
    sb, sb_unit = _surface_brightness(meta)

    if noise_spec:
        p["noise_channels"] = parse_channels(noise_spec, dirty.shape[0])
    if lrs_arcsec and meta.get("cell_arcsec"):
        p["max_scale_px"] = float(lrs_arcsec) / float(meta["cell_arcsec"])
    for k in ("num_scales_2d", "num_levels_1d"):
        if p.get(k) in ("", None):
            p.pop(k, None)

    op = _WatchedOperator(fast, lip, n_iter, frame_every, dv, q, cancel, sb=sb)
    q.put(("progress", dict(fraction=0.0, message="computing thresholds")))
    kw = _kwargs_for(mod.deconvolve, dict(p, operator=op, verbose=True, log=lambda m: q.put(("log", m))))
    dropped = sorted(set(p) - set(kw) - {"operator"})
    if dropped:
        q.put(("log", f"[warn] {mod.__name__}.deconvolve() does not take: {', '.join(dropped)} (ignored)"))
    x, history = mod.deconvolve(psf, dirty, **kw)
    return _finalize(np.asarray(x, dtype=np.float32), history, op.frames, op.frame_iters, op.it,
                     dirty, fast, meta, params, dv, res_dir, q, device="cpu", sb=(sb, sb_unit))


def _finalize(x, history, frames, frame_iters, n_it, dirty, fast, meta, params, dv, res_dir, q, device, sb=(1.0, "")):
    """Save model, residual, restored cube, frames, history and the GIF (all solvers). Frames are moment 0
    in surface brightness (sb = (factor, unit)), at log-spaced iterations unless asked otherwise."""
    q.put(("progress", dict(fraction=1.0, message="saving model, residual, restored cube")))
    np.save(os.path.join(res_dir, "model.npy"), x)
    residual = (dirty - fast.apply(x.astype(np.float64))).astype(np.float32)
    np.save(os.path.join(res_dir, "residual.npy"), residual)

    restored_beam = None
    beam = meta.get("beam")
    cell = meta.get("cell_arcsec")
    if beam and cell:
        from restore import convolve_with_beam
        rb = params.get("restore_beam") or beam
        if isinstance(rb, str):
            rb = [float(v) for v in rb.replace(",", " ").split()]
            if len(rb) == 1:
                rb = [rb[0], rb[0], 0.0]
            if len(rb) != 3:
                raise ValueError("restoring beam: give bmaj, bmin (arcsec) and pa (deg), or one FWHM for a circular beam")
        restored = convolve_with_beam(x, float(cell), *[float(v) for v in rb]).astype(np.float32)
        np.save(os.path.join(res_dir, "restored.npy"), restored)
        restored_beam = [float(v) for v in rb]

    if not frame_iters or frame_iters[-1] != n_it:
        frames = list(frames) + [(x.sum(axis=0) * dv * sb[0]).astype(np.float32)]
        frame_iters = list(frame_iters) + [n_it]
    np.save(os.path.join(res_dir, "frames.npy"), np.stack(frames))
    with open(os.path.join(res_dir, "frames.json"), "w") as f:
        json.dump(frame_iters, f)
    with open(os.path.join(res_dir, "history.json"), "w") as f:
        json.dump(history, f, default=float)

    from .render import make_gif
    q.put(("progress", dict(fraction=1.0, message="rendering the evolution GIF")))
    make_gif(np.stack(frames), frame_iters, os.path.join(res_dir, "evolution.gif"), unit=sb[1], cell=meta.get("cell_arcsec"))
    q.put(("log", f"[done] {n_it} iterations on {device}, model flux {float(x.sum()):.3f} Jy"))
    return dict(iterations=n_it, flux=float(x.sum()), restored_beam=restored_beam, device=device, frame_unit=sb[1])


def parse_channels(spec, nz):
    """'0-9, 80-89' -> [0..9, 80..89] (clipped to the cube)."""
    out = []
    for part in str(spec).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted({c for c in out if 0 <= c < nz})


# ======================================================================
# Devices
# ======================================================================
def devices_job(q=None, cancel=None):
    """What this machine offers: always the CPU, plus every CUDA GPU visible to
    this process (respects CUDA_VISIBLE_DEVICES / the SLURM allocation) and
    Apple's Metal GPU (MPS) when PyTorch is built with it."""
    out = [dict(id="cpu", name=f"CPU ({os.cpu_count()} cores)", kind="cpu")]
    try:
        import torch
    except ImportError:
        return dict(devices=out, torch=None)
    for i in range(torch.cuda.device_count()):
        pr = torch.cuda.get_device_properties(i)
        out.append(dict(id=f"cuda:{i}", name=pr.name, mem_gb=round(pr.total_memory / 1e9, 1), kind="cuda"))
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        out.append(dict(id="mps", name="Apple GPU (Metal, MPS)", kind="mps"))
    return dict(devices=out, torch=torch.__version__)


class _CPUGeneratorTorch:
    """Stand-in for the `torch` module inside gpu_pd when running on MPS:
    random numbers are drawn with a CPU generator (seeded, reproducible on
    any device) and moved to the device. Everything else is plain torch."""

    def __init__(self, torch):
        self._t = torch

    def __getattr__(self, name):
        return getattr(self._t, name)

    def Generator(self, device=None):
        return self._t.Generator("cpu")

    def randn(self, *size, device=None, dtype=None, generator=None, **kw):
        if len(size) == 1 and isinstance(size[0], (tuple, list, self._t.Size)):
            size = tuple(size[0])
        a = self._t.randn(size, dtype=dtype or self._t.float32, generator=generator, **kw)
        return a.to(device) if device is not None else a


def _use_device(G, torch, device, q):
    """Point gpu_pd at `device` (it picks cuda-or-cpu by itself at import)."""
    dev = torch.device(device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"{device} requested but CUDA is not available to this process")
        torch.cuda.set_device(dev)
    if dev.type == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("MPS requested but this PyTorch build / macOS does not provide it")
    G.DEV = dev
    if dev.type == "mps":
        _apply_mps_compat(G, torch, dev)
    q.put(("log", f"[setup] gpu_pd on {dev}" + (f" ({torch.cuda.get_device_name(dev)})" if dev.type == "cuda" else "")))


def _apply_mps_compat(G, torch, dev):
    """Metal has no float64: build the PSF transfer function in float64 on the
    CPU (as gpu_pd does on CUDA) and move the complex64 result; draw random
    numbers on the CPU (see _CPUGeneratorTorch). Only gpu_pd's module state in
    this worker process is changed; the file on disk is not touched."""
    G.torch = _CPUGeneratorTorch(torch)
    Base = G.Op

    class OpMPS(Base):
        def __init__(self, psf, dirty):
            nz, self.ny, self.nx = dirty.shape
            psf = np.asarray(psf, dtype=np.float64)
            psf = psf / psf.max(axis=(1, 2), keepdims=True)
            py, px = np.unravel_index(np.argmax(psf[0]), psf[0].shape)
            self.gy, self.gx = max(psf.shape[1], 2 * self.ny), max(psf.shape[2], 2 * self.nx)
            big = np.zeros((nz, self.gy, self.gx))
            big[:, :psf.shape[1], :psf.shape[2]] = psf
            big = np.roll(big, (-py, -px), axis=(1, 2))
            otf = torch.fft.rfft2(torch.tensor(big, dtype=torch.float64))
            self.otf = otf.to(torch.complex64).to(dev)
            self.d = G.to_t(dirty)
            self.bound = float(self.otf.abs().max())

    G.Op = OpMPS


def _auto_scales(ny, nx, nz, J=None, L=None):
    """J: deepest starlet whose reflect padding (2**J px) still fits the image;
    L: deepest CDF 9/7 level that divides the channel count (gpu_pd packs the
    bands without padding, so nz must be a multiple of 2**L)."""
    jmax = max(1, int(np.floor(np.log2(min(ny, nx)))) - 2)
    J = min(int(J), jmax) if J else min(7, jmax)
    lmax = 0
    while lmax < 6 and nz % (2 ** (lmax + 1)) == 0:
        lmax += 1
    L = min(int(L), lmax) if L not in (None, "") else min(5, lmax)
    return J, L


class _Converged(Exception):
    pass


def torch_pd_job(ds_dir, res_dir, p, meta, q, cancel, device):
    return torch_job(ds_dir, res_dir, p, meta, q, cancel, device, algo="pd")


def torch_job(ds_dir, res_dir, p, meta, q, cancel, device, algo="pd"):
    """PyTorch solvers on CPU / CUDA / MPS, both on gpu_pd's building blocks
    (operator, 2D-1D dictionary, noise lambdas, step size):

      algo="pd"     gpu_pd.solve_pd (Condat-Vu primal-dual), as scripts/make_pd_evolution_gif.py
      algo="fista"  x = max(0, W^-1 soft(W(z - tau grad), tau lambda)) with FISTA momentum,
                    the loop of scripts/make_fista_run.py (reweighting: same convention)
    """
    import torch
    import gpu_pd as G
    from uv_to_image import ShiftInvariantOperator

    _use_device(G, torch, device, q)
    q.put(("progress", dict(fraction=None, message=f"loading cubes onto {device}")))
    dirty = np.load(os.path.join(ds_dir, "dirty.npy")).astype(np.float64)
    psf = np.load(os.path.join(ds_dir, "psf.npy")).astype(np.float64)
    b = int(p.get("rebin") or 1)
    d_s, psf_s = G.rebin(dirty, psf, b) if b > 1 else (dirty, psf)
    nz, ny, nx = d_s.shape
    J, L = _auto_scales(ny, nx, nz, p.get("num_scales_2d"), p.get("num_levels_1d"))
    if p.get("num_levels_1d") not in (None, "") and int(p["num_levels_1d"]) != L:
        q.put(("log", f"[warn] {nz} channels allow at most {L} CDF 9/7 levels (nz must be a multiple of 2**L)"))
    q.put(("log", f"[setup] {algo} on {device}; cube {d_s.shape}" + (f" after {b}x{b} rebinning" if b > 1 else "") +
           f", {J} spatial scales x {L} CDF 9/7 levels"))

    op = G.Op(psf_s, d_s)
    D = G.Dict2D1D(d_s.shape, J, L, dict_2d=p.get("dict_2d") or None)
    q.put(("log", f"[setup] 2D dictionary {D.dict_2d}: {len(D.plane_scale)} spatial planes"
                  + ("" if len(D.plane_scale) == J + 1 else f" (vs {J + 1} for the starlet: memory and time scale with it)")))
    k = float(p.get("k_sigma", 3.0))
    noise = parse_channels(p["noise_channels"], nz) if p.get("noise_channels") else []
    if not noise and not p.get("null_test"):
        noise = _auto_line_free(G, op.d, J)
        if noise:
            q.put(("log", f"[setup] no line-free channels given: detected {_fmt_channels(noise)} "
                          f"(channels whose large-scale structure is not above the outer channels' level)"))
    q.put(("progress", dict(fraction=None, message="noise model and thresholds")))
    if p.get("null_test"):
        if len(noise) < 2:
            raise ValueError("the null test needs line-free channels (to measure the noise power spectrum)")
        ps = G.measured_ps(G.noise_planes(op.d, noise))
        if b > 1:
            raise ValueError("run the null test without rebinning")
        op.d = G.noise_with_ps(ps, tuple(op.d.shape), seed=int(p.get("null_seed") or 12345))
        dirty = op.d.float().cpu().numpy().astype(np.float64)       # residual / dirty panels refer to the noise cube
        q.put(("log", "[null test] the dirty cube is REPLACED by a pure-noise realization with the measured "
                      "power spectrum: the model should come out (close to) zero"))
    calib = p.get("lambda_calibration") or "noise"
    if len(noise) >= 2:
        lam, _, _, s_pix = G.noise_lambdas(D, op.d, noise, k=k)
        lam_noise = lam.clone()   # k x the per sub-band noise, before any later change (for the residual support)
        q.put(("log", f"[setup] lambda = {k} x max(real, white) noise MAD per sub-band, from {len(noise)} line-free channels; pixel sigma {s_pix:.3e}"))
        if calib == "dual":
            lam = _dual_lambdas(G, D, op, noise, k, lam, q)
    else:
        s_pix = 1.4826 * float((op.d - op.d.median()).abs().median())
        white = G.torch.randn(tuple(op.d.shape), device=G.DEV, dtype=G.DT,
                              generator=G.torch.Generator(G.DEV).manual_seed(101)) * s_pix
        lam = k * D.band_mad(D.W(white))
        q.put(("log", "[warn] no line-free channels given: thresholds from WHITE noise only, which "
                      "underestimates the correlated interferometric noise (set line-free channels)"))
    if p.get("coarse_lambda_scale") not in (None, ""):   # optional: scale the coarsest spatial plane's thresholds
        lam = lam.clone()
        lam[-1] = lam[-1] * float(p["coarse_lambda_scale"])
        q.put(("log", f"[setup] coarse-plane thresholds x {float(p['coarse_lambda_scale']):g}"))
    beta = G.field_lipschitz(op)

    n_iter = int(p.get("n_iter") or 5000)
    tol = float(p.get("tol") or 0.0)
    reweight = None
    if p.get("reweight"):
        reweight = dict(start=int(p.get("burn_in_iters") or n_iter // 3), every=int(p.get("reweight_every") or 1000),
                        eps=float(p.get("reweight_eps") or 1.0),
                        # a scaled (e.g. zero) coarse threshold is a deliberate choice: reweighting must not change it
                        # (and dividing by a zero threshold would give 0/0)
                        keep_coarse=bool(p.get("reweight_keep_coarse")) or p.get("coarse_lambda_scale") not in (None, ""))
        if algo == "pd" and (p.get("reweight_ref") or "tau_lambda") == "tau_lambda":
            # pdtau: compare |W x| with the threshold in MODEL units, tau * lambda (FISTA's T0); with the default
            # reference lambda (gradient units) the ratio is beta ~ 1e3 times too small and reweighting does nothing
            reweight["ref"] = lam / beta
        q.put(("log", f"[setup] reweighting every {reweight['every']} iterations from {reweight['start']}, eps {reweight['eps']:g}"
                      + (", reference tau*lambda (pdtau)" if "ref" in reweight else ", reference lambda" if algo == "pd" else "")))
    marks = _frame_marks(n_iter, p.get("frame_every"))
    if p.get("frame_every") and len(marks) < n_iter // max(1, int(p["frame_every"])):
        q.put(("log", f"[warn] snapshot every {p['frame_every']} would keep {n_iter // int(p['frame_every'])} full maps in "
                      f"memory: keeping at most {MAX_SNAPSHOTS} (every {-(-n_iter // MAX_SNAPSHOTS)}); the live view still "
                      f"shows every iteration"))
    sb, sb_unit = _surface_brightness(meta)
    stat_every = max(1, n_iter // 400)
    hist_every = max(1, n_iter // 200)
    dv = abs(float(meta.get("width_kms") or 1.0))
    frames, frame_iters, history = [], [], []
    st = dict(last=0.0, prev=None, x=None, it=0)
    t0 = time.time()
    dmax = float(op.d.max())

    def cb(it, x, g):
        if cancel.is_set():
            raise Cancelled()
        st["x"], st["it"] = x, it
        m0 = (x.sum(0) * (dv * sb)).float().cpu().numpy()     # moment 0, surface brightness
        if b > 1:
            m0 = G.upsample(m0[None], b)[0]
        m0 = m0.astype(np.float32)
        el = time.time() - t0
        # live display, every iteration; the progress travels with the frame so the two always agree
        q.put(("frame", (it, m0, dict(fraction=it / n_iter, message=f"iteration {it}/{n_iter} on {device}  "
                                                                    f"({el:.0f} s, ~{el / max(it, 1) * (n_iter - it):.0f} s left)"))))
        if it in marks:
            frames.append(m0)
            frame_iters.append(it)
        if it % stat_every == 0:
            q.put(("stat", dict(iter=it, residual_rms=float(g.std()), flux=float(x.sum()))))
        # relative change per iteration, measured between iterations it-1 and it
        if it % hist_every == 0 and st["prev"] is not None:
            rel = float((x - st["prev"]).norm() / max(float(x.norm()), 1e-30))
            history.append(dict(iter=it, residual_rms=float(g.std()), flux=float(x.sum()),
                                peak_ratio=float((op.d + g).max()) / dmax, rel_change=rel))
            if it % max(hist_every, 10) == 0:
                q.put(("log", f"iter {it:6d}  flux {float(x.sum()):9.3f}  change {rel:.2e}  resid rms {float(g.std()):.3e}"))
            if tol and it > 50 and rel < tol:
                raise _Converged()
        if (it + 1) % hist_every == 0:
            st["prev"] = x.clone()

    # optional: the model's dirty-beam image must also reproduce the dirty cube's integral over the field, per channel
    # (weight kappa: mu = kappa beta / max |N 1|^2, so kappa = 1 weighs it like the strongest mode of N)
    sup = None   # optional emission mask (.npy, the cube's grid): the model is zero outside it
    if algo == "pd" and p.get("support_npy"):
        m_ = np.load(p["support_npy"]).astype(bool)
        if b > 1:   # a solve-grid pixel is inside if any of its b x b pixels is
            m_ = m_.reshape(m_.shape[0], m_.shape[1] // b, b, m_.shape[2] // b, b).any(axis=(2, 4))
        sup = torch.tensor(m_, device=G.DEV, dtype=G.DT)
        q.put(("log", f"[setup] support mask {os.path.basename(p['support_npy'])}: {100 * m_.mean():.1f}% of the cube"))
    mrs = None   # optional multiresolution support on the residual (Starck): "residual" (every iteration) or "fixed" (dirty)
    if algo == "pd" and p.get("mrs_mode") in ("residual", "fixed"):
        kk = float(p.get("mrs_k") or 3.0)
        sigma_b = lam_noise / k                                    # per sub-band noise, from the noise model
        mrs = dict(sigma=sigma_b, k=kk)
        if p["mrs_mode"] == "fixed":
            mrs["fixed"] = (D.W(op.d).abs() >= kk * sigma_b).to(G.DT)
            q.put(("log", f"[setup] multiresolution support (fixed, from the dirty cube) at {kk:g} sigma: "
                          f"{100 * float(mrs['fixed'].mean()):.1f}% of the coefficients"))
        else:
            q.put(("log", f"[setup] multiresolution support from the residual, every iteration, at {kk:g} sigma"))
    fm, beta_pd = None, beta
    if algo == "pd" and p.get("field_integral_weight") not in (None, "", 0, "0"):
        a1 = op.apply(torch.ones_like(op.d))
        an = float((a1 ** 2).sum(dim=(1, 2)).max())
        fm = dict(a=a1, mu=float(p["field_integral_weight"]) * beta / an)
        beta_pd = beta + fm["mu"] * an
        q.put(("log", f"[setup] field-integral constraint, weight {float(p['field_integral_weight']):g} (step {beta / beta_pd:.3f} x the usual)"))
    q.put(("progress", dict(fraction=0.0, message=f"iterating on {device}")))
    try:
        if algo == "pd":
            x, _, _, _ = G.solve_pd(op, D, lam, n_iter=n_iter, beta=beta_pd, reweight=reweight, print_every=10 ** 9,
                                    log=lambda m: None, callback=cb, flux_match=fm, support=sup, mrs=mrs)
        else:
            x = _torch_fista(G, torch, op, D, lam, beta, n_iter, reweight, cb)
        n_done = n_iter
    except _Converged:
        x, n_done = st["x"], st["it"]
        q.put(("log", f"converged at iteration {n_done}: relative change < {tol:g}"))
    x = x.float().cpu().numpy().astype(np.float64)
    if b > 1:
        x = G.upsample(x, b)
    fast = ShiftInvariantOperator(psf, dirty)
    return _finalize(x.astype(np.float32), history, frames, frame_iters, n_done, dirty, fast, meta,
                     dict(p), dv, res_dir, q, device=device, sb=(sb, sb_unit))


def _auto_line_free(G, d, J):
    """Line-free channels found from the dirty cube alone (scripts/experiments/common.auto_line_channels):
    per-channel std of the large-scale starlet planes, continuum (median of the outer 3 + 3 channels)
    removed; a channel is 'line' if > 3x the median of the outer channels' values, padded by 1 channel.
    Returns [] when the cube is too short or nothing is left."""
    nz = d.shape[0]
    if nz < 10:
        return []
    outer = [0, 1, 2, nz - 3, nz - 2, nz - 1]
    dd = d - d[outer].median(0).values
    j0 = min(4, J - 1)
    pl = G.starlet(dd, J)
    s = np.array([float(pl[j0:J, c].std()) for c in range(nz)])
    ref = np.median(s[outer])
    line = np.where(s > 3 * ref)[0]
    line = {int(c) for i in line for c in (i - 1, i, i + 1) if 0 <= c < nz}
    free = [c for c in range(nz) if c not in line]
    return free if len(free) >= 2 else []


def _fmt_channels(ch):
    """[0, 1, 2, 24, 25] -> '0-2, 24-25'"""
    out, start = [], ch[0]
    for a, b in zip(ch, ch[1:] + [None]):
        if b != a + 1:
            out.append(f"{start}-{a}" if a > start else f"{a}")
            start = b
    return ", ".join(out)


def _dual_lambdas(G, D, op, noise, k, lam_noise, q, seed=7, max_cg=300):
    """lambda_b from the DUAL certificate of pure noise instead of the noise's own
    coefficients. The primal-dual optimum satisfies d - N x = W^T u with |u| <= lambda,
    so x = 0 is the answer for a noise-only cube n exactly when n = W^T u has a solution
    with |u| <= lambda. For the redundant 2D-1D frame W^T W is not the identity, and the
    coefficients u of such a certificate are larger (and distributed differently over the
    sub-bands) than those of W n. Here u = W (W^T W)^-1 n for a noise realization n with
    the measured power spectrum (conjugate gradient), and lambda_b = k x max(MAD_b(u),
    MAD_b(W n)), per spatial scale and spectral band."""
    torch = G.torch
    ps = G.measured_ps(G.noise_planes(op.d, noise))
    n = G.noise_with_ps(ps, tuple(op.d.shape), seed=seed)
    A = lambda v: D.WT(D.W(v))
    x = torch.zeros_like(n)
    r = n.clone()
    pdir = r.clone()
    rs = float((r * r).sum())
    nn = float((n * n).sum())
    for i in range(max_cg):
        Ap = A(pdir)
        a = rs / float((pdir * Ap).sum())
        x += a * pdir
        r -= a * Ap
        rn = float((r * r).sum())
        if rn < 1e-10 * nn:
            break
        pdir = r + (rn / rs) * pdir
        rs = rn
    lam = torch.maximum(k * D.band_mad(D.W(x)), lam_noise)
    sc = torch.tensor(D.plane_scale, device=lam.device)
    ratio = [round(float((lam[sc == j] / lam_noise[sc == j]).median()), 2) for j in range(D.J + 1)]
    q.put(("log", f"[setup] dual-certificate calibration ({i + 1} CG steps, residual {np.sqrt(rn / nn):.1e}): "
                  f"lambda / noise-coefficient lambda per spatial scale (median over bands) = {ratio}"))
    return lam


def _torch_fista(G, torch, op, D, lam, beta, n_iter, reweight, cb):
    """FISTA on the 2D-1D dictionary, exactly the loop of scripts/make_fista_run.py:
    threshold T0 = tau * lambda (the same lambda_b as the primal-dual solve),
    x = max(0, W^-1 soft(W(z - tau grad(z)), T)), momentum on z; reweighting
    T = T0 / (|W x| / T0 + eps) at the scheduled iterations."""
    tau = 1.0 / beta
    T0 = tau * lam
    T = T0
    soft = lambda c, t: torch.sign(c) * torch.clamp(c.abs() - t, min=0)
    x = torch.zeros_like(op.d)
    z = x.clone()
    t = 1.0
    for it in range(1, n_iter + 1):
        if reweight and it > reweight["start"] and (it - 1 - reweight["start"]) % reweight["every"] == 0:
            T = T0 / (D.W(x).abs() / T0 + reweight["eps"])
        g = op.grad(z)
        xn = D.Winv(soft(D.W(z - tau * g), T)).clamp_(min=0)
        tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        z = xn + ((t - 1) / tn) * (xn - x)
        x, t = xn, tn
        cb(it, x, g)
    return x
